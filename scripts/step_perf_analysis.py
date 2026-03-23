"""Per-step GPU performance analysis with MFU/MBU.

Uses TraceLens CSVs for sections 1, 3, 4 (GPU timeline, op groups, GEMM roofline).
Uses raw torch.profiler trace for section 2 (per-step phase attribution).

MI355X peak: 2516.6 bf16 matrix TFLOPS/s, 8 TB/s HBM BW.
"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import pandas as pd

PEAK_BF16_TFLOPS = 2516.6
PEAK_HBM_TBS = 8.0
NUM_PROFILED_STEPS = 10  # warmup=5, active=10


# ─── Section 1 & 3 & 4: from TraceLens CSVs ───────────────────────────────────

def load_tracelens(tl_dir: str) -> dict:
    """Load all relevant TraceLens CSVs from a directory."""
    p = Path(tl_dir)
    data = {}
    for name in ("gpu_timeline", "ops_summary_by_category", "ops_summary",
                  "GEMM", "BinaryElementwise", "UnaryElementwise",
                  "unified_perf_summary", "coll_analysis"):
        f = p / f"{name}.csv"
        if f.exists():
            data[name] = pd.read_csv(f)
    return data


def section1_gpu_timeline(tl: dict, n_steps: int) -> dict:
    """Section 1: GPU time breakdown from TraceLens gpu_timeline.csv."""
    df = tl["gpu_timeline"]
    get = lambda t: float(df[df["type"] == t]["time ms"].values[0]) / n_steps

    total = get("total_time")
    compute = get("computation_time")
    exp_comm = get("exposed_comm_time")
    exp_memcpy = get("exposed_memcpy_time")
    idle = get("idle_time")
    total_comm = get("total_comm_time")
    total_memcpy = get("total_memcpy_time")

    return {
        "total": total, "compute": compute, "exp_comm": exp_comm,
        "exp_memcpy": exp_memcpy, "idle": idle,
        "total_comm": total_comm, "total_memcpy": total_memcpy,
    }


def _cat_agg(df: pd.DataFrame, n_steps: int) -> tuple[float, float, float]:
    """Aggregate (kernel_ms/step, total_gflops/step, total_gb/step) from a TraceLens category CSV."""
    kms = df["Kernel Time (µs)_sum"].sum() / 1000 / n_steps
    gflops = (df["GFLOPS_first"] * df["name_count"]).sum() / n_steps
    mb = (df["Data Moved (MB)_first"] * df["name_count"]).sum() / n_steps
    return kms, gflops, mb / 1024


def section3_op_groups(tl: dict, n_steps: int) -> list[dict]:
    """Section 3: Op group breakdown.

    Kernel time: from ops_summary (name-based classification).
    FLOPS/Bytes: from TraceLens category CSVs (GEMM.csv, BinaryElementwise.csv, etc.).
    """
    ops = tl.get("ops_summary")
    gemm_df = tl.get("GEMM")
    bin_ew = tl.get("BinaryElementwise")
    un_ew = tl.get("UnaryElementwise")
    coll = tl.get("coll_analysis")

    # Kernel time from ops_summary (authoritative)
    emb_kms = attn_kms = gemm_kms = elem_kms = opt_kms = other_kms = 0.0
    if ops is not None:
        for _, r in ops.iterrows():
            name = r["name"].lower()
            cat = str(r.get("Categories", ""))
            kms = r["total_direct_kernel_time_ms"] / n_steps
            if "embedding" in name:
                emb_kms += kms
            elif "attention" in name or "sdpa" in name or "scaled_dot" in name:
                attn_kms += kms
            elif "GEMM" in cat or name in ("aten::mm", "aten::bmm", "aten::addmm", "aten::linear"):
                gemm_kms += kms
            elif "multi_tensor_apply" in cat or "foreach" in name or "_fused_adam" in name:
                opt_kms += kms
            elif "elementwise" in cat or "reduce" in cat or name in (
                    "aten::native_dropout", "aten::cat", "aten::copy_",
                    "aten::gather", "aten::index_select"):
                elem_kms += kms
            else:
                other_kms += kms

    # FLOPS/Bytes from TraceLens category CSVs
    gemm_gflops, gemm_gb = 0.0, 0.0
    if gemm_df is not None and len(gemm_df) > 0:
        _, gemm_gflops, gemm_gb = _cat_agg(gemm_df, n_steps)

    elem_gflops, elem_gb = 0.0, 0.0
    for ew_df in (bin_ew, un_ew):
        if ew_df is not None and len(ew_df) > 0:
            _, g, b = _cat_agg(ew_df, n_steps)
            elem_gflops += g
            elem_gb += b

    # Comm from coll_analysis
    comm_kms = 0.0
    if coll is not None and len(coll) > 0:
        comm_kms = coll["dur_sum"].sum() / n_steps / 1000

    def mkrow(name, kms, gflops=0.0, gb=0.0):
        tflops = gflops / 1e3 / (kms / 1e3) if kms > 0 and gflops > 0 else 0
        mfu = tflops / PEAK_BF16_TFLOPS * 100 if tflops > 0 else 0
        tbs = (gb / 1024) / (kms / 1e3) if kms > 0 and gb > 0 else 0
        mbu = tbs / PEAK_HBM_TBS * 100 if tbs > 0 else 0
        return {"name": name, "kms": kms, "gflops": gflops, "gb": gb,
                "tflops": tflops, "mfu": mfu, "tbs": tbs, "mbu": mbu}

    return [
        mkrow("Embedding lookup", emb_kms),
        mkrow("GEMM (mm/bmm/addmm)", gemm_kms, gemm_gflops, gemm_gb),
        mkrow("Attention (SDPA)", attn_kms),
        mkrow("Elementwise+Reduce", elem_kms, elem_gflops, elem_gb),
        mkrow("Optimizer kernels", opt_kms),
        mkrow("Communication", comm_kms),
        mkrow("Other compute", other_kms),
    ]


# ─── Section 2: per-step phase attribution from raw trace ──────────────────────

def _is_comm_kernel(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("nccl", "rccl", "msccl", "allreduce", "all_to_all",
                                 "allgather", "reducescatter", "ncclkernel", "mscclkernel"))


def section2_phases(trace_path: str) -> list[dict]:
    """Per-step phase breakdown from raw trace using correlation IDs."""
    with open(trace_path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])

    # Find main thread
    main_tid = None
    for e in events:
        if (e.get("cat") == "user_annotation" and e.get("name", "").startswith("ProfilerStep")):
            main_tid = e.get("tid")
            break

    profiler_steps = sorted(
        [e for e in events
         if e.get("name", "").startswith("ProfilerStep") and "dur" in e and e.get("tid") == main_tid],
        key=lambda e: e["ts"],
    )

    # Build cpu_op index for parent lookup
    all_cpu_ops = [e for e in events if e.get("cat") == "cpu_op" and "dur" in e]
    tid_sorted = {}
    tid_ops_map = defaultdict(list)
    for op in all_cpu_ops:
        tid_ops_map[op.get("tid")].append(op)
    for tid, ops in tid_ops_map.items():
        ops.sort(key=lambda e: e["ts"])
        tid_sorted[tid] = (
            [op["ts"] for op in ops],
            [op["ts"] + op["dur"] for op in ops],
            [op["dur"] for op in ops],
            ops,
        )

    def find_parent(rt):
        tid = rt.get("tid")
        entry = tid_sorted.get(tid)
        if not entry:
            return None
        starts, ends, durs, ops = entry
        rt_ts = rt["ts"]
        idx = bisect_right(starts, rt_ts) - 1
        best, best_dur, nearest = None, float("inf"), None
        for i in range(idx, max(idx - 300, -1), -1):
            if i < 0:
                break
            if starts[i] <= rt_ts and ends[i] >= rt_ts and durs[i] < best_dur:
                best, best_dur = ops[i], durs[i]
            if nearest is None and ends[i] <= rt_ts:
                nearest = ops[i]
            if best and nearest:
                break
        return best if best else nearest

    corr_to_rt = {}
    for e in events:
        if e.get("cat") == "cuda_runtime" and "dur" in e:
            cid = e.get("args", {}).get("correlation")
            if cid is not None:
                corr_to_rt[cid] = e

    gpu_events = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    user_annots = [e for e in events if e.get("cat") == "user_annotation" and "dur" in e]

    results = []
    for step_ev in profiler_steps:
        sid = int(step_ev["name"].split("#")[1])
        s_start, s_end = step_ev["ts"], step_ev["ts"] + step_ev["dur"]

        # Phase boundaries
        sa = [a for a in user_annots
              if a["ts"] >= s_start - 10 and a["ts"] + a["dur"] <= s_end + 10
              and not a["name"].startswith("ProfilerStep") and a.get("tid") == main_tid]
        sa.sort(key=lambda a: a["ts"])

        zg = next((a for a in sa if "zero_grad" in a["name"]), None)
        opt = next((a for a in sa if "Optimizer.step" in a["name"]), None)
        dl = next((a for a in sa if "DataLoader" in a["name"]), None)

        fwd_start = (zg["ts"] + zg["dur"]) if zg else s_start
        opt_start = opt["ts"] if opt else s_end
        opt_end = (opt["ts"] + opt["dur"]) if opt else s_end

        # Find backward start: first autograd thread event
        bwd_start = opt_start
        for op in all_cpu_ops:
            if (op.get("tid") != main_tid and "autograd" in op.get("name", "").lower()
                    and op["ts"] >= fwd_start and op["ts"] < opt_start):
                bwd_start = op["ts"]
                break

        phase_us = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0, "other": 0.0}
        comm_us = 0.0
        memcpy_us = 0.0
        compute_us = 0.0

        for gev in gpu_events:
            cid = gev.get("args", {}).get("correlation")
            rt = corr_to_rt.get(cid) if cid else None
            if not rt or rt["ts"] < s_start or rt["ts"] >= s_end:
                continue

            k_dur = gev["dur"]
            if gev.get("cat") in ("gpu_memcpy", "gpu_memset"):
                memcpy_us += k_dur
                continue
            if _is_comm_kernel(gev.get("name", "")):
                comm_us += k_dur
                continue

            compute_us += k_dur
            launch_ts = rt["ts"]
            if launch_ts < fwd_start:
                phase_us["other"] += k_dur
            elif launch_ts < bwd_start:
                phase_us["fwd"] += k_dur
            elif launch_ts < opt_start:
                phase_us["bwd"] += k_dur
            elif launch_ts < opt_end:
                phase_us["opt"] += k_dur
            else:
                phase_us["other"] += k_dur

        busy = min(step_ev["dur"], compute_us + comm_us + memcpy_us)
        exposed_comm = max(0, busy - compute_us - memcpy_us)
        idle = step_ev["dur"] - busy

        # Dataloader+H2D = CPU wall-clock from step start to zero_grad
        # (includes loss.item() sync from prev step, DataLoader.__next__, .to(device))
        dl_h2d_ms = (fwd_start - s_start) / 1000

        results.append({
            "step": sid, "step_ms": step_ev["dur"] / 1000,
            "fwd_ms": phase_us["fwd"] / 1000, "bwd_ms": phase_us["bwd"] / 1000,
            "opt_ms": phase_us["opt"] / 1000, "other_ms": phase_us["other"] / 1000,
            "compute_ms": compute_us / 1000, "comm_ms": comm_us / 1000,
            "exp_comm_ms": exposed_comm / 1000, "memcpy_ms": memcpy_us / 1000,
            "idle_ms": idle / 1000, "dl_cpu_ms": dl_h2d_ms,
        })

    return results


# ─── Report printer ────────────────────────────────────────────────────────────

def print_report(label: str, s1: dict, s2: list[dict], s3: list[dict], n_steps: int):
    print(f"\n{'='*120}")
    print(f"  {label}  (TraceLens: {n_steps} profiled steps)")
    print(f"{'='*120}")

    # ── Section 1 ──
    t = s1
    print(f"\n  1) PER-STEP GPU TIME BREAKDOWN (from TraceLens gpu_timeline, per-step avg)")
    print(f"  {'─'*95}")
    print(f"  {'Total step':40s}: {t['total']:8.2f} ms  (100.0%)")
    print(f"    {'Compute':38s}: {t['compute']:8.2f} ms  ({t['compute']/t['total']*100:5.1f}%)")
    print(f"    {'Exposed comm':38s}: {t['exp_comm']:8.2f} ms  ({t['exp_comm']/t['total']*100:5.1f}%)")
    print(f"    {'Exposed memcpy':38s}: {t['exp_memcpy']:8.2f} ms  ({t['exp_memcpy']/t['total']*100:5.1f}%)")
    print(f"    {'Idle':38s}: {t['idle']:8.2f} ms  ({t['idle']/t['total']*100:5.1f}%)")
    chk = t['compute'] + t['exp_comm'] + t['exp_memcpy'] + t['idle']
    print(f"    {'[check sum]':38s}: {chk:8.2f} ms")
    if t['total_comm'] > 0:
        hidden = t['total_comm'] - t['exp_comm']
        print(f"    {'Total comm (incl. hidden)':38s}: {t['total_comm']:8.2f} ms  (hidden={hidden:.2f}ms)")

    # ── Section 2 ──
    # Exclude warmup (first 2) and outlier steps (step_ms > 2x median of post-warmup)
    skip = min(2, len(s2) - 1)
    post_warmup = s2[skip:]
    if len(post_warmup) > 2:
        step_times = sorted(s["step_ms"] for s in post_warmup)
        median_ms = step_times[len(step_times) // 2]
        ss = [s for s in post_warmup if s["step_ms"] < median_ms * 2]
    else:
        ss = post_warmup
    if not ss:
        ss = post_warmup
    N = len(ss)
    def avg(key): return sum(s[key] for s in ss) / N

    a_fwd = avg("fwd_ms")
    a_bwd = avg("bwd_ms")
    a_opt = avg("opt_ms")
    a_oth = avg("other_ms")
    a_compute = avg("compute_ms")
    a_comm = avg("comm_ms")
    a_exp_comm = avg("exp_comm_ms")
    a_memcpy = avg("memcpy_ms")
    a_idle = avg("idle_ms")
    a_step = avg("step_ms")
    a_dl = avg("dl_cpu_ms")

    print(f"\n  2) PER-STEP PHASE BREAKDOWN (from raw trace, per-step avg over {N} steady-state steps)")
    print(f"  {'─'*95}")
    print(f"  {'':40s} {'ms':>8s} {'%step':>7s}  {'%compute':>10s}")
    print(f"  {'Dataloader+H2D (CPU wall)':40s}: {a_dl:8.2f} {a_dl/a_step*100:6.1f}%")
    print(f"  {'Forward':40s}: {a_fwd:8.2f} {a_fwd/a_step*100:6.1f}%  {a_fwd/a_compute*100:9.1f}%")
    print(f"  {'Backward':40s}: {a_bwd:8.2f} {a_bwd/a_step*100:6.1f}%  {a_bwd/a_compute*100:9.1f}%")
    print(f"  {'Optimizer':40s}: {a_opt:8.2f} {a_opt/a_step*100:6.1f}%  {a_opt/a_compute*100:9.1f}%")
    if a_oth > 0.01:
        print(f"  {'Other':40s}: {a_oth:8.2f} {a_oth/a_step*100:6.1f}%  {a_oth/a_compute*100:9.1f}%")
    print(f"  {'Exposed comm':40s}: {a_exp_comm:8.2f} {a_exp_comm/a_step*100:6.1f}%")
    print(f"  {'Memcpy/memset':40s}: {a_memcpy:8.2f} {a_memcpy/a_step*100:6.1f}%")
    print(f"  {'Idle':40s}: {a_idle:8.2f} {a_idle/a_step*100:6.1f}%")
    print(f"  {'─'*95}")
    total_chk = a_fwd + a_bwd + a_opt + a_oth + a_exp_comm + a_memcpy + a_idle
    print(f"  {'TOTAL':40s}: {total_chk:8.2f} ms  (step={a_step:.2f}ms)")
    if a_comm > a_exp_comm + 0.01:
        print(f"  {'(Total comm incl. hidden)':40s}: {a_comm:8.2f} ms")

    # Per-step table
    print(f"\n  Per-step detail:")
    print(f"  {'step':>5s} {'step_ms':>8s} {'fwd':>7s} {'bwd':>7s} {'opt':>7s} {'exp_cm':>7s} {'memcpy':>7s} {'idle':>7s} {'total':>8s}")
    for s in s2:
        row_total = s["fwd_ms"] + s["bwd_ms"] + s["opt_ms"] + s["other_ms"] + s["exp_comm_ms"] + s["memcpy_ms"] + s["idle_ms"]
        print(f"  {s['step']:5d} {s['step_ms']:8.2f} {s['fwd_ms']:7.2f} {s['bwd_ms']:7.2f} {s['opt_ms']:7.2f}"
              f" {s['exp_comm_ms']:7.2f} {s['memcpy_ms']:7.2f} {s['idle_ms']:7.2f} {row_total:8.2f}")

    # ── Section 3 ──
    print(f"\n  3) OP GROUP BREAKDOWN (from TraceLens, per-step avg) + MFU / MBU")
    print(f"  {'─'*115}")
    hdr = f"  {'Op Group':22s} {'kernel_ms':>10s} {'%step':>7s} {'GFLOPS':>10s} {'TFLOPS/s':>10s} {'MFU%':>7s} {'GB':>10s} {'TB/s':>8s} {'MBU%':>7s}"
    print(hdr)
    print(f"  {'─'*115}")

    total_gflops = 0.0
    total_gb = 0.0
    compute_kms = 0.0

    for g in s3:
        n, kms, gf, gb = g["name"], g["kms"], g["gflops"], g["gb"]
        total_gflops += gf
        total_gb += gb
        if n != "Communication":
            compute_kms += kms

        pct = kms / t["total"] * 100 if t["total"] > 0 else 0
        def fmt(v, f=".1f"):
            return f"{v:{f}}" if v > 0 else "n/a"
        mfu_s = f"{g['mfu']:.2f}%" if g["mfu"] > 0 else "    n/a"
        mbu_s = f"{g['mbu']:.2f}%" if g["mbu"] > 0 else "    n/a"

        print(f"  {n:22s} {kms:10.2f} {pct:6.1f}%"
              f" {fmt(gf):>10s} {fmt(g['tflops']):>10s} {mfu_s:>7s}"
              f" {fmt(gb, '.2f'):>10s} {fmt(g['tbs'], '.3f'):>8s} {mbu_s:>7s}")

    print(f"  {'─'*115}")
    print(f"  {'Compute ops':22s} {compute_kms:10.2f} {compute_kms/t['total']*100:6.1f}%  (TraceLens compute={t['compute']:.2f}ms)")
    print(f"  {'Exposed comm':22s} {t['exp_comm']:10.2f} {t['exp_comm']/t['total']*100:6.1f}%")
    print(f"  {'Exposed memcpy':22s} {t['exp_memcpy']:10.2f} {t['exp_memcpy']/t['total']*100:6.1f}%")
    print(f"  {'Idle':22s} {t['idle']:10.2f} {t['idle']/t['total']*100:6.1f}%")
    step_chk = compute_kms + t['exp_comm'] + t['exp_memcpy'] + t['idle']
    print(f"  {'STEP TOTAL':22s} {step_chk:10.2f}   (TraceLens total={t['total']:.2f}ms)")

    # ── Section 4 ──
    step_s = t["total"] / 1e3
    compute_s = t["compute"] / 1e3
    e2e_tflops = (total_gflops / 1e3) / step_s if step_s > 0 else 0
    e2e_mfu = e2e_tflops / PEAK_BF16_TFLOPS * 100
    e2e_tbs = (total_gb / 1024) / step_s if step_s > 0 else 0
    e2e_mbu = e2e_tbs / PEAK_HBM_TBS * 100
    c_tflops = (total_gflops / 1e3) / compute_s if compute_s > 0 else 0
    c_mfu = c_tflops / PEAK_BF16_TFLOPS * 100
    c_tbs = (total_gb / 1024) / compute_s if compute_s > 0 else 0
    c_mbu = c_tbs / PEAK_HBM_TBS * 100

    print(f"\n  4) E2E METRICS")
    print(f"  {'─'*95}")
    print(f"  {'Step time (TraceLens)':35s}: {t['total']:8.2f} ms")
    print(f"  {'Total GFLOPS/step':35s}: {total_gflops:8.1f}")
    print(f"  {'Total GB moved/step':35s}: {total_gb:8.2f}")
    print(f"  {'E2E TFLOPS/s':35s}: {e2e_tflops:8.2f}")
    print(f"  {'E2E TB/s':35s}: {e2e_tbs:8.4f}")
    print(f"  {'E2E MFU':35s}: {e2e_mfu:8.4f}%")
    print(f"  {'E2E MBU':35s}: {e2e_mbu:8.4f}%")
    print(f"  {'Compute-only TFLOPS/s':35s}: {c_tflops:8.2f}")
    print(f"  {'Compute-only TB/s':35s}: {c_tbs:8.4f}")
    print(f"  {'Compute-only MFU':35s}: {c_mfu:8.4f}%")
    print(f"  {'Compute-only MBU':35s}: {c_mbu:8.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True, help="Raw trace JSONs")
    parser.add_argument("--tracelens", nargs="+", required=True, help="TraceLens output dirs")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels")
    parser.add_argument("--steps", type=int, default=NUM_PROFILED_STEPS)
    args = parser.parse_args()

    for trace, tl_dir, label in zip(args.traces, args.tracelens, args.labels):
        tl = load_tracelens(tl_dir)
        s1 = section1_gpu_timeline(tl, args.steps)
        s2 = section2_phases(trace)
        s3 = section3_op_groups(tl, args.steps)
        print_report(label, s1, s2, s3, args.steps)


if __name__ == "__main__":
    main()
