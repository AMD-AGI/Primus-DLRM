#!/usr/bin/env python3
"""Analyze Chrome traces produced by torch.profiler.

Parses the trace JSON exported by torch.profiler.export_chrome_trace() and
derives per-step phase breakdowns using GPU kernel time attributed to each
phase via correlation IDs (linking CPU launch to GPU execution).

Usage:
    python scripts/analyze_trace.py results/_trace_chrome/trace/trace_0.json
    python scripts/analyze_trace.py trace_1gpu.json trace_2gpu.json --labels 1GPU 2GPU
"""
import argparse
import json
from pathlib import Path


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("traceEvents", data)


def _build_indices(events: list[dict]):
    """Pre-build event indices by category."""
    ua, cpu_ops, kernels, runtimes = [], [], [], []
    for e in events:
        cat = e.get("cat")
        if e.get("ph") != "X":
            continue
        if cat == "user_annotation":
            ua.append(e)
        elif cat == "cpu_op":
            cpu_ops.append(e)
        elif cat == "kernel":
            kernels.append(e)
        elif cat == "cuda_runtime":
            runtimes.append(e)

    corr_to_kernels: dict[int, list[dict]] = {}
    for k in kernels:
        cid = k.get("args", {}).get("correlation")
        if cid is not None:
            corr_to_kernels.setdefault(cid, []).append(k)

    corr_to_runtime: dict[int, dict] = {}
    for r in runtimes:
        cid = r.get("args", {}).get("correlation")
        if cid is not None:
            corr_to_runtime[cid] = r

    return ua, cpu_ops, corr_to_kernels, corr_to_runtime


def extract_steps(ua: list[dict]) -> dict[int, dict]:
    steps: dict[int, dict] = {}
    for e in ua:
        if e["name"].startswith("ProfilerStep#"):
            n = int(e["name"].split("#")[1])
            steps[n] = {"ts": e["ts"], "dur": e["dur"]}
    return steps


PHASE_ORDER = ["dataloader", "forward", "backward", "optimizer", "other", "total"]


def compute_gpu_phases(
    steps: dict[int, dict],
    ua: list[dict],
    cpu_ops: list[dict],
    corr_to_kernels: dict[int, list[dict]],
    corr_to_runtime: dict[int, dict],
) -> list[dict]:
    """Compute GPU kernel time per phase for each step.

    Attributes each GPU kernel to the phase (dataloader/forward/backward/
    optimizer) where it was launched from CPU, using correlation IDs.
    """
    step_anns: dict[int, list[dict]] = {}
    for e in ua:
        if e["name"].startswith("ProfilerStep#"):
            continue
        for sn, sd in steps.items():
            if sd["ts"] <= e["ts"] <= sd["ts"] + sd["dur"]:
                step_anns.setdefault(sn, []).append(e)
                break

    bwd_ops = [e for e in cpu_ops if "Backward" in e.get("name", "")]

    records = []
    for sn in sorted(steps):
        sd = steps[sn]
        s_ts = sd["ts"]
        s_end = s_ts + sd["dur"]

        anns = sorted(step_anns.get(sn, []), key=lambda e: e["ts"])
        zero_grad = next((a for a in anns if "zero_grad" in a["name"]), None)
        opt_step = next((a for a in anns if "Optimizer.step" in a["name"]), None)

        zg_end = (zero_grad["ts"] + zero_grad.get("dur", 0)) if zero_grad else s_ts
        opt_ts = opt_step["ts"] if opt_step else s_end
        opt_end = (opt_step["ts"] + opt_step.get("dur", 0)) if opt_step else s_end

        step_bwd = [e for e in bwd_ops if s_ts <= e["ts"] <= s_end]
        bwd_start = min(e["ts"] for e in step_bwd) if step_bwd else opt_ts

        phase_gpu = {p: 0.0 for p in PHASE_ORDER[:-1]}
        for cid, kerns in corr_to_kernels.items():
            rt = corr_to_runtime.get(cid)
            if rt is None:
                continue
            launch_ts = rt["ts"]
            if not (s_ts <= launch_ts <= s_end):
                continue

            if launch_ts < zg_end:
                phase = "dataloader"
            elif launch_ts < bwd_start:
                phase = "forward"
            elif launch_ts < opt_ts:
                phase = "backward"
            elif launch_ts < opt_end:
                phase = "optimizer"
            else:
                phase = "other"

            for k in kerns:
                phase_gpu[phase] += k.get("dur", 0) / 1000

        phase_gpu["total"] = sum(phase_gpu.values())
        phase_gpu["step"] = sn
        records.append(phase_gpu)

    return records


def stats(values: list[float]) -> dict:
    v = sorted(values)
    n = len(v)
    mean = sum(v) / n
    return {
        "mean": mean,
        "std": (sum((x - mean) ** 2 for x in v) / n) ** 0.5,
        "min": v[0],
        "p50": v[n // 2],
        "p90": v[int(n * 0.9)],
        "p99": v[int(n * 0.99)],
        "max": v[-1],
    }


def print_summary(records: list[dict], label: str, pcts: list[int]):
    header = f"{'phase':>12s}  {'mean':>8s}  {'std':>7s}  {'min':>7s}"
    for p in pcts:
        header += f"  {'p' + str(p):>7s}"
    header += f"  {'max':>7s}  {'%':>5s}"

    print(f"\n=== {label} ({len(records)} steps) ===")
    print(header)
    print("-" * len(header))

    total_mean = stats([r["total"] for r in records])["mean"]

    for col in PHASE_ORDER:
        vals = [r[col] for r in records if col in r]
        if not vals:
            continue
        s = stats(vals)
        pct = s["mean"] / total_mean * 100 if col != "total" else 100.0
        row = f"{col:>12s}  {s['mean']:8.2f}  {s['std']:7.2f}  {s['min']:7.2f}"
        for p in pcts:
            key = f"p{p}"
            row += f"  {s[key]:7.2f}"
        row += f"  {s['max']:7.2f}  {pct:5.1f}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Analyze torch.profiler Chrome traces")
    parser.add_argument("traces", nargs="+", help="Chrome trace JSON files")
    parser.add_argument("--labels", nargs="*", help="Labels for each trace")
    parser.add_argument("--percentiles", nargs="*", type=int, default=[50, 90, 99])
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip first N profiled steps")
    args = parser.parse_args()

    labels = args.labels or [Path(t).parent.name or Path(t).stem for t in args.traces]
    if len(labels) < len(args.traces):
        labels += [Path(t).stem for t in args.traces[len(labels):]]

    all_records = []
    for trace_path, label in zip(args.traces, labels):
        events = load_events(trace_path)
        ua, cpu_ops, corr_to_kernels, corr_to_runtime = _build_indices(events)
        steps = extract_steps(ua)
        records = compute_gpu_phases(steps, ua, cpu_ops, corr_to_kernels, corr_to_runtime)
        if args.skip > 0:
            records = records[args.skip:]
        if not records:
            print(f"\n=== {label}: no steps found ===")
            all_records.append([])
            continue
        print_summary(records, label, args.percentiles)
        all_records.append(records)

    if len(args.traces) > 1:
        print(f"\n=== Comparison (mean GPU ms) ===")
        header = f"{'phase':>12s}" + "".join(f"  {l:>10s}" for l in labels)
        print(header)
        print("-" * len(header))
        for col in PHASE_ORDER:
            row = f"{col:>12s}"
            for recs in all_records:
                vals = [r[col] for r in recs if col in r]
                if vals:
                    row += f"  {sum(vals)/len(vals):10.2f}"
                else:
                    row += f"  {'—':>10s}"
            print(row)
    print()


if __name__ == "__main__":
    main()
