"""Attention kernel benchmark for OneTrans pyramid layers.

Benchmarks per-layer attention shapes from the OneTrans pyramid schedule
across multiple backends. Measures FWD and BWD separately. Uses
torch.profiler to extract exact CUDA kernel durations when --fwd-bwd is
used, enabling kernel-to-kernel comparison with training traces.

Usage:
    # FWD-only, all backends, CUDA-event timing
    python bench_attn_kernel.py

    # FWD + BWD with exact kernel extraction via profiler
    python bench_attn_kernel.py --fwd-bwd --backends fav4

    # Compare against a training trace (kernel-to-kernel)
    python bench_attn_kernel.py --fwd-bwd --backends fav4 --batch 1024 \\
        --trace-json results/trace_step52.json

    # Custom model dimensions
    python bench_attn_kernel.py --d-model 256 --n-heads 8 --n-layers 4

    # Uniform (non-pyramid) schedule
    python bench_attn_kernel.py --no-pyramid
"""
import argparse
import json
import os
import tempfile
from collections import defaultdict

import torch
import torch.nn.functional as F

# ---- Backend registry ----

BACKEND_REGISTRY = {}

KERNEL_PATTERNS = {
    "fav4": ("fwd_sm100", "bwd_sm100"),
    "fav2": ("fwd_sm80", "bwd_sm80"),
    "sdpa": ("fmha_v2", "fmha_v2"),
    "turbo": ("flash_fwd", "flash_bwd"),
}


def register_backend(name, fwd_fn, bwd_ok=True):
    BACKEND_REGISTRY[name] = {"fwd": fwd_fn, "bwd_ok": bwd_ok}


def _load_backends(names):
    loaded = {}
    for name in names:
        try:
            if name == "fav2":
                from flash_attn import flash_attn_func
                register_backend("fav2",
                    lambda q, k, v, fn=flash_attn_func: fn(q, k, v, causal=True))
            elif name == "fav4":
                from flash_attn.cute import flash_attn_func as fa4_fn
                register_backend("fav4",
                    lambda q, k, v, fn=fa4_fn: fn(q, k, v, causal=True)[0])
            elif name == "sdpa":
                def sdpa_causal(q, k, v):
                    return F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                        is_causal=True,
                    ).transpose(1, 2)
                register_backend("sdpa", sdpa_causal)
            elif name == "turbo":
                from primus_turbo.pytorch.ops import flash_attn_func as turbo_fn
                register_backend("turbo",
                    lambda q, k, v, fn=turbo_fn: fn(q, k, v, causal=True))
            else:
                print(f"  [WARN] Unknown backend: {name}")
                continue
            loaded[name] = BACKEND_REGISTRY[name]
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
    return loaded


# ---- Pyramid schedule ----

def pyramid_schedule(l_s, l_ns, n_layers):
    if n_layers == 1:
        return [l_ns]
    return [max(int(round(l_s - i * (l_s - l_ns) / (n_layers - 1))), l_ns)
            for i in range(n_layers)]


def compute_layer_shapes(l_s, l_ns, n_layers, use_pyramid):
    if use_pyramid:
        schedule = pyramid_schedule(l_s, l_ns, n_layers)
    else:
        schedule = [l_s] * n_layers
    kv_len = l_s + l_ns
    layers = []
    for i in range(n_layers):
        q_len = schedule[i] + l_ns
        layers.append((q_len, kv_len))
        kv_len = q_len
    return layers


# ---- Benchmark helpers ----

def _timed(fn, warmup, iters):
    """Run fn for warmup+iters iterations, return average ms via CUDA events."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_fwd(attn_fn, q, k, v, warmup, iters):
    return _timed(lambda: attn_fn(q, k, v), warmup, iters)


def _profile_fwd_bwd(attn_fn, q, k, v, grad_out, warmup, iters, fwd_pat, bwd_pat):
    """Run FWD+BWD under torch.profiler and extract exact kernel durations."""
    from torch.profiler import profile, ProfilerActivity

    torch.cuda.synchronize()
    for _ in range(warmup):
        q.grad = k.grad = v.grad = None
        out = attn_fn(q, k, v)
        out.backward(grad_out, retain_graph=False)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            q.grad = k.grad = v.grad = None
            out = attn_fn(q, k, v)
            out.backward(grad_out, retain_graph=False)
        torch.cuda.synchronize()

    tmp = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(tmp)
    with open(tmp) as f:
        trace = json.load(f)
    os.unlink(tmp)

    fwd_us = bwd_us = 0
    for evt in trace.get("traceEvents", []):
        if evt.get("cat") != "kernel" or evt.get("dur", 0) <= 0:
            continue
        name = evt["name"].lower()
        if fwd_pat in name:
            fwd_us += evt["dur"]
        elif bwd_pat in name:
            bwd_us += evt["dur"]

    return fwd_us / 1000 / iters, bwd_us / 1000 / iters


def _load_trace(path, n_layers):
    """Load a training trace and extract per-layer FWD/BWD kernel averages."""
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    tfwd = sorted([e for e in events if "fwd_sm100" in e.get("name", "").lower()
                   and e.get("cat") == "kernel" and e.get("dur", 0) > 0],
                  key=lambda e: e["ts"])
    tbwd = sorted([e for e in events if "bwd_sm100" in e.get("name", "").lower()
                   and e.get("cat") == "kernel" and e.get("dur", 0) > 0],
                  key=lambda e: e["ts"])

    fwd_by_layer = defaultdict(list)
    for i, e in enumerate(tfwd):
        fwd_by_layer[i % n_layers].append(e["dur"] / 1000)

    bwd_by_layer = defaultdict(list)
    for i, e in enumerate(tbwd):
        bwd_by_layer[n_layers - 1 - (i % n_layers)].append(e["dur"] / 1000)

    fwd_avg = {k: sum(v) / len(v) for k, v in fwd_by_layer.items()}
    bwd_avg = {k: sum(v) / len(v) for k, v in bwd_by_layer.items()}
    steps = len(tfwd) // n_layers if n_layers else 0
    return fwd_avg, bwd_avg, steps


def main():
    parser = argparse.ArgumentParser(description="OneTrans attention kernel benchmark")
    parser.add_argument("--backends", nargs="+", default=["fav2", "fav4", "sdpa"],
                        help="Backends to benchmark (fav2, fav4, sdpa, turbo)")
    parser.add_argument("--batch", type=int, default=2048, help="Per-GPU batch size")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=500)
    parser.add_argument("--n-ns", type=int, default=16)
    parser.add_argument("--no-pyramid", action="store_true")
    parser.add_argument("--fwd-bwd", action="store_true",
                        help="Also benchmark BWD (uses profiler for exact kernel times)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--trace-json", type=str, default=None,
                        help="Training trace JSON for kernel-to-kernel comparison")
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    head_dim = args.d_model // args.n_heads
    n_groups = 3
    l_s = args.history_length * n_groups
    l_ns = args.n_ns

    gpu_name = torch.cuda.get_device_name(0)
    gpu_peaks = {"MI355": 2300, "MI350": 2300, "MI300": 1300, "B200": 2250, "H100": 990, "H200": 990}
    gpu_peak = next((v for k, v in gpu_peaks.items() if k in gpu_name), 1000)

    layers = compute_layer_shapes(l_s, l_ns, args.n_layers, not args.no_pyramid)

    # Load training trace if provided
    trace_fwd, trace_bwd, trace_steps = {}, {}, 0
    if args.trace_json:
        trace_fwd, trace_bwd, trace_steps = _load_trace(args.trace_json, args.n_layers)

    print("=" * 90)
    print("OneTrans Attention Kernel Benchmark")
    print(f"  GPU: {gpu_name} | BF16 peak: {gpu_peak:,} TFLOPS")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, head_dim={head_dim}")
    print(f"  history_length={args.history_length}, n_groups={n_groups}, L_S={l_s}, L_NS={l_ns}")
    print(f"  n_layers={args.n_layers}, pyramid={'ON' if not args.no_pyramid else 'OFF'}")
    print(f"  batch={args.batch}, dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    if args.fwd_bwd:
        print(f"  BWD timing: torch.profiler (exact kernel extraction)")
    if trace_fwd:
        print(f"  trace: {args.trace_json} ({trace_steps} steps)")
    print("=" * 90)

    print(f"\nLoading backends: {args.backends}")
    backends = _load_backends(args.backends)
    if not backends:
        print("No backends available!")
        return
    print(f"Active backends: {list(backends.keys())}\n")

    # ---- FWD-only mode (CUDA events, fast) ----
    if not args.fwd_bwd:
        print("--- FWD ---")
        col_w = 12
        header = f"  {'Layer':<7} {'Q':>6} {'KV':>6} {'FLOPs':>8}"
        for name in backends:
            header += f"  {name + ' ms':>{col_w}} {name + ' TF/s':>{col_w}}"
        if trace_fwd:
            header += f"  {'trace ms':>{col_w}} {'trace TF/s':>{col_w}} {'Δ':>6}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        total_ms = {n: 0.0 for n in backends}
        total_trace = 0.0
        total_flops = 0

        for i, (q_len, kv_len) in enumerate(layers):
            flops = 4 * args.batch * args.n_heads * head_dim * q_len * kv_len
            total_flops += flops

            q_t = torch.randn(args.batch, q_len, args.n_heads, head_dim,
                               device=device, dtype=dtype)
            k_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype)
            v_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype)

            row = f"  L{i:<5} {q_len:>6} {kv_len:>6} {flops / 1e9:>7.1f}G"
            for name, be in backends.items():
                try:
                    ms = bench_fwd(be["fwd"], q_t, k_t, v_t, args.warmup, args.iters)
                    tfs = flops / (ms / 1000) / 1e12
                    total_ms[name] += ms
                    row += f"  {ms:>{col_w}.3f} {tfs:>{col_w}.0f}"
                except Exception as e:
                    row += f"  {'ERR':>{col_w}} {'':>{col_w}}"
                    print(f"    [ERR L{i} {name}]: {e}")

            if trace_fwd:
                tf = trace_fwd.get(i, 0)
                total_trace += tf
                tfs_t = flops / (tf / 1000) / 1e12 if tf > 0 else 0
                first_be = list(backends.keys())[0]
                delta = (tf / total_ms.get(first_be, tf) * len(layers) /
                         (i + 1) - 1) * 100 if total_ms.get(first_be) else 0
                bms = total_ms[first_be] - sum(
                    total_ms[first_be] for _ in [])  # placeholder
                row += f"  {tf:>{col_w}.3f} {tfs_t:>{col_w}.0f}"
                bench_ms_i = ms  # last backend
                delta_i = (tf / bench_ms_i - 1) * 100 if bench_ms_i > 0 else 0
                row += f" {delta_i:>+5.1f}%"

            print(row)
            del q_t, k_t, v_t
            torch.cuda.empty_cache()

        print("  " + "-" * (len(header) - 2))
        summary = f"  {'Total':<7} {'':>6} {'':>6} {total_flops / 1e9:>7.1f}G"
        for name in backends:
            if total_ms[name] > 0:
                tfs = total_flops / (total_ms[name] / 1000) / 1e12
                pct = tfs / gpu_peak * 100
                summary += f"  {total_ms[name]:>{col_w - 2}.02f}ms {tfs:>{col_w - 4}.0f} ({pct:.1f}%)"
            else:
                summary += f"  {'N/A':>{col_w}} {'':>{col_w}}"
        if trace_fwd and total_trace > 0:
            tfs_t = total_flops / (total_trace / 1000) / 1e12
            first_be = list(backends.keys())[0]
            delta_t = (total_trace / total_ms[first_be] - 1) * 100
            summary += f"  {total_trace:>{col_w - 2}.02f}ms {tfs_t:>{col_w - 4}.0f} ({tfs_t / gpu_peak * 100:.1f}%) {delta_t:>+5.1f}%"
        print(summary)
        print(f"\n  Total FLOPs/sample (FWD): {total_flops / args.batch / 1e9:.2f} GFLOP\n")
        return

    # ---- FWD+BWD mode (profiler, exact kernel times) ----
    for name, be in backends.items():
        attn_fn = be["fwd"]
        fwd_pat, bwd_pat = KERNEL_PATTERNS.get(name, ("fwd", "bwd"))

        has_trace = bool(trace_fwd)
        if has_trace:
            header = (f"  {'Layer':<6} {'Q':>5} {'KV':>5}"
                      f" | {'Bench FWD':>10} {'Trace FWD':>10} {'Δ':>6}"
                      f" | {'Bench BWD':>10} {'Trace BWD':>10} {'Δ':>6}")
        else:
            header = (f"  {'Layer':<6} {'Q':>5} {'KV':>5}"
                      f" | {'FWD ms':>10} {'FWD TF/s':>10}"
                      f" | {'BWD ms':>10} {'BWD TF/s':>10}")

        print(f"--- {name}: FWD + BWD (profiled, exact kernel times) ---")
        print(header)
        print("  " + "-" * (len(header) - 2))

        tot_bf = tot_tf = tot_bb = tot_tb = 0

        for i, (q_len, kv_len) in enumerate(layers):
            flops_fwd = 4 * args.batch * args.n_heads * head_dim * q_len * kv_len
            flops_bwd = flops_fwd * 2

            q_t = torch.randn(args.batch, q_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=True)
            k_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=True)
            v_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=True)
            out_tmp = attn_fn(q_t, k_t, v_t)
            grad_out = torch.randn_like(out_tmp)
            del out_tmp

            fwd_ms, bwd_ms = _profile_fwd_bwd(
                attn_fn, q_t, k_t, v_t, grad_out,
                args.warmup, args.iters, fwd_pat, bwd_pat)

            tot_bf += fwd_ms
            tot_bb += bwd_ms

            if has_trace:
                tf = trace_fwd.get(i, 0)
                tb = trace_bwd.get(i, 0)
                tot_tf += tf
                tot_tb += tb
                df = (tf / fwd_ms - 1) * 100 if fwd_ms > 0 else 0
                db = (tb / bwd_ms - 1) * 100 if bwd_ms > 0 else 0
                print(f"  L{i:<4} {q_len:>5} {kv_len:>5}"
                      f" | {fwd_ms:>9.3f}ms {tf:>9.3f}ms {df:>+5.1f}%"
                      f" | {bwd_ms:>9.3f}ms {tb:>9.3f}ms {db:>+5.1f}%")
            else:
                fwd_tfs = flops_fwd / (fwd_ms / 1000) / 1e12 if fwd_ms > 0 else 0
                bwd_tfs = flops_bwd / (bwd_ms / 1000) / 1e12 if bwd_ms > 0 else 0
                print(f"  L{i:<4} {q_len:>5} {kv_len:>5}"
                      f" | {fwd_ms:>9.3f}ms {fwd_tfs:>9.0f}  "
                      f" | {bwd_ms:>9.3f}ms {bwd_tfs:>9.0f}  ")

            del q_t, k_t, v_t, grad_out
            torch.cuda.empty_cache()

        print("  " + "-" * (len(header) - 2))

        flops_fwd_total = sum(4 * args.batch * args.n_heads * head_dim * q * kv
                              for q, kv in layers)
        bf_tfs = flops_fwd_total / (tot_bf / 1000) / 1e12 if tot_bf > 0 else 0
        bb_tfs = (flops_fwd_total * 2) / (tot_bb / 1000) / 1e12 if tot_bb > 0 else 0

        if has_trace:
            df_t = (tot_tf / tot_bf - 1) * 100 if tot_bf > 0 else 0
            db_t = (tot_tb / tot_bb - 1) * 100 if tot_bb > 0 else 0
            tf_tfs = flops_fwd_total / (tot_tf / 1000) / 1e12 if tot_tf > 0 else 0
            tb_tfs = (flops_fwd_total * 2) / (tot_tb / 1000) / 1e12 if tot_tb > 0 else 0
            print(f"  {'Total':<6} {'':>5} {'':>5}"
                  f" | {tot_bf:>8.02f}ms {tot_tf:>8.02f}ms {df_t:>+5.1f}%"
                  f" | {tot_bb:>8.02f}ms {tot_tb:>8.02f}ms {db_t:>+5.1f}%")
            print()
            print(f"  FWD  bench: {bf_tfs:.0f} TF/s ({bf_tfs / gpu_peak * 100:.1f}%)"
                  f"   trace: {tf_tfs:.0f} TF/s ({tf_tfs / gpu_peak * 100:.1f}%)"
                  f"   overhead: {df_t:+.1f}%")
            print(f"  BWD  bench: {bb_tfs:.0f} TF/s ({bb_tfs / gpu_peak * 100:.1f}%)"
                  f"   trace: {tb_tfs:.0f} TF/s ({tb_tfs / gpu_peak * 100:.1f}%)"
                  f"   overhead: {db_t:+.1f}%")
        else:
            print(f"  {'Total':<6} {'':>5} {'':>5}"
                  f" | {tot_bf:>8.02f}ms {bf_tfs:>9.0f}  "
                  f" | {tot_bb:>8.02f}ms {bb_tfs:>9.0f}  ")
            print()
            print(f"  FWD: {bf_tfs:.0f} TF/s ({bf_tfs / gpu_peak * 100:.1f}% peak)")
            print(f"  BWD: {bb_tfs:.0f} TF/s ({bb_tfs / gpu_peak * 100:.1f}% peak)")

        print(f"\n  Total FLOPs/sample: FWD {flops_fwd_total / args.batch / 1e9:.2f}"
              f"  BWD {flops_fwd_total * 2 / args.batch / 1e9:.2f} GFLOP\n")


if __name__ == "__main__":
    main()
