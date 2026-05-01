"""Extract GEMM shapes, durations, and TF/s from a training trace.

Correlates GPU kernels (nvjet/cutlass) to CPU ops (aten::mm/addmm) via
External id, groups by (M,N,K) shape, and reports TF/s sorted by time share.

Usage:
    python scripts/analyze_gemm_trace.py --trace-json results/.../trace_step52.json
    python scripts/analyze_gemm_trace.py --trace-json trace.json --min-flops 1
"""
import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="Extract GEMM TF/s from training trace")
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--min-flops", type=float, default=0,
                        help="Skip shapes with fewer GFLOPs than this")
    parser.add_argument("--peak", type=float, default=0,
                        help="GPU BF16 peak TF/s (auto-detect if 0)")
    args = parser.parse_args()

    with open(args.trace_json) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    # GPU kernel -> External id
    gpu_by_ext = defaultdict(list)
    for e in events:
        if e.get("cat") == "kernel" and e.get("dur", 0) > 0:
            ext_id = e.get("args", {}).get("External id")
            if ext_id is not None:
                gpu_by_ext[ext_id].append(e)

    # CPU op -> shape (only aten::mm / aten::addmm with actual GPU work)
    cpu_by_ext = {}
    for e in events:
        if e.get("cat") != "cpu_op":
            continue
        name = e.get("name", "")
        args_e = e.get("args", {})
        dims = args_e.get("Input Dims")
        ext_id = args_e.get("External id")
        if not dims or ext_id is None:
            continue
        shape = None
        if name == "aten::addmm" and len(dims) >= 3 and len(dims[1]) == 2 and len(dims[2]) == 2:
            M, K = dims[1]; K2, N = dims[2]
            if K == K2:
                shape = (M, N, K)
        elif name == "aten::mm" and len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
            M, K = dims[0]; K2, N = dims[1]
            if K == K2:
                shape = (M, N, K)
        if shape:
            cpu_by_ext[ext_id] = shape

    # Walk GPU kernels, correlate to shapes
    # GEMM kernel name patterns:
    #   NVIDIA: nvjet (cuBLAS), cutlass
    #   AMD:    Cijk_ (Tensile/hipBLASLt), rocblas_*gemm
    # Attention kernels to exclude:
    #   NVIDIA: fwd_sm100/bwd_sm100/bwd_preprocess/bwd_postprocess (FA4/FA2/FA3)
    #   AMD:    Fmha*Kernel (CK / aiter Triton)
    GEMM_PATS = ["nvjet", "cutlass", "cijk_", "rocblas"]
    ATTN_PATS = ["fwd_sm100", "bwd_sm100", "bwd_preprocess", "bwd_postprocess",
                 "fmhafwdkernel", "fmhabwd"]
    ops = []
    for ext_id, kernels in gpu_by_ext.items():
        is_gemm = any(any(g in k["name"].lower() for g in GEMM_PATS) for k in kernels)
        if not is_gemm:
            continue
        if any(any(p in k["name"].lower() for p in ATTN_PATS) for k in kernels):
            continue
        shape = cpu_by_ext.get(ext_id)
        if not shape:
            continue
        dur_ms = sum(k["dur"] for k in kernels) / 1000
        ops.append({"shape": shape, "dur_ms": dur_ms, "n_kernels": len(kernels),
                     "kernel": kernels[0]["name"][:60]})

    # Detect steps. ProfilerStep#N may appear multiple times per step (multiple
    # threads emit the same span); dedupe by name to count unique step indices.
    step_names = set()
    for e in events:
        n = e.get("name", "")
        if "ProfilerStep" in n and e.get("ph") == "X" and e.get("dur", 0) > 0:
            step_names.add(n)
    n_steps = len(step_names) if step_names else 1

    # Group by shape
    shape_stats = defaultdict(lambda: {"count": 0, "total_ms": 0, "kernel": "", "n_kernels": 0})
    for o in ops:
        s = shape_stats[o["shape"]]
        s["count"] += 1
        s["total_ms"] += o["dur_ms"]
        if not s["kernel"]:
            s["kernel"] = o["kernel"]
            s["n_kernels"] = o["n_kernels"]

    total_gemm_ms = sum(v["total_ms"] for v in shape_stats.values())

    # Auto-detect GPU peak
    gpu_peak = args.peak
    if gpu_peak == 0:
        gpu_peaks = {"MI355": 2300, "MI350": 2300, "MI300": 1300,
                     "B200": 2250, "H100": 990, "H200": 990}
        for e in events:
            dev_name = e.get("args", {}).get("name", "")
            if dev_name:
                for k, v in gpu_peaks.items():
                    if k in dev_name:
                        gpu_peak = v
                        break
            if gpu_peak:
                break
        if not gpu_peak:
            gpu_peak = 2250

    # Filter
    shapes = sorted(shape_stats.items(), key=lambda x: -x[1]["total_ms"])
    if args.min_flops > 0:
        shapes = [(k, v) for k, v in shapes
                  if 2 * k[0] * k[1] * k[2] / 1e9 >= args.min_flops]

    print(f"GEMM Analysis from Training Trace")
    print(f"  trace: {args.trace_json}")
    print(f"  profiled steps: {n_steps}")
    print(f"  total GEMM time: {total_gemm_ms:.2f} ms ({total_gemm_ms / n_steps:.2f} ms/step)")
    print(f"  GPU peak: {gpu_peak:,} TF/s")
    print()

    print(f"{'#':>3} {'M':>10} {'N':>6} {'K':>6} {'Ops':>4} {'kn/op':>5}"
          f" | {'Avg ms':>8} {'TF/s':>7} {'% peak':>7}"
          f" | {'% time':>7} {'Cum %':>6}"
          f" | Kernel")
    print("-" * 115)

    cum_pct = 0
    total_flops = 0
    for i, ((M, N, K), info) in enumerate(shapes):
        flops = 2 * M * N * K
        avg_ms = info["total_ms"] / info["count"]
        tfs = flops / (avg_ms / 1000) / 1e12 if avg_ms > 0 else 0
        pct_peak = tfs / gpu_peak * 100
        time_pct = info["total_ms"] / total_gemm_ms * 100
        cum_pct += time_pct
        total_flops += flops * info["count"]

        print(f"{i+1:>3} {M:>10} {N:>6} {K:>6} {info['count']:>4} {info['n_kernels']:>5}"
              f" | {avg_ms:>7.3f}ms {tfs:>6.0f}  {pct_peak:>6.1f}%"
              f" | {time_pct:>6.1f}% {cum_pct:>5.1f}%"
              f" | {info['kernel']}")

    print("-" * 115)
    agg_tfs = total_flops / (total_gemm_ms / 1000) / 1e12 if total_gemm_ms > 0 else 0
    print(f"\nAggregate: {agg_tfs:.0f} TF/s ({agg_tfs / gpu_peak * 100:.1f}% peak)"
          f"  |  {len(shapes)} shapes, {sum(v['count'] for v in shape_stats.values())} ops"
          f"  |  {total_gemm_ms / n_steps:.2f} ms/step")


if __name__ == "__main__":
    main()
