"""Attention kernel benchmark for OneTrans pyramid layers.

Benchmarks per-layer attention shapes from the OneTrans pyramid schedule
across multiple backends. Supports FWD-only and FWD+BWD, configurable
batch size, model dimensions, and backend selection.

Usage:
    # All available backends, default OneTrans-Large shapes
    python bench_attn_kernel.py

    # Specific backends only
    python bench_attn_kernel.py --backends fav2 fav4 sdpa

    # Custom model dimensions
    python bench_attn_kernel.py --d-model 256 --n-heads 8 --n-layers 4 --history-length 20 --n-ns 8

    # Custom batch size per GPU
    python bench_attn_kernel.py --batch 1024

    # Include FWD+BWD benchmark
    python bench_attn_kernel.py --fwd-bwd

    # Uniform (non-pyramid) schedule for comparison
    python bench_attn_kernel.py --no-pyramid
"""
import argparse
import time
import torch
import torch.nn.functional as F

# ---- Backend registry ----

BACKEND_REGISTRY = {}


def register_backend(name, fwd_fn, bwd_ok=True):
    BACKEND_REGISTRY[name] = {"fwd": fwd_fn, "bwd_ok": bwd_ok}


def _make_fav2_backend(backend_module, label):
    """Build a flash_attn_func wrapper that swaps the FA dispatcher to
    ``backend_module`` (the CK C-extension or the aiter Triton interface)
    just-in-time, since flash_attn.flash_attn_interface reads
    ``flash_attn_gpu`` at call time."""
    import flash_attn.flash_attn_interface as fa_iface
    from flash_attn import flash_attn_func

    def _fn(q, k, v):
        fa_iface.flash_attn_gpu = backend_module
        return flash_attn_func(q, k, v, causal=True)

    _fn.__name__ = label
    return _fn


def _load_backends(names):
    loaded = {}
    for name in names:
        try:
            if name == "fav2":
                from flash_attn import flash_attn_func
                register_backend("fav2",
                    lambda q, k, v, fn=flash_attn_func: fn(q, k, v, causal=True))
            elif name == "fav2_ck":
                # CK C++ extension built from flash-attention source
                import flash_attn_2_cuda as ck_backend
                register_backend("fav2_ck",
                    _make_fav2_backend(ck_backend, "fav2_ck"))
            elif name == "fav2_triton":
                # AMD Triton kernels shipped in aiter
                from aiter.ops.triton._triton_kernels.flash_attn_triton_amd \
                    import flash_attn_2 as triton_backend
                register_backend("fav2_triton",
                    _make_fav2_backend(triton_backend, "fav2_triton"))
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


# ---- Benchmark ----

def bench_kernel(fn, q, k, v, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(q, k, v)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


def bench_fwd_bwd(fn, q, k, v, warmup, iters):
    torch.cuda.synchronize()
    for _ in range(warmup):
        out = fn(q, k, v)
        out.sum().backward()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v)
        out.sum().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000


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
    parser.add_argument("--fwd-bwd", action="store_true", help="Also benchmark FWD+BWD")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
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

    print("=" * 90)
    print(f"OneTrans Attention Kernel Benchmark")
    print(f"  GPU: {gpu_name} | BF16 peak: {gpu_peak:,} TFLOPS")
    print(f"  d_model={args.d_model}, n_heads={args.n_heads}, head_dim={head_dim}")
    print(f"  history_length={args.history_length}, n_groups={n_groups}, L_S={l_s}, L_NS={l_ns}")
    print(f"  n_layers={args.n_layers}, pyramid={'ON' if not args.no_pyramid else 'OFF'}")
    print(f"  batch={args.batch}, dtype={args.dtype}, warmup={args.warmup}, iters={args.iters}")
    print("=" * 90)

    print(f"\nLoading backends: {args.backends}")
    backends = _load_backends(args.backends)
    if not backends:
        print("No backends available!")
        return
    print(f"Active backends: {list(backends.keys())}\n")

    for mode in ["FWD", "FWD+BWD"] if args.fwd_bwd else ["FWD"]:
        need_grad = mode == "FWD+BWD"
        print(f"--- {mode} ---")
        col_w = 12
        header = f"  {'Layer':<7} {'Q':>6} {'KV':>6} {'FLOPs':>8}"
        for name in backends:
            header += f"  {name+' ms':>{col_w}} {name+' TF/s':>{col_w}}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        total_ms = {n: 0.0 for n in backends}
        total_flops = 0

        for i, (q_len, kv_len) in enumerate(layers):
            flops = 4 * args.batch * args.n_heads * head_dim * q_len * kv_len
            if mode == "FWD+BWD":
                flops *= 3
            total_flops += flops

            k_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=need_grad)
            v_t = torch.randn(args.batch, kv_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=need_grad)
            q_t = torch.randn(args.batch, q_len, args.n_heads, head_dim,
                               device=device, dtype=dtype, requires_grad=need_grad)

            row = f"  L{i:<5} {q_len:>6} {kv_len:>6} {flops/1e9:>7.1f}G"
            for name, be in backends.items():
                fn = be["fwd"]
                try:
                    if need_grad:
                        ms = bench_fwd_bwd(fn, q_t, k_t, v_t, args.warmup, args.iters)
                    else:
                        ms = bench_kernel(fn, q_t, k_t, v_t, args.warmup, args.iters)
                    tfs = flops / (ms / 1000) / 1e12
                    total_ms[name] += ms
                    row += f"  {ms:>{col_w}.3f} {tfs:>{col_w}.0f}"
                except Exception as e:
                    row += f"  {'ERR':>{col_w}} {'':>{col_w}}"
                    print(f"    [ERR L{i} {name}]: {e}")
            print(row)

            del q_t, k_t, v_t
            torch.cuda.empty_cache()

        print("  " + "-" * (len(header) - 2))
        summary = f"  {'Total':<7} {'':>6} {'':>6} {total_flops/1e9:>7.1f}G"
        for name in backends:
            if total_ms[name] > 0:
                tfs = total_flops / (total_ms[name] / 1000) / 1e12
                pct = tfs / gpu_peak * 100
                summary += f"  {total_ms[name]:>{col_w-2}.2f}ms {tfs:>{col_w-4}.0f} ({pct:.1f}%)"
            else:
                summary += f"  {'N/A':>{col_w}} {'':>{col_w}}"
        print(summary)
        print(f"\n  Total FLOPs/sample ({mode}): {total_flops/args.batch/1e9:.2f} GFLOP")
        print()


if __name__ == "__main__":
    main()
