"""Standalone attention kernel benchmark. No model, no data — just Q/K/V matmuls."""
import torch
import time
import sys

B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
S = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
H = int(sys.argv[3]) if len(sys.argv) > 3 else 64
D = int(sys.argv[4]) if len(sys.argv) > 4 else 128
WARMUP = 10
ITERS = 50

device = torch.device("cuda")
dtype = torch.bfloat16

q = torch.randn(B, S, H, D, device=device, dtype=dtype)
k = torch.randn(B, S, H, D, device=device, dtype=dtype)
v = torch.randn(B, S, H, D, device=device, dtype=dtype)

causal_flops = 4 * B * H * D * S * (S + 1) / 2
full_flops = 4 * B * H * D * S * S

print(f"Shape: B={B}, S={S}, H={H}, D={D}")
print(f"Causal FLOPs: {causal_flops/1e12:.3f} TFLOP")
print(f"Full FLOPs:   {full_flops/1e12:.3f} TFLOP")
print(f"MI350X BF16 peak: 2,300 TFLOPS")
print()

def bench(fn, label):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / ITERS * 1000
    causal_tfs = causal_flops / (avg_ms / 1000) / 1e12
    full_tfs = full_flops / (avg_ms / 1000) / 1e12
    print(f"  {label:<20} {avg_ms:>8.2f}ms  causal={causal_tfs:>7.0f} TF/s ({causal_tfs/2300*100:>5.1f}%)  full={full_tfs:>7.0f} TF/s ({full_tfs/2300*100:>5.1f}%)")

print("FWD only:")
try:
    from primus_turbo.pytorch.ops import flash_attn_func as turbo_fn
    bench(lambda: turbo_fn(q, k, v, causal=True), "Turbo (causal)")
    bench(lambda: turbo_fn(q, k, v, causal=False), "Turbo (full)")
except Exception as e:
    print(f"  Turbo: {e}")

try:
    from flash_attn import flash_attn_func as flash_fn
    bench(lambda: flash_fn(q, k, v, causal=True), "Flash (causal)")
    bench(lambda: flash_fn(q, k, v, causal=False), "Flash (full)")
except Exception as e:
    print(f"  Flash: {e}")

qt = q.transpose(1, 2)
kt = k.transpose(1, 2)
vt = v.transpose(1, 2)
bench(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True), "SDPA (causal)")
bench(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt), "SDPA (full)")

print("\nFWD + BWD:")
q.requires_grad_(True)
k.requires_grad_(True)
v.requires_grad_(True)

def fwd_bwd(fn):
    out = fn()
    out.sum().backward()

try:
    from primus_turbo.pytorch.ops import flash_attn_func as turbo_fn
    bench(lambda: fwd_bwd(lambda: turbo_fn(q, k, v, causal=True)), "Turbo (causal)")
except Exception as e:
    print(f"  Turbo: {e}")

try:
    from flash_attn import flash_attn_func as flash_fn
    bench(lambda: fwd_bwd(lambda: flash_fn(q, k, v, causal=True)), "Flash (causal)")
except Exception as e:
    print(f"  Flash: {e}")

qt = q.transpose(1, 2).detach().requires_grad_(True)
kt = k.transpose(1, 2).detach().requires_grad_(True)
vt = v.transpose(1, 2).detach().requires_grad_(True)
bench(lambda: fwd_bwd(lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)), "SDPA (causal)")
