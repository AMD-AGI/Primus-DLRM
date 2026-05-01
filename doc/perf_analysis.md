# Performance Analysis

## NVIDIA B200

**Setup**

- GPU: NVIDIA B200 (BF16 peak: 2,250 TFLOPS)
- Container: `nvcr.io/nvidia/pytorch:26.01-py3` (PyTorch 2.10.0a0, CUDA 13.1)
- TorchRec: 1.4.0 (`--no-deps`)
- FBGEMM-GPU: source-built from v1.5.0 tag (`TORCH_CUDA_ARCH_LIST=10.0`)
- FlashAttention-4: `flash-attn-4==4.0.0b10` (CuTeDSL, sm100)
- Model: OneTrans Large (d_model=384, n_heads=6, head_dim=64, 8 layers, pyramid schedule)
- Sequence config: history_length=500, n_groups=3, L_S=1500, L_NS=16
- Batch: 1024 per GPU, BF16, `torch.compile` (Inductor backend)
- Causal attention, no dropout

### Attention (FlashAttention-4 / CuTeDSL)

**Methodology**

Standalone benchmark and end-to-end (E2E) training trace both use `torch.profiler`
to extract exact CUDA kernel durations for `fwd_sm100` (forward) and `bwd_sm100`
(backward main kernel). This ensures a kernel-to-kernel comparison with no autograd,
launch overhead, or auxiliary kernel contamination on either side.

**Per-Layer Results (FWD — `fwd_sm100`)**

| Layer | Q | KV | Standalone | TF/s | E2E Trace | TF/s | Overhead |
|-------|------:|------:|-----------:|-----:|----------:|-----:|--------:|
| L0 | 1516 | 1516 | 3.544 ms | 1020 | 3.532 ms | 1024 | −0.3% |
| L1 | 1304 | 1516 | 4.112 ms | 756 | 4.125 ms | 754 | +0.3% |
| L2 | 1092 | 1304 | 3.172 ms | 706 | 3.276 ms | 684 | +3.3% |
| L3 | 880 | 1092 | 2.291 ms | 660 | 2.510 ms | 602 | +9.5% |
| L4 | 668 | 880 | 1.532 ms | 604 | 1.779 ms | 520 | +16.1% |
| L5 | 456 | 668 | 0.957 ms | 501 | 1.092 ms | 439 | +14.1% |
| L6 | 244 | 456 | 0.427 ms | 410 | 0.496 ms | 353 | +16.2% |
| L7 | 32 | 244 | 0.218 ms | 56 | 0.256 ms | 48 | +17.5% |
| **Total** | | | **16.25 ms** | **742** | **17.07 ms** | **707** | **+5.0%** |

**Per-Layer Results (BWD — `bwd_sm100`)**

| Layer | Q | KV | Standalone | TF/s | E2E Trace | TF/s | Overhead |
|-------|------:|------:|-----------:|-----:|----------:|-----:|--------:|
| L0 | 1516 | 1516 | 8.656 ms | 835 | 9.228 ms | 784 | +6.6% |
| L1 | 1304 | 1516 | 9.338 ms | 667 | 10.174 ms | 612 | +9.0% |
| L2 | 1092 | 1304 | 7.109 ms | 630 | 7.497 ms | 598 | +5.5% |
| L3 | 880 | 1092 | 5.011 ms | 604 | 5.290 ms | 572 | +5.6% |
| L4 | 668 | 880 | 3.839 ms | 482 | 4.225 ms | 438 | +10.1% |
| L5 | 456 | 668 | 2.493 ms | 384 | 2.855 ms | 336 | +14.5% |
| L6 | 244 | 456 | 1.249 ms | 280 | 1.466 ms | 239 | +17.3% |
| L7 | 32 | 244 | 0.505 ms | 49 | 0.589 ms | 42 | +16.6% |
| **Total** | | | **38.20 ms** | **632** | **41.32 ms** | **584** | **+8.2%** |

**Kernel Composition**

Each FA4 attention call launches:
- FWD: 1 kernel (`fwd_sm100`)
- BWD: 3 kernels (`bwd_preprocess` + `bwd_sm100` + `bwd_postprocess`)

The `bwd_sm100` kernel carries ~89% of total BWD time (41.3 ms out of 46.8 ms per step).
The `bwd_preprocess` and `bwd_postprocess` add ~3.1 ms and ~2.3 ms respectively.

**Training Context**

- FA4 attention (FWD+BWD) consumes **58.4 ms per training step** (all 8 layers combined),
  which is **17.4% of the 336 ms step time** at 24,400 samples/s throughput.
- No NCCL communication overlaps with FA4 kernels (they run on separate streams with
  zero temporal overlap).
- Between consecutive FA4 FWD kernels, the GPU executes MLP/FFN projections (nvJet GEMMs),
  RMSNorm (Triton fused kernels), and output gating. These inter-layer operations take
  2.6–14.6 ms depending on layer size.

**Overhead Analysis**

The E2E overhead relative to standalone benchmark comes from L2 cache and memory bandwidth
contention caused by inter-layer operations (MLP, norm, projections) running immediately
before each attention kernel:
- Large layers (L0–L1, Q≥1304): ≤3% FWD overhead — sufficient compute to amortize cache effects.
- Small layers (L5–L7, Q≤456): 14–17% FWD overhead — shorter kernels are more sensitive to
  cache pollution and kernel launch scheduling latency.
- BWD overhead follows the same pattern but is slightly higher overall (+8.2% vs +5.0%),
  likely due to additional memory bandwidth pressure from gradient tensors.

### GEMM (cuBLAS / nvJet)

GEMM shapes and TF/s extracted directly from GPU kernel durations in the E2E training trace,
correlated to CPU ops (`aten::mm` / `aten::addmm`) via `External id` for (M, N, K) shape
inference. Per-op kernel durations are summed when cuBLAS launches multiple kernels per op.

**Aggregate: 1,092 TF/s (48.5% peak) | 88 shapes, 645 ops | 33.13 ms/step**

| # | M | N | K | Ops | kn/op | Avg ms | TF/s | % peak | % time | Cum % | Kernel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1536000 | 1536 | 384 | 10 | 2 | 1.510 | 1200 | 53.3% | 4.6% | 4.6% | nvjet_128x256_2cta_v_bz_TNT |
| 2 | 1536000 | 384 | 1536 | 10 | 2 | 1.399 | 1295 | 57.6% | 4.2% | 8.8% | nvjet_192x160_2cta_v_bz_TNT |
| 3 | 1152 | 384 | 1536000 | 10 | 2 | 1.367 | 994 | 44.2% | 4.1% | 12.9% | nvjet_96x128_splitK_NTN |
| 4 | 1536000 | 1152 | 384 | 10 | 2 | 1.317 | 1032 | 45.9% | 4.0% | 16.9% | nvjet_128x256_h_bz_TNT |
| 5 | 1318912 | 1536 | 384 | 10 | 2 | 1.290 | 1206 | 53.6% | 3.9% | 20.8% | nvjet_128x256_2cta_v_bz_TNT |
| 6 | 1318912 | 384 | 1536 | 10 | 2 | 1.194 | 1303 | 57.9% | 3.6% | 24.4% | nvjet_192x160_2cta_v_bz_TNT |
| 7 | 1101824 | 1536 | 384 | 10 | 2 | 1.068 | 1217 | 54.1% | 3.2% | 27.6% | nvjet_128x256_2cta_v_bz_TNT |
| 8 | 1536000 | 384 | 1152 | 10 | 2 | 1.050 | 1295 | 57.5% | 3.2% | 30.8% | nvjet_192x160_2cta_v_bz_NNT |
| 9 | 1101824 | 384 | 1536 | 10 | 2 | 1.000 | 1299 | 57.8% | 3.0% | 33.8% | nvjet_192x160_2cta_v_bz_TNT |
| 10 | 1536 | 384 | 1536000 | 5 | 2 | 1.797 | 1009 | 44.8% | 2.7% | 36.5% | nvjet_96x128_splitK_NTN |
| 11 | 384 | 1536 | 1536000 | 5 | 2 | 1.750 | 1036 | 46.0% | 2.6% | 39.1% | nvjet_96x128_splitK_NTN |
| 12 | 884736 | 1536 | 384 | 10 | 1 | 0.840 | 1242 | 55.2% | 2.5% | 41.7% | nvjet_128x256_2cta_v_bz_TNT |
| 13 | 884736 | 384 | 1536 | 10 | 1 | 0.802 | 1302 | 57.9% | 2.4% | 44.1% | nvjet_192x128_2cta_h_bz_TNT |
| 14 | 384 | 1536 | 1101824 | 5 | 2 | 1.568 | 829 | 36.8% | 2.4% | 46.5% | nvjet_96x128_splitK_NTN |
| 15 | 384 | 1536 | 1318912 | 5 | 2 | 1.513 | 1029 | 45.7% | 2.3% | 48.8% | nvjet_96x128_splitK_NTN |
| 16 | 1536 | 384 | 1318912 | 5 | 2 | 1.464 | 1063 | 47.0% | 2.2% | 51.0% | nvjet_96x128_splitK_NTN |
| 17 | 667648 | 1536 | 384 | 10 | 1 | 0.665 | 1185 | 52.7% | 2.0% | 53.0% | nvjet_128x256_2cta_v_bz_TNT |
| 18 | 384 | 1536 | 884736 | 5 | 2 | 1.312 | 796 | 35.4% | 2.0% | 55.0% | nvjet_96x128_splitK_NTN |
| 19 | 667648 | 384 | 1536 | 10 | 1 | 0.629 | 1253 | 55.7% | 1.9% | 56.9% | nvjet_128x256_h_bz_TNT |
| 20 | 384 | 1536 | 667648 | 5 | 2 | 1.229 | 641 | 28.5% | 1.9% | 58.7% | nvjet_96x128_splitK_NTN |
| 21 | 1536 | 384 | 1101824 | 5 | 2 | 1.215 | 1077 | 47.6% | 1.8% | 60.5% | nvjet_96x128_splitK_NTN |
| 22 | 1152 | 384 | 1318912 | 5 | 2 | 1.167 | 1000 | 44.4% | 1.8% | 62.3% | nvjet_96x128_splitK_NTN |
| 23 | 1318912 | 1152 | 384 | 5 | 2 | 1.127 | 1035 | 46.0% | 1.7% | 64.0% | nvjet_128x256_h_bz_TNT |
| 24 | 1152 | 384 | 1101824 | 5 | 2 | 0.969 | 1006 | 44.7% | 1.5% | 65.5% | nvjet_96x128_splitK_NTN |
| 25 | 1536 | 384 | 884736 | 5 | 2 | 0.954 | 1094 | 48.6% | 1.4% | 66.9% | nvjet_96x128_splitK_NTN |
| 26 | 1101824 | 1152 | 384 | 5 | 2 | 0.932 | 1046 | 46.5% | 1.4% | 68.3% | nvjet_128x256_h_bz_TNT |
| 27 | 384 | 1536 | 450560 | 5 | 2 | 0.932 | 571 | 25.4% | 1.4% | 69.7% | nvjet_96x128_splitK_NTN |
| 28 | 1318912 | 384 | 1152 | 5 | 2 | 0.908 | 1286 | 57.1% | 1.4% | 71.1% | nvjet_192x160_2cta_v_bz_NNT |
| 29 | 512000 | 384 | 384 | 30 | 1 | 0.148 | 1023 | 45.4% | 1.3% | 72.4% | nvjet_128x256_bias_TNT |
| 30 | 1552384 | 384 | 384 | 10 | 2 | 0.436 | 1050 | 46.7% | 1.3% | 73.7% | nvjet_192x160_2cta_v_bz_TNT |
| 31 | 450560 | 384 | 1536 | 10 | 1 | 0.421 | 1261 | 56.1% | 1.3% | 75.0% | nvjet_128x256_h_bz_TNT |
| 32 | 884736 | 1152 | 384 | 5 | 1 | 0.827 | 947 | 42.1% | 1.2% | 76.3% | nvjet_128x256_h_bz_TNT |
| 33 | 450560 | 1536 | 384 | 10 | 1 | 0.412 | 1290 | 57.3% | 1.2% | 77.5% | nvjet_128x256_2cta_v_bz_TNT |
| 34 | 1101824 | 384 | 1152 | 5 | 2 | 0.771 | 1265 | 56.2% | 1.2% | 78.7% | nvjet_192x160_2cta_v_bz_NNT |
| 35 | 1152 | 384 | 884736 | 5 | 2 | 0.768 | 1019 | 45.3% | 1.2% | 79.8% | nvjet_96x128_splitK_NTN |
| 36 | 1335296 | 384 | 384 | 10 | 2 | 0.371 | 1063 | 47.2% | 1.1% | 80.9% | nvjet_192x160_2cta_v_bz_TNT |
| 37 | 1536 | 384 | 667648 | 5 | 2 | 0.704 | 1119 | 49.7% | 1.1% | 82.0% | nvjet_96x128_splitK_NTN |
| 38 | 884736 | 384 | 1152 | 5 | 1 | 0.635 | 1232 | 54.8% | 1.0% | 83.0% | nvjet_192x128_2cta_v_bz_NNT |
| 39 | 1118208 | 384 | 384 | 10 | 2 | 0.308 | 1072 | 47.7% | 0.9% | 83.9% | nvjet_192x160_2cta_v_bz_TNT |
| 40 | 1152 | 384 | 667648 | 5 | 2 | 0.578 | 1022 | 45.4% | 0.9% | 84.8% | nvjet_96x128_splitK_NTN |
| 41 | 667648 | 1152 | 384 | 5 | 1 | 0.549 | 1076 | 47.8% | 0.8% | 85.6% | nvjet_128x256_h_bz_TNT |
| 42 | 901120 | 384 | 384 | 10 | 1 | 0.260 | 1020 | 45.3% | 0.8% | 86.4% | nvjet_128x256_h_bz_TNT |
| 43 | 384 | 384 | 512000 | 15 | 2 | 0.169 | 893 | 39.7% | 0.8% | 87.2% | nvjet_96x128_splitK_NTN |
| 44 | 384 | 384 | 1552384 | 5 | 2 | 0.482 | 950 | 42.2% | 0.7% | 87.9% | nvjet_96x128_splitK_NTN |
| 45 | 667648 | 384 | 1152 | 5 | 1 | 0.476 | 1241 | 55.2% | 0.7% | 88.6% | nvjet_128x256_h_bz_NNT |
| 46 | 1536 | 384 | 450560 | 5 | 2 | 0.468 | 1137 | 50.5% | 0.7% | 89.3% | nvjet_96x128_splitK_NTN |
| 47 | 233472 | 384 | 1536 | 10 | 1 | 0.211 | 1306 | 58.1% | 0.6% | 89.9% | nvjet_192x128_2cta_v_bz_TNT |
| 48 | 233472 | 1536 | 384 | 10 | 1 | 0.211 | 1307 | 58.1% | 0.6% | 90.6% | nvjet_128x256_2cta_v_bz_TNT |
| 49 | 384 | 384 | 1335296 | 5 | 2 | 0.421 | 936 | 41.6% | 0.6% | 91.2% | nvjet_96x128_splitK_NTN |
| 50 | 384 | 1536 | 233472 | 5 | 2 | 0.411 | 670 | 29.8% | 0.6% | 91.8% | nvjet_96x128_splitK_NTN |
| 51 | 1152 | 384 | 450560 | 5 | 2 | 0.390 | 1022 | 45.4% | 0.6% | 92.4% | nvjet_96x128_splitK_NTN |
| 52 | 512000 | 384 | 192 | 15 | 1 | 0.128 | 588 | 26.1% | 0.6% | 93.0% | nvjet_128x512_bias_TNT |
| 53 | 684032 | 384 | 384 | 10 | 1 | 0.190 | 1064 | 47.3% | 0.6% | 93.6% | nvjet_192x128_2cta_v_bz_TNT |
| 54 | 450560 | 1152 | 384 | 5 | 1 | 0.357 | 1117 | 49.6% | 0.5% | 94.1% | nvjet_192x128_2cta_v_bz_TNT |
| 55 | 384 | 384 | 1118208 | 5 | 2 | 0.346 | 953 | 42.4% | 0.5% | 94.6% | nvjet_96x128_splitK_NTN |
| 56 | 450560 | 384 | 1152 | 5 | 1 | 0.317 | 1257 | 55.9% | 0.5% | 95.1% | nvjet_128x256_h_bz_NNT |
| 57 | 384 | 192 | 512000 | 15 | 2 | 0.104 | 723 | 32.2% | 0.5% | 95.6% | nvjet_96x128_splitK_NTN |
| 58 | 384 | 384 | 901120 | 5 | 2 | 0.278 | 956 | 42.5% | 0.4% | 96.0% | nvjet_96x128_splitK_NTN |
| 59 | 466944 | 384 | 384 | 10 | 1 | 0.136 | 1011 | 44.9% | 0.4% | 96.4% | nvjet_128x256_h_bz_TNT |
| 60 | 512000 | 192 | 384 | 15 | 1 | 0.090 | 835 | 37.1% | 0.4% | 96.8% | nvjet_192x128_h_bz_NNT |
| 61 | 1536 | 384 | 233472 | 5 | 2 | 0.242 | 1139 | 50.6% | 0.4% | 97.2% | nvjet_96x128_splitK_NTN |
| 62 | 384 | 384 | 684032 | 5 | 2 | 0.216 | 934 | 41.5% | 0.3% | 97.5% | nvjet_96x128_splitK_NTN |
| 63 | 1152 | 384 | 233472 | 5 | 2 | 0.205 | 1007 | 44.7% | 0.3% | 97.8% | nvjet_96x128_splitK_NTN |
| 64 | 233472 | 1152 | 384 | 5 | 1 | 0.185 | 1114 | 49.5% | 0.3% | 98.1% | nvjet_192x128_2cta_v_bz_TNT |
| 65 | 233472 | 384 | 1152 | 5 | 1 | 0.157 | 1315 | 58.4% | 0.2% | 98.3% | nvjet_192x128_2cta_v_bz_NNT |
| 66 | 384 | 384 | 466944 | 5 | 2 | 0.156 | 886 | 39.4% | 0.2% | 98.6% | nvjet_96x128_splitK_NTN |
| 67 | 249856 | 384 | 384 | 10 | 1 | 0.075 | 980 | 43.5% | 0.2% | 98.8% | nvjet_128x256_h_bz_TNT |
| 68 | 1024 | 396 | 6144 | 5 | 1 | 0.118 | 42 | 1.9% | 0.2% | 99.0% | cutlass_80_tensorop_bf16 |
| 69 | 1024 | 6144 | 6144 | 10 | 1 | 0.055 | 1400 | 62.2% | 0.2% | 99.1% | nvjet_192x128_2cta_h_bias_TNT |
| 70 | 384 | 384 | 249856 | 5 | 2 | 0.094 | 788 | 35.0% | 0.1% | 99.3% | nvjet_96x128_splitK_NTN |
| 71 | 6144 | 396 | 1024 | 5 | 1 | 0.058 | 87 | 3.8% | 0.1% | 99.4% | cutlass_80_tensorop_bf16 |
| 72 | 384 | 1536 | 16384 | 5 | 2 | 0.056 | 342 | 15.2% | 0.1% | 99.5% | nvjet_96x128_splitK_NTN |
| 73 | 1536 | 384 | 16384 | 5 | 2 | 0.054 | 357 | 15.8% | 0.1% | 99.5% | nvjet_96x128_splitK_NTN |
| 74 | 16384 | 384 | 1536 | 10 | 1 | 0.025 | 759 | 33.7% | 0.1% | 99.6% | nvjet_192x128_2cta_h_bz_TNT |
| 75 | 6144 | 6144 | 1024 | 5 | 1 | 0.049 | 1580 | 70.2% | 0.1% | 99.7% | nvjet_128x256_2cta_v_bz_NTT |
| 76 | 16384 | 1536 | 384 | 10 | 1 | 0.024 | 814 | 36.2% | 0.1% | 99.8% | nvjet_128x192_2cta_v_bz_TNT |
| 77 | 1024 | 6144 | 396 | 5 | 1 | 0.038 | 131 | 5.8% | 0.1% | 99.8% | cutlass_80_tensorop_bf16 |
| 78 | 32768 | 384 | 384 | 10 | 1 | 0.014 | 669 | 29.7% | 0.0% | 99.9% | nvjet_192x128_2cta_h_bz_TNT |
| 79 | 384 | 384 | 32768 | 5 | 2 | 0.026 | 376 | 16.7% | 0.0% | 99.9% | nvjet_96x64_splitK_NTN |
| 80 | 64 | 18 | 1024 | 5 | 1 | 0.013 | 0 | 0.0% | 0.0% | 99.9% | cutlass_80_wmma_bf16 |
| 81 | 1024 | 384 | 6144 | 5 | 1 | 0.011 | 422 | 18.7% | 0.0% | 99.9% | nvjet_64x48_bias_TNT |
| 82 | 384 | 6144 | 1024 | 5 | 1 | 0.009 | 548 | 24.4% | 0.0% | 100.0% | nvjet_128x128_2cta_v_bz_NTT |
| 83 | 1024 | 6144 | 384 | 5 | 1 | 0.008 | 583 | 25.9% | 0.0% | 100.0% | nvjet_192x128_2cta_h_bz_NNT |
| 84 | 1024 | 1 | 384 | 5 | 2 | 0.006 | 0 | 0.0% | 0.0% | 100.0% | elementwise_kernel |
| 85 | 64 | 128 | 1024 | 5 | 1 | 0.004 | 4 | 0.2% | 0.0% | 100.0% | nvjet_64x32_2cta_h_bz_NTT |
| 86 | 1 | 384 | 1024 | 5 | 1 | 0.004 | 0 | 0.0% | 0.0% | 100.0% | nvjet_32x64_v_bz_NNN |
| 87 | 1024 | 64 | 18 | 5 | 1 | 0.003 | 1 | 0.0% | 0.0% | 100.0% | cutlass_80_wmma_bf16 |
| 88 | 1024 | 64 | 128 | 5 | 1 | 0.003 | 6 | 0.3% | 0.0% | 100.0% | nvjet_8x64_bias_TNN |

**Shape categories by TF/s efficiency:**
- **>1200 TF/s (>53% peak):** FFN down-projections (M~1M, N=384, K=1536) — best efficiency
- **1000–1200 TF/s (44–53%):** FFN up-projections, large attention projections
- **800–1000 TF/s (35–44%):** Backward weight gradients (splitK shapes)
- **500–800 TF/s (25–35%):** Small backward splitK, embedding projections
- **<500 TF/s (<22%):** Tiny shapes, CUTLASS fallbacks (N=396, N=18)

### E2E Training

**Performance**

| Metric | Value |
|---|---|
| Throughput | 24,400 samples/s (8x B200) |
| TFLOPS/GPU | 327–328 |
| MFU | 14.5–14.6% |
| Step time | 335.7 ms |

**GPU Kernel Time Breakdown (per step, % of 335.7 ms wall-clock step time)**

| Category | ms/step | % of step |
|---|---:|---:|
| Fused ops (Triton) | 105.55 | 31.4% |
| GEMM (cuBLAS) | 68.02 | 20.3% |
| Attention (FA4) | 63.86 | 19.0% |
| Elementwise | 60.22 | 17.9% |
| NCCL — exposed | 16.76 | 5.0% |
| NCCL — overlapped | 14.58 | 4.3% |
| Embeddings (FBGEMM) | 12.46 | 3.7% |
| Misc (optimizer, sort, etc.) | 12.01 | 3.6% |

Total kernel time (353 ms) exceeds wall-clock step time (336 ms) because ~14.6 ms of
NCCL communication overlaps with compute on a separate CUDA stream. Only 16.8 ms of
NCCL is exposed (not overlapped), adding 5% to step time.

**Kernel TF/s by category:**

| Category | TF/s | % of B200 peak |
|---|---:|---:|
| Attention FWD (`fwd_sm100`) | 707 | 31.4% |
| Attention BWD (`bwd_sm100`) | 584 | 26.0% |
| GEMM (cuBLAS aggregate) | 1,092 | 48.5% |

GEMM achieves higher % peak than attention because cuBLAS benefits from Inductor's
algorithm autotuning, while FA4 is a fixed custom kernel. Attention kernels are
memory-bandwidth-bound at smaller pyramid layers (L5–L7), pulling down the aggregate.
