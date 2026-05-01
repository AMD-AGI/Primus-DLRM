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

### Communications (NCCL)

Per-comm-kind breakdown extracted from the E2E training trace (5 active steps).
Each row groups all kernels of the same `(purpose, dtype, size-bucket)` and reports
the representative kernel signature, message volume and bandwidth metrics, and time
decomposition into EXPOSED (critical-path) vs HIDDEN (overlapped with compute on a
separate CUDA stream). Sorted by exposed time descending.

```
profiled steps : 5
step wall time : 339.2 ms (avg across 5 ProfilerStep events)
NCCL kernels   : 155
total NCCL time: 31.3 ms/step (9.2% of step)
exposed (crit) : 17.0 ms/step (5.0% of step)
overlapped     : 14.3 ms/step (4.2% of step)
per-rank NVLink5 peak: 50 GB/s × 18 links = 900 GB/s (busBw % column references this)
```

| # | purpose | kernel | dtype | #/step | vol/call | ms/call | algBw (% peak) | busBw (% peak) | tot ms | %step | EXP ms | %exp | where |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Embedding data a2a (FWD dispatch / BWD grad) | `ncclDevKernel_SendRecv` | Float (4B) | 2.0× | 7.81 GB | 8.38 ms | 804.1 GB/s (89.3%) | **703.6 GB/s (78.2%)** | 16.76 | 4.9% | **16.76** | **4.9%** | **EXPOSED (100%)** |
| 2 | DDP gradient AllReduce (small bucket) | `ncclDevKernel_AllReduce` | Float (4B) | 25.0× | 73.4 MB | 0.52 ms | 121.6 GB/s (13.5%) | 212.7 GB/s (23.6%) | 12.95 | 3.8% | 0.18 | 0.1% | HIDDEN |
| 3 | DDP gradient AllReduce (big bucket) | `ncclDevKernel_AllReduce` | Float (4B) | 1.0× | 305.5 MB | 0.62 ms | 286.7 GB/s (31.9%) | 501.7 GB/s (55.7%) | 0.62 | 0.2% | 0.08 | 0.0% | HIDDEN |
| 4 | KJT lengths a2a (per-row offsets) | `ncclDevKernel_SendRecv` | Long (8B) | 1.0× | 244.3 MB | 0.98 ms | 217.0 GB/s (24.1%) | 189.9 GB/s (21.1%) | 0.98 | 0.3% | 0.02 | 0.0% | HIDDEN |
| 5 | KJT splits a2a (per-feature metadata) | `ncclDevKernel_SendRecv` | Long (8B) | 1.0× | 384.0 B | 0.01 ms | 23.0 MB/s (0.0%) | 20.2 MB/s (0.0%) | 0.01 | 0.0% | 0.01 | 0.0% | EXPOSED (71%) |
| 6 | KJT keys a2a (sparse feature indices) | `ncclDevKernel_SendRecv` | Int (4B) | 1.0× | 852.0 KB | 0.03 ms | 17.5 GB/s (1.9%) | 15.3 GB/s (1.7%) | 0.03 | 0.0% | 0.00 | 0.0% | HIDDEN |
| | **TOTAL** | | | 31.0× | | | | | **31.34** | **9.2%** | **17.05** | **5.0%** | EXP+HID |


## AMD MI355X

**Setup**

- GPU: AMD Instinct MI355X (gfx950, 256 CUs, BF16 peak: 2,300 TFLOPS)
- Container: `rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424`
- ROCm: 7.2.53211 | PyTorch: 2.10.0+git94c6e04 | Python: 3.12.3
- TorchRec: 1.4.0 | FBGEMM-GPU: 2026.4.24 (bundled, ROCm 7.2)
- FlashAttention-2 (CK backend, yiding12 ROCm dev branch, commit 7222cad5)
- RCCL: 2.27.7 (no MSCCL++ — `ENABLE_MSCCLPP` not compiled in)
- Model: OneTrans Large — same config as B200 (d_model=384, n_heads=6, head_dim=64,
  8 layers, pyramid; L_S=1500, L_NS=16; batch=1024 per GPU; BF16; `torch.compile` Inductor)
- Causal attention, no dropout, hipBLASLt offline tuning override applied
  (`configs/hipblaslt_tune/onetrans_large_5b_all138_full.txt`, +6.4% e2e gain)

### E2E Training

**Performance**

| Metric | Value |
|---|---|
| Throughput | ~23,500 samples/s (8x MI355X, batch=8192) |
| TFLOPS/GPU | 316 |
| MFU | 13.7% |
| Step time | 341 ms |

**GPU Kernel Time Breakdown (per step, % of 341 ms wall-clock step time)**

| Category | ms/step | % of step |
|---|---:|---:|
| NCCL — overlapped | 131.90 | 38.7% |
| GEMM (hipBLASLt/Tensile) | 90.10 | 26.4% |
| Fused ops (Triton/Inductor) | 88.91 | 26.1% |
| Attention (CK Flash) | 77.87 | 22.8% |
| NCCL — exposed | 33.53 | 9.8% |
| Elementwise/Reduce | 29.66 | 8.7% |
| Embeddings (FBGEMM) | 19.56 | 5.7% |
| Misc (optimizer, etc.) | 10.58 | 3.1% |

Total kernel time across all categories sums to 482 ms; this exceeds the 341 ms wall
step time because **132 ms (39%) of NCCL** runs overlapped with compute on a separate
HIP stream and contributes nothing to wall time. Only **34 ms (10%)** of NCCL is
exposed (on the critical path), about 2× B200's 5%.

**NCCL breakdown by collective type (per step, classified by kernel `Collective name` arg):**

| Collective | Total ms | Exposed ms | Overlapped ms | Hidden % |
|---|---:|---:|---:|---:|
| AllToAll  | 125.0 | 29.7 | 95.3  | 76.3% |
| AllReduce |  36.4 |  0.9 | 35.5  | **97.5%** |
| Other     |   4.0 |  3.0 |  1.1  | 26.6% |

AllReduce is essentially free on this run (97.5% hidden behind compute, just 0.9 ms
exposed) — DDP gradient sync overlaps almost perfectly with FWD compute of the
following step. **All NCCL exposed time is dominated by AllToAll** (29.7 of 33.5 ms),
specifically the keys a2a + tail of the large data a2a. This is the opposite of the
naïve assumption that gradient AllReduce dominates comm cost on AMD.

**Kernel TF/s by category:**

| Category | TF/s | % of MI355X peak |
|---|---:|---:|
| Attention FWD (`FmhaFwdKernel`) | 651 | 28.3% |
| Attention BWD (DQDKDV+OGD+CQG) | 509 | 22.1% |
| GEMM (Tensile aggregate) | 836 | 36.4% |

GEMM achieves the highest % peak among the three but it is **substantially below
B200's 48.5%** for the same logical workload. The gap comes mostly from non-square
backward-weight (splitK) shapes — e.g. shape `(384, 1536, 1.5M)` runs at 635 TF/s
on MI355X (28% peak) vs 1009 TF/s on B200 (45% peak). hipBLASLt offline tuning
recovered +6.4% e2e but the residual gap appears to be Tensile kernel selection +
lower achievable HBM bandwidth utilization on the splitK back-prop shapes.

### Attention (FlashAttention-2 / CK)

**Methodology**

Standalone benchmark uses `scripts/bench_attn_save_trace.py` (sequential layer
iteration with profiler-saved trace). E2E uses `torch.profiler` trace at step 52
(2 warmup + 5 active steps). Per-layer matching uses kernel grid dimensions:
`FmhaFwdKernel` grid is `(n_heads=6, q_seq_tiles, batch=1024)` so q-seq-tile count
identifies the layer; `FmhaBwdDQDKDVKernel` grid is `(kv_seq_tiles, n_heads=6, batch)`
so kv-tile count identifies the layer for BWD. Three layers (L0–L2) share the same
kv-tile count of 6; their per-layer BWD time is reported as the average within that
group.

**Per-Layer Results (FWD — `FmhaFwdKernel`)**

| Layer | Q | KV | Standalone | TF/s | E2E Trace | TF/s | Overhead |
|-------|------:|------:|-----------:|-----:|----------:|-----:|--------:|
| L0 | 1516 | 1516 | 4.068 ms | 889 | 4.730 ms | 764 | +16.3% |
| L1 | 1304 | 1516 | 4.041 ms | 770 | 4.606 ms | 675 | +14.0% |
| L2 | 1092 | 1304 | 3.039 ms | 737 | 3.376 ms | 663 | +11.1% |
| L3 | 880 | 1092 | 2.157 ms | 701 | 2.374 ms | 637 | +10.0% |
| L4 | 668 | 880 | 1.627 ms | 568 | 1.823 ms | 507 | +12.1% |
| L5 | 456 | 668 | 0.985 ms | 486 | 1.085 ms | 441 | +10.2% |
| L6 | 244 | 456 | 0.446 ms | 392 | 0.444 ms | 394 | −0.4% |
| L7 | 32 | 244 | 0.091 ms | 135 | 0.102 ms | 121 | +11.7% |
| **Total** | | | **16.45 ms** | **733** | **18.54 ms** | **651** | **+12.7%** |

**Per-Layer Results (BWD — sum of `OGradDotO + DQDKDV + ConvertQGrad`)**

| Layer | Q | KV | Standalone | TF/s | E2E Trace | TF/s | Overhead |
|-------|------:|------:|-----------:|-----:|----------:|-----:|--------:|
| L0 | 1516 | 1516 | 13.441 ms | 672 | 13.234 ms | 683 | −1.5% |
| L1 | 1304 | 1516 | 12.745 ms | 610 | 13.051 ms | 596 | +2.4% |
| L2 | 1092 | 1304 | 10.930 ms | 512 | 12.903 ms | 434 | +18.0% |
| L3 | 880 | 1092 | 7.932 ms | 476 | 8.249 ms | 458 | +4.0% |
| L4 | 668 | 880 | 6.328 ms | 365 | 6.297 ms | 367 | −0.5% |
| L5 | 456 | 668 | 3.548 ms | 338 | 3.380 ms | 354 | −4.8% |
| L6 | 244 | 456 | 1.701 ms | 257 | 1.744 ms | 251 | +2.5% |
| L7 | 32 | 244 | 0.342 ms | 90 | 0.352 ms | 87 | +3.0% |
| **Total** | | | **56.97 ms** | **530** | **59.21 ms** | **509** | **+3.9%** |

**Kernel Composition**

Each CK FA-2 attention call launches:
- FWD: 1 kernel (`FmhaFwdKernel`)
- BWD: 3 kernels (`FmhaBwdOGradDotOKernel` + `FmhaBwdDQDKDVKernel` + `FmhaBwdConvertQGradKernel`)

The `FmhaBwdDQDKDVKernel` (main BWD) carries ~90% of total BWD time (53.7 ms of 59.2 ms
in E2E). `OGradDotOKernel` adds 3.0 ms and `ConvertQGradKernel` 2.7 ms. The CK kernel
grid for FWD is `(n_heads=6, ⌈Q/128⌉, batch=1024)` with block `(256,1,1)` = 4 wavefronts
per block. Even the smallest layer (L7, 32-token Q) launches 6 144 blocks — 3× over the
GPU's 2 048 concurrent-block capacity (256 CU × 32 waves/CU ÷ 4 waves/block) — so all
256 CUs are saturated for every layer in both standalone and E2E. Kernel selection is
shape-driven, not workload-driven.

**Training Context**

- CK FA-2 attention (FWD+BWD) consumes **77.75 ms per training step** (all 8 layers
  combined), which is **22.8% of the 341 ms step time** at ~23,500 samples/s throughput.
- FWD attention for the 6 largest layers (L0–L5, ~17 ms total per step) **overlaps
  100% with the EmbeddingShardingDist keys all-to-all** — the 87 ms `all_to_allv`
  on int32 indices that runs concurrently on the comm stream throughout most of the
  FWD pass. The two smallest layers (L6, L7) finish in <0.5 ms each and fall outside
  this overlap window. Per-kernel verification: 30 of 40 FmhaFwd events show 100%
  lifetime overlap with the keys a2a, 0% with AllReduce; 10 events (the L6/L7 pairs)
  show 0% overlap with any NCCL.
- BWD attention has **zero NCCL overlap** in our trace (verified for all 40 DQDKDV
  events) — by the time BWD starts, the next-step keys a2a hasn't been launched yet,
  and the AllReduce queue runs on a non-contending HIP stream. The +3.9% BWD overhead
  is from background GEMM/embedding HBM contention, not communication.
- Between consecutive FA-2 FWD kernels the GPU runs MLP/FFN projections (Tensile GEMMs),
  RMSNorm (Triton), and output gating — same compute pattern as B200.

**Overhead Analysis**

The E2E overhead pattern differs from B200's. On B200 the smallest layers (L5–L7) suffer
the largest FWD overhead (14–17%) because cache pollution dominates short kernels. On
MI355X the **largest** layers (L0–L1) suffer the largest FWD overhead (+14–16%):
- The keys a2a (87 ms) is in flight throughout the FWD pass, sharing HBM bandwidth
  with FA-2. Larger FWD kernels stay on-chip longer and are exposed to more of the
  a2a's HBM traffic, so they pay a larger fraction of contention cost.
- Per-shape grid/block configs are **identical** between bench and E2E (we verified by
  matching `(grid, block)` tuples in `scripts/compare_attn_grid_block_v2.py`); the kernel
  selection itself does not adapt to runtime contention.
- BWD overhead is much lower (+3.9% vs +12.7% FWD) because by the time BWD starts the
  keys a2a window has closed and BWD runs in a near-quiet comm period. Combined with
  DQDKDV's higher arithmetic intensity (recompute + dQ/dK/dV in one kernel), this
  leaves only background GEMM/embedding contention as the BWD overhead source.

### GEMM (hipBLASLt / Tensile)

GEMM shapes and TF/s extracted directly from GPU kernel durations in the E2E training
trace, correlated to CPU ops (`aten::mm` / `aten::addmm`) via `External id` for (M, N, K)
shape inference. Per-op kernel durations are summed when hipBLASLt launches multiple
kernels per op (split-K cases). Trace already includes the hipBLASLt offline tuning
override (138 hot shapes pre-tuned via full `hipblaslt-bench --requested_solution -1`
sweep, +6.4% e2e gain over RCCL defaults).

**Aggregate: 836 TF/s (36.4% peak) | 89 shapes, 650 ops | 86.54 ms/step**

| # | M | N | K | Ops | kn/op | Avg ms | TF/s | % peak | % time | Cum % | Kernel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1536000 | 1536 | 384 | 10 | 1 | 2.352 | 770 | 33.5% | 5.4% | 5.4% | Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1 |
| 2 | 1318912 | 1536 | 384 | 10 | 1 | 1.862 | 835 | 36.3% | 4.3% | 9.7% | Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1 |
| 3 | 1536000 | 1152 | 384 | 10 | 1 | 1.857 | 732 | 31.8% | 4.3% | 14.0% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 4 | 1536000 | 384 | 1536 | 10 | 1 | 1.813 | 1000 | 43.5% | 4.2% | 18.2% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 5 | 1101824 | 1536 | 384 | 10 | 1 | 1.588 | 818 | 35.6% | 3.7% | 21.9% | Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1 |
| 6 | 1318912 | 384 | 1536 | 10 | 1 | 1.472 | 1057 | 45.9% | 3.4% | 25.3% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 7 | 1536000 | 384 | 1152 | 10 | 1 | 1.460 | 931 | 40.5% | 3.4% | 28.7% | Cijk_Ailk_Bljk_BBS_BH_MT192x384x64 |
| 8 | 384 | 1536 | 1536000 | 5 | 2 | 2.854 | 635 | 27.6% | 3.3% | 32.0% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 9 | 884736 | 1536 | 384 | 10 | 1 | 1.350 | 773 | 33.6% | 3.1% | 35.1% | Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1 |
| 10 | 1152 | 384 | 1536000 | 10 | 2 | 1.304 | 1042 | 45.3% | 3.0% | 38.1% | Cijk_Ailk_Bjlk_BBS_BH_MT192x320x64 (splitK) |
| 11 | 384 | 1536 | 1318912 | 5 | 2 | 2.494 | 624 | 27.1% | 2.9% | 41.0% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 12 | 1101824 | 384 | 1536 | 10 | 1 | 1.244 | 1045 | 45.4% | 2.9% | 43.9% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 13 | 667648 | 1536 | 384 | 10 | 1 | 1.071 | 735 | 32.0% | 2.5% | 46.3% | Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1 |
| 14 | 384 | 1536 | 1101824 | 5 | 2 | 2.107 | 617 | 26.8% | 2.4% | 48.8% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 15 | 884736 | 384 | 1536 | 10 | 1 | 0.991 | 1054 | 45.8% | 2.3% | 51.1% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 16 | 1536 | 384 | 1536000 | 5 | 2 | 1.711 | 1059 | 46.0% | 2.0% | 53.0% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 17 | 384 | 1536 | 884736 | 5 | 2 | 1.687 | 619 | 26.9% | 1.9% | 55.0% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 18 | 512000 | 384 | 384 | 30 | 1 | 0.277 | 545 | 23.7% | 1.9% | 56.9% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |
| 19 | 1536 | 384 | 1318912 | 5 | 2 | 1.608 | 967 | 42.1% | 1.9% | 58.8% | Cijk_Ailk_Bjlk_BBS_BH_MT192x192x64 (splitK) |
| 20 | 1318912 | 1152 | 384 | 5 | 1 | 1.553 | 751 | 32.7% | 1.8% | 60.6% | Cijk_Alik_Bljk_BBS_BH_MT384x256x32 |

(Full table for all 89 shapes is available via `python3 scripts/analyze_gemm_trace.py
--trace-json results/.../trace_step52.json --peak 2300`. Shapes 21–89 collectively
contribute the remaining 39.4% of GEMM time.)

**Shape categories by TF/s efficiency:**
- **>1000 TF/s (>43% peak):** FFN down-projections (M~1M, N=384, K=1536) — best efficiency,
  but only ~45% of peak (vs B200's 57% for the same logical shapes).
- **700–1000 TF/s (30–43%):** FFN up-projections, large attention projections.
- **500–700 TF/s (22–30%):** Backward weight gradients (splitK shapes).
- **<500 TF/s (<22%):** Tiny shapes, fallback shaders.

### Communications (RCCL)

Per-comm-kind breakdown extracted from the same E2E training trace (5 active steps,
hipBLASLt-tuned baseline). Each row groups all kernels of the same `(purpose, dtype,
size-bucket)` and reports the representative kernel signature, message-volume and
bandwidth metrics, and time decomposition into EXPOSED (critical-path) vs HIDDEN
(overlapped with compute on a separate stream). Sorted by exposed time descending,
so the single comm kind that actually moves wall-clock time is at the top.

```
profiled steps : 5
step wall time : 341.8 ms (avg across 5 ProfilerStep events)
NCCL kernels   : 155 (after dropping 4 trace-boundary kernels with no Collective name args)
total NCCL time: 161.4 ms/step (47.2% of step)
exposed (crit) : 30.6 ms/step ( 8.9% of step)
overlapped     : 130.8 ms/step (38.3% of step)
per-rank XGMI peak: 128 GB/s × 7 links = 896 GB/s (busBw % column references this)
```

| # | purpose | kernel | grid | block | dtype | #/step | vol/call | ms/call | algBw (% peak) | busBw (% peak) | tot ms | %step | EXP ms | %exp | where |
|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Embedding data a2a (FWD dispatch / BWD grad) | `ncclDevKernel_G1` | (64,1,1)  | (256,1,1) | Float (4B) |  2.0× |   7.81 GB | 14.83 ms | 447.3 GB/s (49.9%)  | **391.4 GB/s (43.7%)** | 29.65 |  8.7% | **29.65** | **8.7%** | **EXPOSED (100%)** |
| 2 | DDP gradient AllReduce (small bucket)        | `ncclDevKernel_G1` | (112,1,1) | (256,1,1) | Float (4B) | 25.0× |  73.0 MB |  1.33 ms |  54.8 GB/s ( 6.1%)  |  95.9 GB/s (10.7%)     | 33.25 |  9.7% |  0.88     | 0.3%     | HIDDEN |
| 3 | DDP gradient AllReduce (single big bucket)   | `ncclDevKernel_G1` | (112,1,1) | (256,1,1) | Float (4B) |  1.0× | 316.2 MB |  3.16 ms |  51.1 GB/s ( 5.7%)  |  89.5 GB/s (10.0%)     |  3.16 |  0.9% |  0.02     | 0.0%     | HIDDEN |
| 4 | KJT keys a2a (sparse feature indices)        | `ncclDevKernel_G1` | (57,1,1)  | (256,1,1) | Int (4B)   |  1.0× | 852.0 KB | **87.16 ms** | **4.9 MB/s (0.0%)** | **4.3 MB/s (0.0%)**    | **87.16** | **25.5%** | 0.01 | 0.0% | HIDDEN |
| 5 | KJT splits a2a (per-feature lengths metadata)| `mscclKernel_Sum`  | (15,1,1)  | (256,1,1) | Long (8B)  |  1.0× |  384.0 B |  6.83 ms |  28.1 KB/s (0.0%)   |  24.6 KB/s (0.0%)      |  6.83 |  2.0% |  0.00     | 0.0%     | HIDDEN |
| 6 | KJT lengths a2a (per-row offsets)            | `ncclDevKernel_G1` | (64,1,1)  | (256,1,1) | Long (8B)  |  1.0× | 244.3 MB |  1.34 ms | 160.0 GB/s (17.9%)  | 140.0 GB/s (15.6%)     |  1.34 |  0.4% |  0.00     | 0.0%     | HIDDEN |
| | **TOTAL** | | | | | 31.0× | | | | | **161.39** | **47.2%** | **30.56** | **8.9%** | EXP+HID |

Column meanings:
- `vol/call` = `(In + Out) × dtype_bytes` per rank — total per-call I/O volume per rank
  (asymmetric for a2a-v scatter/gather, exactly 2× msg-size for symmetric collectives).
- `algBw` = `msg_size_per_rank / time` — perceived per-rank bandwidth (nccl-tests
  "algBw"; uses `max(In, Out) × dtype_bytes` as the size).
- `busBw` = `algBw × bus_factor` — bandwidth on the bus (nccl-tests convention,
  comparable to per-link peak); `bus_factor = (N-1)/N` for AllToAll and `2(N-1)/N`
  for AllReduce.
- `ms/call` = total kernel time ÷ kernel count for the group.
- `where` = `EXPOSED (X%)` if >50% on critical path, `HIDDEN` if <5%, else `mixed (X%)`.

Variants of row 2 (DDP small-bucket AR uses several grid sizes; the dominant `(112,1,1)`
is shown above): `ncclDevKernel_Generic_1` also fires at `grid=(110,1,1)` and `(111,1,1)`.

## Comparing MI355X vs B200

Both runs train OneTrans Large (1.23B dense params) on 8 GPUs intra-node, batch=1024
per GPU, BF16, `torch.compile`. BF16 peaks: B200 = 2,250 TF/s, MI355X = 2,300 TF/s
(within 2%, so absolute TF/s and % peak are directly comparable).

### E2E

| Metric | B200 | MI355X | MI355X / B200 |
|---|---:|---:|---:|
| Throughput (samples/s) | 24,400 | ~23,500 | 0.96× |
| TFLOPS/GPU | 327–328 | 316 | 0.96× |
| MFU | 14.5–14.6% | 13.7% | 0.94× |
| Step time | 335.7 ms | 341 ms | 1.02× |

MI355X is ~4% behind B200 on overall throughput despite a ~2% higher peak BF16. The
gap comes mainly from GEMM efficiency (see below), partly compensated by similar
attention TF/s and similar exposed comm.

### Attention (FlashAttention-2 CK on AMD vs FlashAttention-4 CuTeDSL on NVIDIA)

Aggregate per training step (8 layers combined):

| Phase | B200 standalone | B200 e2e | MI355X standalone | MI355X e2e | MI355 / B200 (e2e) |
|---|---:|---:|---:|---:|---:|
| FWD time | 16.25 ms | 17.07 ms | 16.45 ms | 18.54 ms | 1.09× slower |
| FWD TF/s | 742 | 707 | 733 | 651 | 0.92× |
| BWD time | 38.20 ms | 41.32 ms | 56.97 ms | 59.21 ms | 1.43× slower |
| BWD TF/s | 632 | 584 | 530 | 509 | 0.87× |
| **FWD+BWD total** | 54.45 ms | 58.39 ms | 73.42 ms | 77.75 ms | **1.33× slower** |
| E2E overhead | +5.0% | +12.7% FWD, +3.9% BWD | | | |

CK FA-2 BWD on MI355X is ~1.4× slower than FA-4 BWD on B200 (the largest single
attention gap). FA-4's CuTeDSL kernel benefits from sm_100 TMA + WGMMA scheduling
that CK FA-2 hasn't yet incorporated. FWD is closer (~1.1× slower).

### GEMM (cuBLAS+nvJet on B200 vs hipBLASLt+Tensile on MI355X)

Aggregate:

| Metric | B200 | MI355X | MI355X / B200 |
|---|---:|---:|---:|
| Total GEMM time / step | 33.13 ms | 86.54 ms | **2.61× slower** |
| Aggregate TF/s | 1,092 | 836 | 0.77× |
| Aggregate % peak | 48.5% | 36.4% | — |
| Unique shapes | 88 | 89 | — |
| Total ops / step | ~129 | ~130 | — |

Same-shape per-kernel comparison (shapes appearing in both top-time tables):

| Shape (M, N, K) | role | B200 TF/s (% peak) | MI355X TF/s (% peak) | MI / B200 |
|---|---|---:|---:|---:|
| **FFN forward (M ≫ N, K)** | | | | |
| (1 536 000, 1 536, 384) | FFN up-proj | 1 200 (53%) | 770 (34%) | **0.64×** |
| (1 536 000, 384, 1 536) | FFN down-proj | 1 295 (58%) | 1 000 (44%) | 0.77× |
| (1 318 912, 1 536, 384) | FFN up-proj | 1 206 (54%) | 835 (36%) | 0.69× |
| (1 318 912, 384, 1 536) | FFN down-proj | 1 303 (58%) | 1 057 (46%) | 0.81× |
| (1 101 824, 384, 1 536) | FFN down-proj | 1 299 (58%) | 1 045 (45%) | 0.80× |
| (884 736, 1 536, 384) | FFN up-proj | 1 242 (55%) | 773 (34%) | **0.62×** |
| (884 736, 384, 1 536) | FFN down-proj | 1 302 (58%) | 1 054 (46%) | 0.81× |
| **Attention projection (M ≫ N, K)** | | | | |
| (1 536 000, 1 152, 384) | qkv proj | 1 032 (46%) | 732 (32%) | 0.71× |
| (1 536 000, 384, 1 152) | out proj | 1 295 (58%) | 931 (40%) | 0.72× |
| **Backward weight-grad (splitK, K ≫ M, N)** | | | | |
| (1 152, 384, 1 536 000) | dW for attn proj | 994 (44%) | 1 042 (45%) | **1.05×** |
| (1 536, 384, 1 536 000) | dW for FFN | 1 009 (45%) | 1 059 (46%) | **1.05×** |
| (384, 1 536, 1 536 000) | dW for FFN | 1 036 (46%) | 635 (28%) | **0.61×** |
| (384, 1 536, 1 318 912) | dW for FFN | 1 029 (46%) | 624 (27%) | 0.61× |
| (384, 1 536, 1 101 824) | dW for FFN | 829 (37%) | 617 (27%) | 0.74× |

Pattern:
- **Forward "tall-skinny" GEMMs** (M ≫ N, K): MI355X gets 0.6–0.8× of B200's per-shape TF/s.
- **Backward weight-gradient splitK with M ∈ {1152, 1536}**: MI355X is **competitive
  (1.05× — slightly faster)** because Tensile's MT256x256 tile naturally fits
  these MN dimensions.
- **Backward weight-gradient splitK with M = 384**: MI355X falls to **0.6×** —
  Tensile splitK kernel (`MT192x192`) is poorly tuned for this aspect ratio. This
  single shape pattern (M=384) accounts for the bulk of the GEMM aggregate gap.

### Communication (NCCL on B200 vs RCCL on MI355X)

| Metric | B200 | MI355X | MI355X / B200 |
|---|---:|---:|---:|
| Total NCCL GPU time / step | 31.34 ms | 161.39 ms | **5.15× more** |
| **Exposed (critical path) / step** | **16.76 ms** | **30.56 ms** | **1.82× more** |
| Hidden (overlapped) / step | 14.58 ms | 130.83 ms | 8.97× more |
| Hidden % | 47% | 81% | — |
| Total per-rank wire bytes / step | (similar) | 14.54 GB | — |

On both platforms, the **single critical-path collective is the embedding data a2a**
(FWD scatter + BWD gather of the 6.6 GB per-rank tensor):

| Embedding data a2a | B200 | MI355X | MI355X / B200 |
|---|---:|---:|---:|
| ms/call | 8.38 ms | 14.83 ms | 1.77× slower |
| algBw (% peak) | 804 GB/s (89% of NVLink5 peak) | 447 GB/s (50% of XGMI peak) | 0.56× |
| busBw (% peak) | 704 GB/s (78%) | 391 GB/s (44%) | 0.56× |
| Critical-path contribution / step | 16.76 ms | 29.65 ms | 1.77× |

NVIDIA achieves ~89% of NVLink5 peak for the data a2a; AMD achieves ~50% of XGMI peak.
The 1.8× exposed-comm gap on this single collective accounts for ~13 ms of MI355X's
5 ms step-time deficit vs B200; the remainder comes from GEMM and BWD attention.

MI355X has **5× more total NCCL GPU time** because of (a) the int32 keys-a2a RCCL
pathology (87 ms/step, fully hidden but using 25% of step time on the comm stream)
and (b) AllReduce gradient sync that takes 33 ms on MI355X vs 13 ms on B200 (smaller
buckets at lower busBw). Both are well-overlapped with compute and don't affect
wall-clock time on either platform — the only e2e-relevant comm gap is the data a2a.

### Bottom line

| | B200 | MI355X | gap explanation |
|---|---:|---:|---|
| E2E TFLOPS/GPU | 327 | 316 (0.97×) | small overall gap |
| Attention | strong (sm_100 TMA/WGMMA) | weaker (CK FA-2) | BWD 1.4× slower, FWD 1.1× slower |
| GEMM | 48% peak | 36% peak (0.77×) | splitK M=384 path poorly tuned in Tensile |
| Critical-path comm | 17 ms | 31 ms (1.8× more) | a2a busBw 78% NVLink vs 44% XGMI peak |

MI355X reaches 96% of B200's e2e despite individual subsystems being 0.6–0.8× as
efficient because: AllReduce (the comm cost most users worry about) is fully hidden
on both, fused-ops/embeddings are similar, and the overall step is paced by the
slowest stream (compute on B200, slightly more comm-on-the-side on MI355X). Closing
the remaining 4% gap requires improving (in priority): Tensile splitK for M=384 shapes,
CK FA-2 BWD kernel selection, and embedding data a2a busBw (would need MSCCL++).
