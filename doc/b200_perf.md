# OneTrans Large Performance on NVIDIA B200

Benchmark results for the OneTrans Large model on 8x NVIDIA B200 GPUs with
FlashAttention-4 (FAv4), DMP+pipeline distributed strategy, and torch.compile.

## Environment

| Component | Version |
|---|---|
| GPU | 8x NVIDIA B200 (182 GB HBM, sm_100) |
| Container | `nvcr.io/nvidia/pytorch:26.01-py3` |
| PyTorch | 2.10.0a0+a36e1d3 (NVIDIA fork) |
| CUDA | 13.1 (Forward Compatibility on driver 580.126.09) |
| FBGEMM_GPU | 1.5.0+cu130 |
| TorchRec | 1.4.0 |
| FlashAttention-2 | 2.7.4.post1 (pre-installed in container) |
| FlashAttention-4 | 4.0.0b10 (CuTeDSL, `flash-attn-4[cu13]`) |
| NCCL | 2.29.2+cuda13.1 |
| Python | 3.12 |

### FA4 + torch.compile integration

FlashAttention-4 beta does not yet register custom ops for torch.compile
(see [RFC #2154](https://github.com/Dao-AILab/flash-attention/issues/2154)).
The FA4 call is wrapped with `@torch._dynamo.disable` so Inductor compiles
everything else (FFN, norms, projections, embeddings) while FA4's internal
CuTeDSL JIT handles the attention kernels independently.

## Model Configuration

OneTrans Large from arXiv:2510.26104:

| Parameter | Value |
|---|---|
| Model type | OneTrans (Pyramid Transformer) |
| Parameters | 1.23B (dense) |
| d_model | 384 |
| n_heads | 6 (head_dim=64) |
| n_layers | 8 |
| L_S (S-tokens) | 1500 (history_length=500 × 3 groups) |
| L_NS (NS-tokens) | 16 |
| Pyramid schedule | ON (L_S shrinks 1500→16 across 8 layers) |
| Precision | BF16 (torch.amp autocast) |
| Embedding tables | 4 tables, ~15M total embeddings |

## Standalone Attention Kernel Benchmark

Per-layer FA4 kernel performance on a single B200 GPU, batch=1024,
causal attention with pyramid Q/KV shapes.

### FWD only

| Layer | Q_len | KV_len | FLOPs | Time (ms) | TF/s | % of peak |
|---|---|---|---|---|---|---|
| L0 | 1516 | 1516 | 3614.8G | 3.444 | 1050 | 46.7% |
| L1 | 1304 | 1516 | 3109.3G | 4.022 | 773 | 34.4% |
| L2 | 1092 | 1304 | 2239.7G | 3.117 | 719 | 31.9% |
| L3 | 880 | 1092 | 1511.5G | 2.255 | 670 | 29.8% |
| L4 | 668 | 880 | 924.6G | 1.511 | 612 | 27.2% |
| L5 | 456 | 668 | 479.1G | 0.945 | 507 | 22.5% |
| L6 | 244 | 456 | 175.0G | 0.422 | 414 | 18.4% |
| L7 | 32 | 244 | 12.3G | 0.212 | 58 | 2.6% |
| **Total** | | | **12066.3G** | **15.93ms** | **758** | **33.7%** |

### FWD + BWD

| Layer | Q_len | KV_len | FLOPs | Time (ms) | TF/s | % of peak |
|---|---|---|---|---|---|---|
| L0 | 1516 | 1516 | 10844.5G | 15.544 | 698 | 31.0% |
| L1 | 1304 | 1516 | 9328.0G | 16.553 | 564 | 25.1% |
| L2 | 1092 | 1304 | 6719.1G | 12.948 | 519 | 23.1% |
| L3 | 880 | 1092 | 4534.4G | 9.486 | 478 | 21.2% |
| L4 | 668 | 880 | 2773.8G | 7.127 | 389 | 17.3% |
| L5 | 456 | 668 | 1437.3G | 4.700 | 306 | 13.6% |
| L6 | 244 | 456 | 525.0G | 2.426 | 216 | 9.6% |
| L7 | 32 | 244 | 36.8G | 1.011 | 36 | 1.6% |
| **Total** | | | **36199.0G** | **69.80ms** | **519** | **23.1%** |

FLOPs per sample (FWD): 11.78 GFLOP | FLOPs per sample (FWD+BWD): 35.35 GFLOP

### Backend comparison (B=2048, all 8 pyramid layers combined, FWD only)

| Backend | Total (ms) | TF/s | % of peak |
|---|---|---|---|
| FAv2 (FlashAttention-2) | 61.35 | 393 | 17.5% |
| **FAv4 (FlashAttention-4)** | **31.76** | **760** | **33.8%** |
| SDPA (cuDNN) | 30.71 | 786 | 34.9% |

FAv4 is **1.93x faster than FAv2**. SDPA (cuDNN) is slightly faster than FAv4
on these asymmetric pyramid shapes due to cuDNN's flexible kernel dispatch.

## End-to-End Training Performance

OneTrans Large, synthetic data, 8x B200, DMP+pipeline, torch.compile=ON.

### Best configuration: FAv4 + DMP + pipeline

| Setting | Value |
|---|---|
| attention_impl | fav4 |
| distributed | DMP + TrainPipelineSparseDist |
| torch_compile | ON (Inductor, FA4 excluded via dynamo.disable) |
| batch_size | 8192 (1024/GPU) |

| Metric | Value |
|---|---|
| Steady-state throughput | ~24,000 samples/s |
| TFLOPS/GPU | 322 |
| MFU | 14.3% |
| GPU memory | 121 GB / 178 GB (68%) |
| Training time (100 steps) | 74.1s |
| Estimated FLOPs/sample (fwd+bwd) | 107.3 GFLOP |

### Attention kernel overhead in training

Comparing standalone FA4 benchmark to actual training trace (step 52):

| Layer | Bench (ms) | Trace (ms) | Overhead |
|---|---|---|---|
| L0 (1516×1516) | 3.444 | 3.605 | 5% |
| L1 (1304×1516) | 4.022 | 4.114 | 2% |
| L2 (1092×1304) | 3.117 | 3.201 | 3% |
| L3 (880×1092) | 2.255 | 2.330 | 3% |
| L4 (668×880) | 1.511 | 1.694 | 12% |
| L5 (456×668) | 0.945 | 1.010 | 7% |
| L6 (244×456) | 0.422 | 0.449 | 6% |
| L7 (32×244) | 0.212 | 0.230 | 9% |
| **Total** | **15.93ms** | **16.63ms** | **4.4%** |

FA4 kernels achieve **95.6% of standalone benchmark throughput** in actual
training — only 4.4% overhead from the DMP pipeline scheduling and DDP sync.


