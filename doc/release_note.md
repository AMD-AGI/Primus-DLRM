# v0.1.0 Release Notes

**Tag:** `v0.1.0`
**Date:** March 24, 2026

## Overview

Initial release of the OneTrans DLRM benchmark suite for AMD MI355X GPUs. Supports single-node and multi-node distributed training with TorchRec embedding sharding on ROCm 7.1.

## Features

### Model
- **OneTrans** unified causal Transformer with split-history behavior pools, mixed parameterization, and pyramid S-token pruning
- **DLRM++ baseline** with TorchRec EmbeddingBagCollection and configurable interaction modules (ConcatMLP, Dot, DCNv2)
- Multi-task prediction heads (listen+, like, dislike, played_ratio)

### Data
- Yambda dataset support (50M / 500M / 5B variants)
- Automated preprocessing with global temporal split and session segmentation
- Memory-mapped data storage for fast loading
- Optional counter features (user/item/cross, multi-window)

### Training
- BF16 mixed precision via `torch.amp.autocast`
- TF32 matmul mode (`allow_tf32` config option)
- TorchRec `TrainPipelineSparseDist` for pipelined embedding distribution
- FBGEMM fused embedding optimizers
- Gradient clipping, cosine LR schedule with warmup

### Distributed
- **Single-node DMP + pipeline** — 8 GPUs with TorchRec DistributedModelParallel
- **Multi-node DDP** — across 2+ nodes with RCCL
- **Multi-node DMP + pipeline** — with `--ainic` flag for AINIC clusters
- Docker-based launch via `launch_slurm_docker.sh`
- Bare metal launch via `launch_slurm.sh`

### AINIC Multi-Node Support
- Custom RCCL + `librccl-anp.so` network plugin for inter-node RDMA over AINIC adapters
- `--ainic` flag in `launch_slurm_docker.sh` enables automatic RCCL/AINIC setup
- AINIC-tuned NCCL parameters for optimal transport configuration

### Evaluation
- NDCG@{10,50,100} and Recall@{10,50,100}
- Fast eval (top-5K items) and full-catalog eval modes

## Verified Stack

| Component | Version |
|---|---|
| PyTorch | 2.10.0+rocm7.1 |
| TorchRec | 1.4.0 |
| FBGEMM-GPU | 1.5.0+rocm7.1 |
| ROCm | 7.1.1 |
| GPU | AMD Instinct MI355X (gfx950) |
| Docker | `tasimage/primus:pr-609-ainic` |

## Known Limitations

- Multi-node DMP in Docker requires `--ainic` flag (uses custom RCCL, not PyTorch bundled)
