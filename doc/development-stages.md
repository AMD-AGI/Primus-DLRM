# DLRM Development Stages

This document consolidates the project development plan across all stages.
For detailed specs, see the individual `stage*-spec.md` files.

---

## Stage 1A: Baseline DLRM++

**Goal:** End-to-end DLRM++ implementation on single GPU with multi-task prediction.

### Data Pipeline
- **Dataset:** Yambda (50M / 500M / 5B variants) from HuggingFace
- **Preprocessing:** Global temporal split (300d train / 30min gap / 1d test), session segmentation (30-min inactivity gap), per-user history construction
- **Features:** User history (last-L pooled embeddings), candidate item features, optional counter features (user/item/cross, multi-window)
- **Labels:** Natural events — listen+ (played_ratio >= 50%), like, dislike, played_ratio

### Model
- **Architecture:** Two-tower (user + item) with TorchRec EmbeddingBagCollection (pooled) + EmbeddingCollection (unpooled)
- **Interaction:** ConcatMLP (default), DotInteraction, or DCNv2
- **Heads:** Multi-task — listen_plus (BCE), like (BCE), dislike (BCE), played_ratio (MSE)
- **Training:** TorchRec TrainPipelineSparseDist, fused embedding optimizers via FBGEMM TBE, BF16 AMP

### Evaluation
- Fast eval: top-5K popular items (~11s)
- Full-catalog eval: all items with non-zero popularity (~508K for 50M, ~10 min)
- Metrics: NDCG@{10,50,100}, Recall@{10,50,100}

### Results
- NDCG@10 = 0.0050 (epoch 2, full-catalog) — 15x gap vs SASRec (0.0748), expected due to mean-pooling destroying sequential info

---

## Stage 1B: OneTrans

**Goal:** Replace DLRM++ pooling with a unified causal Transformer for joint sequence modeling and feature interaction.

### Architecture (from [OneTrans, WWW 2026](https://arxiv.org/abs/2510.26104))
- **S-tokens:** User history split into 3 behavior pools (listen+, like, skip), each with per-behavior MLP tokenizer. Concatenated to form `[B, 3L, d_model]`
- **NS-tokens:** Candidate + user context features projected via auto-split tokenizer into `[B, n_ns, d_model]`
- **Mixed parameterization:** S-tokens share Q/K/V+FFN weights; each NS-token gets its own
- **Pyramid stack:** Progressive S-token query pruning per layer, funneling information into NS-tokens
- **Prediction:** Final NS-token states → multi-task heads (same as Stage 1A)

### Key Design Decisions
- Single new file: `primus_dlrm/models/onetrans.py`
- Reuses everything else: same batch format, loss, trainer, evaluation
- Per-position `nn.Embedding` instead of `nn.EmbeddingBag` (no mean-pooling)
- Batched NS Q/K/V via `torch.einsum` (avoids Python loop)

### Scaling Configs
| Config | d_model | n_heads | n_layers | n_ns | ~Dense Params |
|--------|:-------:|:-------:|:--------:|:----:|:-------------:|
| OneTransS | 128 | 4 | 4 | 8 | ~8M |
| OneTransM | 192 | 6 | 6 | 12 | ~30M |
| OneTransL | 256 | 8 | 8 | 16 | ~80M |

### Phase 2: Cross-Candidate KV Caching
- `encode_user()` → cache S-side K/V per layer (once per unique user)
- `score_candidates()` → NS-side only, using cached K/V
- Evaluation speedup: ~1000x for 10K candidates (O(L + N·L_NS) vs O(N·L))

---

## Stage 2: Distributed Training

**Goal:** Multi-node training with embedding sharding (DMP) and dense replication (DDP/FSDP) on AMD MI355X.

### Phase 1: TorchRec Embedding Migration
- Replace `nn.Embedding`/`EmbeddingBag` with TorchRec `EBC` + `EC`
- Unified `TorchRecEmbeddings` module with `forward(features_dict)` API
- Shared physical tables with multiple feature names
- Verification: loss parity within ±5% at steps 1000/3000/5000

### Phase 2: Distributed Infrastructure
- Process group init (torchrun-compatible, RCCL backend)
- Model wrapping: `DistributedModelParallel` (DMP) + DDP/FSDP
- `DistributedSampler`, rank-0-only logging/checkpointing
- Launch: `torchrun --nproc_per_node=8 --nnodes=N scripts/run_distributed.py`

### Phase 3: Auto Sharding Planner
- Heuristic-based: row-wise (tables > 10MB), table-wise (< 10MB), column-wise (dim >= 256)
- Dense strategy: DDP if < 10% HBM, else FSDP

### Phase 4: 5B Dataset Support
- Chunked preprocessing (10M rows/chunk via `polars.scan_parquet`)
- Memory-mapped event store (`np.load(mmap_mode="r")`)

### Phase 5: Convergence Parity
- Compare 1/2/8 GPU runs against single-GPU baseline
- Pass: loss within ±5% at steps 1000/3000/5000

### Known Issues
- **TorchRec DMP segfault:** `SplitTableBatchedEmbeddingBagsCodegen` crashes during all-to-all for row-wise sharded tables on ROCm 7.1/MI355X. DDP/FSDP without DMP work correctly. Tracked as platform issue.
- **Contrastive loss under DDP:** `forward_cross()` bypasses DDP wrapper. Contrastive loss disabled for distributed training.

---

## Stage 3: Embedding Sharding with TorchRec DMP

**Goal:** Enable TorchRec `DistributedModelParallel` for embedding sharding, unlocking scalability to 5B dataset and 32+ GPUs.

### Phase 0: FBGEMM Enablement (First Priority)
- Validate FBGEMM TBE kernels on ROCm (`SplitTableBatchedEmbeddingBagsCodegen`)
- Test DMP with each sharding type individually: DATA_PARALLEL → TABLE_WISE → ROW_WISE
- Test RCCL all-to-all independently
- Root-cause any segfaults from Stage 2's failed DMP attempt

### Phase 1: DMP Integration
- Meta device init for embeddings (`torch.device("meta")`)
- `EmbeddingShardingPlanner` with hardware-aware `Topology`
- Split optimizer: dense (AdamW) + embedding (DMP fused TBE)
- Convergence parity: loss ±2%, NDCG ±5% vs replicated baseline

### Phase 2: Performance
- `TrainPipelineSparseDist` for comm-compute overlap
- Quantized all-to-all (FP32 → FP16/BF16, 2x volume reduction)
- Hardware-aware topology (XGMI 900 GB/s intra, AINIC 100 GB/s inter)

### Phase 3: Scale Testing
- 5B dataset with large tables requiring sharding
- 32-128 hashed cross-feature tables stress test
- 2-node (16 GPU) and 4-node (32 GPU) scaling runs

### Memory Savings (OneTrans, 8 GPUs, Row-Wise)
| | Replicated | Sharded | Savings |
|---|:---:|:---:|:---:|
| 50M dataset | 495 MB/GPU | 62 MB/GPU | 87.5% |
| 5B dataset | 12 GB/GPU | 1.5 GB/GPU | 87.5% |

### Fallback: Native PyTorch Sharding
If TorchRec DMP is fundamentally broken on ROCm: `nn.Embedding` + `dist.all_to_all_single` with custom `autograd.Function`. More code, no FBGEMM, but works on any NCCL backend.

### Sharding Types (TorchRec)
| Type | Description | Use Case |
|------|-------------|----------|
| DATA_PARALLEL | Full copy on each GPU, allreduce grads | Baseline / fallback |
| TABLE_WISE | Whole table on one GPU | Many small-medium tables |
| ROW_WISE | Rows split across GPUs | Large tables (item, user) |
| COLUMN_WISE | Embedding dim split | Large embedding dims (>=256) |
| GRID_SHARD | 2D (rows × columns) | Extreme-scale tables |
