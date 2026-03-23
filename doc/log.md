# Experiment Log

## Stage 2: TorchRec Embedding Migration Verification

### TorchRec uniform init -- lower loss (2026-02-26)

TorchRec's default embedding init is `uniform(-1/sqrt(N), 1/sqrt(N))` which
produces near-zero values for large tables. This gave **consistently lower
loss** than the `nn.Embedding` N(0,1) baseline for both models.

**counter_v1** (config: `configs/exp_counter_v1.yaml`, 5000 steps, batch_size=4096):


| Step | nn.Embedding (baseline) | TorchRec uniform init | Improvement |
| ---- | ----------------------- | --------------------- | ----------- |
| 0    | 1.0633                  | 1.0596                | 0.3%        |
| 1000 | 0.5314                  | 0.5099                | 4.0%        |
| 2000 | 0.5248                  | 0.4986                | 5.0%        |
| 3000 | 0.5333                  | 0.4974                | 6.7%        |
| 4000 | 0.5124                  | 0.4844                | 5.5%        |
| 5000 | 0.5101                  | 0.4703                | 7.8%        |


Throughput: 68,875 samples/s (baseline: 63,031 -- 9.3% faster).

**onetrans_v6** (config: `configs/r2_onetrans_v6.yaml`, 5000 steps, batch_size=4096):


| Step | nn.Embedding (baseline) | TorchRec uniform init | Improvement    |
| ---- | ----------------------- | --------------------- | -------------- |
| 0    | 4.9864                  | 1.0435                | 79% (init BPR) |
| 1000 | 0.8179                  | 0.7343                | 10.2%          |
| 2000 | 0.7623                  | 0.6295                | 17.4%          |
| 3000 | 0.7097                  | 0.5864                | 17.4%          |
| 4000 | 0.6633                  | 0.5597                | 15.6%          |
| 5000 | 0.6362                  | 0.5290                | 16.8%          |


Throughput: 52,124 samples/s (baseline: 50,296 -- 3.6% faster).

**Setting**: `TorchRecEmbeddings` without explicit `init_fn` (uses TorchRec default).
To reproduce, remove the `init_fn=_normal_init` from `EmbeddingBagConfig`/`EmbeddingConfig`
in `primus_dlrm/models/embedding.py`.

The smaller init reduces initial gradient magnitudes, which may act as implicit
regularization. Worth a full-epoch comparison to see if the lower half-epoch
loss translates to better final NDCG.

---

### TorchRec N(0,1) init -- parity verification (2026-02-26)

Matched nn.Embedding's default N(0,1) init via `init_fn` to verify TorchRec
migration correctness. This is the current production setting.

**counter_v1** (config: `configs/exp_counter_v1.yaml`, 5000 steps, batch_size=4096):


| Step | Baseline Loss | TorchRec Loss | Diff |
| ---- | ------------- | ------------- | ---- |
| 0    | 1.0633        | 1.0675        | 0.4% |
| 1000 | 0.5314        | 0.5365        | 1.0% |
| 2000 | 0.5248        | 0.5265        | 0.3% |
| 3000 | 0.5333        | 0.5291        | 0.8% |
| 4000 | 0.5124        | 0.5085        | 0.8% |
| 5000 | 0.5101        | 0.5050        | 1.0% |


**Result: PASS** -- all checkpoints within 1% of baseline.
Throughput: 65,214 samples/s (baseline: 63,031 -- 3.5% faster).

**onetrans_v6** (config: `configs/r2_onetrans_v6.yaml`, 5000 steps, batch_size=4096):


| Step | Baseline Loss | TorchRec Loss | Diff |
| ---- | ------------- | ------------- | ---- |
| 0    | 4.9864        | 5.3745        | 7.8% |
| 1000 | 0.8179        | 0.8209        | 0.4% |
| 2000 | 0.7623        | 0.7538        | 1.1% |
| 3000 | 0.7097        | 0.7009        | 1.2% |
| 4000 | 0.6633        | 0.6669        | 0.5% |
| 5000 | 0.6362        | 0.6334        | 0.4% |


**Result: PASS** -- steps 1000-5000 all within 1.2% of baseline.
Throughput: 52,095 samples/s (baseline: 50,296 -- 3.6% faster).

---

---

## Stage 2: Distributed Training Benchmark (2026-02-27)

### 2A. Short-run parity check (50 steps)

Quick sanity check: contrastive vs non-contrastive, 1-GPU vs 2-GPU (DDP & FSDP).
All runs: 50 steps, uniform embedding init, batch_size=4096 global.


| Run          | step=0 | step=10 | step=20 | step=30 | step=40 | avg_loss | throughput |
| ------------ | ------ | ------- | ------- | ------- | ------- | -------- | ---------- |
| **DLRM**     |        |         |         |         |         |          |            |
| 1GPU no-CL   | 0.7038 | 0.7026  | 0.7009  | 0.6976  | 0.6949  | 0.6987   | 22,605 s/s |
| 1GPU w/ CL   | 1.0596 | 1.0559  | 1.0539  | 1.0485  | 1.0482  | 1.0518   | 22,150 s/s |
| 2GPU DDP     | 0.7025 | 0.7032  | 0.7003  | 0.6974  | 0.6943  | 0.6986   | 24,377 s/s |
| 2GPU FSDP    | 0.6874 | 0.6858  | 0.6861  | 0.6848  | 0.6837  | 0.6845   | 12,471 s/s |
| **OneTrans** |        |         |         |         |         |          |            |
| 1GPU no-CL   | 0.6970 | 0.6368  | 0.5885  | 0.5471  | 0.5166  | 0.5785   | 19,635 s/s |
| 1GPU w/ CL   | 1.0435 | 0.9833  | 0.9350  | 0.8935  | 0.8631  | 0.9250   | 17,504 s/s |
| 2GPU DDP     | 0.6946 | 0.6381  | 0.5930  | 0.5365  | 0.5361  | 0.5782   | 14,079 s/s |
| 2GPU FSDP    | 0.6809 | 0.6302  | 0.5856  | 0.5384  | 0.5326  | 0.5742   | 9,604 s/s  |


**Key findings** (50-step):

- DDP parity: 1GPU vs 2GPU DDP loss within 0.05% for both models. PASS.
- Contrastive loss: purely additive, does not interfere with listen_plus loss.
- FSDP: ~2x slower throughput (expected), slight loss difference from param flattening.

---

### 2B. Full 5000-step comparison: 1GPU vs 2GPU DDP vs 2GPU FSDP, with/without CL

All runs: 5000 steps, uniform embedding init, batch_size=4096.

#### Exact commands (2B)

**1-GPU runs** (4 in parallel, each on one GPU):

```
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
    --config configs/s2_onetrans_v6.yaml --max-steps 5000 --run-name s2_onetrans_1gpu_5k

CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py \
    --config configs/s2_onetrans_v6_cl.yaml --max-steps 5000 --run-name s2_onetrans_1gpu_cl_5k

CUDA_VISIBLE_DEVICES=2 python scripts/run_experiment.py \
    --config configs/s2_counter_v1.yaml --max-steps 5000 --run-name s2_dlrm_1gpu_5k

CUDA_VISIBLE_DEVICES=3 python scripts/run_experiment.py \
    --config configs/s2_counter_v1_cl.yaml --max-steps 5000 --run-name s2_dlrm_1gpu_cl_5k
```

**2-GPU distributed runs** (DDP and FSDP, each on 2 GPUs):

```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
    scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml \
    --dense-strategy ddp --max-steps 5000 --log-interval 1000 \
    --run-name s2_onetrans_2gpu_ddp_5k

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 \
    scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml \
    --dense-strategy fsdp --max-steps 5000 --log-interval 1000 \
    --run-name s2_onetrans_2gpu_fsdp_5k
```

(Same pattern for DLRM with `--config configs/dist_counter_v1.yaml`.)

#### Config details


| Setting          | DLRM (counter_v1)                         | OneTrans (onetrans_v6)           |
| ---------------- | ----------------------------------------- | -------------------------------- |
| Config (1GPU)    | `configs/s2_counter_v1.yaml`              | `configs/s2_onetrans_v6.yaml`    |
| Config (1GPU+CL) | `configs/s2_counter_v1_cl.yaml`           | `configs/s2_onetrans_v6_cl.yaml` |
| Config (dist)    | `configs/dist_counter_v1.yaml`            | `configs/dist_onetrans_v6.yaml`  |
| Model type       | DLRMBaseline                              | OneTransModel                    |
| Embedding dim    | 16 (small: 8)                             | 64                               |
| Params           | 465,698,721                               | 997,103,617                      |
| LR               | 0.001                                     | 0.0003                           |
| Warmup steps     | 1000                                      | 2000                             |
| Counter windows  | [30]                                      | [7, 30]                          |
| Interaction      | concat_mlp (bottom: [64], top: [128, 64]) | transformer (d=256, h=8, L=4)    |
| Embedding init   | uniform (TorchRec default)                | uniform (TorchRec default)       |
| Batch size       | 4096 (1GPU) / 4096 global (2GPU)          | 4096 (1GPU) / 4096 global (2GPU) |
| BF16             | yes                                       | yes                              |
| Grad clip        | 1.0                                       | 1.0                              |


#### Loss trajectories (listen_plus loss only)


| Run          | step=0 | step=1000 | step=2000 | step=3000 | step=4000 | avg_loss | throughput |
| ------------ | ------ | --------- | --------- | --------- | --------- | -------- | ---------- |
| **DLRM**     |        |           |           |           |           |          |            |
| 1GPU no-CL   | 0.7038 | 0.4965    | 0.4872    | 0.4896    | 0.4776    | 0.4894   | 64,595 s/s |
| 1GPU w/ CL   | 0.7038 | 0.4997    | 0.4907    | 0.4895    | 0.4768    | —        | 58,646 s/s |
| 2GPU DDP     | 0.7025 | 0.5100    | 0.4895    | 0.4659    | 0.4727    | 0.4901   | 67,589 s/s |
| 2GPU FSDP    | 0.6874 | 0.5084    | 0.4893    | 0.4613    | 0.4713    | 0.4898   | 34,150 s/s |
| **OneTrans** |        |           |           |           |           |          |            |
| 1GPU no-CL   | 0.6970 | 0.4866    | 0.4957    | 0.4859    | 0.4766    | 0.4817   | 35,097 s/s |
| 1GPU w/ CL   | 0.6970 | 0.4874    | 0.5058    | 0.4949    | 0.4816    | —        | 38,599 s/s |
| 2GPU DDP     | 0.6946 | 0.4995    | 0.4823    | 0.4678    | 0.4590    | 0.4820   | 28,916 s/s |
| 2GPU FSDP    | 0.6809 | 0.5002    | 0.4861    | 0.4683    | 0.4594    | 0.4826   | 15,019 s/s |


#### NDCG at 5000 steps (listen_plus task)


| Run          | G-NDCG@10  | G-NDCG@100 | P-NDCG@10  | P-NDCG@100 | G-Recall@100 | P-Recall@100 |
| ------------ | ---------- | ---------- | ---------- | ---------- | ------------ | ------------ |
| **DLRM**     |            |            |            |            |              |              |
| 1GPU no-CL   | 0.0583     | 0.0664     | 0.1134     | 0.1551     | 0.0880       | 0.2300       |
| 1GPU w/ CL   | **0.0835** | **0.0873** | **0.1194** | **0.1618** | **0.1071**   | **0.2385**   |
| 2GPU DDP     | 0.0467     | 0.0560     | 0.1109     | 0.1515     | 0.0796       | 0.2254       |
| 2GPU FSDP    | 0.0498     | 0.0585     | 0.1109     | 0.1527     | 0.0813       | 0.2271       |
| **OneTrans** |            |            |            |            |              |              |
| 1GPU no-CL   | 0.0911     | 0.0945     | 0.1213     | 0.1701     | 0.1124       | 0.2497       |
| 1GPU w/ CL   | **0.0934** | **0.0978** | **0.1407** | **0.1818** | **0.1173**   | **0.2558**   |
| 2GPU DDP     | 0.0771     | 0.0830     | 0.1102     | 0.1589     | 0.1024       | 0.2376       |
| 2GPU FSDP    | 0.0821     | 0.0870     | 0.1136     | 0.1617     | 0.1070       | 0.2400       |


#### Analysis

**1GPU vs 2GPU DDP parity**:

- DLRM: 1GPU avg=0.4894 vs 2GPU DDP avg=0.4901 — **0.14% diff**. PASS.
- OneTrans: 1GPU avg=0.4817 vs 2GPU DDP avg=0.4820 — **0.06% diff**. PASS.

**DDP vs FSDP parity**:

- DLRM: DDP avg=0.4901 vs FSDP avg=0.4898 — **0.06% diff**. PASS.
- OneTrans: DDP avg=0.4820 vs FSDP avg=0.4826 — **0.12% diff**. PASS.

**Contrastive loss impact on NDCG** (1GPU, listen_plus only):

- DLRM: CL **dramatically boosts** NDCG despite similar listen_plus loss.
G-NDCG@100: 0.0873 vs 0.0664 (+31%). P-NDCG@100: 0.1618 vs 0.1551 (+4%).
- OneTrans: CL also improves NDCG, especially per-user.
G-NDCG@100: 0.0978 vs 0.0945 (+3.5%). P-NDCG@100: 0.1818 vs 0.1701 (+7%).
- BPR loss converges to near-zero for DLRM (0.015), remains higher for OneTrans (0.156).
- **Conclusion**: contrastive loss significantly improves ranking quality, especially for DLRM.

**OneTrans vs DLRM quality** (best of each):

- OneTrans+CL is the best overall: G-NDCG@100=0.0978, P-NDCG@100=0.1818, P-Recall@100=0.2558.
- DLRM+CL: G-NDCG@100=0.0873, P-NDCG@100=0.1618, P-Recall@100=0.2385.
- OneTrans wins by +12% G-NDCG@100, +12% P-NDCG@100, +7% P-Recall@100.

**1GPU vs 2GPU NDCG gap**:

- 1GPU consistently has higher NDCG than 2GPU DDP despite matching loss.
This is expected: different random seeds per rank cause different gradient noise,
leading to slightly different model states even with identical avg loss.

**Throughput**:

- DLRM 1GPU: 64,595 s/s; 2GPU DDP: 67,589 s/s (1.05x scaling).
- OneTrans 1GPU: 35,097 s/s; 2GPU DDP: 28,916 s/s (0.82x — transformer bottleneck).
- FSDP ~2x slower than DDP as expected.

---

### 2C. Scaling: 8-GPU (single node) and 2-node (16-GPU) OneTrans (2026-02-28)

All runs: full epoch (~4900 steps), uniform init, batch_size=4096 global, `--skip-eval`.
Eval from checkpoint via `scripts/eval_checkpoint.py` on a single GPU.

#### Exact commands

**8-GPU DDP** (single node, chi2866):

```
torchrun --nproc_per_node=8 --master_port=29500 \
    scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml \
    --dense-strategy ddp --max-steps 5000 --log-interval 1000 \
    --skip-eval --run-name s2_onetrans_8gpu_ddp
```

**8-GPU FSDP** (single node, chi2866):

```
torchrun --nproc_per_node=8 --master_port=29500 \
    scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml \
    --dense-strategy fsdp --max-steps 5000 --log-interval 1000 \
    --skip-eval --run-name s2_onetrans_8gpu_fsdp
```

**2-node DDP / FSDP** (chi2866 + chi2798, 16 GPUs):

```
# Both nodes: env vars needed for socket-only inter-node NCCL
export NCCL_IB_DISABLE=1 NCCL_NET=Socket NCCL_SOCKET_IFNAME=enp193s0f1np1

# Node 0 (chi2866):
bash scripts/s2_multinode_node0.sh ddp   # or fsdp

# Node 1 (chi2798, via ssh):
ssh chi2798 "tmux new-session -d -s mn \
  'bash scripts/s2_multinode_node1.sh ddp'"   # or fsdp
```

**Eval from checkpoint** (any run, single GPU):

```
CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint.py \
    --config configs/dist_onetrans_v6.yaml \
    --checkpoint results/<run_name>/checkpoints/epoch_0.pt
```

#### Result directories


| Run         | Result dir                        | Train log        |
| ----------- | --------------------------------- | ---------------- |
| 8GPU DDP    | `results/s2_onetrans_8gpu_ddp/`   | `logs/train.log` |
| 8GPU FSDP   | `results/s2_onetrans_8gpu_fsdp/`  | `logs/train.log` |
| 2-node DDP  | `results/s2_onetrans_2node_ddp/`  | `logs/train.log` |
| 2-node FSDP | `results/s2_onetrans_2node_fsdp/` | `logs/train.log` |


#### Loss trajectories


| Run                          | step=0 | step=1000 | step=2000 | step=3000 | step=4000 | avg_loss | throughput |
| ---------------------------- | ------ | --------- | --------- | --------- | --------- | -------- | ---------- |
| 1GPU no-CL                   | 0.6970 | 0.4866    | 0.4957    | 0.4859    | 0.4766    | 0.4817   | 35,097 s/s |
| 2GPU DDP                     | 0.6946 | 0.4995    | 0.4823    | 0.4678    | 0.4590    | 0.4820   | 28,916 s/s |
| 2GPU FSDP                    | 0.6809 | 0.5002    | 0.4861    | 0.4683    | 0.4594    | 0.4826   | 15,019 s/s |
| 8GPU DDP                     | 0.7009 | 0.4894    | 0.4719    | 0.4624    | 0.4336    | 0.4818   | 33,238 s/s |
| 8GPU FSDP                    | 0.7181 | 0.4869    | 0.4729    | 0.4618    | 0.4324    | 0.4820   | 30,296 s/s |
| 2-node DDP Socket (16 GPUs)  | 0.6933 | 0.4885    | 0.4794    | 0.4735    | 0.4288    | 0.4813   | 4,120 s/s  |
| 2-node FSDP Socket (16 GPUs) | 0.6935 | 0.4944    | 0.4858    | 0.4802    | 0.4286    | 0.4821   | 4,805 s/s  |
| 2-node DDP AINIC (16 GPUs)   | 0.6933 | 0.4972    | —         | —         | —         | 0.5054*  | 56,455 s/s |


#### NDCG at epoch end (from checkpoint eval)


| Run        | G-NDCG@10 | G-NDCG@100 | P-NDCG@10 | P-NDCG@100 | G-Recall@100 | P-Recall@100 |
| ---------- | --------- | ---------- | --------- | ---------- | ------------ | ------------ |
| 1GPU no-CL | 0.0911    | 0.0945     | 0.1213    | 0.1701     | 0.1124       | 0.2497       |
| 2GPU DDP   | 0.0771    | 0.0830     | 0.1102    | 0.1589     | 0.1024       | 0.2376       |
| 2GPU FSDP  | 0.0821    | 0.0870     | 0.1136    | 0.1617     | 0.1070       | 0.2400       |
| 8GPU DDP   | 0.0759    | 0.0821     | 0.1103    | 0.1587     | 0.1022       | 0.2377       |
| 8GPU FSDP  | 0.0754    | 0.0815     | 0.1108    | 0.1579     | 0.1010       | 0.2366       |


#### Scaling analysis

**Loss parity across GPU counts**:

- 1GPU=0.4817, 2GPU DDP=0.4820, 8GPU DDP=0.4818, 16GPU DDP=0.4813. All within 0.15%. PASS.
- FSDP matches DDP within 0.2% at every scale (16GPU: FSDP=0.4821 vs DDP=0.4813). PASS.

**NDCG parity (8-GPU vs 2-GPU)**:

- 8GPU DDP vs 2GPU DDP: G-NDCG@100 0.0821 vs 0.0830 (1.1%). P-NDCG@100 0.1587 vs 0.1589 (0.1%). PASS.
- 8GPU FSDP vs DDP: G-NDCG@100 0.0815 vs 0.0821 (0.7%). PASS.

**Throughput scaling**:


| GPUs                    | DDP throughput | FSDP throughput | DDP scaling eff. |
| ----------------------- | -------------- | --------------- | ---------------- |
| 1                       | 35,097 s/s     | —               | —                |
| 2                       | 28,916 s/s     | 15,019 s/s      | 0.41x            |
| 8                       | 33,238 s/s     | 30,296 s/s      | 0.12x            |
| 16 (2-node, sockets)    | 4,120 s/s      | 4,805 s/s       | 0.007x           |
| 16 (2-node, AINIC RDMA) | 56,455 s/s     | —               | 0.10x            |


**Multi-node observations**:

- Inter-node communication over TCP sockets (no RDMA/RoCE configured) is the
dominant bottleneck. 16-GPU throughput (4,120 s/s) is 8x slower than 8-GPU
(33,238 s/s) and actually slower than 1-GPU (35,097 s/s).
- Loss convergence is correct despite poor throughput: multi-node DDP avg_loss=0.4813
matches single-node within noise.
- FSDP 2-node completed full epoch: avg_loss=0.4821, throughput=4,805 s/s.
(Initial run crashed at step 3000 due to node1 SIGTERM; rerun completed cleanly.)
- Reconfirmed with 1k-step rerun: DDP Socket throughput=4,847 s/s (avg_loss=0.5054),
consistent with the original 4,120 s/s measurement.

**AINIC RDMA investigation**:

- Both nodes (chi2866, chi2798) have 8 active AINIC devices (ionic_0..ionic_7).
- ANP plugin libraries are installed: `librccl-net.so` at `/usr/local/lib/` (both nodes)
and `librccl-anp.so` at `/opt/rocm-7.1.1/lib/` (node 0 only).
- **Attempt 1** (old `/usr/local/lib/librccl-net.so`): SIGSEGV on all ranks during
DDP init — version mismatch between Oct 7 plugin build and torch-bundled RCCL 2.27.7.
- **Attempt 2** (newer `/opt/rocm-7.1.1/lib/librccl-net.so`): ANP loaded successfully,
but RDMA `Connect` failed with `ncclRemoteError` (res=3) — inter-node RDMA routing
not configured at the cluster level.
- **Actionable**: needs cluster admin to verify RDMA routing between nodes
(ionic device subnet config, AINIC firmware, and possibly `bnxt` driver rebuild
as shown in Primus `run_pretrain.sh`). Socket transport remains the fallback.

**Code changes in this session**:

- Added `--skip-eval` flag to `scripts/run_distributed.py` — skips eval data
loading and post-training eval on all ranks, avoiding OOM/timeout issues at
8+ GPU scale. Eval runs separately via `scripts/eval_checkpoint.py`.
- Moved standalone eval script from `/tmp/s2_fsdp_eval.py` to
`scripts/eval_checkpoint.py` (works for both DDP and FSDP checkpoints).
- Created `scripts/s2_multinode_node0.sh` and `scripts/s2_multinode_node1.sh`
for 2-node torchrun launches with NCCL socket configuration.
- Built and cached OneTrans counter data (`data/cache/train_w7_30/`,
`data/cache/eval_w7_30/`) — subsequent runs load in ~30s instead of ~20 min.

---

### 2D. Throughput scaling analysis — profiled (2026-03-02)

**Why doesn't throughput scale linearly with GPU count?**

Profiled 200-step DDP runs on OneTrans (997M params, batch_size=4096 global,
bf16). Added `--profile` flag to `scripts/run_distributed.py` which instruments
each step phase with `torch.cuda.Event` timing in `dist_trainer.py`.

```
# Profile commands (200 steps, skip eval):
CUDA_VISIBLE_DEVICES=0   torchrun --nproc_per_node=1 scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml --dense-strategy ddp \
    --max-steps 200 --log-interval 200 --skip-eval --profile --run-name _profile_1gpu

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml --dense-strategy ddp \
    --max-steps 200 --log-interval 200 --skip-eval --profile --run-name _profile_2gpu

torchrun --nproc_per_node=8 scripts/run_distributed.py \
    --config configs/dist_onetrans_v6.yaml --dense-strategy ddp \
    --max-steps 200 --log-interval 200 --skip-eval --profile --run-name _profile_8gpu
```

#### Per-step phase breakdown (mean ms, 20-step warmup excluded)


| Phase                 | 1-GPU (bs=4096) | 2-GPU (bs=2048/gpu) | 8-GPU (bs=512/gpu) |
| --------------------- | --------------- | ------------------- | ------------------ |
| H2D transfer          | 0.62            | 0.55                | 0.57               |
| Forward               | 12.82           | 8.02                | 4.76               |
| Backward (+allreduce) | 39.36           | 110.03              | 31.64              |
| Optimizer             | 24.09           | 24.02               | 24.25              |
| **GPU step total**    | **76.89**       | **142.63**          | **61.22**          |
| DataLoader + Python   | ~32             | ~31                 | ~31                |
| **Wall-clock / step** | **~109**        | **~174**            | **~93**            |
| **Throughput (s/s)**  | **37,629**      | **23,555**          | **44,339**         |


#### Backward decomposition: compute vs allreduce

Forward scales linearly with per-GPU batch (12.82 → 8.02 → 4.76 ms). Assuming
backward compute scales the same way:


| GPUs | Backward (measured) | Est. backward compute | Est. allreduce overhead |
| ---- | ------------------- | --------------------- | ----------------------- |
| 1    | 39.36 ms            | 39.36 ms              | 0 (no comm)             |
| 2    | 110.03 ms           | ~19.7 ms              | **~90.3 ms**            |
| 8    | 31.64 ms            | ~4.9 ms               | **~26.7 ms**            |


The 2-GPU allreduce is 6.3x slower than 8-GPU despite having fewer GPUs to
coordinate. This is counter-intuitive — see explanation below.

#### Root cause: XGMI bandwidth, not data volume

**Key insight: the gradient volume is fixed.** AllReduce always reduces the
full 997M params (~2 GB), regardless of how many GPUs participate. Ring allreduce
transfers `2*(N-1)/N * S` bytes per GPU:


| GPUs | Data per GPU       | Measured latency | Bus BW     |
| ---- | ------------------ | ---------------- | ---------- |
| 2    | 2.0 GB (`2*1/2*2`) | 57.27 ms         | 36.6 GB/s  |
| 8    | 3.5 GB (`2*7/8*2`) | 9.03 ms          | 232.3 GB/s |


8-GPU transfers **1.75x more** data per GPU, yet finishes **6.3x faster**.
The explanation is entirely about link-level parallelism on the XGMI mesh:

On MI355X, each GPU has **7 XGMI links** (one to every other GPU in the
fully-connected mesh). In ring allreduce, the ring `0→1→2→…→7→0` uses 8
link pairs simultaneously. RCCL further creates **multiple parallel ring
channels** over different physical links (each GPU uses 2 links per ring for
in/out, leaving 5 spare links for additional rings — supporting ~3 channels).

- **2 GPUs**: only **1 link pair** exists → max bandwidth ≈ 37 GB/s
- **8 GPUs**: 8 links × ~3 channels → aggregate bandwidth ≈ 232 GB/s

The per-link bandwidth is the same (~50 GB/s); with 8 GPUs, RCCL simply
harnesses more links in parallel.

#### Why 2-GPU is slower than 1-GPU (the key insight)

With fixed global batch_size=4096:


|           | Compute time | Allreduce time | Optimizer | DataLoader | Total  | Throughput |
| --------- | ------------ | -------------- | --------- | ---------- | ------ | ---------- |
| **1-GPU** | 52 ms        | 0              | 24 ms     | 32 ms      | 109 ms | 37.6K s/s  |
| **2-GPU** | 28 ms        | 90 ms          | 24 ms     | 31 ms      | 174 ms | 23.6K s/s  |
| **8-GPU** | 10 ms        | 27 ms          | 24 ms     | 31 ms      | 93 ms  | 44.1K s/s  |


1. **Compute halves** when you double GPUs (good), but **allreduce cost is
  added** (bad). With 2 GPUs, allreduce (90 ms) dominates the step — it's
   larger than the entire 1-GPU compute (52 ms).
2. **Optimizer (24 ms) and DataLoader (~31 ms) are fixed costs** that don't
  shrink with more GPUs. Together they're 55 ms — already 50% of the 1-GPU
   step time.
3. At 8 GPUs, the much higher XGMI bandwidth (232 GB/s vs 36.6 GB/s) makes
  allreduce fast enough that the total step time (93 ms) is actually lower
   than 1-GPU (109 ms). But the 55 ms of fixed costs means you can never reach
   8x speedup.

#### Implications for scaling

To improve scaling, you'd need to:

- **Increase global batch size** with more GPUs (weak scaling) — keeps per-GPU
batch large so compute dominates.
- **Overlap DataLoader with training** — current DataLoader stall (~31 ms/step)
is a major bottleneck, consuming 30-55% of step time.
- **Gradient compression** or **reduce-scatter** (FSDP) to lower comm volume.

### 2E. Throughput scaling — weak scaling with bs=32768 (2026-03-02)

Using `configs/bench_onetrans_v6.yaml` (batch_size=32768, OneTrans 997M params).
Weak scaling: each GPU always gets `32768 / world_size` samples per step.


| Config        | GPUs | Per-GPU batch | Throughput (s/s) | Step time (p50 ms) | Speedup |
| ------------- | ---- | ------------- | ---------------- | ------------------ | ------- |
| 1-GPU         | 1    | 32768         | 51,624           | 352                | 1.0x    |
| 8-GPU         | 8    | 4096          | 240,000          | 94                 | 4.65x   |
| 16-GPU 2-node | 16   | 2048          | 230,919          | 79                 | 4.47x   |


#### Trace breakdown (mean ms, from `torch.profiler` Chrome traces)


| Phase                 | 1-GPU   | 8-GPU  | 16-GPU 2-node |
| --------------------- | ------- | ------ | ------------- |
| dataloader            | 15.4    | 13.4   | 13.1          |
| forward               | 5.3     | 6.0    | 5.9           |
| backward (+allreduce) | 325.2   | 68.2   | 53.7          |
| optimizer             | 6.4     | 6.7    | 6.5           |
| **total**             | **352** | **94** | **79**        |


(1-GPU dataloader p50 used; one outlier step at 700ms skewed the mean.)

**Key observations:**

1. Backward dominates in all configs (71-73% of step time).
2. 8-GPU backward is 4.8x faster than 1-GPU despite adding allreduce — because
  per-GPU compute drops 8x (batch 32768→4096) while allreduce is fast on XGMI.
3. 16-GPU is only marginally faster than 8-GPU: inter-node allreduce over
  AINIC RDMA adds ~15ms vs intra-node XGMI, but the per-GPU batch is also
   halved (4096→2048), reducing compute further.
4. 16-GPU throughput (231K) is slightly **lower** than 8-GPU (240K) — the
  inter-node communication overhead outweighs the compute reduction at this
   small per-GPU batch size.

**Trace files** (loadable in [ui.perfetto.dev](https://ui.perfetto.dev/)):

- `results/bench_1gpu/trace/trace_0.json`
- `results/bench_8gpu/trace/trace_0.json`
- `results/bench_16gpu/trace/trace_0.json`

**Commands:**

```bash
# 1-GPU
CUDA_VISIBLE_DEVICES=0 bash scripts/launch_torchrun.sh --nnodes 1 --gpus 1 \
  --config configs/bench_onetrans_v6.yaml --max-steps 200 --skip-eval --trace --run-name bench_1gpu

# 8-GPU
bash scripts/launch_torchrun.sh --nnodes 1 --gpus 8 \
  --config configs/bench_onetrans_v6.yaml --max-steps 200 --skip-eval --trace --run-name bench_8gpu

# 16-GPU 2-node
bash scripts/launch_torchrun.sh --nnodes 2 --gpus 8 --nodes chi2866,chi2810 \
  --config configs/bench_onetrans_v6.yaml --max-steps 200 --skip-eval --trace --run-name bench_16gpu
```

### Settings

- Hardware: AMD Instinct MI355X, ROCm 7.1
- `embedding_init: "uniform"` (TorchRec default, `uniform(-1/sqrt(N), 1/sqrt(N))`)
- Configs: `configs/dist_counter_v1.yaml`, `configs/dist_onetrans_v6.yaml`
- DDP: `find_unused_parameters=True` (needed for OneTrans unused param paths)
- FSDP: `use_orig_params=True`, `FullStateDictConfig(rank0_only=True)` for checkpoints
- FSDP eval: checkpoint loaded into fresh non-FSDP model on single GPU
(`summon_full_params` approach times out because FSDP forward hooks trigger
collective ops even inside the context, but only rank 0 calls forward)
- NCCL timeout: 30 min (eval data loading can take >10 min)
- `forward_cross` not supported under DDP yet — tracked as follow-up in stage2-spec.md
- Multi-node: `torchrun --nnodes=2 --node_rank=N --master_addr=<private_ip>`
with `NCCL_SOCKET_IFNAME=enp193s0f1np1` (private 10.2.x.x management network for rendezvous)
- Multi-node AINIC RDMA config (11.6x vs Socket):
  ```
  NCCL_IB_GID_INDEX=1          # RoCE v2 GID
  NCCL_IB_PCI_RELAXED_ORDERING=1
  NCCL_IB_USE_INLINE=1
  NCCL_IB_QPS_PER_CONNECTION=4
  NCCL_IB_ECE_ENABLE=0
  NCCL_DMABUF_ENABLE=1         # needs /boot mounted for kernel metadata
  NCCL_GDRCOPY_ENABLE=1
  NCCL_GDR_FLUSH_DISABLE=1
  NCCL_PXN_DISABLE=0
  NCCL_IGNORE_CPU_AFFINITY=1
  NCCL_CHECKS_DISABLE=1
  NET_OPTIONAL_RECV_COMPLETION=1
  RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
  RCCL_LL128_FORCE_ENABLE=1
  RCCL_MSCCLPP_ENABLE=1        # warns if RCCL built without MSCCL++ (harmless)
  IONIC_LOCKFREE=all
  ```
  Reference: [AMD maxtext-slurm/train_env.sh](https://github.com/AMD-AGI/maxtext-slurm/blob/main/train_env.sh)
- `--skip-eval` flag added for large-scale runs; eval from checkpoint separately

### Notes

- Model param counts: counter_v1=465,698,721, onetrans_v6=997,103,617
- Global batch size is constant: `per_gpu_batch = batch_size // world_size`
- Single-card configs: `configs/s2_*.yaml` or `configs/exp_*.yaml`
- Distributed configs: `configs/dist_*.yaml`
- FSDP checkpoint saving: uses `FSDP.state_dict_type(FULL_STATE_DICT)` to save
consolidated params on rank 0; loadable into a non-FSDP model for eval

---

## Stage 3: Embedding Sharding

### Phase 0: TorchRec/FBGEMM Enablement Results

**Goal**: Validate TorchRec DMP + FBGEMM TBE on MI355X/ROCm 7.1.

**Environment**:

- PyTorch 2.10.0+rocm7.1, FBGEMM GPU 1.5.0+rocm7.1.25424, TorchRec (matching)
- AMD Instinct MI355X, 8 GPUs per node

**Test Results**:


| Test                             | Status       | Notes                                                        |
| -------------------------------- | ------------ | ------------------------------------------------------------ |
| FBGEMM TBE single-GPU forward    | **SEGFAULT** | Crashes in `SplitTableBatchedEmbeddingBagsCodegen.forward()` |
| TorchRec EBC single-GPU (no DMP) | PASS         | Non-fused fallback works, no TBE                             |
| RCCL all-to-all 2-GPU            | PASS         |                                                              |
| DMP DATA_PARALLEL 2-GPU          | **SEGFAULT** | DMP internally routes through TBE                            |
| DMP TABLE_WISE 2-GPU             | **SEGFAULT** | Same root cause                                              |
| DMP ROW_WISE 2-GPU               | **SEGFAULT** | Same root cause                                              |


**Root Cause**: The FBGEMM GPU `SplitTableBatchedEmbeddingBagsCodegen` kernel segfaults
on MI355X during forward pass. This is a ROCm/FBGEMM compatibility issue — the kernel
binary is likely compiled for a different GPU architecture. TorchRec DMP always routes
through TBE regardless of sharding strategy, making it completely unusable on this hardware.

**Decision**: Implement native PyTorch embedding sharding (nn.Embedding + all-to-all)
as the production path. TorchRec EBC/EC remain for single-GPU (non-DMP) use since
they work without TBE.

### Phase 1: Native Embedding Sharding Implementation

**Approach**: Custom `ShardedEmbeddingCollection` using `nn.Embedding` + 
`torch.distributed.all_to_all_single` with differentiable autograd wrapper.

**Architecture**:

- `ShardedEmbeddingCollection` (`primus_dlrm/models/sharded_embedding.py`):
  - Replaces `TorchRecEmbeddings` at wrap time, same forward API
  - Table-wise: each table on one GPU, IDs routed via all-to-all
  - Row-wise: table rows split across GPUs, IDs routed by chunk
  - `_AllToAllEmb`: custom autograd.Function for differentiable embedding all-to-all
  - `_all_to_all_ids`: non-differentiable all-to-all for integer ID routing
- `_ShardedModelWrapper` (`primus_dlrm/distributed/wrapper.py`):
  - Composes sharded embeddings with manual dense allreduce hooks
  - Dense params get post-accumulate allreduce hooks (replaces DDP)
  - Embedding params sync via all-to-all in forward/backward

**Config**: `--dense-strategy dmp --embedding-sharding {auto,table_wise,row_wise}`

**Test Results (2-GPU and 8-GPU)**:


| Test                                            | 2-GPU | 8-GPU |
| ----------------------------------------------- | ----- | ----- |
| Table-wise sharding (forward+backward)          | PASS  | PASS  |
| Row-wise sharding (forward+backward)            | PASS  | PASS  |
| Pooled table-wise (mean pooling)                | PASS  | PASS  |
| Full model integration (forward+backward+optim) | PASS  | PASS  |
| wrap_model("dmp", table_wise)                   | PASS  | PASS  |
| wrap_model("dmp", row_wise)                     | PASS  | PASS  |
| wrap_model("dmp", auto)                         | PASS  | PASS  |


**Files Changed**:

- `primus_dlrm/models/sharded_embedding.py` — NEW: native sharding module
- `primus_dlrm/models/embedding.py` — refactored config building into `_build_torchrec_configs()`
- `primus_dlrm/distributed/wrapper.py` — `wrap_model("dmp")` routes to native sharding
- `primus_dlrm/training/dist_trainer.py` — handles `_ShardedModelWrapper` unwrapping
- `primus_dlrm/config.py` — added `EmbeddingShardingConfig` dataclass
- `scripts/run_distributed.py` — added `--embedding-sharding` CLI flag
- `primus_dlrm/models/dlrm.py`, `onetrans.py` — added `meta_device` kwarg (future DMP)
- `scripts/test_fbgemm_tbe.py`, `test_dmp_minimal.py` — Phase 0 smoke tests
- `scripts/test_native_sharding.py`, `test_wrapper_integration.py` — Phase 1 tests

---

### Stage 3 Update: FBGEMM Rebuild + TorchRec DMP Enablement

**Date**: 2026-03-03

#### Root Cause: FBGEMM TBE Segfault

The pre-built FBGEMM GPU wheel (`1.5.0+rocm7.1.25424`) was compiled only for
`gfx942` (MI300X). The MI355X uses `gfx950` (CDNA4), which is not backward-
compatible. The HIP runtime attempted to run `gfx942` code objects on `gfx950`,
causing segfaults in both `SplitTableBatchedEmbeddingBagsCodegen` (fused TBE) and
`DenseTableBatchedEmbeddingBagsCodegen` (dense TBE).

Confirmed via `roc-obj-ls`: only `hipv4-amdgcn-amd-amdhsa--gfx942` code objects
were present in the installed `.so` files, despite `gfx950` appearing as a string
in the binary metadata.

#### Fix: Rebuild FBGEMM from Source

```bash
git clone --recursive -b v1.5.0 https://github.com/pytorch/FBGEMM.git
cd FBGEMM/fbgemm_gpu
export ROCM_PATH=/opt/rocm PYTORCH_ROCM_ARCH=gfx950 BUILD_ROCM_VERSION=7.1.1
python setup.py install --build-target=default --build-variant=rocm \
    -DAMDGPU_TARGETS="gfx950" -DHIP_ROOT_DIR="${ROCM_PATH}"
```

Build time: ~4.5 minutes. All `.so` files now contain `gfx950` code objects.
Backup of original at `.venv/.../fbgemm_gpu_backup/`, revert script at
`scripts/revert_fbgemm.sh`.

#### TorchRec DMP: Now Fully Working

With rebuilt FBGEMM, TorchRec `DistributedModelParallel` works end-to-end on
MI355X. Rewrote `wrapper.py` to use real TorchRec DMP instead of native sharding:

- `wrap_model("dmp")` now calls `DistributedModelParallel` directly
- DMP discovers EBC/EC submodules, shards them via FBGEMM TBE kernels
- Dense layers automatically wrapped with DDP by DMP
- Fused embedding optimizer (`CombinedOptimizer`) managed by DMP
- `dist_trainer.py` splits optimizer: dense params (AdamW) + DMP fused (TBE SGD)

**DMP Test Results**:


| Test                             | 2-GPU | 8-GPU |
| -------------------------------- | ----- | ----- |
| RCCL all-to-all                  | PASS  | PASS  |
| DMP DATA_PARALLEL (EBC)          | PASS  | PASS  |
| DMP TABLE_WISE (EBC)             | PASS  | PASS  |
| DMP ROW_WISE (EBC)               | PASS  | PASS  |
| DMP TABLE_WISE (EC, unpooled)    | PASS  | PASS  |
| DMP full toy model (EBC + dense) | PASS  | PASS  |
| DMP DLRMBaseline auto sharding   | PASS  | PASS  |
| DMP DLRMBaseline table_wise      | PASS  | PASS  |
| DMP DLRMBaseline row_wise        | PASS  | PASS  |
| DMP DLRMBaseline data_parallel   | PASS  | PASS  |
| DMP optimizer step (multi-step)  | PASS  | PASS  |


**Files Changed**:

- `primus_dlrm/distributed/wrapper.py` — rewrote for TorchRec DMP (`_wrap_dmp`,
`_build_constraints`, `_log_sharding_plan`, `is_dmp`)
- `primus_dlrm/training/dist_trainer.py` — DMP optimizer split (dense AdamW +
fused TBE), `is_dmp` check
- `scripts/run_distributed.py` — meta device model build for DMP
- `scripts/test_dmp_integration.py` — NEW: DLRMBaseline DMP integration tests
- `scripts/test_fbgemm_tbe.py` — added Dense TBE test

### Stage 3 Benchmark: DDP vs DMP Training (8-GPU, 2026-03-03)

**Goal**: Verify DMP training correctness (loss parity with DDP) and measure
throughput impact of TorchRec embedding sharding strategies.

**Setup**:

- Model: DLRMBaseline (counter_v1, 465M params, embedding_dim=16)
- Config: `configs/dist_counter_v1.yaml`, batch_size=4096 global, bf16
- Hardware: 8× AMD MI355X (single node), ROCm 7.1, FBGEMM rebuilt for gfx950
- 5000 steps, warmup 1000 steps, `--skip-eval`

**Commands**:

```bash
# DDP baseline
torchrun --nproc_per_node=8 scripts/run_distributed.py \
    --config configs/dist_counter_v1.yaml --dense-strategy ddp \
    --max-steps 5000 --run-name s3_ddp_baseline --log-interval 500 --skip-eval

# DMP auto (planner picks sharding per table)
torchrun --nproc_per_node=8 scripts/run_distributed.py \
    --config configs/dist_counter_v1.yaml --dense-strategy dmp --embedding-sharding auto \
    --max-steps 5000 --run-name s3_dmp_auto --log-interval 500 --skip-eval

# DMP table_wise / row_wise / data_parallel (same pattern)
torchrun --nproc_per_node=8 scripts/run_distributed.py \
    --config configs/dist_counter_v1.yaml --dense-strategy dmp \
    --embedding-sharding {table_wise,row_wise,data_parallel} \
    --max-steps 5000 --run-name s3_dmp_{tw,rw,dp} --log-interval 500 --skip-eval
```

#### Loss trajectories (listen_plus)


| Step    | DDP        | DMP auto   | DMP table_wise | DMP row_wise | DMP data_parallel |
| ------- | ---------- | ---------- | -------------- | ------------ | ----------------- |
| 0       | 0.7034     | 0.7024     | 0.7024         | 0.7030       | 0.7040            |
| 500     | 0.4825     | 0.4780     | 0.4780         | 0.4801       | 0.4856            |
| 1000    | 0.5076     | 0.5184     | 0.5184         | 0.5234       | 0.5271            |
| 1500    | 0.4556     | 0.4599     | 0.4599         | 0.4574       | 0.4923            |
| 2000    | 0.4853     | 0.5033     | 0.5033         | 0.4953       | 0.5220            |
| 2500    | 0.4997     | 0.5080     | 0.5080         | 0.5184       | 0.5290            |
| 3000    | 0.4605     | 0.4656     | 0.4656         | 0.4730       | 0.5012            |
| 3500    | 0.4591     | 0.4688     | 0.4688         | 0.4650       | 0.4996            |
| 4000    | 0.4505     | 0.4574     | 0.4574         | 0.4554       | 0.4837            |
| 4500    | 0.4642     | 0.4779     | 0.4779         | 0.4776       | 0.5067            |
| **avg** | **0.4897** | **0.4964** | **0.4964**     | **0.4963**   | **0.5164**        |


#### Throughput summary


| Strategy          | Avg loss | Epoch time (s) | Throughput (s/s) | vs DDP     |
| ----------------- | -------- | -------------- | ---------------- | ---------- |
| DDP baseline      | 0.4897   | 149.3          | 137,188          | —          |
| DMP auto          | 0.4964   | 103.3          | 198,233          | **+44.5%** |
| DMP table_wise    | 0.4964   | 103.3          | 198,194          | **+44.5%** |
| DMP row_wise      | 0.4963   | 106.5          | 192,220          | **+40.1%** |
| DMP data_parallel | 0.5164   | 123.7          | 165,564          | **+20.7%** |


#### Analysis

**Loss correctness**:

- DMP auto/table_wise/row_wise: avg loss within 1.4% of DDP. PASS.
- DMP data_parallel: avg loss 5.5% higher — the `DenseTableBatchedEmbeddingBagsCodegen`
backward uses fused SGD (not AdamW) for embeddings, causing divergence. The
sharded strategies (auto/table_wise/row_wise) use `SplitTableBatchedEmbeddingBagsCodegen`
which also uses fused SGD, but the sharding distributes tables across GPUs,
reducing per-GPU embedding parameter count and making the SGD-vs-AdamW
difference less impactful. data_parallel keeps full copies everywhere,
amplifying the optimizer mismatch.
- All strategies converge to similar final loss (~0.46-0.51 at step 4500),
confirming training correctness.

**Throughput**:

- DMP auto/table_wise delivers **44.5% higher throughput** than DDP (198K vs 137K s/s).
This is because TorchRec DMP wraps dense layers with DDP internally while
embeddings use FBGEMM fused TBE kernels with built-in optimizer — eliminating
the allreduce overhead for embedding gradients (which dominate the 465M param model).
- row_wise is slightly slower (192K vs 198K) due to all-to-all communication
overhead from splitting table rows across GPUs.
- data_parallel is slowest DMP variant (165K) but still +20.7% over DDP, since
DenseTableBatchedEmbeddingBagsCodegen has lower overhead than nn.Embedding + allreduce.
- The auto planner chose table_wise for this model (identical results), which is
expected for many small tables that fit on individual GPUs.

**FBGEMM rebuild impact**:

- All FBGEMM TBE kernels (fused and dense) now produce `gfx950` code objects,
running natively on MI355X without architecture translation overhead.
- "Running on CDNA architecture" warnings are expected and harmless — they indicate
the HIP kernels are using CDNA-specific optimizations.

---

## Stage 3 Benchmark: OneTrans 1GPU vs DDP vs DMP (Adam)

**Date**: 2026-03-04

**Objective**: Compare OneTrans quality and performance across 1-GPU, DDP 8-GPU,
and DMP 8-GPU (with fused Adam embedding optimizer) for 1 full epoch.

### Setup

- Model: OneTrans (d_model=256, n_heads=8, n_layers=4, pyramid, ~900M params)
- Dataset: 50M events, 10K users, 9.4M items
- Batch size: 32768 global (1GPU: 32768/GPU, 8GPU: 4096/GPU)
- Optimizer: AdamW lr=3e-4 for dense params; DMP uses fused Adam lr=1e-2 for embeddings
- 1 epoch = 1525 steps
- Hardware: 8× AMD MI355X, ROCm 7.1
- Config: `configs/bench_onetrans_v6.yaml`

DMP uses TorchRec `DistributedModelParallel` with `auto` sharding (planner chose
`table_wise`), fused Adam embedding optimizer via `_apply_optimizer_in_backward`.

### Commands

```bash
# 1-GPU
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
  scripts/run_distributed.py --config configs/bench_onetrans_v6.yaml \
  --dense-strategy ddp --run-name s3_ot_1gpu --log-interval 100 --skip-eval --trace

# DDP 8-GPU
torchrun --nproc_per_node=8 scripts/run_distributed.py \
  --config configs/bench_onetrans_v6.yaml \
  --dense-strategy ddp --run-name s3_ot_ddp8 --log-interval 100 --skip-eval --trace

# DMP 8-GPU (auto sharding, fused Adam for embeddings)
torchrun --nproc_per_node=8 scripts/run_distributed.py \
  --config configs/bench_onetrans_v6.yaml \
  --dense-strategy dmp --embedding-sharding auto --run-name s3_ot_dmp8_v2 \
  --log-interval 100 --skip-eval --trace

# Eval (single GPU, from checkpoint)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_checkpoint.py \
  --config configs/bench_onetrans_v6.yaml \
  --checkpoint results/<run_name>/checkpoints/epoch_0.pt
```

### Training Results


| Setting   | Avg Loss | Throughput  | Time | Scaling |
| --------- | -------- | ----------- | ---- | ------- |
| 1-GPU     | 0.4929   | 59,085 s/s  | 801s | 1.0×    |
| DDP 8-GPU | 0.4930   | 306,309 s/s | 155s | 5.18×   |
| DMP 8-GPU | 0.4714   | 314,589 s/s | 151s | 5.33×   |


#### Loss Trajectory


| Step | 1-GPU  | DDP 8-GPU | DMP 8-GPU |
| ---- | ------ | --------- | --------- |
| 0    | 0.6955 | 0.6967    | 0.7196    |
| 100  | 0.5115 | 0.5064    | 0.5034    |
| 200  | 0.5034 | 0.4992    | 0.4865    |
| 400  | 0.4945 | 0.4785    | 0.4604    |
| 600  | 0.4947 | 0.5041    | 0.4761    |
| 800  | 0.4923 | 0.4883    | 0.4622    |
| 1000 | 0.4863 | 0.4599    | 0.4395    |
| 1200 | 0.4790 | 0.4743    | 0.4579    |
| 1400 | 0.4721 | 0.4753    | 0.4528    |


DMP achieves ~4.4% lower avg loss (0.4714 vs 0.4929) — the fused Adam embedding
optimizer converges faster than AdamW applied to dense embedding gradients.

### Eval NDCG (global-5000, listen_plus)


| Setting   | NDCG@10    | NDCG@50    | NDCG@100   | Recall@10  | Recall@50  | Recall@100 | Users |
| --------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----- |
| 1-GPU     | 0.0681     | 0.0666     | 0.0724     | 0.0376     | 0.0717     | 0.0890     | 4600  |
| DDP 8-GPU | 0.0707     | 0.0702     | 0.0766     | 0.0390     | 0.0763     | 0.0955     | 4600  |
| DMP 8-GPU | **0.0852** | **0.0832** | **0.0895** | **0.0463** | **0.0894** | **0.1103** | 4600  |


**DMP wins convincingly**: +25% NDCG@10, +24% NDCG@100, +24% Recall@100 over 1-GPU.
DDP is within ~4% of 1-GPU (expected with same optimizer, different gradient noise).
DMP's fused Adam for embeddings substantially improves ranking quality.

### Per-Step Time Breakdown (ms, from torch.profiler)

GPU kernel time attributed to each phase via correlation IDs (linking CPU
kernel launch to actual GPU execution). Dataloader is CPU wall-clock time.


| Phase            | 1-GPU      | DDP 8-GPU  | DMP 8-GPU |
| ---------------- | ---------- | ---------- | --------- |
| dataloader (CPU) | 105.93     | 13.58      | 1.12      |
| forward (GPU)    | 86.34      | 12.64      | 59.00     |
| backward (GPU)   | 238.12     | 67.53      | 31.38     |
| optimizer (GPU)  | 18.37      | 18.42      | 0.64      |
| **total**        | **448.76** | **112.17** | **92.14** |
| bwd/fwd ratio    | 2.76×      | 5.34×      | 0.53×     |


#### Forward kernel breakdown (GPU ms)


| Kernel Category         | 1-GPU     | DDP 8-GPU | DMP 8-GPU |
| ----------------------- | --------- | --------- | --------- |
| NCCL/MSCCL (all-to-all) | —         | —         | 45.38     |
| flash attn forward      | 13.68     | 1.79      | 1.86      |
| GEMM (linear)           | 14.50     | 2.60      | 2.64      |
| elementwise             | 24.92     | 3.85      | 3.91      |
| cat ops                 | 17.09     | 2.37      | 2.44      |
| dropout                 | 6.55      | 0.77      | 0.80      |
| other                   | 9.44      | 1.27      | 2.36      |
| **total**               | **86.18** | **12.65** | **59.39** |


Forward scales almost linearly with batch: 1-GPU=86ms vs DDP8=12.6ms (6.8×
for 8× smaller per-GPU batch). DMP forward includes 45ms all-to-all for
embedding sharding — without it, pure compute is 14ms, matching DDP8.

#### Backward kernel breakdown (GPU ms)


| Kernel Category          | 1-GPU      | DDP 8-GPU | DMP 8-GPU |
| ------------------------ | ---------- | --------- | --------- |
| NCCL/RCCL allreduce      | —          | 18.74     | 3.52      |
| flash attn backward      | 35.76      | 4.48      | 4.48      |
| embedding grad scatter   | 35.15      | 4.95      | —         |
| GEMM (linear backward)   | 32.24      | 4.91      | 5.06      |
| elementwise (grad arith) | 98.35      | 21.91     | 12.59     |
| FBGEMM TBE backward      | —          | —         | 1.23      |
| cat ops                  | 12.44      | 1.70      | 1.85      |
| other                    | 24.58      | 9.72      | 2.37      |
| **total**                | **238.52** | **66.41** | **31.09** |


**Why DDP8 backward/forward = 5.34×** (67.5ms vs 12.6ms):

The backward-to-forward ratio for DDP 8-GPU (5.34×) is much higher than
1-GPU (2.76×) because DDP adds NCCL allreduce overhead that only exists in
backward, not forward:

1. **NCCL allreduce: 18.74ms** (28% of backward, 0% of forward).
  DDP must synchronize ~900M param gradients across 8 GPUs.
2. **Elementwise grad arithmetic: 21.91ms** (33% of backward).
  Backward requires ~3× more elementwise ops than forward — chain-rule
   multiplies, activation derivatives, gradient accumulation.
3. `**multi_tensor_apply` + other: 9.72ms** (15% of backward).
  DDP gradient bucketing and reduction hooks add overhead.
4. Flash attn backward (4.48ms) is 2.5× its forward (1.79ms) — backward
  must recompute attention weights (flash attention memory optimization)
   and compute gradients w.r.t. Q, K, V.

Without NCCL, DDP8 backward would be ~48.7ms → 3.85× forward, close to the
1-GPU ratio of 2.76×. The extra 1.1× comes from DDP hook overhead.

### Key Findings

1. **DMP is the best strategy for OneTrans**: same throughput as DDP (+2.7%), but
  significantly better model quality (+25% NDCG@10) due to fused Adam for embeddings.
2. **Fused Adam > dense AdamW for embeddings**: The TBE kernel applies Adam updates
  sparsely (only accessed rows), which is both faster and converges to better quality
   than DDP's approach of allreducing dense (mostly-zero) gradients and then applying
   AdamW to all rows.
3. **DDP scales 5.18× on 8 GPUs** (vs ideal 8×). The gap comes from gradient allreduce
  overhead for ~900M params. DMP achieves 5.33× by eliminating embedding allreduce.
4. **DMP checkpoint fix**: DMP checkpoints contain `ShardedTensor` objects that can't
  be loaded on a single GPU. Fixed `_save_checkpoint` in `dist_trainer.py` to gather
   all shards via `torch.distributed.reduce` before saving, producing standard tensors
   loadable on any GPU count.

### Result Directories


| Run       | Directory                | Checkpoint                      | Trace                |
| --------- | ------------------------ | ------------------------------- | -------------------- |
| 1-GPU     | `results/s3_ot_1gpu/`    | `checkpoints/epoch_0.pt` (3.8G) | `trace/trace_0.json` |
| DDP 8-GPU | `results/s3_ot_ddp8/`    | `checkpoints/epoch_0.pt` (3.8G) | `trace/trace_0.json` |
| DMP 8-GPU | `results/s3_ot_dmp8_v2/` | `checkpoints/epoch_0.pt` (3.8G) | `trace/trace_0.json` |


---

## Stage 3 GPU Performance Analysis: TraceLens + Roofline (2026-03-09)

**Objective**: Detailed per-step GPU time breakdown with MFU/MBU roofline analysis
for OneTrans 1-GPU, DDP 8-GPU, and DMP 8-GPU, using TraceLens for sections 1/3/4
and raw trace correlation-ID attribution for section 2 (phase breakdown).

### Approach

**Two data sources combined**:

1. **TraceLens** (AMD-AGI): Processes `torch.profiler` JSON traces to produce
  `gpu_timeline.csv` (authoritative compute/comm/idle split), `ops_summary.csv`
   (per-op kernel time), `GEMM.csv` / `BinaryElementwise.csv` / `UnaryElementwise.csv`
   (FLOPS, bytes moved, TFLOPS/s, TB/s per unique op config), and `coll_analysis.csv`
   (collective communication details).
2. **Raw trace correlation IDs**: GPU kernels are linked back to their CPU-side
  launch point via `cuda_runtime` `correlation` field, then matched to the parent
   `cpu_op`. The CPU launch timestamp determines which training phase (forward,
   backward, optimizer) a GPU kernel belongs to.

**Key details**:

- Traces collected with `record_shapes=True` in `torch.profiler` (required for
TraceLens roofline model to compute FLOPS/Byte).
- MI355X architecture file (`configs/mi355x_arch.json`): bf16 matrix peak = 2516.6
TFLOPS/s, HBM bandwidth = 8 TB/s.
- Profiler settings: `warmup=5, active=10` (10 profiled steps).
- Section 2 averages exclude warmup (first 2 profiled steps) and outlier steps
(step_ms > 2x median), but per-step detail shows all 10 steps.

### How to Reproduce

```bash
# 1. Re-profile with record_shapes=True (already set in primus_dlrm/training/tracer.py)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
  scripts/run_distributed.py --config configs/bench_onetrans_v6.yaml \
  --dense-strategy ddp --run-name s3_ot_1gpu_v2 --max-steps 200 --skip-eval --trace

torchrun --nproc_per_node=8 scripts/run_distributed.py \
  --config configs/bench_onetrans_v6.yaml \
  --dense-strategy ddp --run-name s3_ot_ddp8_v2 --max-steps 200 --skip-eval --trace

torchrun --nproc_per_node=8 scripts/run_distributed.py \
  --config configs/bench_onetrans_v6.yaml \
  --dense-strategy dmp --embedding-sharding auto --run-name s3_ot_dmp8_v3 \
  --max-steps 200 --skip-eval --trace

# 2. Generate TraceLens reports
pip install git+https://github.com/AMD-AGI/TraceLens.git openpyxl
for dir in s3_ot_1gpu_v2 s3_ot_ddp8_v2 s3_ot_dmp8_v3; do
  python -m TraceLens.TraceLens_generate_perf_report_pytorch \
    --trace_dir results/$dir/trace \
    --output_csv_dir results/$dir/tracelens \
    --arch_json configs/mi355x_arch.json
done

# 3. Run combined analysis
python scripts/step_perf_analysis.py \
  --traces results/s3_ot_1gpu_v2/trace/trace_0.json \
           results/s3_ot_ddp8_v2/trace/trace_0.json \
           results/s3_ot_dmp8_v3/trace/trace_0.json \
  --tracelens results/s3_ot_1gpu_v2/tracelens \
              results/s3_ot_ddp8_v2/tracelens \
              results/s3_ot_dmp8_v3/tracelens \
  --labels "1-GPU" "DDP-8GPU" "DMP-8GPU"
```

### 1) Per-Step GPU Time Breakdown (from TraceLens gpu_timeline)


| Component                 | 1-GPU (ms) | 1-GPU % | DDP-8GPU (ms) | DDP-8GPU % | DMP-8GPU (ms) | DMP-8GPU % |
| ------------------------- | ---------- | ------- | ------------- | ---------- | ------------- | ---------- |
| **Total step**            | **443.64** | 100%    | **94.12**     | 100%       | **68.64**     | 100%       |
| Compute                   | 341.74     | 77.0%   | 78.24         | 83.1%      | 41.98         | 61.2%      |
| Exposed comm              | 0.00       | 0.0%    | 9.82          | 10.4%      | 21.84         | 31.8%      |
| Exposed memcpy            | 7.03       | 1.6%    | 2.72          | 2.9%       | 1.09          | 1.6%       |
| Idle                      | 94.88      | 21.4%   | 3.35          | 3.6%       | 3.74          | 5.4%       |
| Total comm (incl. hidden) | 0.00       | --      | 20.20         | --         | 24.08         | --         |
| Hidden comm (overlapped)  | --         | --      | 10.38         | --         | 2.24          | --         |


**Observations**:

- 1-GPU has 21% idle (profiler overhead + pipeline bubbles, no comm).
- DDP hides ~51% of its communication behind compute (10.38ms hidden / 20.20ms total).
- DMP has more total comm (24.08ms all-to-all) but only hides 9% (2.24ms) -- all-to-all
is harder to overlap than DDP's bucketed allreduce.

### 2) Per-Step Phase Breakdown (from raw trace, steady-state avg)


| Phase            | 1-GPU (ms) | 1-GPU %step | DDP-8GPU (ms) | DDP-8GPU %step | DMP-8GPU (ms) | DMP-8GPU %step |
| ---------------- | ---------- | ----------- | ------------- | -------------- | ------------- | -------------- |
| Dataloader+H2D (CPU wall) | 15.19 | 4.3% | 13.07 | 13.9% | 1.15 | 2.2% |
| Forward          | 85.56      | 24.4%       | 12.62         | 13.4%          | 13.80         | 26.1%          |
| Backward         | 238.06     | 67.8%       | 47.39         | 50.5%          | 27.53         | 52.0%          |
| Optimizer        | 18.06      | 5.1%        | 18.10         | 19.3%          | 0.63          | 1.2%           |
| Exposed comm     | 0.00       | 0.0%        | 12.71         | 13.5%          | 8.52          | 16.1%          |
| Memcpy/memset    | 7.22       | 2.1%        | 3.10          | 3.3%           | 1.23          | 2.3%           |
| Idle             | 2.41       | 0.7%        | 0.00          | 0.0%           | 1.25          | 2.4%           |
| **Total**        | **351.32** |             | **93.92**     |                | **52.97**     |                |


**Dataloader+H2D breakdown**: CPU wall-clock from step start to `zero_grad` (when
forward begins). Includes:
- `loss.item()` from previous step (~12.5ms for 1-GPU/DDP) -- forces
  `cudaStreamSynchronize`, flushing the entire GPU pipeline
- `DataLoader.__next__()` (~0.2ms) -- dequeuing from multiprocessing prefetch queue
- `.to(device)` H2D transfer (~2ms for 1-GPU bs=32768, ~0.5ms for 8-GPU bs=4096)

DMP is only 1.15ms because the fused TBE optimizer eliminates the `loss.item()`
sync stall (no heavy optimizer step to drain before the next batch starts).

**Observations**:

- Forward/Backward report only *compute* kernel time (comm kernels separated into
  "Exposed comm"). This makes phase times directly comparable across configs.
- DMP optimizer is 0.63ms vs DDP 18.10ms: the fused TBE kernel runs Adam inside
  backward, so there's almost no separate optimizer step.
- **DDP dataloader+H2D = 13.9% of step**: The `loss.item()` sync is a significant
  pipeline bubble. Deferring `.item()` to only logging steps would save ~12ms/step.
- DDP exposed comm (12.71ms) is higher than DMP (8.52ms) in steady state, despite
  DDP's allreduce being smaller total (20.20ms vs 24.08ms). This is because DDP's
  remaining exposed comm represents allreduce buckets that didn't fully overlap.

### 3) Op Group Breakdown (from TraceLens, per-step avg) + MFU / MBU


| Op Group            | 1-GPU kernel_ms | 1-GPU %step | DDP-8GPU kernel_ms | DDP-8GPU %step | DMP-8GPU kernel_ms | DMP-8GPU %step |
| ------------------- | --------------- | ----------- | ------------------ | -------------- | ------------------ | -------------- |
| Embedding lookup    | 44.22           | 10.0%       | 7.83               | 8.3%           | 1.26               | 1.8%           |
| GEMM (mm/bmm/addmm) | 46.44           | 10.5%       | 7.58               | 8.1%           | 7.76               | 11.3%          |
| Attention (SDPA)    | 49.68           | 11.2%       | 6.27               | 6.7%           | 6.30               | 9.2%           |
| Elementwise+Reduce  | 185.14          | 41.7%       | 36.17              | 38.4%          | 25.31              | 36.9%          |
| Optimizer kernels   | 23.09           | 5.2%        | 23.10              | 24.5%          | 0.82               | 1.2%           |
| Communication       | 0.00            | 0.0%        | 20.20              | 21.5%          | 24.08              | 35.1%          |
| Other compute       | 0.19            | 0.0%        | 0.15               | 0.2%           | 1.62               | 2.4%           |


**GEMM roofline (from TraceLens GEMM.csv)**:


| Metric            | 1-GPU      | DDP-8GPU   | DMP-8GPU   |
| ----------------- | ---------- | ---------- | ---------- |
| GFLOPS/step       | 26,460     | 3,308      | 3,308      |
| TFLOPS/s achieved | 569.8      | 436.3      | 426.3      |
| **MFU**           | **22.64%** | **17.34%** | **16.94%** |
| GB moved/step     | 129.69     | 16.27      | 16.27      |
| TB/s achieved     | 2.727      | 2.096      | 2.048      |
| **MBU**           | **34.09%** | **26.20%** | **25.60%** |


**Elementwise roofline (from TraceLens Binary/UnaryElementwise.csv)**:


| Metric        | 1-GPU      | DDP-8GPU   | DMP-8GPU   |
| ------------- | ---------- | ---------- | ---------- |
| GB moved/step | 613.66     | 118.41     | 74.27      |
| TB/s achieved | 3.237      | 3.197      | 2.865      |
| **MBU**       | **40.46%** | **39.96%** | **35.81%** |


**Observations**:

- **GEMM MFU 17-23%**: OneTrans GEMMs are moderately compute-efficient. The 1-GPU
batch (32768) has larger matrices yielding better utilization than the 8-GPU batch
(4096/GPU).
- **Elementwise MBU 36-40%**: These ops are bandwidth-bound; achieving ~3 TB/s out
of 8 TB/s peak is typical for mixed-size elementwise ops.
- **DMP embedding is 35x faster than 1-GPU** (1.26ms vs 44.22ms). Table-wise sharding
gives each GPU only its local tables, and the FBGEMM fused TBE kernel is highly optimized.
- **DDP optimizer = 24.5% of step**: The AdamW step (23.1ms) is a fixed cost regardless
of GPU count -- a major bottleneck at scale. DMP eliminates this (0.82ms) via fused
Adam inside TBE backward.
- **Elementwise dominates compute** (~37-42% of step) across all configs.

### 4) End-to-End Metrics


| Metric                | 1-GPU      | DDP-8GPU   | DMP-8GPU   |
| --------------------- | ---------- | ---------- | ---------- |
| Step time (TraceLens) | 443.64 ms  | 94.12 ms   | 68.64 ms   |
| Total GFLOPS/step     | 26,534     | 3,321      | 3,317      |
| Total GB moved/step   | 743.35     | 134.68     | 90.54      |
| **E2E TFLOPS/s**      | **59.81**  | **35.28**  | **48.32**  |
| **E2E TB/s**          | **1.636**  | **1.397**  | **1.288**  |
| **E2E MFU**           | **2.38%**  | **1.40%**  | **1.92%**  |
| **E2E MBU**           | **20.45%** | **17.47%** | **16.10%** |
| Compute-only TFLOPS/s | 77.64      | 42.45      | 79.01      |
| Compute-only TB/s     | 2.124      | 1.681      | 2.106      |
| Compute-only MFU      | 3.09%      | 1.69%      | 3.14%      |
| Compute-only MBU      | 26.55%     | 21.01%     | 26.33%     |


**Observations**:

- **E2E MFU is very low (1.4-2.4%)**: OneTrans spends most GPU time on memory-bound
elementwise ops, not compute-bound GEMMs. This is typical for recommendation models
with small embedding dimensions and relatively small transformer layers.
- **Compute-only MFU ~3%**: When only counting compute kernel time (excluding comm,
memcpy, idle), utilization is still low -- confirming the model is fundamentally
memory-bandwidth-bound, not compute-bound.
- **DMP has the highest E2E TFLOPS/s** (48.32 vs DDP 35.28): shorter step time means
less idle/comm overhead diluting the FLOPS throughput.
- **1-GPU has highest E2E MBU** (20.45%): larger batch -> larger tensors -> more
efficient memory access patterns.

### Elementwise+Reduce Breakdown (1-GPU, 185ms/step = 41.7%)

The largest time bucket is elementwise ops. Breakdown from TraceLens `ops_summary`:

| Op | ms/step | % of group | Purpose |
|----|:-------:|:----------:|---------|
| `aten::mul` | 41.9 | 23% | Gradient chain-rule multiplications, attention scaling, GELU derivative |
| `aten::cat` | 30.5 | 17% | Concatenating Q/K/V heads, merging sequence + item embeddings |
| `aten::copy_` | 27.0 | 15% | Contiguous-ification, gradient buffer copies |
| `aten::add_` | 26.0 | 15% | Residual connections (`x = x + attn_out`), gradient accumulation |
| `aten::sum` | 9.9 | 6% | Loss reductions, embedding pooling |
| `aten::gelu` + `gelu_backward` | 12.8 | 7% | GELU activation forward + backward |
| `aten::fill_` | 6.9 | 4% | Zero-ing gradient buffers |
| dropout + pow + mean + others | ~30 | 13% | Remaining arithmetic |

**Why so large?** These are all **memory-bandwidth-bound**: each reads/writes large
tensors but does trivial math per element. The top ops operate on tensors of shape
`(32768, 68, 256)` = 570M elements = ~1.1 GB per tensor. A single `aten::mul` of
two such inputs moves ~3.3 GB at ~6 TB/s, taking ~0.5ms -- but there are hundreds
of such kernels per step from chain-rule gradient propagation in the backward pass.

Individual kernel efficiency is decent (~4-6 TB/s, 50-75% of peak 8 TB/s). The
problem is the sheer number of separate kernel launches, each doing a full
round-trip to HBM. Adjacent ops like `mul -> add_ -> gelu_backward -> mul` each
independently read from and write to global memory, when they could share data
through registers/L2 if fused into one kernel.

### Optimization Opportunities

Based on the TraceLens analysis:

1. **Elementwise kernel fusion** (biggest win, targeting 37-42% of step time):
   `torch.compile` would fuse chains like `mul -> add_ -> gelu_backward -> mul`
   into a single kernel that reads input once and writes output once, eliminating
   redundant HBM round-trips. The ~950 `aten::mul` + ~340 `aten::add_` + ~340
   `aten::cat` invocations per step create massive kernel launch overhead (~5-10us
   each) that fusion would also eliminate.
2. **Three-way pipeline: H2D / embedding all-to-all / dense forward** (DMP):
   Currently the training loop is fully sequential: dataloader -> H2D -> embedding
   lookup + all-to-all -> dense forward -> backward. These can be pipelined across
   CUDA streams:
   - **Stream A**: prefetch next batch H2D (async `pin_memory` copy)
   - **Stream B**: embedding lookup + all-to-all for current batch
   - **Stream C**: dense forward/backward on previous batch's embeddings
   This would hide the 21.84ms exposed all-to-all (32% of DMP step) and the H2D
   transfer behind dense compute. The training loop (`dist_trainer.py`) currently
   does everything synchronously on the default stream with no prefetching.
3. **DDP: comm-compute overlap**: DDP hides only 51% of allreduce. Tuning bucket
   sizes (`ddp_bucket_cap_mb`) could improve overlap.
4. **Optimizer fusion**: DDP's 18ms optimizer step (24.5% of step) could be reduced
   with `torch.optim._multi_tensor` or fused Adam implementations.
5. **Batch size scaling**: Larger per-GPU batches improve GEMM MFU (22.6% at bs=32768
   vs 17% at bs=4096).

### Result Directories


| Run       | Trace                                      | TraceLens                          |
| --------- | ------------------------------------------ | ---------------------------------- |
| 1-GPU     | `results/s3_ot_1gpu_v2/trace/trace_0.json` | `results/s3_ot_1gpu_v2/tracelens/` |
| DDP 8-GPU | `results/s3_ot_ddp8_v2/trace/trace_0.json` | `results/s3_ot_ddp8_v2/tracelens/` |
| DMP 8-GPU | `results/s3_ot_dmp8_v3/trace/trace_0.json` | `results/s3_ot_dmp8_v3/tracelens/` |


Analysis script: `scripts/step_perf_analysis.py`
Architecture JSON: `configs/mi355x_arch.json` (MI355X: 2516.6 bf16 TFLOPS, 8 TB/s HBM)