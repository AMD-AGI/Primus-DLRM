# OneTrans DLRM Benchmark

A TorchRec-based implementation of [OneTrans (WWW 2026)](https://arxiv.org/abs/2510.26104) for music recommendation on AMD MI350+ GPUs, trained on the [Yambda](https://huggingface.co/datasets/yandex/YamBDA) dataset.

## OneTrans

OneTrans is a unified causal Transformer that jointly performs user behavior sequence modeling and feature interaction in a single stack:

- **S-tokens**: User history is split into 3 behavior pools (listen+, like, skip). Each pool is tokenized by a per-behavior MLP and concatenated into a sequence of S-tokens.
- **NS-tokens**: Candidate item and user context features are projected via an auto-split tokenizer into a set of non-sequential tokens.
- **Mixed parameterization**: S-tokens share Q/K/V and FFN weights; each NS-token has its own token-specific projections.
- **Pyramid stack**: S-token queries are progressively pruned layer-by-layer, funneling information into NS-tokens.
- **Prediction**: Final NS-token states feed into multi-task heads (listen+, like, dislike, played_ratio).

## Verified Stack

| Component  | Version                         |
|------------|---------------------------------|
| Python     | 3.12.3                          |
| PyTorch    | 2.10.0+rocm7.1                  |
| TorchRec   | 1.4.0                           |
| FBGEMM_GPU | 1.5.0+rocm7.1                   |
| ROCm       | 7.1.1                           |
| GPU        | 8x AMD Instinct MI355X (gfx950) |
| Docker     | `rocm/primus:v26.1`             |

## Quick Start

### Download Data

```bash
docker run --rm \
    -v $(pwd):/workspace/dlrm -w /workspace/dlrm \
    rocm/primus:v26.1 \
    bash -c "pip install -q datasets fsspec polars pyarrow && \
        python scripts/download_data.py --size 50m --data-dir data/raw"
```

### Preprocess

```bash
docker run --rm \
    -v $(pwd):/workspace/dlrm -w /workspace/dlrm \
    -e PYTHONPATH=/workspace/dlrm \
    rocm/primus:v26.1 \
    bash -c "pip install -q polars pyarrow pyyaml && \
        python scripts/preprocess.py --raw-dir data/raw --out-dir data/processed --size 50m"
```

This builds per-user timelines, applies the temporal train/test split (300d train / 30min gap / 1d test), segments sessions, and saves under `data/processed/`.

### Verify Environment

```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    rocm/primus:v26.1 \
    python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'ROCm: {torch.version.hip}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
import torchrec; print(f'TorchRec {torchrec.__version__}')
import fbgemm_gpu; print('FBGEMM_GPU OK')
"
```

Expected:
```
PyTorch 2.10.0+rocm7.1
ROCm: 7.1.25424
GPUs: 8x AMD Instinct MI355X
TorchRec 1.4.0
FBGEMM_GPU OK
```

### Train (single node, 8 GPUs)

```bash
docker run --rm --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v $(pwd):/workspace/dlrm -w /workspace/dlrm \
    -e PYTHONPATH=/workspace/dlrm \
    rocm/primus:v26.1 \
    bash -c "pip install -q polars pyarrow pyyaml tqdm datasets && \
        torchrun --nproc_per_node=8 --standalone \
        scripts/run_distributed.py \
        --config configs/bench_onetrans_v6.yaml \
        --dense-strategy dmp --pipeline \
        --max-steps 5000 --run-name onetrans_bench"
```

### Train (multi-node via Slurm)

```bash
bash scripts/launch_slurm_docker.sh \
    --nnodes 2 \
    --config configs/bench_onetrans_v6.yaml \
    --run-name onetrans_2n \
    --pipeline \
    --image rocm/primus:v26.1
```

### Evaluate

```bash
docker run --rm --network=host --ipc=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    -v $(pwd):/workspace/dlrm -w /workspace/dlrm \
    -e PYTHONPATH=/workspace/dlrm \
    rocm/primus:v26.1 \
    bash -c "pip install -q polars pyarrow pyyaml tqdm && \
        python scripts/eval_checkpoint.py \
        --config configs/bench_onetrans_v6.yaml \
        --checkpoint results/onetrans_bench/checkpoints/epoch_0.pt \
        --num-candidates 5000"
```

Metrics: NDCG@{10,50,100} and Recall@{10,50,100} on the held-out test set.
