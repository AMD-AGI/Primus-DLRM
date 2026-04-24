#!/usr/bin/env bash
# Launch distributed training on a single NVIDIA node via srun + Docker.
# Designed for nvcr.io/nvidia/pytorch containers on B200 GPUs.
#
# Usage:
#   bash scripts/launch_srun_nvidia.sh \
#       --reservation gh-chcai-8d263168 \
#       --nodelist hungry-hippo-fin-03-8 \
#       --config configs/bench_onetrans_large_synthetic_flash.yaml \
#       --max-steps 100 --run-name b200_onetrans_large_flash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

GPUS=8
CONFIG=""
MAX_STEPS=100
RUN_NAME="nvidia_onetrans"
RESULTS_DIR="results"
DENSE_STRATEGY="ddp"
LOG_INTERVAL=10
RESERVATION=""
NODELIST=""
DOCKER_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
SKIP_EVAL="--skip-eval"
PIPELINE=""
TRACE=""
TRACE_STEPS=""
TRACE_WARMUP=5
TRACE_ACTIVE=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)           GPUS="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --max-steps)      MAX_STEPS="$2"; shift 2;;
        --run-name)       RUN_NAME="$2"; shift 2;;
        --results-dir)    RESULTS_DIR="$2"; shift 2;;
        --dense-strategy) DENSE_STRATEGY="$2"; shift 2;;
        --log-interval)   LOG_INTERVAL="$2"; shift 2;;
        --reservation)    RESERVATION="$2"; shift 2;;
        --nodelist)       NODELIST="$2"; shift 2;;
        --image)          DOCKER_IMAGE="$2"; shift 2;;
        --skip-eval)      SKIP_EVAL="--skip-eval"; shift;;
        --pipeline)       PIPELINE="--pipeline"; shift;;
        --trace)          TRACE="--trace"; shift;;
        --trace-steps)    TRACE_STEPS="$2"; shift 2;;
        --trace-warmup)   TRACE_WARMUP="$2"; shift 2;;
        --trace-active)   TRACE_ACTIVE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi
if [[ -z "$RESERVATION" ]]; then
    echo "Error: --reservation is required"
    exit 1
fi
if [[ -z "$NODELIST" ]]; then
    echo "Error: --nodelist is required"
    exit 1
fi

LOG_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/logs"
CHECKPOINT_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"

echo "============================================"
echo "Node:          $NODELIST"
echo "Reservation:   $RESERVATION"
echo "GPUs:          $GPUS"
echo "Config:        $CONFIG"
echo "Strategy:      $DENSE_STRATEGY"
echo "Run name:      $RUN_NAME"
echo "Docker image:  $DOCKER_IMAGE"
echo "Max steps:     $MAX_STEPS"
echo "============================================"

srun --reservation="$RESERVATION" \
     --nodelist="$NODELIST" \
     --qos=reservation-only \
     --chdir=/tmp \
     bash -c '
docker run --rm \
    --runtime=nvidia \
    --gpus all \
    --network=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=64g \
    -v "'"$PROJECT_DIR"'":"/workspace/dlrm" \
    -w "/workspace/dlrm" \
    -e NCCL_DEBUG=WARN \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e GLOO_SOCKET_IFNAME=eth0 \
    -e TORCH_CUDA_ARCH_LIST=10.0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "'"$DOCKER_IMAGE"'" \
    bash -c "
        pip install fbgemm-gpu==1.5.0+cu130 --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -3
        pip install torchrec==1.4.0 polars pyarrow pyyaml tqdm datasets psutil 2>&1 | tail -3
        pip install 'flash-attn-4==4.0.0b10' 'flash-attn-4[cu13]' 2>&1 | tail -3
        export PYTHONPATH=/workspace/dlrm:\${PYTHONPATH:-}
        echo \"Starting torchrun...\"
        torchrun \
            --nproc_per_node='"$GPUS"' \
            --standalone \
            scripts/run_distributed.py \
            --config \"'"$CONFIG"'\" \
            --dense-strategy \"'"$DENSE_STRATEGY"'\" \
            --max-steps '"$MAX_STEPS"' \
            --log-interval '"$LOG_INTERVAL"' \
            --run-name \"'"$RUN_NAME"'\" \
            --results-dir \"'"$RESULTS_DIR"'\" \
            --trace-steps \"'"$TRACE_STEPS"'\" \
            --trace-warmup '"$TRACE_WARMUP"' \
            --trace-active '"$TRACE_ACTIVE"' \
            '"$SKIP_EVAL"' '"$TRACE"' '"$PIPELINE"'
    "
'

echo "Training complete. Exit code: $?"
