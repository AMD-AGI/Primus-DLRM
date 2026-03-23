#!/usr/bin/env bash
# Quick multi-node launch via torchrun + SSH (no Slurm reservation).
# Use for short tests only -- nodes stay "idle" in Slurm and could be claimed.
#
# Usage:
#   # Single-node, 2 GPUs
#   bash scripts/launch_torchrun.sh --nnodes 1 --gpus 2 --config configs/dist_counter_v1.yaml --max-steps 100
#
#   # 2 nodes, 8 GPUs each
#   bash scripts/launch_torchrun.sh --nnodes 2 --gpus 8 --config configs/dist_onetrans_v6.yaml --max-steps 5000
#
#   # Specific nodes, skip eval
#   bash scripts/launch_torchrun.sh --nnodes 2 --gpus 8 --nodes chi2866,chi2798 --config configs/dist_onetrans_v6.yaml --skip-eval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin"

# Defaults
NNODES=1
GPUS=8
CONFIG=""
MAX_STEPS=0
RUN_NAME="dist_test"
RESULTS_DIR="results"
DENSE_STRATEGY="ddp"
NODES=""
RDZV_PORT=29500
LOG_INTERVAL=1000
SKIP_EVAL=""
TRACE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)         NNODES="$2"; shift 2;;
        --gpus)           GPUS="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --max-steps)      MAX_STEPS="$2"; shift 2;;
        --run-name)       RUN_NAME="$2"; shift 2;;
        --results-dir)    RESULTS_DIR="$2"; shift 2;;
        --dense-strategy) DENSE_STRATEGY="$2"; shift 2;;
        --nodes)          NODES="$2"; shift 2;;
        --rdzv-port)      RDZV_PORT="$2"; shift 2;;
        --log-interval)   LOG_INTERVAL="$2"; shift 2;;
        --skip-eval)      SKIP_EVAL="--skip-eval"; shift;;
        --trace)          TRACE="--trace"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi

# Pick idle nodes from Slurm if not specified
if [[ "$NNODES" -gt 1 && -z "$NODES" ]]; then
    NODES=$(sinfo -N -h -t idle -p mi355x -o "%N" 2>/dev/null | head -n "$NNODES" | paste -sd,)
    if [[ -z "$NODES" ]]; then
        echo "Error: no idle nodes found. Specify --nodes manually."
        exit 1
    fi
    echo "Auto-selected idle nodes: $NODES"
fi

# AINIC RDMA transport (aligned with AMD maxtext-slurm/train_env.sh)
NCCL_ENV=$(cat <<'ENV'
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=1
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_USE_INLINE=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_ECE_ENABLE=0
export NCCL_CROSS_NIC=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_DMABUF_ENABLE=1
export NCCL_GDRCOPY_ENABLE=1
export NCCL_GDR_FLUSH_DISABLE=1
export NCCL_PXN_DISABLE=0
export NET_OPTIONAL_RECV_COMPLETION=1
export RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
export RCCL_LL128_FORCE_ENABLE=1
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCL_ENABLE=0
export IONIC_LOCKFREE=all
export NCCL_CHECKS_DISABLE=1
ENV
)

eval "$NCCL_ENV"

SCRIPT_ARGS=(
    scripts/run_distributed.py
    --config "$CONFIG"
    --dense-strategy "$DENSE_STRATEGY"
    --max-steps "$MAX_STEPS"
    --log-interval "$LOG_INTERVAL"
    --run-name "$RUN_NAME"
    --results-dir "$RESULTS_DIR"
)
[[ -n "$SKIP_EVAL" ]] && SCRIPT_ARGS+=("$SKIP_EVAL")
[[ -n "$TRACE" ]] && SCRIPT_ARGS+=("$TRACE")

if [[ "$NNODES" -eq 1 ]]; then
    echo "=== Single-node: $GPUS GPUs ==="
    cd "$PROJECT_DIR"
    source "$VENV/activate"
    export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

    "$VENV/torchrun" \
        --nproc_per_node="$GPUS" \
        --nnodes=1 \
        "${SCRIPT_ARGS[@]}"
else
    IFS=',' read -ra NODE_LIST <<< "$NODES"
    MASTER_ADDR="${NODE_LIST[0]}"
    echo "=== Multi-node: $NNODES nodes x $GPUS GPUs ==="
    echo "    Nodes: $NODES"
    echo "    Master: $MASTER_ADDR:$RDZV_PORT"

    TORCHRUN_ARGS=(
        --nproc_per_node="$GPUS"
        --nnodes="$NNODES"
        --rdzv_backend=c10d
        --rdzv_endpoint="${MASTER_ADDR}:${RDZV_PORT}"
    )

    REMOTE_CMD="cd $PROJECT_DIR && source $VENV/activate && export PYTHONPATH=$PROJECT_DIR:\${PYTHONPATH:-}; $NCCL_ENV; $VENV/torchrun ${TORCHRUN_ARGS[*]} ${SCRIPT_ARGS[*]}"

    PIDS=()
    for ((i=1; i<NNODES; i++)); do
        NODE="${NODE_LIST[$i]}"
        echo "  Launching on $NODE (node $i)..."
        ssh -o StrictHostKeyChecking=no "$NODE" "$REMOTE_CMD" \
            2>&1 | sed "s/^/[$NODE] /" &
        PIDS+=($!)
    done

    echo "  Launching on $MASTER_ADDR (node 0, master)..."
    ssh -o StrictHostKeyChecking=no "$MASTER_ADDR" "$REMOTE_CMD" \
        2>&1 | sed "s/^/[$MASTER_ADDR] /"
    MASTER_EXIT=$?

    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    exit $MASTER_EXIT
fi
