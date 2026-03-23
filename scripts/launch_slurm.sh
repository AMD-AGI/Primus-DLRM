#!/usr/bin/env bash
# Submit a distributed training job via Slurm sbatch.
#
# Usage:
#   # 1 node, 8 GPUs, full training
#   bash scripts/launch_slurm.sh --nnodes 1 --config configs/dist_onetrans_v6.yaml --run-name onetrans_1n
#
#   # 2 nodes, 16 GPUs, quick test
#   bash scripts/launch_slurm.sh --nnodes 2 --config configs/dist_counter_v1.yaml --max-steps 1000 --run-name counter_2n
#
#   # 4 nodes, DDP, specific time limit
#   bash scripts/launch_slurm.sh --nnodes 4 --dense-strategy ddp --config configs/dist_onetrans_v6.yaml --time 4:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV="$PROJECT_DIR/.venv/bin"

# Defaults
NNODES=1
GPUS=8
CONFIG=""
MAX_STEPS=0
RUN_NAME="dist_slurm"
RESULTS_DIR="results"
DENSE_STRATEGY="ddp"
PARTITION="mi355x"
TIME="12:00:00"
JOB_NAME="dlrm_dist"
LOG_INTERVAL=1000
NCCL_IF=""
SKIP_EVAL=""
TRACE=""
PIPELINE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)         NNODES="$2"; shift 2;;
        --gpus)           GPUS="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --max-steps)      MAX_STEPS="$2"; shift 2;;
        --run-name)       RUN_NAME="$2"; shift 2;;
        --results-dir)    RESULTS_DIR="$2"; shift 2;;
        --dense-strategy) DENSE_STRATEGY="$2"; shift 2;;
        --partition)      PARTITION="$2"; shift 2;;
        --time)           TIME="$2"; shift 2;;
        --job-name)       JOB_NAME="$2"; shift 2;;
        --log-interval)   LOG_INTERVAL="$2"; shift 2;;
        --nccl-if)        NCCL_IF="$2"; shift 2;;
        --skip-eval)      SKIP_EVAL="--skip-eval"; shift;;
        --trace)          TRACE="--trace"; shift;;
        --pipeline)       PIPELINE="--pipeline"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi

LOG_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"

SBATCH_SCRIPT=$(mktemp /tmp/dlrm_sbatch_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NNODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=$GPUS
#SBATCH --cpus-per-task=32
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm_%j.out
#SBATCH --error=$LOG_DIR/slurm_%j.err
#SBATCH --exclusive

set -euo pipefail

export PROJECT_DIR="$PROJECT_DIR"
export VENV="$VENV"

# Resolve master IP from the management interface.
# Use ens9np0 for NCCL/Gloo bootstrap (AINIC management plane).
# If NCCL_IF was explicitly set, use that instead.
NCCL_IF="$NCCL_IF"
if [[ -z "\$NCCL_IF" ]]; then
    NCCL_IF=ens9np0
fi
MASTER_ADDR=\$(ip -4 addr show "\$NCCL_IF" | awk '/inet /{sub(/\/.*/, "", \$2); print \$2}')
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=\$NCCL_IF
export GLOO_SOCKET_IFNAME=\$NCCL_IF
export NCCL_DEBUG=WARN

# AINIC RDMA transport (aligned with Primus start_training_dsv3.sh)
# Use only ionic (AINIC) HCAs for inter-node RDMA; exclude mlx5 which
# causes ibv_modify_qp timeouts on this cluster.
export NCCL_IB_HCA=ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1
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

echo "============================================"
echo "Job ID:        \$SLURM_JOB_ID"
echo "Nodes:         \$SLURM_NODELIST (\$SLURM_NNODES nodes)"
echo "GPUs/node:     $GPUS"
echo "Master:        \$MASTER_ADDR:\$MASTER_PORT"
echo "Config:        $CONFIG"
echo "Strategy:      $DENSE_STRATEGY"
echo "Run name:      $RUN_NAME"
echo "============================================"

cd "\$PROJECT_DIR"
source "\$VENV/activate"
export PYTHONPATH="\$PROJECT_DIR:\${PYTHONPATH:-}"

# srun launches one task per node; each task runs torchrun which spawns
# $GPUS workers. SLURM_PROCID is set per-task by srun (0, 1, ..., N-1).
srun --kill-on-bad-exit=1 bash -c '
    "\$VENV/torchrun" \\
        --nproc_per_node=$GPUS \\
        --nnodes=\$SLURM_NNODES \\
        --node_rank=\$SLURM_PROCID \\
        --master_addr=\$MASTER_ADDR \\
        --master_port=\$MASTER_PORT \\
        scripts/run_distributed.py \\
        --config "$CONFIG" \\
        --dense-strategy "$DENSE_STRATEGY" \\
        --max-steps $MAX_STEPS \\
        --log-interval $LOG_INTERVAL \\
        --run-name "$RUN_NAME" \\
        --results-dir "$RESULTS_DIR" \\
        $SKIP_EVAL $TRACE $PIPELINE
'

echo "Training complete. Exit code: \$?"
SBATCH_EOF

echo "Generated sbatch script: $SBATCH_SCRIPT"
echo "---"
cat "$SBATCH_SCRIPT"
echo "---"

JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo ""
echo "Submitted Slurm job: $JOB_ID"
echo "  Monitor: squeue -j $JOB_ID"
echo "  Logs:    tail -f $LOG_DIR/slurm_${JOB_ID}.out"
echo "  Cancel:  scancel $JOB_ID"
