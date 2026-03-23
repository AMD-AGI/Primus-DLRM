#!/usr/bin/env bash
# Submit a distributed training job via Slurm using Docker containers.
#
# Uses Docker with --device=/dev/kfd --device=/dev/dri for ROCm GPU access.
# The project directory is bind-mounted into the container and pip installs
# any missing dependencies (polars, pyarrow, etc.) at container startup.
#
# Usage:
#   # 1 node, 8 GPUs
#   bash scripts/launch_slurm_docker.sh --nnodes 1 --config configs/bench_onetrans_v6.yaml \
#       --run-name docker_1n_test --max-steps 50
#
#   # 2 nodes, 16 GPUs
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config configs/bench_onetrans_v6.yaml \
#       --run-name docker_2n
#
#   # Custom image
#   bash scripts/launch_slurm_docker.sh --nnodes 1 --config configs/bench_onetrans_v6.yaml \
#       --image myregistry/primus:latest
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
NNODES=1
GPUS=8
CONFIG=""
MAX_STEPS=0
RUN_NAME="docker_dist"
RESULTS_DIR="results"
DENSE_STRATEGY="ddp"
PARTITION="amd-aig"
TIME="12:00:00"
JOB_NAME="dlrm_docker"
LOG_INTERVAL=1000
NCCL_IF=""
SKIP_EVAL=""
TRACE=""
PIPELINE=""
DOCKER_IMAGE="tasimage/primus:pr-609-ainic"

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
        --image)          DOCKER_IMAGE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"
    exit 1
fi

LOG_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/logs"
mkdir -p "$LOG_DIR"

SBATCH_SCRIPT=$(mktemp /tmp/dlrm_docker_sbatch_XXXXXX.sh)

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
export DOCKER_IMAGE="$DOCKER_IMAGE"

# Resolve master IP from the management interface.
NCCL_IF="$NCCL_IF"
if [[ -z "\$NCCL_IF" ]]; then
    NCCL_IF=ens9np0
fi
MASTER_ADDR=\$(ip -4 addr show "\$NCCL_IF" | awk '/inet /{sub(/\/.*/, "", \$2); print \$2}')
MASTER_PORT=29500
export MASTER_ADDR MASTER_PORT

echo "============================================"
echo "Job ID:        \$SLURM_JOB_ID"
echo "Nodes:         \$SLURM_NODELIST (\$SLURM_NNODES nodes)"
echo "GPUs/node:     $GPUS"
echo "Master:        \$MASTER_ADDR:\$MASTER_PORT"
echo "Config:        $CONFIG"
echo "Strategy:      $DENSE_STRATEGY"
echo "Run name:      $RUN_NAME"
echo "Docker image:  $DOCKER_IMAGE"
echo "============================================"

cd "\$PROJECT_DIR"

# srun launches one task per node; each task starts a Docker container
# that runs torchrun with the appropriate per-node configuration.
srun --kill-on-bad-exit=1 bash -c '
    docker run --rm \
        --network=host \
        --ipc=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -v "\$PROJECT_DIR":"/workspace/dlrm" \
        -w "/workspace/dlrm" \
        -e MASTER_ADDR="\$MASTER_ADDR" \
        -e MASTER_PORT="\$MASTER_PORT" \
        -e NCCL_SOCKET_IFNAME='"$NCCL_IF"' \
        -e GLOO_SOCKET_IFNAME='"$NCCL_IF"' \
        -e NCCL_DEBUG=WARN \
        -e NCCL_IB_HCA=ionic_0:1,ionic_2:1,ionic_3:1,ionic_4:1,ionic_5:1,ionic_7:1,ionic_8:1,ionic_9:1 \
        -e NCCL_IB_GID_INDEX=1 \
        -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
        -e NCCL_IB_USE_INLINE=1 \
        -e NCCL_IB_QPS_PER_CONNECTION=4 \
        -e NCCL_IB_ECE_ENABLE=0 \
        -e NCCL_CROSS_NIC=0 \
        -e NCCL_IGNORE_CPU_AFFINITY=1 \
        -e NCCL_DMABUF_ENABLE=1 \
        -e NCCL_GDRCOPY_ENABLE=1 \
        -e NCCL_GDR_FLUSH_DISABLE=1 \
        -e NCCL_PXN_DISABLE=0 \
        -e NET_OPTIONAL_RECV_COMPLETION=1 \
        -e RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0 \
        -e RCCL_LL128_FORCE_ENABLE=1 \
        -e RCCL_MSCCLPP_ENABLE=1 \
        -e RCCL_MSCCL_ENABLE=0 \
        -e IONIC_LOCKFREE=all \
        -e NCCL_CHECKS_DISABLE=1 \
        -e SLURM_PROCID="\$SLURM_PROCID" \
        "\$DOCKER_IMAGE" \
        bash -c "
            pip install --quiet --no-cache-dir polars pyarrow pyyaml tqdm datasets pytest 2>/dev/null
            export PYTHONPATH=/workspace/dlrm:\${PYTHONPATH:-}
            torchrun \
                --nproc_per_node='$GPUS' \
                --nnodes=\$SLURM_NNODES \
                --node_rank=\$SLURM_PROCID \
                --master_addr=\$MASTER_ADDR \
                --master_port=\$MASTER_PORT \
                scripts/run_distributed.py \
                --config \"$CONFIG\" \
                --dense-strategy \"$DENSE_STRATEGY\" \
                --max-steps $MAX_STEPS \
                --log-interval $LOG_INTERVAL \
                --run-name \"$RUN_NAME\" \
                --results-dir \"$RESULTS_DIR\" \
                $SKIP_EVAL $TRACE $PIPELINE
        "
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
echo "  Logs:    tail -f $LOG_DIR/slurm_\${JOB_ID}.out"
echo "  Cancel:  scancel $JOB_ID"
