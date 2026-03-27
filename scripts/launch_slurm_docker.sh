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
#
#   # Multi-node with AINIC (required for multi-node on AINIC clusters)
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config configs/bench_onetrans_v6.yaml \
#       --run-name 2n_dmp --pipeline --ainic
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
TRACE_STEPS=""
PIPELINE=""
USE_AINIC=""
NODELIST=""
DEPENDENCY=""
DOCKER_IMAGE="tasimage/primus:pr-563-ainic"

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
        --trace-steps)    TRACE_STEPS="$2"; shift 2;;
        --pipeline)       PIPELINE="--pipeline"; shift;;
        --ainic)          USE_AINIC=1; shift;;
        --nodelist)       NODELIST="$2"; shift 2;;
        --dependency)     DEPENDENCY="$2"; shift 2;;
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
$([ -n "$NODELIST" ] && echo "#SBATCH --nodelist=$NODELIST")
$([ -n "$DEPENDENCY" ] && echo "#SBATCH --dependency=$DEPENDENCY")

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
echo "AINIC:         ${USE_AINIC:-disabled}"
echo "============================================"

cd "\$PROJECT_DIR"

# Kill any stale Docker containers on all allocated nodes to free GPU
# resources and ports (e.g. 29500) left behind by cancelled jobs.
srun --kill-on-bad-exit=1 --export=ALL bash -c '
    STALE=\$(docker ps -q 2>/dev/null)
    if [ -n "\$STALE" ]; then
        echo "[\$(hostname)] Stopping stale containers: \$STALE"
        docker kill \$STALE 2>/dev/null || true
    fi
'

# srun launches one task per node; each task starts a Docker container
# that runs torchrun with the appropriate per-node configuration.
# --export=ALL passes all sbatch env vars (NCCL_*, RCCL_*, etc.) to srun tasks.
srun --kill-on-bad-exit=1 --export=ALL bash -c '
    # Docker run flags:
    #   --network=host       Share host network namespace (required for NCCL/RCCL)
    #   --ipc=host           Share host IPC namespace (shared memory for GPU comms)
    #   --device=/dev/kfd    ROCm GPU compute device
    #   --device=/dev/dri    GPU DRM/display device
    #   --device=/dev/infiniband  RDMA/InfiniBand devices for inter-node comms
    #   --privileged         Full host device access (required for RDMA/GDR)
    #   --cap-add=SYS_PTRACE     Allow ptrace (debugging, profiling)
    #   --cap-add=CAP_SYS_ADMIN  Allow sysfs access (GPU topology, perf counters)
    #   --cap-add=IPC_LOCK       Allow memory locking (RDMA pinned memory)
    #   --security-opt seccomp=unconfined  Disable seccomp (GPU driver syscalls)
    #   --group-add video    Access to GPU device nodes
    #
    # Environment variables:
    #   HSA_NO_SCRATCH_RECLAIM=1     Prevent GPU scratch memory reclaim (stability)
    #   HSA_KERNARG_POOL_SIZE=12M    Larger kernel argument pool (prevent OOM)
    #   GPU_MAX_HW_QUEUES=4          Limit GPU hardware queues (stability)
    #   NCCL_IB_HCA=ionic_*          Use only AINIC HCAs for inter-node RDMA
    #   NCCL_IB_GID_INDEX=1          RoCE GID index for AINIC
    #   NCCL_DMABUF_ENABLE=1         DMA-BUF for intra-node GPU direct (overridden by AINIC)
    #   NCCL_GDRCOPY_ENABLE=1        GPU Direct RDMA copy
    #   RCCL_MSCCLPP_ENABLE=1        MSCCL++ protocol for RCCL
    #   IONIC_LOCKFREE=all           Lock-free mode for ionic AINIC adapters
    docker run --rm \
        --network=host \
        --ipc=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --device=/dev/infiniband \
        --group-add video \
        --privileged \
        --cap-add=SYS_PTRACE \
        --cap-add=CAP_SYS_ADMIN \
        --cap-add=IPC_LOCK \
        --security-opt seccomp=unconfined \
        -v "\$PROJECT_DIR":"/workspace/dlrm" \
        -w "/workspace/dlrm" \
        -e MASTER_ADDR="\$MASTER_ADDR" \
        -e MASTER_PORT="\$MASTER_PORT" \
        -e NCCL_SOCKET_IFNAME='$NCCL_IF' \
        -e GLOO_SOCKET_IFNAME='$NCCL_IF' \
        -e NCCL_DEBUG=WARN \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e HSA_KERNARG_POOL_SIZE=12582912 \
        -e GPU_MAX_HW_QUEUES=4 \
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
            # ----------------------------------------------------------
            # AINIC setup (--ainic flag)
            # Uses custom RCCL build + AINIC network plugin (librccl-anp.so)
            # instead of PyTorch bundled RCCL. Required for multi-node on
            # AINIC clusters — without this, inter-node RCCL collectives
            # crash with SIGSEGV because the bundled RCCL lacks the AINIC
            # network transport.
            #
            # WARNING: The NCCL/RCCL tuning values below (IB_TC, FIFO_TC,
            # MAX_P2P_CHANNELS, IB_QPS_PER_CONNECTION, DMABUF_ENABLE) are
            # specific to the AINIC cluster hardware and network topology.
            # These may need adjustment for different clusters or AINIC
            # firmware versions. Refer to Primus/examples/run_pretrain.sh
            # for the canonical AINIC configuration.
            # ----------------------------------------------------------
            if [ '"$USE_AINIC"' = '1' ] && [ -f /opt/amd-anp/build/librccl-anp.so ]; then
                # Load AINIC network plugin for RCCL
                export NCCL_NET_PLUGIN=librccl-anp.so
                # Override LD_LIBRARY_PATH: custom RCCL at /opt/rccl replaces
                # PyTorch bundled librccl.so; AINIC plugin at /opt/amd-anp
                export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:/opt/rccl/build/release:/opt/amd-anp/build:/opt/ompi/lib
                # AINIC-tuned NCCL parameters (override defaults set above)
                export NCCL_DMABUF_ENABLE=0          # AINIC does not use DMA-BUF
                export NCCL_IB_QPS_PER_CONNECTION=1  # fewer queue pairs for AINIC
                export NCCL_IB_TC=41                 # traffic class for QoS
                export NCCL_IB_FIFO_TC=185           # FIFO traffic class
                export NCCL_MAX_P2P_CHANNELS=56      # max peer-to-peer channels
                echo AINIC_enabled
            fi

            pip install --no-cache-dir polars pyarrow pyyaml tqdm datasets pytest psutil
            # Distributed Shampoo optimizer -- requires Python 3.12+
            # Only installed when config contains dense_optimizer: shampoo
            grep -q shampoo /workspace/dlrm/$CONFIG 2>/dev/null && pip install --no-cache-dir git+https://github.com/facebookresearch/optimizers.git && echo distributed-shampoo_installed
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
                --trace-steps \"$TRACE_STEPS\" \
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
