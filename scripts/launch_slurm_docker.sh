#!/usr/bin/env bash
# Submit a distributed training job via Slurm using Docker containers.
#
# Uses Docker with --device=/dev/kfd --device=/dev/dri for ROCm GPU access.
# The project directory is bind-mounted into the container and pip installs
# any missing dependencies (polars, pyarrow, etc.) at container startup.
#
# RDMA / AINIC handling is fully auto-detected per node at launch time:
#   * Host RDMA devices (any /sys/class/infiniband/*) -> NCCL_IB_HCA built
#     from the actually-present HCAs (no hardcoded list, no missing devices).
#   * Host userspace IB drivers (lib{ionic,mlx5,mlx4,efa}-rdmav*.so) ->
#     bind-mounted into the container so they match the host kernel ABI.
#     Without this, kernel-ABI mismatches silently force NCCL onto the
#     NET/Socket fallback (~5% of expected RDMA bandwidth).
#   * Host /boot/config-* -> mounted read-only when present so NCCL can
#     verify CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY (DMABUF support).
#   * AINIC RCCL plugin -> probed inside the container at:
#       /opt/amd-anp/build/librccl-anp.so   (legacy: needs custom RCCL too)
#       /opt/rocm/lib/librccl-anp.so        (new: ships in rocm/pyt-* images)
#       /opt/rocm-*/lib/librccl-anp.so
#     LD_LIBRARY_PATH is updated automatically; no NCCL_NET_PLUGIN override
#     is required (RCCL auto-loads matching plugins).
#
# Override knobs (use sparingly):
#   --no-rdma       Disable RDMA detection (force NET/Socket); useful for
#                   single-node-only debugging of the launcher itself.
#   --legacy-ainic  Use the old --ainic codepath (custom RCCL at /opt/rccl,
#                   NCCL_IB_TC=41, NCCL_IB_FIFO_TC=185, NCCL_MAX_P2P_CHANNELS=56).
#                   Only needed for clusters where the legacy AINIC firmware
#                   requires those QoS classes. Newer images do NOT need it.
#
# Usage:
#   # 1 node, 8 GPUs (RDMA auto-detected)
#   bash scripts/launch_slurm_docker.sh --nnodes 1 --config configs/bench_onetrans_v6.yaml \
#       --run-name docker_1n_test --max-steps 50
#
#   # 2 nodes, 16 GPUs (RDMA auto-detected, libionic auto-mounted)
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config configs/bench_onetrans_v6.yaml \
#       --run-name docker_2n
#
#   # Custom image
#   bash scripts/launch_slurm_docker.sh --nnodes 1 --config configs/bench_onetrans_v6.yaml \
#       --image myregistry/primus:latest
#
#   # Force the legacy AINIC tuning (only if the new auto path won't bring up RDMA)
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config ... --legacy-ainic
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
NNODES=1
GPUS=8
CONFIG=""
DATA_ROOT=""
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
TRACE_WARMUP=5
TRACE_ACTIVE=10
PIPELINE=""
USE_AINIC=""           # legacy: force --ainic codepath (custom RCCL, special TCs)
NO_RDMA=""             # set to 1 to force NET/Socket regardless of HW
NODELIST=""
DEPENDENCY=""
DOCKER_IMAGE="tasimage/primus:pr-563-ainic"

while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)         NNODES="$2"; shift 2;;
        --gpus)           GPUS="$2"; shift 2;;
        --config)         CONFIG="$2"; shift 2;;
        --data-root)      DATA_ROOT="$2"; shift 2;;
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
        --trace-warmup)   TRACE_WARMUP="$2"; shift 2;;
        --trace-active)   TRACE_ACTIVE="$2"; shift 2;;
        --pipeline)       PIPELINE="--pipeline"; shift;;
        --ainic|--legacy-ainic) USE_AINIC=1; shift;;
        --no-rdma)        NO_RDMA=1; shift;;
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

# Build optional flags at sbatch-generation time so they are either present
# (with a value) or completely absent in the rendered torchrun command line.
# Embedding the conditional inside the unquoted heredoc rendered the empty
# DATA_ROOT as `--data-root ""`, which torchrun treated as missing argument.
DATA_ROOT_FLAG=""
[[ -n "$DATA_ROOT" ]] && DATA_ROOT_FLAG="--data-root $DATA_ROOT"

# Pre-create output dirs and set permissions before Docker launch.
# NFS root_squash maps container root to nobody, so dirs must be
# world-writable and train.log must be pre-created for the
# FileHandler inside the container to open it.
LOG_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/logs"
CHECKPOINT_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/checkpoints"
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
touch "$LOG_DIR/train.log"
chmod 666 "$LOG_DIR/train.log"
chmod 777 "$LOG_DIR" "$CHECKPOINT_DIR"

SBATCH_SCRIPT=$(mktemp /tmp/dlrm_docker_sbatch_XXXXXX.sh)

cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NNODES
#SBATCH --ntasks-per-node=1
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
if [ "${NO_RDMA:-}" = "1" ]; then
    echo "RDMA mode:     disabled (--no-rdma) -> NET/Socket"
elif [ "${USE_AINIC:-}" = "1" ]; then
    echo "RDMA mode:     legacy AINIC (--legacy-ainic) -> custom RCCL + librccl-anp.so"
else
    echo "RDMA mode:     auto-detect (host HCAs + libionic mount + AINIC plugin probe)"
fi
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

# srun launches one task per node; each task auto-detects the per-node RDMA
# environment, then starts a Docker container that runs torchrun.
# --export=ALL passes all sbatch env vars (NCCL_*, RCCL_*, etc.) to srun tasks.
srun --kill-on-bad-exit=1 --export=ALL bash -c '
    # ==========================================================
    # Host-side RDMA detection (runs once per node, on the host)
    # ==========================================================
    # We dynamically build RDMA_VOLS (extra -v mounts) and RDMA_ENVS
    # (extra -e env vars) based on what is actually on this host.
    # This keeps the same launcher working across:
    #   - single-node CPU/GPU only (no IB hardware)
    #   - AINIC clusters where the kernel module ABI matches the
    #     container userspace driver out of the box
    #   - AINIC clusters where we must override the container userspace
    #     driver with the host one (kernel ABI mismatch -> Socket fallback)
    RDMA_VOLS=()
    RDMA_ENVS=()
    HCA_LIST=""
    if [ "'"$NO_RDMA"'" != "1" ] && [ -d /sys/class/infiniband ] && \
       [ -n "\$(ls /sys/class/infiniband 2>/dev/null)" ]; then
        # Build NCCL_IB_HCA from devices actually present (drop the dead
        # hardcoded ionic_0,2,3,4,5,7,8,9 list).  Format: dev1:1,dev2:1,...
        HCA_LIST=\$(ls /sys/class/infiniband 2>/dev/null | sort | sed "s/\$/:1/" | paste -sd,)
        RDMA_ENVS+=("-e" "NCCL_IB_HCA=\$HCA_LIST")

        # Mount the host userspace driver for whichever RDMA provider is
        # present.  libibverbs loads the driver named lib<provider>-rdmav*.so
        # at runtime, and that build MUST match the host kernel module ABI.
        # If the container ships a newer driver, ibv_devices reports
        # "Driver X does not support the kernel ABI" and NCCL silently
        # falls back to NET/Socket (~5% of expected RDMA bandwidth).
        for drv in ionic mlx5 mlx4 efa irdma rxe; do
            for libpath in /usr/lib/x86_64-linux-gnu/libibverbs/lib\${drv}-rdmav*.so; do
                if [ -e "\$libpath" ]; then
                    RDMA_VOLS+=("-v" "\$libpath:\$libpath:ro")
                fi
            done
        done

        # Mount /boot read-only so NCCL can verify CONFIG_PCI_P2PDMA /
        # CONFIG_DMABUF_MOVE_NOTIFY in /boot/config-\$(uname -r) before
        # enabling DMABUF.  Without this NCCL prints "DMABUF not enabled".
        if compgen -G "/boot/config-*" >/dev/null 2>&1; then
            RDMA_VOLS+=("-v" "/boot:/boot:ro")
        fi
    fi

    echo "[\$(hostname)] RDMA detected: \${HCA_LIST:-none}"
    echo "[\$(hostname)] RDMA bind-mounts: \${#RDMA_VOLS[@]} entries"

    # Docker run flags:
    #   --network=host           Share host network ns (required for NCCL/RCCL)
    #   --ipc=host               Share host IPC ns (shared memory for GPU comms)
    #   --device=/dev/kfd        ROCm GPU compute device
    #   --device=/dev/dri        GPU DRM/display device
    #   --device=/dev/infiniband RDMA/InfiniBand devices for inter-node comms
    #   --privileged             Full host device access (required for RDMA/GDR)
    #   --cap-add=SYS_PTRACE     Allow ptrace (debugging, profiling)
    #   --cap-add=CAP_SYS_ADMIN  Allow sysfs access (GPU topology, perf counters)
    #   --cap-add=IPC_LOCK       Allow memory locking (RDMA pinned memory)
    #   --security-opt seccomp=unconfined  Disable seccomp (GPU driver syscalls)
    #   --group-add video        Access to GPU device nodes
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
        "\${RDMA_VOLS[@]}" \
        -e MASTER_ADDR="\$MASTER_ADDR" \
        -e MASTER_PORT="\$MASTER_PORT" \
        -e NCCL_SOCKET_IFNAME='$NCCL_IF' \
        -e GLOO_SOCKET_IFNAME='$NCCL_IF' \
        -e NCCL_DEBUG=WARN \
        -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e HSA_KERNARG_POOL_SIZE=12582912 \
        -e GPU_MAX_HW_QUEUES=4 \
        "\${RDMA_ENVS[@]}" \
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
            # ==========================================================
            # Container-side AINIC plugin auto-detect
            # ==========================================================
            # Probe known librccl-anp.so locations. When found, prepend
            # its directory to LD_LIBRARY_PATH so RCCL auto-loads it.
            # No NCCL_NET_PLUGIN env override needed; RCCL scans
            # LD_LIBRARY_PATH for librccl-net*.so / librccl-anp.so itself.
            ANP_PATH=\"\"
            for p in /opt/amd-anp/build/librccl-anp.so \\
                     /opt/rocm/lib/librccl-anp.so \\
                     /opt/rocm-*/lib/librccl-anp.so; do
                # The glob is intentional; expand and pick first match.
                for q in \\\$p; do
                    if [ -f \"\\\$q\" ]; then ANP_PATH=\"\\\$q\"; break 2; fi
                done
            done
            if [ -n \"\\\$ANP_PATH\" ]; then
                ANP_DIR=\\\$(dirname \"\\\$ANP_PATH\")
                export LD_LIBRARY_PATH=\"\\\$ANP_DIR:\\\${LD_LIBRARY_PATH:-}\"
                echo \"[anp] plugin: \\\$ANP_PATH\"
            else
                echo \"[anp] no librccl-anp.so found (RCCL will pick best NET transport)\"
            fi

            # ==========================================================
            # Legacy AINIC override (--legacy-ainic / --ainic)
            # ==========================================================
            # Use custom RCCL build at /opt/rccl + AINIC firmware-specific
            # QoS classes. Only enable for clusters where the new
            # auto-loaded path will not bring up RDMA (typically very
            # early AINIC firmware). Newer images do NOT need this; the
            # auto-detect block above handles them.
            if [ '"$USE_AINIC"' = '1' ] && [ -f /opt/amd-anp/build/librccl-anp.so ]; then
                export NCCL_NET_PLUGIN=librccl-anp.so
                export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:/opt/rccl/build/release:/opt/amd-anp/build:/opt/ompi/lib
                export NCCL_DMABUF_ENABLE=0          # legacy AINIC firmware: no DMABUF
                export NCCL_IB_QPS_PER_CONNECTION=1  # legacy AINIC firmware: 1 QP
                export NCCL_IB_TC=41                 # legacy AINIC firmware: QoS TC
                export NCCL_IB_FIFO_TC=185           # legacy AINIC firmware: FIFO TC
                export NCCL_MAX_P2P_CHANNELS=56      # legacy AINIC firmware: P2P chans
                echo legacy_ainic_enabled
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
                $DATA_ROOT_FLAG \
                --dense-strategy \"$DENSE_STRATEGY\" \
                --max-steps $MAX_STEPS \
                --log-interval $LOG_INTERVAL \
                --run-name \"$RUN_NAME\" \
                --results-dir \"$RESULTS_DIR\" \
                --trace-steps \"$TRACE_STEPS\" \
                --trace-warmup $TRACE_WARMUP \
                --trace-active $TRACE_ACTIVE \
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
