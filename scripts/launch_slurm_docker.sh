#!/usr/bin/env bash
# Submit a distributed training job via Slurm using Docker containers.
# Supports both AMD ROCm and NVIDIA CUDA platforms with auto-detection.
#
# Platform detection:
#   --platform auto (default): detects /dev/kfd (AMD) or nvidia-smi (NVIDIA)
#   --platform amd:            force AMD ROCm path
#   --platform nvidia:         force NVIDIA CUDA path
#
# Submission modes:
#   --submit sbatch (default): generate and submit an sbatch script
#   --submit srun:             run interactively via srun (requires --reservation)
#
# Usage:
#   # AMD: 2 nodes via sbatch
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config configs/... \
#       --run-name amd_2n --partition amd-aig
#
#   # NVIDIA: 2 nodes via srun with reservation
#   bash scripts/launch_slurm_docker.sh --nnodes 2 --config configs/... \
#       --run-name nv_2n --submit srun --reservation gh-chcai-xxx \
#       --nodelist "hungry-hippo-fin-03-[1-2]"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ---- Defaults ----
# Mirrors run_distributed.py / Slurm semantics. Only --config is required.
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
NCCL_IF=""                 # bootstrap interface (auto-set per platform below)
SKIP_EVAL=""
TRACE=""
TRACE_STEPS=""
TRACE_WARMUP=5
TRACE_ACTIVE=10
PIPELINE=""
USE_AINIC=""               # 1 = legacy AINIC custom-RCCL path; default uses auto-detected librccl-anp.so
NO_RDMA=""                 # 1 = force NET/Socket fallback (debug only; ~5% of RDMA bandwidth)
NODELIST=""
DEPENDENCY=""
RESERVATION=""             # required for --submit srun
QOS=""                     # auto-set to "reservation-only" when --reservation is used with srun
RDZV_PORT=29500
PLATFORM="auto"            # amd | nvidia | auto (probe /dev/kfd or nvidia-smi)
SUBMIT="sbatch"            # sbatch (queue) or srun (interactive, requires --reservation)
DOCKER_IMAGE=""            # platform-specific default applied below if empty

# Performance knobs (optional; off by default to keep launcher backward-compatible).
# Pass these to recover the canonical 316 TF/s/GPU baseline on AMD MI355X with 5b config.
ATTN_IMPL=""               # --attention-impl override -> {fav2 (CK), turbo (aiter), sdpa, fav4}
FA_WHEEL=""                # local CK FlashAttention wheel; force-installed before torchrun
HIPBLASLT_TUNE_FILE=""     # offline GEMM tuning override (~7% on MI355X; use ours at configs/hipblaslt_tune/)
# polars-u64-idx (always installed): 64-bit row index polars, required for 5b-scale
# (>2^32 row) caches; drop-in replacement for stock polars at ~negligible cost on smaller data.

while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes|--nodes) NNODES="$2"; shift 2;;
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
        --reservation)    RESERVATION="$2"; shift 2;;
        --qos)            QOS="$2"; shift 2;;
        --rdzv-port)      RDZV_PORT="$2"; shift 2;;
        --platform)       PLATFORM="$2"; shift 2;;
        --submit)         SUBMIT="$2"; shift 2;;
        --image)          DOCKER_IMAGE="$2"; shift 2;;
        # perf knobs:
        --attention-impl|--attn-impl) ATTN_IMPL="$2"; shift 2;;
        --fa-wheel)       FA_WHEEL="$2"; shift 2;;
        --hipblaslt-tune) HIPBLASLT_TUNE_FILE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required"; exit 1
fi

# ---- Platform detection ----
# Probe the login/submit host (not the compute node) -- both clusters we run on
# are homogeneous so this is a safe proxy for the compute-node platform.
if [[ "$PLATFORM" == "auto" ]]; then
    if [[ -e /dev/kfd ]]; then
        PLATFORM="amd"
    elif command -v nvidia-smi &>/dev/null; then
        PLATFORM="nvidia"
    else
        echo "Error: cannot auto-detect platform. Use --platform amd|nvidia."; exit 1
    fi
fi

# Per-cluster image / interface defaults. Override via --image and --nccl-if when
# moving to a new cluster (and update these defaults if the new cluster becomes the norm).
#   AMD default:    a ROCm image that bundles librccl-anp.so under /opt/rocm*/lib
#                   plus polars / torchrec / fbgemm-gpu (auto-detected at runtime)
#   NVIDIA default: a CUDA PyTorch base image; matches scripts/launch_srun_nvidia.sh
[[ -z "$DOCKER_IMAGE" ]] && {
    [[ "$PLATFORM" == "amd" ]] && DOCKER_IMAGE="rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424" || DOCKER_IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"
}
[[ -z "$NCCL_IF" ]] && {
    [[ "$PLATFORM" == "amd" ]] && NCCL_IF="ens9np0" || NCCL_IF="eth0"
}
[[ "$SUBMIT" == "srun" && -z "$QOS" && -n "$RESERVATION" ]] && QOS="reservation-only"

if [[ "$SUBMIT" == "srun" ]]; then
    [[ -z "$RESERVATION" ]] && { echo "Error: --reservation required with --submit srun"; exit 1; }
    [[ -z "$NODELIST" ]]    && { echo "Error: --nodelist required with --submit srun"; exit 1; }
fi

DATA_ROOT_FLAG=""
[[ -n "$DATA_ROOT" ]] && DATA_ROOT_FLAG="--data-root $DATA_ROOT"

LOG_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/logs"
CHECKPOINT_DIR="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/checkpoints"
# NFS root_squash maps container root -> nobody. Pre-create dirs world-writable
# so the in-container FileHandler can open train.log.
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
touch "$LOG_DIR/train.log" 2>/dev/null || true
chmod 666 "$LOG_DIR/train.log" 2>/dev/null || true
chmod 777 "$LOG_DIR" "$CHECKPOINT_DIR" 2>/dev/null || true

echo "============================================"
echo "Platform:      $PLATFORM"
echo "Submit:        $SUBMIT"
echo "Nodes:         $NNODES"
[[ -n "$NODELIST" ]]    && echo "Nodelist:      $NODELIST"
[[ -n "$RESERVATION" ]] && echo "Reservation:   $RESERVATION"
echo "GPUs/node:     $GPUS"
echo "Config:        $CONFIG"
echo "Strategy:      $DENSE_STRATEGY"
echo "Run name:      $RUN_NAME"
echo "Docker image:  $DOCKER_IMAGE"
echo "Max steps:     $MAX_STEPS"
echo "============================================"

# ==============================================================================
# Generate the srun payload script that runs on each compute node.
# Built in three layered heredocs to keep escaping sane:
#   EOF_HEADER  -- 'quoted' literal: shebang + Slurm env probing
#   EOF_VARS    -- unquoted: $-substituted constants captured at generation time
#   EOF_BODY    -- 'quoted' literal: the per-node logic that runs on every task
# Written to the shared FS so every Slurm task on every node can exec the same file.
# ==============================================================================
SRUN_PAYLOAD="$PROJECT_DIR/$RESULTS_DIR/$RUN_NAME/.srun_payload.sh"

cat > "$SRUN_PAYLOAD" << 'EOF_HEADER'
#!/usr/bin/env bash
set -euo pipefail

# Resolve master address (rank-0 hostname) and per-task node rank from Slurm env.
# Falls back to localhost / rank 0 when invoked outside Slurm (smoke tests).
if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
else
    MASTER_ADDR=${MASTER_ADDR:-localhost}
fi
NODE_RANK=${SLURM_NODEID:-0}
EOF_HEADER

# Inject platform-aware constants (expanded at generation time on the submit host).
cat >> "$SRUN_PAYLOAD" << EOF_VARS
PLATFORM="$PLATFORM"
PROJECT_DIR="$PROJECT_DIR"
DOCKER_IMAGE="$DOCKER_IMAGE"
NCCL_IF="$NCCL_IF"
NO_RDMA="$NO_RDMA"
USE_AINIC="$USE_AINIC"
NNODES=$NNODES
GPUS=$GPUS
RDZV_PORT=$RDZV_PORT
CONFIG="$CONFIG"
DATA_ROOT_FLAG="$DATA_ROOT_FLAG"
DENSE_STRATEGY="$DENSE_STRATEGY"
MAX_STEPS=$MAX_STEPS
LOG_INTERVAL=$LOG_INTERVAL
RUN_NAME="$RUN_NAME"
RESULTS_DIR="$RESULTS_DIR"
TRACE_STEPS="$TRACE_STEPS"
TRACE_WARMUP=$TRACE_WARMUP
TRACE_ACTIVE=$TRACE_ACTIVE
SKIP_EVAL="$SKIP_EVAL"
TRACE="$TRACE"
PIPELINE="$PIPELINE"
ATTN_IMPL="$ATTN_IMPL"
FA_WHEEL="$FA_WHEEL"
HIPBLASLT_TUNE_FILE="$HIPBLASLT_TUNE_FILE"
EOF_VARS

cat >> "$SRUN_PAYLOAD" << 'EOF_BODY'

echo "[node $NODE_RANK] MASTER_ADDR=$MASTER_ADDR  HOSTNAME=$(hostname)  PLATFORM=$PLATFORM"

# ==================================================================
# Host-side RDMA detection (AMD only; NVIDIA path uses --gpus all + bundled NCCL)
# ==================================================================
# Builds two arrays passed to docker run:
#   RDMA_VOLS  -- bind-mounts of host userspace IB drivers + /boot
#   RDMA_ENVS  -- NCCL_IB_HCA pointing at devices that actually exist
# This block is the difference between RDMA running at full speed and silently
# falling back to NET/Socket (~5% of expected throughput, was the original bug).
RDMA_VOLS=()
RDMA_ENVS=()
HCA_LIST=""

if [[ "$PLATFORM" == "amd" && "$NO_RDMA" != "1" ]] && \
   [ -d /sys/class/infiniband ] && \
   [ -n "$(ls /sys/class/infiniband 2>/dev/null)" ]; then
    # Build NCCL_IB_HCA from devices actually present (replaces hardcoded list).
    HCA_LIST=$(ls /sys/class/infiniband 2>/dev/null | sort | sed "s/$/:1/" | paste -sd,)
    RDMA_ENVS+=(-e "NCCL_IB_HCA=$HCA_LIST")

    # Bind-mount the *host*'s userspace driver for every RDMA provider that has one.
    # libibverbs loads lib<provider>-rdmav*.so at runtime and that build MUST match
    # the host kernel module's ABI. If the container ships a newer driver, ibv_devices
    # logs "Driver X does not support the kernel ABI" and NCCL silently picks Socket.
    for drv in ionic mlx5 mlx4 efa irdma rxe; do
        for libpath in /usr/lib/x86_64-linux-gnu/libibverbs/lib${drv}-rdmav*.so; do
            [ -e "$libpath" ] && RDMA_VOLS+=(-v "$libpath:$libpath:ro")
        done
    done
    # NCCL needs to read /boot/config-$(uname -r) to enable DMABUF. Without this
    # mount, NCCL_DMABUF_ENABLE=1 prints "DMABUF not enabled" and silently disables itself.
    compgen -G "/boot/config-*" >/dev/null 2>&1 && RDMA_VOLS+=(-v "/boot:/boot:ro")
fi
echo "[$(hostname)] RDMA detected: ${HCA_LIST:-none}"

# ==================================================================
# Build docker run command (assembled into an array so quoting is sane)
# ==================================================================
# --network=host: required so NCCL can use the host's RDMA fabric
# --ipc=host    : shared memory for intra-node GPU comms (DMA-BUF, GDRCopy)
DOCKER_CMD=(docker run --rm --network=host --ipc=host)

if [[ "$PLATFORM" == "amd" ]]; then
    # /dev/kfd: ROCm compute device (HSA queues)
    # /dev/dri: GPU DRM device (sysfs / topology)
    # SYS_PTRACE + CAP_SYS_ADMIN: needed for kineto/profiler perf counters
    # IPC_LOCK: pinned memory for RDMA queues
    DOCKER_CMD+=(
        --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband
        --group-add video --privileged
        --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN --cap-add=IPC_LOCK
        --security-opt seccomp=unconfined
    )
else
    # NVIDIA runtime injects the GPU device files; --shm-size matches DGX defaults.
    DOCKER_CMD+=(
        --runtime=nvidia --gpus all --privileged
        --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=64g
        -v /dev/infiniband:/dev/infiniband
    )
fi

DOCKER_CMD+=(
    -v "$PROJECT_DIR":"/workspace/dlrm" -w "/workspace/dlrm"
    "${RDMA_VOLS[@]}"
    -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=$RDZV_PORT
    -e NODE_RANK="$NODE_RANK"
    -e NCCL_SOCKET_IFNAME="$NCCL_IF" -e GLOO_SOCKET_IFNAME="$NCCL_IF"
    "${RDMA_ENVS[@]}"
    -e SLURM_PROCID="${SLURM_PROCID:-0}"
)

# Optional perf knobs
[[ -n "$HIPBLASLT_TUNE_FILE" ]] && DOCKER_CMD+=(-e "HIPBLASLT_TUNING_OVERRIDE_FILE=$HIPBLASLT_TUNE_FILE")
# Mount the tune file's containing dir read-only so it's visible inside the container
if [[ -n "$HIPBLASLT_TUNE_FILE" && -e "$HIPBLASLT_TUNE_FILE" ]]; then
    DOCKER_CMD+=(-v "$(dirname "$HIPBLASLT_TUNE_FILE"):$(dirname "$HIPBLASLT_TUNE_FILE"):ro")
fi
# Mount the FA wheel so the in-container pip install can find it
if [[ -n "$FA_WHEEL" && -e "$FA_WHEEL" ]]; then
    DOCKER_CMD+=(-v "$FA_WHEEL:$FA_WHEEL:ro")
fi

# Platform-specific NCCL/RCCL env vars.
# AMD block is the canonical AINIC tune (sourced from AMD-AGI/maxtext-slurm/train_env.sh):
# QoS-related vars (TC, FIFO_TC, MAX_P2P_CHANNELS) are NOT here -- they belong to the
# legacy AINIC firmware path and are only set inside the container if --legacy-ainic is on.
if [[ "$PLATFORM" == "amd" ]]; then
    DOCKER_CMD+=(
        -e NCCL_DEBUG=WARN
        # ROCm runtime stability:
        -e HSA_NO_SCRATCH_RECLAIM=1 -e HSA_KERNARG_POOL_SIZE=12582912
        -e GPU_MAX_HW_QUEUES=4
        # IB/RoCE basics (NCCL_IB_HCA itself is set in RDMA_ENVS above)
        -e NCCL_IB_GID_INDEX=1 -e NCCL_IB_PCI_RELAXED_ORDERING=1
        -e NCCL_IB_USE_INLINE=1 -e NCCL_IB_QPS_PER_CONNECTION=4
        -e NCCL_IB_ECE_ENABLE=0 -e NCCL_CROSS_NIC=0
        -e NCCL_IGNORE_CPU_AFFINITY=1
        # GPU-direct RDMA path (depends on /boot mount + DMABUF support):
        -e NCCL_DMABUF_ENABLE=1 -e NCCL_GDRCOPY_ENABLE=1
        -e NCCL_GDR_FLUSH_DISABLE=1 -e NCCL_PXN_DISABLE=0
        -e NET_OPTIONAL_RECV_COMPLETION=1
        -e RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING=0
        # MSCCL++ for tuned multi-node algos; classic MSCCL off (LL kernels conflict):
        -e RCCL_LL128_FORCE_ENABLE=1 -e RCCL_MSCCLPP_ENABLE=1
        -e RCCL_MSCCL_ENABLE=0 -e IONIC_LOCKFREE=all
        -e NCCL_CHECKS_DISABLE=1
    )
else
    # NVIDIA: minimal NCCL config; expandable_segments helps with KJT-style grad scatter.
    DOCKER_CMD+=(
        -e NCCL_DEBUG=INFO
        -e NCCL_IBEXT_DISABLE=1 -e NCCL_IB_PKEY=1
        -e TORCH_CUDA_ARCH_LIST=10.0
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )
fi

DOCKER_CMD+=("$DOCKER_IMAGE")

# ==================================================================
# Container entry script (inline; runs inside the container as bash -c)
# ==================================================================
# Single-quoted heredoc-as-string so $-vars stay literal until the *container's*
# bash evaluates them. Outer constants are interpolated via '"$VAR"' splices.
CONTAINER_SCRIPT='
set -e

# ---- AMD: AINIC plugin auto-detect + legacy override ----
# Most modern ROCm images ship librccl-anp.so under /opt/rocm-X.Y.Z/lib; some
# legacy images put it under /opt/amd-anp/build. Probe both, prepend the dir
# to LD_LIBRARY_PATH and let RCCL auto-load it (no NCCL_NET_PLUGIN required).
if [ "'"$PLATFORM"'" = "amd" ]; then
    ANP_PATH=""
    for p in /opt/amd-anp/build/librccl-anp.so \
             /opt/rocm/lib/librccl-anp.so \
             /opt/rocm-*/lib/librccl-anp.so; do
        for q in $p; do
            [ -f "$q" ] && { ANP_PATH="$q"; break 2; }
        done
    done
    if [ -n "$ANP_PATH" ]; then
        export LD_LIBRARY_PATH="$(dirname "$ANP_PATH"):${LD_LIBRARY_PATH:-}"
        echo "[anp] plugin: $ANP_PATH"
    else
        echo "[anp] no librccl-anp.so found"
    fi
    if [ "'"$USE_AINIC"'" = "1" ] && [ -f /opt/amd-anp/build/librccl-anp.so ]; then
        export NCCL_NET_PLUGIN=librccl-anp.so
        export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu/libibverbs:/opt/rccl/build/release:/opt/amd-anp/build:/opt/ompi/lib
        export NCCL_DMABUF_ENABLE=0 NCCL_IB_QPS_PER_CONNECTION=1
        export NCCL_IB_TC=41 NCCL_IB_FIFO_TC=185 NCCL_MAX_P2P_CHANNELS=56
        echo legacy_ainic_enabled
    fi
fi

# ---- Pip installs ----
# polars-u64-idx (64-bit row index) is mandatory: stock polars wraps at 2^32 rows
# which crashes during 5b cache load. Drop-in replacement; ~negligible cost on smaller data.
if [ "'"$PLATFORM"'" = "amd" ]; then
    pip install --no-cache-dir polars-u64-idx pyarrow pyyaml tqdm datasets pytest psutil numba 2>&1 | tail -3
    grep -q shampoo /workspace/dlrm/'"$CONFIG"' 2>/dev/null && \
        pip install --no-cache-dir git+https://github.com/facebookresearch/optimizers.git 2>&1 | tail -3
else
    pip install /workspace/dlrm/pip_cache/fbgemm_gpu_nightly-2026.4.29-cp312-cp312-linux_x86_64.whl 2>&1 | tail -3
    pip install --no-deps torchrec==1.4.0 2>&1 | tail -3
    pip install polars-u64-idx pyarrow pyyaml tqdm datasets psutil torchmetrics tensordict pyre-extensions iopath typing-inspect 2>&1 | tail -3
    pip install "flash-attn-4==4.0.0b10" "flash-attn-4[cu13]" 2>&1 | tail -3
fi
# Optional CK FlashAttention wheel (yiding12 / fav2 path).
if [ -n "'"$FA_WHEEL"'" ] && [ -e "'"$FA_WHEEL"'" ]; then
    pip install --force-reinstall --no-deps "'"$FA_WHEEL"'" 2>&1 | tail -3
fi
# Surface tuning override file path inside the container if it was provided.
if [ -n "${HIPBLASLT_TUNING_OVERRIDE_FILE:-}" ]; then
    echo "[hipblaslt] tuning override: $HIPBLASLT_TUNING_OVERRIDE_FILE  ($([ -e "$HIPBLASLT_TUNING_OVERRIDE_FILE" ] && echo OK || echo MISSING))"
fi

export PYTHONPATH=/workspace/dlrm:${PYTHONPATH:-}
ATTN_FLAG=""
[ -n "'"$ATTN_IMPL"'" ] && ATTN_FLAG="--attention-impl '"$ATTN_IMPL"'"
echo "[node $NODE_RANK] Starting torchrun (nnodes='"$NNODES"', rdzv=$MASTER_ADDR:'"$RDZV_PORT"')..."
torchrun \
    --nproc_per_node='"$GPUS"' \
    --nnodes='"$NNODES"' \
    --node_rank=$NODE_RANK \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:'"$RDZV_PORT"' \
    scripts/run_distributed.py \
    --config "'"$CONFIG"'" \
    '"$DATA_ROOT_FLAG"' \
    --dense-strategy "'"$DENSE_STRATEGY"'" \
    --max-steps '"$MAX_STEPS"' \
    --log-interval '"$LOG_INTERVAL"' \
    --run-name "'"$RUN_NAME"'" \
    --results-dir "'"$RESULTS_DIR"'" \
    --trace-steps "'"$TRACE_STEPS"'" \
    --trace-warmup '"$TRACE_WARMUP"' \
    --trace-active '"$TRACE_ACTIVE"' \
    $ATTN_FLAG \
    '"$SKIP_EVAL"' '"$TRACE"' '"$PIPELINE"'
'

"${DOCKER_CMD[@]}" bash -c "$CONTAINER_SCRIPT"
EOF_BODY

chmod +x "$SRUN_PAYLOAD"

# ==============================================================================
# Submit the job
# Two paths share the same SRUN_PAYLOAD:
#   srun (interactive) -- requires --reservation; blocks until done
#   sbatch (queued)    -- generates a thin sbatch wrapper that srun's the payload
# ==============================================================================
if [[ "$SUBMIT" == "srun" ]]; then
    SRUN_ARGS=(
        --reservation="$RESERVATION"
        --nodelist="$NODELIST"
        --nodes="$NNODES"
        --ntasks="$NNODES"
        --ntasks-per-node=1
        --chdir=/tmp
    )
    [[ -n "$QOS" ]] && SRUN_ARGS+=(--qos="$QOS")

    echo "Running via srun..."
    srun "${SRUN_ARGS[@]}" bash "$SRUN_PAYLOAD"
    EXIT_CODE=$?
    rm -f "$SRUN_PAYLOAD" 2>/dev/null || true
    echo "Training complete. Exit code: $EXIT_CODE"

else
    # sbatch path: write a wrapper script to /tmp that:
    #   1) prints job-level metadata into the slurm.out
    #   2) kills any stale containers from a previous (cancelled) run
    #   3) srun's the shared SRUN_PAYLOAD on each allocated node
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

echo "============================================"
echo "Job ID:        \$SLURM_JOB_ID"
echo "Nodes:         \$SLURM_NODELIST (\$SLURM_NNODES nodes)"
echo "Platform:      $PLATFORM"
echo "GPUs/node:     $GPUS"
echo "Config:        $CONFIG"
echo "Strategy:      $DENSE_STRATEGY"
echo "Run name:      $RUN_NAME"
echo "Docker image:  $DOCKER_IMAGE"
echo "============================================"

cd "$PROJECT_DIR"

# Kill stale containers on every allocated node (frees GPU + port 29500 from
# crashed/cancelled previous runs). Safe no-op when nothing is running.
srun --kill-on-bad-exit=1 --export=ALL bash -c '
    STALE=\$(docker ps -q 2>/dev/null)
    if [ -n "\$STALE" ]; then
        echo "[\$(hostname)] Stopping stale containers: \$STALE"
        docker kill \$STALE 2>/dev/null || true
    fi
'

# Launch the actual training payload (one task per node; payload spawns the docker container).
srun --kill-on-bad-exit=1 --export=ALL bash "$SRUN_PAYLOAD"

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
fi
