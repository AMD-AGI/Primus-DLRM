#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

COMMON_ARGS="--processed-dir data/processed --results-dir results --eval-top-k 5000"

echo "=== Launching 7 comparison experiments in parallel ==="
echo "Start time: $(date)"

python scripts/run_experiment.py \
    --config configs/exp_dlrm_dot.yaml \
    --run-name dlrm_dot \
    --device cuda:0 \
    $COMMON_ARGS \
    > results/dlrm_dot_stdout.log 2>&1 &
PID_0=$!
echo "[GPU 0] dlrm_dot (PID=$PID_0)"

python scripts/run_experiment.py \
    --config configs/exp_dlrm_dcnv2.yaml \
    --run-name dlrm_dcnv2 \
    --device cuda:1 \
    $COMMON_ARGS \
    > results/dlrm_dcnv2_stdout.log 2>&1 &
PID_1=$!
echo "[GPU 1] dlrm_dcnv2 (PID=$PID_1)"

python scripts/run_experiment.py \
    --config configs/exp_onetrans_small.yaml \
    --run-name onetrans_small \
    --device cuda:2 \
    $COMMON_ARGS \
    > results/onetrans_small_stdout.log 2>&1 &
PID_2=$!
echo "[GPU 2] onetrans_small (PID=$PID_2)"

python scripts/run_experiment.py \
    --config configs/exp_onetrans_base.yaml \
    --run-name onetrans_base \
    --device cuda:3 \
    $COMMON_ARGS \
    > results/onetrans_base_stdout.log 2>&1 &
PID_3=$!
echo "[GPU 3] onetrans_base (PID=$PID_3)"

python scripts/run_experiment.py \
    --config configs/exp_onetrans_deep.yaml \
    --run-name onetrans_deep \
    --device cuda:4 \
    $COMMON_ARGS \
    > results/onetrans_deep_stdout.log 2>&1 &
PID_4=$!
echo "[GPU 4] onetrans_deep (PID=$PID_4)"

python scripts/run_experiment.py \
    --config configs/exp_onetrans_wide.yaml \
    --run-name onetrans_wide \
    --device cuda:5 \
    $COMMON_ARGS \
    > results/onetrans_wide_stdout.log 2>&1 &
PID_5=$!
echo "[GPU 5] onetrans_wide (PID=$PID_5)"

python scripts/run_experiment.py \
    --config configs/exp_onetrans_long.yaml \
    --run-name onetrans_long \
    --device cuda:6 \
    $COMMON_ARGS \
    > results/onetrans_long_stdout.log 2>&1 &
PID_6=$!
echo "[GPU 6] onetrans_long (PID=$PID_6)"

echo ""
echo "All 7 jobs launched. Waiting for completion..."
echo "PIDs: $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6"

FAILED=0
for PID in $PID_0 $PID_1 $PID_2 $PID_3 $PID_4 $PID_5 $PID_6; do
    if ! wait $PID; then
        echo "FAILED: PID $PID"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=== All jobs finished ==="
echo "End time: $(date)"
if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED job(s) failed. Check stdout logs in results/"
    exit 1
fi
echo "All succeeded. Run: python scripts/generate_report.py"
