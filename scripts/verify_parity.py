#!/usr/bin/env python3
"""Convergence parity verification: compare training across GPU counts.

Usage:
    # Single-node parity (Phase 5A)
    python scripts/verify_parity.py --config configs/dist_onetrans_v6.yaml \
        --gpu-counts 1,2,4,8 --max-steps 5000 --mode single-node

    # Multi-node parity (Phase 5B) -- requires slurm or manual launch
    python scripts/verify_parity.py --config configs/dist_onetrans_v6.yaml \
        --node-counts 1,2,4 --gpus-per-node 8 --max-steps 5000 --mode multi-node
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# Reference losses from baseline runs (50M dataset)
BASELINE_LOSSES = {
    "counter_v1": {
        0: 1.0633, 1000: 0.5314, 2000: 0.5248,
        3000: 0.5333, 4000: 0.5124, 5000: 0.5101,
    },
    "onetrans_v6": {
        0: 4.9864, 1000: 0.8179, 2000: 0.7623,
        3000: 0.7097, 4000: 0.6633, 5000: 0.6362,
    },
}


def parse_losses_from_log(log_path: Path) -> dict[int, float]:
    """Extract step -> loss from a train.log file."""
    losses = {}
    pattern = re.compile(r"step=(\d+)\s*\|\s*loss=([\d.]+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                losses[step] = loss
    return losses


def compare_losses(
    run_losses: dict[int, float],
    baseline_losses: dict[int, float],
    tolerance: float = 0.05,
    check_steps: list[int] | None = None,
) -> tuple[bool, list[str]]:
    """Compare run losses against baseline at specified steps."""
    if check_steps is None:
        check_steps = [1000, 3000, 5000]

    passed = True
    messages = []

    for step in check_steps:
        if step not in run_losses:
            messages.append(f"  step={step}: MISSING from run log")
            passed = False
            continue
        if step not in baseline_losses:
            messages.append(f"  step={step}: MISSING from baseline")
            continue

        run_val = run_losses[step]
        base_val = baseline_losses[step]
        rel_diff = abs(run_val - base_val) / max(abs(base_val), 1e-8)

        if rel_diff > tolerance:
            messages.append(
                f"  step={step}: FAIL -- run={run_val:.4f} vs baseline={base_val:.4f} "
                f"(diff={rel_diff:.2%} > {tolerance:.0%})"
            )
            passed = False
        else:
            messages.append(
                f"  step={step}: PASS -- run={run_val:.4f} vs baseline={base_val:.4f} "
                f"(diff={rel_diff:.2%})"
            )

    # Check for NaN/Inf
    for step, loss in run_losses.items():
        if loss != loss or abs(loss) == float("inf"):
            messages.append(f"  step={step}: FAIL -- NaN/Inf detected (loss={loss})")
            passed = False

    return passed, messages


def run_single_node_parity(
    config_path: str,
    gpu_counts: list[int],
    max_steps: int,
    results_dir: str,
    run_name_prefix: str,
) -> dict[int, dict]:
    """Run training at different GPU counts on a single node."""
    results = {}

    for n_gpus in gpu_counts:
        run_name = f"{run_name_prefix}_gpu{n_gpus}"
        run_dir = Path(results_dir) / run_name
        log_path = run_dir / "logs" / "train.log"

        logger.info(f"\n{'='*60}")
        logger.info(f"Running with {n_gpus} GPU(s)...")
        logger.info(f"{'='*60}")

        if n_gpus == 1:
            cmd = [
                sys.executable, "scripts/run_experiment.py",
                "--config", config_path,
                "--device", "cuda:0",
                "--run-name", run_name,
                "--results-dir", results_dir,
                "--max-steps", str(max_steps),
                "--log-interval", "1000",
                "--epochs", "1",
            ]
        else:
            cmd = [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "scripts/run_distributed.py",
                "--config", config_path,
                "--run-name", run_name,
                "--results-dir", results_dir,
                "--max-steps", str(max_steps),
            ]

        logger.info(f"Command: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            logger.error(f"Training failed with exit code {proc.returncode}")
            logger.error(proc.stderr[-2000:] if proc.stderr else "no stderr")
            results[n_gpus] = {"status": "failed", "returncode": proc.returncode}
            continue

        if log_path.exists():
            losses = parse_losses_from_log(log_path)
            results[n_gpus] = {"status": "completed", "losses": losses}
            logger.info(f"  Extracted {len(losses)} loss checkpoints")
        else:
            results[n_gpus] = {"status": "completed", "losses": {}}
            logger.info("  No log file found")

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify convergence parity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-counts", default="1,2,8",
                        help="Comma-separated GPU counts for single-node")
    parser.add_argument("--max-steps", type=int, default=5001)
    parser.add_argument("--results-dir", default="results/parity")
    parser.add_argument("--run-name-prefix", default="parity")
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--baseline", default=None,
                        help="Baseline name (counter_v1 or onetrans_v6)")
    parser.add_argument("--compare-log", default=None,
                        help="Path to baseline log to compare against")
    args = parser.parse_args()

    gpu_counts = [int(x) for x in args.gpu_counts.split(",")]

    results = run_single_node_parity(
        config_path=args.config,
        gpu_counts=gpu_counts,
        max_steps=args.max_steps,
        results_dir=args.results_dir,
        run_name_prefix=args.run_name_prefix,
    )

    # Get baseline losses
    baseline_losses = {}
    if args.baseline and args.baseline in BASELINE_LOSSES:
        baseline_losses = BASELINE_LOSSES[args.baseline]
    elif args.compare_log:
        baseline_losses = parse_losses_from_log(Path(args.compare_log))

    # Compare results
    logger.info(f"\n{'='*60}")
    logger.info("PARITY VERIFICATION RESULTS")
    logger.info(f"{'='*60}")

    all_passed = True

    # Compare each run against baseline (if available)
    if baseline_losses:
        for n_gpus, res in sorted(results.items()):
            if res["status"] != "completed":
                logger.info(f"\n{n_gpus} GPU(s): FAILED to run")
                all_passed = False
                continue

            passed, messages = compare_losses(
                res.get("losses", {}), baseline_losses, args.tolerance,
            )
            status = "PASS" if passed else "FAIL"
            logger.info(f"\n{n_gpus} GPU(s) vs baseline: {status}")
            for msg in messages:
                logger.info(msg)
            if not passed:
                all_passed = False

    # Cross-compare between GPU counts
    gpu_results = {k: v for k, v in results.items() if v["status"] == "completed"}
    if len(gpu_results) > 1:
        ref_gpus = min(gpu_results.keys())
        ref_losses = gpu_results[ref_gpus].get("losses", {})

        for n_gpus, res in sorted(gpu_results.items()):
            if n_gpus == ref_gpus:
                continue
            passed, messages = compare_losses(
                res.get("losses", {}), ref_losses, args.tolerance,
            )
            status = "PASS" if passed else "FAIL"
            logger.info(f"\n{n_gpus} GPU(s) vs {ref_gpus} GPU(s): {status}")
            for msg in messages:
                logger.info(msg)
            if not passed:
                all_passed = False

    logger.info(f"\n{'='*60}")
    logger.info(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    logger.info(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
