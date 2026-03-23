#!/usr/bin/env python3
"""Run a full experiment: train for N epochs with periodic eval, log everything."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from primus_dlrm.config import Config
from primus_dlrm.data.dataset import (
    YambdaEvalDataset,
    YambdaTrainDataset,
    collate_scoring_pairs,
)
from primus_dlrm.evaluation.metrics import evaluate_ranking, evaluate_ranking_peruser
from primus_dlrm.models.dlrm import DLRMBaseline
from primus_dlrm.training.losses import InBatchBPRLoss, MultiTaskLoss

def build_model(config, num_users, num_items, num_artists, num_albums, audio_input_dim, device, tasks):
    num_counter_windows = len(config.data.counter_windows_days) if config.data.enable_counters else 0
    if config.model.model_type == "onetrans":
        from primus_dlrm.models.onetrans import OneTransModel
        return OneTransModel(
            config=config.model, num_users=num_users, num_items=num_items,
            num_artists=num_artists, num_albums=num_albums,
            audio_input_dim=audio_input_dim, device=device, tasks=tasks,
            num_counter_windows=num_counter_windows,
        )
    else:
        return DLRMBaseline(
            config=config.model, num_users=num_users, num_items=num_items,
            num_artists=num_artists, num_albums=num_albums,
            audio_input_dim=audio_input_dim, device=device, tasks=tasks,
            num_counter_windows=num_counter_windows,
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run DLRM++ experiment")
    parser.add_argument("--config", default="configs/dlrm_baseline.yaml")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--eval-top-k", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--run-name", default="baseline_v1")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--eval-every-steps", type=int, default=0,
                        help="Run eval every N steps within an epoch (0 = end of epoch only)")
    parser.add_argument("--resume-epoch", type=int, default=None,
                        help="Resume from this epoch's checkpoint (skip training, run eval)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after this many training steps (0 = full run)")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.epochs is not None:
        config.train.epochs = args.epochs
    device = torch.device(args.device)
    processed_dir = Path(args.processed_dir)
    torch.manual_seed(config.train.seed)

    run_dir = Path(args.results_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    for d in (run_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Add file logging to run-specific log directory
    file_handler = logging.FileHandler(log_dir / "train.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # --- Data ---
    logger.info("Loading training dataset...")
    train_dataset = YambdaTrainDataset(config.data, processed_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_scoring_pairs,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(train_loader)

    logger.info("Loading eval dataset...")
    eval_dataset = YambdaEvalDataset(config.data, processed_dir)
    pop = eval_dataset.item_popularity
    candidate_items = np.argsort(-pop)[: args.eval_top_k]

    # --- Model ---
    logger.info("Building model...")
    num_users = int(train_dataset.store.unique_uids.max()) + 1
    active_tasks = [k for k, v in config.train.loss_weights.items() if v > 0]
    logger.info(f"Active tasks: {active_tasks}")
    model = build_model(
        config=config,
        num_users=num_users,
        num_items=train_dataset.num_items,
        num_artists=train_dataset.num_artists,
        num_albums=train_dataset.num_albums,
        audio_input_dim=train_dataset.audio_dim,
        device=device,
        tasks=active_tasks,
    )
    num_params = sum(p.numel() for p in model.parameters())
    num_emb_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "emb" in n
    )
    num_dense_params = num_params - num_emb_params
    logger.info(
        f"Model: {num_params:,} params total "
        f"({num_emb_params:,} embedding, {num_dense_params:,} dense)"
    )

    # --- Optimizer + Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    total_steps = config.train.epochs * steps_per_epoch
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=config.train.warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - config.train.warmup_steps, 1))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[config.train.warmup_steps])

    loss_fn = MultiTaskLoss(weights=config.train.loss_weights)
    use_contrastive = config.train.contrastive_weight > 0
    contrastive_loss_fn = None
    if use_contrastive:
        contrastive_loss_fn = InBatchBPRLoss(temperature=config.train.contrastive_temperature)
        logger.info(f"Contrastive loss enabled: weight={config.train.contrastive_weight}, temp={config.train.contrastive_temperature}")

    # --- Log experiment config ---
    exp_info = {
        "run_name": args.run_name,
        "config_file": args.config,
        "device": args.device,
        "num_params": num_params,
        "num_emb_params": num_emb_params,
        "num_dense_params": num_dense_params,
        "train_samples": len(train_dataset),
        "train_positions": len(train_dataset._positions),
        "eval_users": len(eval_dataset),
        "eval_top_k": args.eval_top_k,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "data_config": {
            "dataset_size": config.data.dataset_size,
            "history_length": config.data.history_length,
            "enable_counters": config.data.enable_counters,
            "counter_windows_days": config.data.counter_windows_days if config.data.enable_counters else [],
        },
        "model_config": {
            "model_type": config.model.model_type,
            "embedding_dim": config.model.embedding_dim,
            "dropout": config.model.dropout,
            **({"onetrans": {
                "d_model": config.model.onetrans.d_model,
                "n_heads": config.model.onetrans.n_heads,
                "n_layers": config.model.onetrans.n_layers,
                "n_ns_tokens": config.model.onetrans.n_ns_tokens,
                "use_pyramid": config.model.onetrans.use_pyramid,
            }} if config.model.model_type == "onetrans" else {
                "interaction_type": config.model.interaction_type,
                "bottom_mlp_dims": config.model.bottom_mlp_dims,
                "top_mlp_dims": config.model.top_mlp_dims,
            }),
        },
        "train_config": {
            "batch_size": config.train.batch_size,
            "lr": config.train.lr,
            "weight_decay": config.train.weight_decay,
            "epochs": config.train.epochs,
            "warmup_steps": config.train.warmup_steps,
            "bf16": config.train.bf16,
            "grad_clip": config.train.grad_clip,
            "loss_weights": config.train.loss_weights,
            "contrastive_weight": config.train.contrastive_weight,
            "contrastive_temperature": config.train.contrastive_temperature,
        },
    }
    logger.info(f"Experiment config: {json.dumps(exp_info, indent=2)}")

    # --- Resume or Train ---
    global_step = 0
    start_epoch = 0
    history = {"train_loss": [], "epoch_results": []}
    exp_start = time.time()

    if args.resume_epoch is not None:
        ckpt_path = ckpt_dir / f"epoch_{args.resume_epoch}.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            global_step = ckpt.get("global_step", (args.resume_epoch + 1) * steps_per_epoch)
            start_epoch = args.resume_epoch + 1
            logger.info(f"Resumed from {ckpt_path} (epoch={args.resume_epoch}, step={global_step})")
        else:
            logger.error(f"Checkpoint not found: {ckpt_path}")
            return

    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_task_losses = {}
        epoch_start = time.time()
        num_batches = 0

        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if use_contrastive:
                    preds, cross_scores = model.forward_with_cross_scores(batch, cross_task="listen_plus")
                else:
                    preds = model(batch)
                labels = {t: batch[t] for t in active_tasks}
                total_loss, task_losses = loss_fn(preds, labels)

                if use_contrastive:
                    bpr_loss = contrastive_loss_fn(cross_scores, batch["listen_plus"])
                    total_loss = total_loss + config.train.contrastive_weight * bpr_loss
                    task_losses["bpr"] = bpr_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_val = total_loss.item()
            epoch_loss += loss_val
            for k, v in task_losses.items():
                epoch_task_losses[k] = epoch_task_losses.get(k, 0.0) + (v.item() if hasattr(v, 'item') else float(v))
            num_batches += 1

            if global_step % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                throughput = num_batches * config.train.batch_size / elapsed
                logger.info(
                    f"epoch={epoch} step={global_step} | "
                    f"loss={loss_val:.4f} | lr={lr:.6f} | "
                    f"throughput={throughput:.0f} samples/s | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in task_losses.items())
                )

            if args.eval_every_steps > 0 and global_step > 0 and global_step % args.eval_every_steps == 0:
                ckpt_path = ckpt_dir / f"step_{global_step}.pt"
                torch.save({"epoch": epoch, "global_step": global_step, "model_state_dict": model.state_dict()}, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                logger.info(f"Mid-epoch eval at step {global_step}...")
                step_eval = {"step": global_step}
                results_global = evaluate_ranking(model, eval_dataset, device, candidate_item_ids=candidate_items, ks=[10, 50, 100], task="listen_plus")
                logger.info(f"  [global-{args.eval_top_k}] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(results_global.items())))
                step_eval["global"] = results_global

                results_peruser = evaluate_ranking_peruser(model, eval_dataset, device, ks=[10, 50, 100], task="listen_plus", top_n=100)
                logger.info(f"  [peruser-100] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(results_peruser.items())))
                step_eval["peruser"] = results_peruser

                history["epoch_results"].append(step_eval)
                model.train()

            global_step += 1

            if args.max_steps > 0 and global_step >= args.max_steps:
                logger.info(f"Reached --max-steps={args.max_steps}, stopping.")
                break

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches
        avg_task = {k: float(v) / num_batches for k, v in epoch_task_losses.items()}
        throughput = num_batches * config.train.batch_size / epoch_time

        logger.info(
            f"=== Epoch {epoch} done === "
            f"avg_loss={avg_loss:.4f} | time={epoch_time:.1f}s | "
            f"throughput={throughput:.0f} samples/s"
        )

        # --- Save checkpoint first ---
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")

        epoch_result = {
            "epoch": epoch,
            "avg_loss": round(float(avg_loss), 6),
            "avg_task_losses": {k: round(v, 6) for k, v in avg_task.items()},
            "epoch_time_s": round(epoch_time, 1),
            "throughput_samples_s": round(throughput, 0),
            "lr_end": optimizer.param_groups[0]["lr"],
        }

        # --- Eval ---
        logger.info(f"Running evaluation...")
        eval_results = {}

        results_global = evaluate_ranking(model, eval_dataset, device, candidate_item_ids=candidate_items, ks=[10, 50, 100], task="listen_plus")
        eval_results["global"] = {k: round(v, 6) if isinstance(v, float) else v for k, v in results_global.items()}
        logger.info(f"  [global-{args.eval_top_k}] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(results_global.items())))

        results_peruser = evaluate_ranking_peruser(model, eval_dataset, device, ks=[10, 50, 100], task="listen_plus", top_n=100)
        eval_results["peruser"] = {k: round(v, 6) if isinstance(v, float) else v for k, v in results_peruser.items()}
        logger.info(f"  [peruser-100] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(results_peruser.items())))

        epoch_result["eval"] = eval_results
        history["epoch_results"].append(epoch_result)
        history["train_loss"].append(avg_loss)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    total_time = time.time() - exp_start
    logger.info(f"Experiment complete: {config.train.epochs} epochs in {total_time:.1f}s")

    # --- Save results ---
    exp_info["total_time_s"] = round(total_time, 1)
    exp_info["history"] = history
    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(exp_info, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # --- Print summary ---
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    for er in history["epoch_results"]:
        if "epoch" not in er:
            continue
        gl = er.get("eval", {}).get("global", {})
        pu = er.get("eval", {}).get("peruser", {})
        logger.info(
            f"  Epoch {er['epoch']}: loss={er.get('avg_loss', 0):.4f} | "
            f"global: ndcg@10={gl.get('ndcg@10', 0):.4f} ndcg@100={gl.get('ndcg@100', 0):.4f} | "
            f"peruser: ndcg@10={pu.get('ndcg@10', 0):.4f} ndcg@100={pu.get('ndcg@100', 0):.4f} | "
            f"throughput={er.get('throughput_samples_s', 0):.0f} s/s"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
