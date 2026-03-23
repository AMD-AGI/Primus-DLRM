#!/usr/bin/env python3
"""Integration test: TorchRec DMP wrapping DLRMBaseline with real EBC+EC.

Tests forward, backward, and optimizer step with DMP for various sharding
strategies. Validates that the wrap_model() + DistributedTrainer path works.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/test_dmp_integration.py
    torchrun --nproc_per_node=8 scripts/test_dmp_integration.py
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist

from primus_dlrm.config import ModelConfig
from primus_dlrm.distributed.wrapper import wrap_model, is_dmp
from primus_dlrm.models.dlrm import DLRMBaseline


def init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def _build_model(device):
    """Build a small DLRMBaseline with meta device for embeddings."""
    config = ModelConfig(
        model_type="dlrm",
        embedding_dim=16,
        bottom_mlp_dims=[64],
        top_mlp_dims=[64, 32],
        interaction_type="concat_mlp",
        dropout=0.0,
    )
    return DLRMBaseline(
        config=config,
        num_users=100, num_items=200, num_artists=50, num_albums=80,
        audio_input_dim=32, device=torch.device("cpu"),
        tasks=["listen_plus"], meta_device=True,
    )


def _make_batch(B, device):
    """Create a synthetic training batch matching DLRMBaseline._lookup_all."""
    L = 5
    batch = {
        "uid": torch.randint(0, 100, (B,), device=device),
        "item_id": torch.randint(1, 200, (B,), device=device),
        "artist_id": torch.randint(1, 50, (B,), device=device),
        "album_id": torch.randint(1, 80, (B,), device=device),
        "audio_embed": torch.randn(B, 32, device=device),
        "listen_plus": torch.randint(0, 2, (B,), device=device).float(),
    }
    for prefix in ["hist_lp", "hist_like", "hist_skip"]:
        for entity in ["item", "artist", "album"]:
            key = f"{prefix}_{entity}_ids"
            if entity == "item":
                max_id = 200
            elif entity == "artist":
                max_id = 50
            else:
                max_id = 80
            batch[key] = torch.randint(0, max_id, (B, L), device=device)
    return batch


def test_dmp_dlrm(rank, world_size, device, sharding_strategy):
    """Test DLRMBaseline wrapped with DMP for a given sharding strategy."""
    model = _build_model(device)
    wrapped = wrap_model(model, device, dense_strategy="dmp",
                         embedding_sharding=sharding_strategy)

    assert is_dmp(wrapped), "Model should be DMP-wrapped"

    batch = _make_batch(B=8, device=device)
    preds = wrapped(batch)

    assert "listen_plus" in preds, f"Missing prediction key, got {list(preds.keys())}"
    assert preds["listen_plus"].shape == (8,), f"Wrong shape: {preds['listen_plus'].shape}"
    log(rank, f"{sharding_strategy}: forward OK, pred shape={preds['listen_plus'].shape}")

    loss = preds["listen_plus"].sum()
    loss.backward()
    log(rank, f"{sharding_strategy}: backward OK")

    fused_opt = wrapped.fused_optimizer
    log(rank, f"{sharding_strategy}: fused_optimizer={type(fused_opt).__name__}")


def test_dmp_optimizer_step(rank, world_size, device):
    """Full train step: forward + backward + optimizer step with DMP."""
    from torch.optim import AdamW

    model = _build_model(device)
    wrapped = wrap_model(model, device, dense_strategy="dmp",
                         embedding_sharding="auto")

    dense_params = [
        p for n, p in wrapped.named_parameters()
        if p.requires_grad and "ebc" not in n and "ec" not in n and "embedding" not in n
    ]
    optimizer = AdamW(dense_params, lr=1e-3)

    batch = _make_batch(B=8, device=device)

    optimizer.zero_grad()
    preds = wrapped(batch)
    loss = preds["listen_plus"].sum()
    loss.backward()
    optimizer.step()

    log(rank, f"optimizer step: loss={loss.item():.4f}")

    # Run a second step to verify stability
    optimizer.zero_grad()
    preds2 = wrapped(batch)
    loss2 = preds2["listen_plus"].sum()
    loss2.backward()
    optimizer.step()

    log(rank, f"second step: loss={loss2.item():.4f}, no crash")


def main():
    rank, world_size, device = init()
    log(rank, f"world_size={world_size}, device={device}, "
              f"GPU={torch.cuda.get_device_name(device)}")

    tests = [
        ("DMP DLRM auto", lambda r, w, d: test_dmp_dlrm(r, w, d, "auto")),
        ("DMP DLRM table_wise", lambda r, w, d: test_dmp_dlrm(r, w, d, "table_wise")),
        ("DMP DLRM row_wise", lambda r, w, d: test_dmp_dlrm(r, w, d, "row_wise")),
        ("DMP DLRM data_parallel", lambda r, w, d: test_dmp_dlrm(r, w, d, "data_parallel")),
        ("DMP optimizer step", test_dmp_optimizer_step),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        dist.barrier()
        if rank == 0:
            print(f"\n[TEST] {name}", flush=True)
        dist.barrier()
        try:
            fn(rank, world_size, device)
            passed += 1
            result = "PASSED"
        except Exception as e:
            failed += 1
            result = f"FAILED: {e}"
            if rank == 0:
                traceback.print_exc()
        dist.barrier()
        if rank == 0:
            print(f"  {result}\n", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
        print(f"{'='*50}")

    dist.destroy_process_group()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
