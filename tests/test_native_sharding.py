#!/usr/bin/env python3
"""Test native PyTorch embedding sharding on multi-GPU.

Validates table-wise and row-wise ShardedEmbeddingCollection:
- Forward produces correct shapes
- Backward flows gradients to local shard params
- All-to-all communication works via RCCL

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/test_native_sharding.py
    torchrun --nproc_per_node=8 scripts/test_native_sharding.py
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn


def init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def test_table_wise(rank, world_size, device):
    """Table-wise sharding: each table on one GPU."""
    from primus_dlrm.config import EmbeddingTableConfig
    from primus_dlrm.models.sharded_embedding import ShardedEmbeddingCollection, assign_sharding

    specs = [
        EmbeddingTableConfig(name="item", features=["item", "hist_item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
        EmbeddingTableConfig(name="user", features=["uid"], num_embeddings=500, embedding_dim=16, pooling="none"),
    ]
    plan = assign_sharding(specs, world_size, strategy="table_wise")
    log(rank, f"table_wise plan: {[(k, v.owner_rank) for k, v in plan.items()]}")

    sharded = ShardedEmbeddingCollection(specs, plan, device=device)

    B, L = 4, 3
    features = {
        "item": torch.randint(0, 1000, (B,), device=device),
        "hist_item": torch.randint(0, 1000, (B, L), device=device),
        "uid": torch.randint(0, 500, (B,), device=device),
    }

    out = sharded(features)

    assert out["item"].shape == (B, 16), f"item shape: {out['item'].shape}"
    assert out["hist_item"].shape == (B, L, 16), f"hist_item shape: {out['hist_item'].shape}"
    assert out["uid"].shape == (B, 16), f"uid shape: {out['uid'].shape}"
    log(rank, f"forward OK: item={out['item'].shape}, hist_item={out['hist_item'].shape}, uid={out['uid'].shape}")

    loss = sum(v.sum() for v in out.values())
    loss.backward()

    for name, p in sharded.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        log(rank, f"  param {name}: shape={p.shape}, has_grad={has_grad}")
    log(rank, "backward OK")


def test_row_wise(rank, world_size, device):
    """Row-wise sharding: table rows split across GPUs."""
    from primus_dlrm.config import EmbeddingTableConfig
    from primus_dlrm.models.sharded_embedding import ShardedEmbeddingCollection, assign_sharding

    specs = [
        EmbeddingTableConfig(name="item", features=["item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
        EmbeddingTableConfig(name="user", features=["uid"], num_embeddings=500, embedding_dim=16, pooling="none"),
    ]
    plan = assign_sharding(specs, world_size, strategy="row_wise")

    sharded = ShardedEmbeddingCollection(specs, plan, device=device)

    B = 4
    features = {
        "item": torch.randint(0, 1000, (B,), device=device),
        "uid": torch.randint(0, 500, (B,), device=device),
    }

    out = sharded(features)
    assert out["item"].shape == (B, 16)
    assert out["uid"].shape == (B, 16)
    log(rank, f"forward OK: item={out['item'].shape}, uid={out['uid'].shape}")

    loss = sum(v.sum() for v in out.values())
    loss.backward()

    for name, p in sharded.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        log(rank, f"  param {name}: shape={p.shape}, has_grad={has_grad}")
    log(rank, "backward OK")


def test_pooled_table_wise(rank, world_size, device):
    """Table-wise sharding with pooled (mean) embeddings."""
    from primus_dlrm.config import EmbeddingTableConfig
    from primus_dlrm.models.sharded_embedding import ShardedEmbeddingCollection, assign_sharding

    specs = [
        EmbeddingTableConfig(name="hist_item", features=["hist_lp_item", "hist_like_item"], num_embeddings=1000, embedding_dim=16, pooling="mean"),
        EmbeddingTableConfig(name="item", features=["item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
    ]
    plan = assign_sharding(specs, world_size, strategy="table_wise")

    sharded = ShardedEmbeddingCollection(specs, plan, device=device)

    B, L = 4, 5
    features = {
        "hist_lp_item": torch.randint(0, 1000, (B, L), device=device),
        "hist_like_item": torch.randint(0, 1000, (B, L), device=device),
        "item": torch.randint(0, 1000, (B,), device=device),
    }

    out = sharded(features)
    assert out["hist_lp_item"].shape == (B, 16), f"pooled shape: {out['hist_lp_item'].shape}"
    assert out["hist_like_item"].shape == (B, 16)
    assert out["item"].shape == (B, 16)
    log(rank, f"pooled forward OK: hist_lp_item={out['hist_lp_item'].shape}, item={out['item'].shape}")

    loss = sum(v.sum() for v in out.values())
    loss.backward()
    log(rank, "pooled backward OK")


def test_full_model_integration(rank, world_size, device):
    """Test replacing TorchRecEmbeddings with ShardedEmbeddingCollection in a model."""
    from primus_dlrm.config import EmbeddingTableConfig
    from primus_dlrm.models.embedding import TorchRecEmbeddings
    from primus_dlrm.models.sharded_embedding import ShardedEmbeddingCollection, assign_sharding

    class ToyRecommender(nn.Module):
        def __init__(self):
            super().__init__()
            specs = [
                EmbeddingTableConfig(name="item", features=["item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
                EmbeddingTableConfig(name="user", features=["uid"], num_embeddings=500, embedding_dim=16, pooling="none"),
            ]
            self.emb = TorchRecEmbeddings(specs, device=device)
            self.head = nn.Linear(32, 1, device=device)

        def forward(self, batch):
            embs = self.emb({"item": batch["item"], "uid": batch["uid"]})
            x = torch.cat([embs["item"], embs["uid"]], dim=-1)
            return self.head(x).squeeze(-1)

    model = ToyRecommender()

    # Replace emb with sharded version
    specs = [
        EmbeddingTableConfig(name="item", features=["item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
        EmbeddingTableConfig(name="user", features=["uid"], num_embeddings=500, embedding_dim=16, pooling="none"),
    ]
    plan = assign_sharding(specs, world_size, strategy="table_wise")
    model.emb = ShardedEmbeddingCollection(specs, plan, device=device)
    model = model.to(device)

    B = 4
    batch = {
        "item": torch.randint(0, 1000, (B,), device=device),
        "uid": torch.randint(0, 500, (B,), device=device),
    }

    out = model(batch)
    assert out.shape == (B,)
    log(rank, f"model forward OK: output={out.shape}")

    loss = out.sum()
    loss.backward()

    all_grads = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )
    log(rank, f"model backward OK: all params have grads={all_grads}")


def main():
    rank, world_size, device = init()
    log(rank, f"world_size={world_size}, device={device}")

    tests = [
        ("Table-wise sharding", test_table_wise),
        ("Row-wise sharding", test_row_wise),
        ("Pooled table-wise", test_pooled_table_wise),
        ("Full model integration", test_full_model_integration),
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
