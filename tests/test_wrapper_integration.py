#!/usr/bin/env python3
"""Test the full wrap_model integration with dmp mode on multi-GPU.

This tests the complete path: model creation -> wrap_model(dmp) ->
forward -> backward -> optimizer step.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/test_wrapper_integration.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    from primus_dlrm.config import EmbeddingTableConfig
    from primus_dlrm.models.embedding import TorchRecEmbeddings
    from primus_dlrm.distributed.wrapper import wrap_model

    class ToyDLRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = TorchRecEmbeddings([
                EmbeddingTableConfig(name="hist_item", features=["hist_lp_item", "hist_like_item"], num_embeddings=1000, embedding_dim=16, pooling="mean"),
                EmbeddingTableConfig(name="item", features=["item"], num_embeddings=1000, embedding_dim=16, pooling="none"),
                EmbeddingTableConfig(name="user", features=["uid"], num_embeddings=500, embedding_dim=16, pooling="none"),
            ], device=device)
            self.head = nn.Sequential(
                nn.Linear(48, 32, device=device),
                nn.ReLU(),
                nn.Linear(32, 1, device=device),
            )

        def forward(self, batch):
            embs = self.emb({
                "hist_lp_item": batch["hist_lp_item"],
                "hist_like_item": batch["hist_like_item"],
                "item": batch["item"],
                "uid": batch["uid"],
            })
            x = torch.cat([
                embs["hist_lp_item"],
                embs["item"],
                embs["uid"],
            ], dim=-1)
            return self.head(x).squeeze(-1)

    model = ToyDLRM()
    print(f"[rank {rank}] Model created", flush=True)

    for strategy in ["table_wise", "row_wise", "auto"]:
        dist.barrier()
        if rank == 0:
            print(f"\n[TEST] wrap_model dmp, strategy={strategy}", flush=True)

        test_model = ToyDLRM()
        from primus_dlrm.config import Config
        dummy_config = Config()
        dummy_config.distributed.dense_strategy = "dmp"
        dummy_config.distributed.embedding_sharding.strategy = strategy
        wrapped = wrap_model(test_model, device, dummy_config)

        B, L = 4, 5
        batch = {
            "hist_lp_item": torch.randint(0, 1000, (B, L), device=device),
            "hist_like_item": torch.randint(0, 1000, (B, L), device=device),
            "item": torch.randint(0, 1000, (B,), device=device),
            "uid": torch.randint(0, 500, (B,), device=device),
        }

        out = wrapped(batch)
        print(f"[rank {rank}] forward OK: {out.shape}", flush=True)

        loss = out.sum()
        loss.backward()
        print(f"[rank {rank}] backward OK", flush=True)

        opt = AdamW(wrapped.parameters(), lr=0.001)
        opt.step()
        print(f"[rank {rank}] optimizer step OK", flush=True)

        dist.barrier()
        if rank == 0:
            print(f"  PASSED\n", flush=True)

    dist.barrier()
    if rank == 0:
        print("All integration tests passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
