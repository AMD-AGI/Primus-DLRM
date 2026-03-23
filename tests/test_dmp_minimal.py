#!/usr/bin/env python3
"""Phase 0.2–0.5: Validate TorchRec DMP on multi-GPU with ROCm.

Tests RCCL all-to-all, then DMP with DATA_PARALLEL, TABLE_WISE, and ROW_WISE
sharding — escalating complexity to isolate failures.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 scripts/test_dmp_minimal.py
    torchrun --nproc_per_node=8 scripts/test_dmp_minimal.py
"""
import os
import sys
import traceback

import torch
import torch.distributed as dist
from torchrec import KeyedJaggedTensor


def init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, dist.get_world_size(), device


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


# ---- Test 0: RCCL all-to-all ----

def test_rccl_all_to_all(rank, world_size, device):
    """Verify dist.all_to_all_single works on RCCL."""
    send = torch.arange(world_size * 4, device=device, dtype=torch.float32).reshape(world_size, 4)
    send = send + rank * 100
    recv = torch.zeros_like(send)
    dist.all_to_all_single(recv, send)
    log(rank, f"all_to_all: sent {send.shape}, received {recv.shape}, "
              f"recv[0]={recv[0].tolist()}")


# ---- Test 1: DMP with DATA_PARALLEL ----

def _make_ebc_kjt(keys, B, L, max_id, device):
    """Build a valid KJT where sum(lengths) == len(values)."""
    n_keys = len(keys)
    lengths = torch.full((B * n_keys,), L, dtype=torch.int32, device=device)
    values = torch.randint(0, max_id, (B * n_keys * L,), device=device)
    return KeyedJaggedTensor.from_lengths_sync(keys=keys, values=values, lengths=lengths)


def test_dmp_data_parallel(rank, world_size, device):
    """DMP with DATA_PARALLEL = replicated tables, no all-to-all needed."""
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.types import ShardingType

    ebc = EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(name="t0", embedding_dim=16,
                               num_embeddings=1000, feature_names=["f0"]),
            EmbeddingBagConfig(name="t1", embedding_dim=16,
                               num_embeddings=500, feature_names=["f1"]),
        ],
        device=torch.device("meta"),
    )

    constraints = {
        "t0": ParameterConstraints(sharding_types=[ShardingType.DATA_PARALLEL.value]),
        "t1": ParameterConstraints(sharding_types=[ShardingType.DATA_PARALLEL.value]),
    }
    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)

    plan = planner.collective_plan(ebc, [EmbeddingBagCollectionSharder()], dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=ebc, device=device,
        sharders=[EmbeddingBagCollectionSharder()],
        plan=plan,
    )

    kjt = _make_ebc_kjt(["f0", "f1"], B=4, L=5, max_id=100, device=device)

    out = model(kjt)
    log(rank, f"DATA_PARALLEL: f0={out['f0'].shape}, f1={out['f1'].shape}")

    loss = out["f0"].sum() + out["f1"].sum()
    loss.backward()
    log(rank, "DATA_PARALLEL: backward OK")


# ---- Test 2: DMP with TABLE_WISE ----

def test_dmp_table_wise(rank, world_size, device):
    """DMP with TABLE_WISE = each table on one GPU."""
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.types import ShardingType

    ebc = EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(name="t0", embedding_dim=16,
                               num_embeddings=1000, feature_names=["f0"]),
            EmbeddingBagConfig(name="t1", embedding_dim=16,
                               num_embeddings=500, feature_names=["f1"]),
        ],
        device=torch.device("meta"),
    )

    constraints = {
        "t0": ParameterConstraints(sharding_types=[ShardingType.TABLE_WISE.value]),
        "t1": ParameterConstraints(sharding_types=[ShardingType.TABLE_WISE.value]),
    }
    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)

    plan = planner.collective_plan(ebc, [EmbeddingBagCollectionSharder()], dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=ebc, device=device,
        sharders=[EmbeddingBagCollectionSharder()],
        plan=plan,
    )

    kjt = _make_ebc_kjt(["f0", "f1"], B=4, L=5, max_id=100, device=device)

    out = model(kjt)
    log(rank, f"TABLE_WISE: f0={out['f0'].shape}, f1={out['f1'].shape}")

    loss = out["f0"].sum() + out["f1"].sum()
    loss.backward()
    log(rank, "TABLE_WISE: backward OK")


# ---- Test 3: DMP with ROW_WISE ----

def test_dmp_row_wise(rank, world_size, device):
    """DMP with ROW_WISE = table rows split across GPUs."""
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.types import ShardingType

    ebc = EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(name="t0", embedding_dim=16,
                               num_embeddings=1000, feature_names=["f0"]),
            EmbeddingBagConfig(name="t1", embedding_dim=16,
                               num_embeddings=500, feature_names=["f1"]),
        ],
        device=torch.device("meta"),
    )

    constraints = {
        "t0": ParameterConstraints(sharding_types=[ShardingType.ROW_WISE.value]),
        "t1": ParameterConstraints(sharding_types=[ShardingType.ROW_WISE.value]),
    }
    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)

    plan = planner.collective_plan(ebc, [EmbeddingBagCollectionSharder()], dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=ebc, device=device,
        sharders=[EmbeddingBagCollectionSharder()],
        plan=plan,
    )

    kjt = _make_ebc_kjt(["f0", "f1"], B=4, L=5, max_id=100, device=device)

    out = model(kjt)
    log(rank, f"ROW_WISE: f0={out['f0'].shape}, f1={out['f1'].shape}")

    loss = out["f0"].sum() + out["f1"].sum()
    loss.backward()
    log(rank, "ROW_WISE: backward OK")


# ---- Test 4: DMP with EmbeddingCollection (unpooled) ----

def test_dmp_ec_table_wise(rank, world_size, device):
    """DMP with EmbeddingCollection (unpooled) + TABLE_WISE."""
    from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.embedding import EmbeddingCollectionSharder
    from torchrec.distributed.types import ShardingType

    ec = EmbeddingCollection(
        tables=[
            EmbeddingConfig(name="t0", embedding_dim=16,
                            num_embeddings=1000, feature_names=["f0"]),
            EmbeddingConfig(name="t1", embedding_dim=16,
                            num_embeddings=500, feature_names=["f1"]),
        ],
        device=torch.device("meta"),
    )

    constraints = {
        "t0": ParameterConstraints(sharding_types=[ShardingType.TABLE_WISE.value]),
        "t1": ParameterConstraints(sharding_types=[ShardingType.TABLE_WISE.value]),
    }
    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology, constraints=constraints)

    plan = planner.collective_plan(ec, [EmbeddingCollectionSharder()], dist.GroupMember.WORLD)

    model = DistributedModelParallel(
        module=ec, device=device,
        sharders=[EmbeddingCollectionSharder()],
        plan=plan,
    )

    B, L = 4, 3
    n_keys = 2
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["f0", "f1"],
        values=torch.randint(0, 100, (B * n_keys * L,), device=device),
        lengths=torch.full((B * n_keys,), L, dtype=torch.int32, device=device),
    )

    out = model(kjt)
    f0_vals = out["f0"].values()
    f1_vals = out["f1"].values()
    log(rank, f"EC TABLE_WISE: f0 values={f0_vals.shape}, f1 values={f1_vals.shape}")

    loss = f0_vals.sum() + f1_vals.sum()
    loss.backward()
    log(rank, "EC TABLE_WISE: backward OK")


# ---- Test 5: DMP wrapping a full nn.Module (not just EBC) ----

def test_dmp_full_model(rank, world_size, device):
    """DMP wrapping a model that has EBC + dense layers (like our real models)."""
    import torch.nn as nn
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
    from torchrec.distributed import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ebc = EmbeddingBagCollection(
                tables=[
                    EmbeddingBagConfig(name="t0", embedding_dim=16,
                                       num_embeddings=1000, feature_names=["f0"]),
                ],
                device=torch.device("meta"),
            )
            self.dense = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, kjt):
            emb_out = self.ebc(kjt)
            return self.dense(emb_out["f0"])

    model = ToyModel()
    topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(topology=topology)

    plan = planner.collective_plan(model, [EmbeddingBagCollectionSharder()], dist.GroupMember.WORLD)

    dmp = DistributedModelParallel(
        module=model, device=device,
        sharders=[EmbeddingBagCollectionSharder()],
        plan=plan,
    )

    kjt = _make_ebc_kjt(["f0"], B=4, L=5, max_id=100, device=device)

    out = dmp(kjt)
    log(rank, f"Full model: output={out.shape}")

    loss = out.sum()
    loss.backward()
    log(rank, "Full model: backward OK")

    # Verify DMP created a fused optimizer
    fused_opt = dmp.fused_optimizer
    log(rank, f"Full model: fused_optimizer type={type(fused_opt).__name__}")


def main():
    rank, world_size, device = init()
    log(rank, f"world_size={world_size}, device={device}, "
              f"GPU={torch.cuda.get_device_name(device)}")

    tests = [
        ("RCCL all-to-all", test_rccl_all_to_all),
        ("DMP DATA_PARALLEL (EBC)", test_dmp_data_parallel),
        ("DMP TABLE_WISE (EBC)", test_dmp_table_wise),
        ("DMP ROW_WISE (EBC)", test_dmp_row_wise),
        ("DMP TABLE_WISE (EC, unpooled)", test_dmp_ec_table_wise),
        ("DMP full model (EBC + dense)", test_dmp_full_model),
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
