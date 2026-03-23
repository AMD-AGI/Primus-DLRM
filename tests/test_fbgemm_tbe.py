#!/usr/bin/env python3
"""Phase 0.1: Validate FBGEMM GPU TBE kernels work on ROCm/MI355X.

Single-GPU test — no distributed setup needed.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_fbgemm_tbe.py
"""
import sys
import torch


def test_tbe_training():
    """Test SplitTableBatchedEmbeddingBagsCodegen (training variant)."""
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        ComputeDevice,
        EmbeddingLocation,
        SplitTableBatchedEmbeddingBagsCodegen,
    )

    device = torch.device("cuda:0")
    T = 2
    op = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (1000, 16, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
            (500, 32, EmbeddingLocation.DEVICE, ComputeDevice.CUDA),
        ],
        device=device,
    )

    B, L = 8, 5
    indices = torch.randint(0, 500, (B * T * L,), device=device, dtype=torch.long)
    offsets = torch.arange(0, B * T * L + 1, L, dtype=torch.long, device=device)

    out = op(indices, offsets)
    print(f"  TBE forward: output shape={out.shape}, dtype={out.dtype}")
    assert out.shape[0] == B and out.shape[1] == 16 + 32, f"Unexpected shape: {out.shape}"

    loss = out.sum()
    loss.backward()
    print("  TBE backward: OK")


def test_dense_tbe():
    """Test DenseTableBatchedEmbeddingBagsCodegen (non-fused, no optimizer)."""
    from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
        DenseTableBatchedEmbeddingBagsCodegen,
    )

    device = torch.device("cuda:0")
    T = 2
    op = DenseTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[
            (1000, 16),
            (500, 32),
        ],
        use_cpu=False,
    )
    op = op.to(device)

    B, L = 8, 5
    indices = torch.randint(0, 500, (B * T * L,), device=device, dtype=torch.long)
    offsets = torch.arange(0, B * T * L + 1, L, dtype=torch.long, device=device)

    out = op(indices, offsets)
    print(f"  Dense TBE forward: output shape={out.shape}, dtype={out.dtype}")
    assert out.shape[0] == B and out.shape[1] == 16 + 32, f"Unexpected shape: {out.shape}"

    loss = out.sum()
    loss.backward()
    print("  Dense TBE backward: OK")


def test_torchrec_ebc_single_gpu():
    """Test TorchRec EmbeddingBagCollection on single GPU (no DMP)."""
    from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor

    device = torch.device("cuda:0")
    ebc = EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(name="t0", embedding_dim=16,
                               num_embeddings=1000, feature_names=["f0"]),
            EmbeddingBagConfig(name="t1", embedding_dim=32,
                               num_embeddings=500, feature_names=["f1"]),
        ],
        device=device,
    )

    B, L = 4, 5
    lengths = torch.full((B * 2,), L, dtype=torch.int32, device=device)
    values = torch.randint(0, 100, (B * 2 * L,), device=device)
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["f0", "f1"],
        values=values,
        lengths=lengths,
    )

    out = ebc(kjt)
    f0 = out["f0"]
    f1 = out["f1"]
    print(f"  EBC forward: f0={f0.shape}, f1={f1.shape}")
    assert f0.shape == (B, 16)
    assert f1.shape == (B, 32)

    loss = f0.sum() + f1.sum()
    loss.backward()
    print("  EBC backward: OK")


def test_torchrec_ec_single_gpu():
    """Test TorchRec EmbeddingCollection on single GPU (no DMP)."""
    from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor

    device = torch.device("cuda:0")
    ec = EmbeddingCollection(
        tables=[
            EmbeddingConfig(name="t0", embedding_dim=16,
                            num_embeddings=1000, feature_names=["f0"]),
        ],
        device=device,
    )

    B, L = 4, 3
    kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["f0"],
        values=torch.randint(0, 100, (B * L,), device=device),
        lengths=torch.full((B,), L, dtype=torch.int32, device=device),
    )

    out = ec(kjt)
    vals = out["f0"].values()
    print(f"  EC forward: values shape={vals.shape}")
    assert vals.shape == (B * L, 16)

    loss = vals.sum()
    loss.backward()
    print("  EC backward: OK")


def main():
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    import fbgemm_gpu
    print(f"FBGEMM GPU: {fbgemm_gpu.__version__}")
    import torchrec
    print(f"TorchRec: {torchrec.__version__}")
    print()

    tests = [
        ("FBGEMM Dense TBE (no fused opt)", test_dense_tbe),
        ("FBGEMM Fused TBE (training)", test_tbe_training),
        ("TorchRec EBC (single GPU)", test_torchrec_ebc_single_gpu),
        ("TorchRec EC (single GPU)", test_torchrec_ec_single_gpu),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        print(f"[TEST] {name}")
        try:
            fn()
            print(f"  PASSED\n")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}\n")
            failed += 1

    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
