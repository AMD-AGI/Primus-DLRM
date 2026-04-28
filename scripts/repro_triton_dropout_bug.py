"""Minimal reproducer for the aiter Triton FA2 dropout assertion bug.

Bug: When ``flash_attn_func(q, k, v, dropout_p>0, causal=True)`` is called via
the aiter Triton backend (FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE), the call
fails with::

    AssertionError: [fwd] return_softmax=False but sd_mask is not None

at ``aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py:203``.

Root cause: the Triton ``fwd`` interface allocates ``sd_mask`` whenever
``dropout_p > 0`` OR ``return_softmax=True``, but its post-call assertion
expects ``sd_mask is None`` whenever ``return_softmax=False``. The flash_attn
wrapper passes ``return_softmax=False`` always, so any ``dropout_p > 0`` trips
the assertion.

The CK backend (FLASH_ATTENTION_TRITON_AMD_ENABLE unset/FALSE) is unaffected.

Run with::

    # Reproduces the assertion error
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE python3 repro_triton_dropout_bug.py

    # Same call succeeds with the CK backend
    FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE python3 repro_triton_dropout_bug.py
"""
from __future__ import annotations

import os
import sys

import torch
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import USE_TRITON_ROCM


def main() -> int:
    print(f"FLASH_ATTENTION_TRITON_AMD_ENABLE={os.environ.get('FLASH_ATTENTION_TRITON_AMD_ENABLE', 'unset')}")
    print(f"USE_TRITON_ROCM={USE_TRITON_ROCM}")
    print(f"torch={torch.__version__}, hip={torch.version.hip}")
    print(f"device={torch.cuda.get_device_name(0)}")

    # Small OneTrans-Large-ish layer (BS=4 to keep it fast)
    B, S, H, D = 4, 256, 6, 64
    dtype = torch.bfloat16

    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=True)

    print(f"\nQKV shape (BSHD): {tuple(q.shape)} dtype={dtype}")

    # 1. dropout_p = 0  -- should always work
    print("\n[1] dropout_p=0.0, causal=True:")
    out0 = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    print(f"    OK; out shape={tuple(out0.shape)}")

    # 2. dropout_p > 0  -- breaks on Triton, works on CK
    print("\n[2] dropout_p=0.1, causal=True (the suspected bug):")
    try:
        out1 = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
        print(f"    OK; out shape={tuple(out1.shape)}")
        return 0
    except AssertionError as e:
        print(f"    AssertionError: {e}")
        print("    (this is the Triton AMD dropout bug)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
