"""OneTrans: unified Transformer for sequence modeling + feature interaction.

Uses split-history pools (listen+, like, skip) with per-behavior MLP tokenizers.
Supports optional counter features and in-batch contrastive loss.
All categorical embeddings backed by TorchRec EmbeddingCollection.
"""
from __future__ import annotations

import logging
import math

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from torch import Tensor

from primus_dlrm.config import Config
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TorchRecEmbeddings

try:
    from primus_turbo.pytorch.ops import flash_attn_func as _turbo_flash_attn
    _HAS_TURBO_ATTN = True
except ImportError:
    _HAS_TURBO_ATTN = False

try:
    from flash_attn import flash_attn_func as _flash_attn
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False

try:
    from flash_attn.cute import flash_attn_func as _flash_attn_4_raw
    import torch._dynamo
    @torch._dynamo.disable
    def _flash_attn_4(q, k, v, causal=True):
        return _flash_attn_4_raw(q, k, v, causal=causal)[0]
    _HAS_FLASH_ATTN_4 = True
except ImportError:
    _HAS_FLASH_ATTN_4 = False



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

@torch.fx.wrap
def pyramid_schedule(l_s: int, l_ns: int, n_layers: int) -> list[int]:
    """Number of S-token queries at each layer, linearly shrinking."""
    if n_layers == 1:
        return [l_ns]
    schedule = []
    for l in range(n_layers):
        q = l_s - l * (l_s - l_ns) / (n_layers - 1)
        schedule.append(max(int(round(q)), l_ns))
    return schedule


def build_pyramid_mask(q_len: int, kv_len: int, device: torch.device) -> Tensor:
    """Causal mask [q_len, kv_len] accounting for query offset."""
    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1) + offset
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= q_idx


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class MixedAttention(nn.Module):
    """Multi-head attention with shared S-token projections and
    batched token-specific NS-token projections."""

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 dropout: float = 0.0, attention_impl: str = "sdpa"):
        super().__init__()
        if attention_impl == "turbo" and not _HAS_TURBO_ATTN:
            raise RuntimeError(
                "attention_impl='turbo' requires primus_turbo package. "
            )
        if attention_impl in ("fav2", "fav2_varlen") and not _HAS_FLASH_ATTN:
            raise RuntimeError(
                f"attention_impl={attention_impl!r} requires flash_attn package. "
            )
        if attention_impl == "fav4" and not _HAS_FLASH_ATTN_4:
            raise RuntimeError(
                "attention_impl='fav4' requires flash-attn-4 package "
                "(pip install flash-attn-4). Optimized for Blackwell GPUs."
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_ns = n_ns_tokens
        self.attention_impl = attention_impl

        self.s_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.ns_qkv_weight = nn.Parameter(torch.empty(n_ns_tokens, 3 * d_model, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.s_qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_qkv_weight.data[i])

    def forward(
        self,
        x: Tensor,
        n_s: int,
        mask: Tensor | None = None,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        real_mask: Tensor | None = None,
        real_indices: Tensor | None = None,
    ) -> Tensor:
        B, T, _ = x.shape
        s_part = x[:, :n_s]
        ns_part = x[:, n_s:]

        s_qkv = self.s_qkv(s_part).reshape(B, n_s, 3, self.n_heads, self.d_head)
        s_q, s_k, s_v = s_qkv.unbind(2)

        ns_qkv = torch.einsum("btn,tmn->btm", ns_part, self.ns_qkv_weight)
        ns_qkv = ns_qkv.reshape(B, self.n_ns, 3, self.n_heads, self.d_head)
        ns_q, ns_k, ns_v = ns_qkv.unbind(2)

        # Q/K/V in BSHD: [B, seq, n_heads, d_head]
        q = torch.cat([s_q, ns_q], dim=1)
        k = torch.cat([s_k, ns_k], dim=1)
        v = torch.cat([s_v, ns_v], dim=1)

        if mask is not None:
            q_len = mask.shape[0]
            if q_len < T:
                q = q[:, T - q_len:]

        drop_p = self.attn_drop if self.training else 0.0

        # Jagged path: when real_indices is provided, pack via index_select
        # (no nonzero/boolean indexing inside the compiled block — graph stays
        # whole, no aten.nonzero graph break, no IndexPut backward), run
        # varlen attention, then scatter back via index_copy.
        if cu_seqlens is not None and real_indices is not None:
            q_flat = q.reshape(B * T, self.n_heads, self.d_head)
            k_flat = k.reshape(B * T, self.n_heads, self.d_head)
            v_flat = v.reshape(B * T, self.n_heads, self.d_head)
            q_packed = q_flat.index_select(0, real_indices)  # [total_real, H, D]
            k_packed = k_flat.index_select(0, real_indices)
            v_packed = v_flat.index_select(0, real_indices)
            out_packed = _flash_attn_varlen(
                q_packed, k_packed, v_packed,
                cu_seqlens, cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True, dropout_p=drop_p,
            )
            out_flat = q_flat.new_zeros(B * T, self.n_heads, self.d_head)
            out_flat.index_copy_(0, real_indices, out_packed)
            out = out_flat.view(B, T, self.n_heads, self.d_head)
        elif self.attention_impl == "turbo":
            out = _turbo_flash_attn(q, k, v, causal=True, dropout_p=drop_p)
        elif self.attention_impl == "fav4":
            out = _flash_attn_4(q, k, v, causal=True)
        elif self.attention_impl == "fav2":
            out = _flash_attn(q, k, v, causal=True, dropout_p=drop_p)
        elif self.attention_impl == "fav2_varlen":
            # Pack [B, T, H, D] -> [B*T, H, D] with cu_seqlens = [0, T, 2T, ..., B*T].
            # All sequences same length T; this is jagged/varlen attention applied
            # to a regular workload to test if the varlen kernel is better tuned
            # than the dense one for the same shapes (no FLOP savings expected).
            B_q, T_q = q.shape[0], q.shape[1]
            T_k = k.shape[1]
            q_packed = q.reshape(B_q * T_q, q.shape[2], q.shape[3])
            k_packed = k.reshape(B_q * T_k, k.shape[2], k.shape[3])
            v_packed = v.reshape(B_q * T_k, v.shape[2], v.shape[3])
            cu_seqlens_q = torch.arange(0, (B_q + 1) * T_q, T_q,
                                        device=q.device, dtype=torch.int32)
            cu_seqlens_k = torch.arange(0, (B_q + 1) * T_k, T_k,
                                        device=q.device, dtype=torch.int32)
            out_packed = _flash_attn_varlen(
                q_packed, k_packed, v_packed,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q=T_q, max_seqlen_k=T_k,
                causal=True, dropout_p=drop_p,
            )
            out = out_packed.reshape(B_q, T_q, q.shape[2], q.shape[3])
        elif self.attention_impl == "sdpa":
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            attn_mask = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask, dropout_p=drop_p,
            ).transpose(1, 2)
        else:
            raise ValueError(
                f"Unknown attention_impl: {self.attention_impl!r}. "
                f"Supported: 'sdpa', 'fav2', 'fav2_varlen', 'fav4', 'turbo'"
            )

        out = out.reshape(B, out.shape[1], self.d_model)
        return self.out_proj(out)


class MixedFFN(nn.Module):
    """FFN with shared weights for S-tokens and batched token-specific weights for NS-tokens."""

    def __init__(self, d_model: int, ffn_dim: int, n_ns_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.n_ns = n_ns_tokens
        self.d_model = d_model

        self.s_w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.s_w2 = nn.Linear(ffn_dim, d_model, bias=False)

        self.ns_w1 = nn.Parameter(torch.empty(n_ns_tokens, ffn_dim, d_model))
        self.ns_w2 = nn.Parameter(torch.empty(n_ns_tokens, d_model, ffn_dim))

        self.dropout = nn.Dropout(dropout)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_uniform_(self.s_w1.weight)
        nn.init.xavier_uniform_(self.s_w2.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_w1.data[i])
            nn.init.xavier_uniform_(self.ns_w2.data[i])

    def forward(self, x: Tensor, n_s: int) -> Tensor:
        s_part = x[:, :n_s]
        ns_part = x[:, n_s:]

        s_out = self.s_w2(self.dropout(F.gelu(self.s_w1(s_part))))
        ns_h = F.gelu(torch.einsum("btn,tmn->btm", ns_part, self.ns_w1))
        ns_out = torch.einsum("btm,tnm->btn", self.dropout(ns_h), self.ns_w2)

        return torch.cat([s_out, ns_out], dim=1)


class OneTransBlock(nn.Module):
    """Pre-norm causal Transformer block with mixed parameterization."""

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 ffn_mult: int = 4, dropout: float = 0.0,
                 attention_impl: str = "sdpa"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MixedAttention(d_model, n_heads, n_ns_tokens, dropout,
                                   attention_impl=attention_impl)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MixedFFN(d_model, d_model * ffn_mult, n_ns_tokens, dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None,
        q_len: int,
        n_s_tokens: int,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        real_mask: Tensor | None = None,
        real_indices: Tensor | None = None,
    ) -> Tensor:
        kv_len = x.shape[1]
        n_ns = kv_len - n_s_tokens

        normed = self.norm1(x)
        attn_out = self.attn(
            normed, n_s=n_s_tokens, mask=mask,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            real_mask=real_mask, real_indices=real_indices,
        )
        x_tail = x[:, kv_len - q_len:]
        z = attn_out + x_tail

        n_s_out = q_len - n_ns
        ffn_out = self.ffn(self.norm2(z), n_s=n_s_out)
        return z + ffn_out


# ---------------------------------------------------------------------------
# Phase B2: Packed (jagged) variants
#
# These run the transformer with S-tokens in PACKED form ([total_S_real, D])
# and NS-tokens in DENSE form ([B, n_ns, D]). NS stays dense because its
# Q/K/V/FFN weights are per-position (n_ns separate weight matrices) and
# index-selecting per token would be slow. The two streams are interleaved
# only at the FA-varlen boundary using precomputed gather/scatter indices.
#
# Pack/unpack indices are computed ONCE per batch in OneTransModel._backbone
# (eager mode) and threaded through all layers, so the per-block compiled
# forward never sees any data-dependent ops (no nonzero, no boolean indexing).
# ---------------------------------------------------------------------------

class MixedAttentionPacked(nn.Module):
    """Packed-input variant of MixedAttention.

    Inputs:
      s_packed:    [total_S_real, D]
      ns_dense:    [B, n_ns, D]
      cu_seqlens:  [B+1] int32 — cumulative (S_real_i + n_ns) per user
      max_seqlen:  static int upper bound (L_S + n_ns)
      s_to_packed: [total_S_real] int64 — for each S-token, its position in the
                   interleaved attention buffer (cu_seqlens[i] + within-user-pos)
      ns_to_packed:[B*n_ns] int64 — for each (i, j) NS slot, its position in the
                   attention buffer (cu_seqlens[i] + S_real_i + j)

    Outputs:
      s_packed_out: [total_S_real, D]
      ns_dense_out: [B, n_ns, D]
    """

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_ns = n_ns_tokens

        self.s_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.ns_qkv_weight = nn.Parameter(torch.empty(n_ns_tokens, 3 * d_model, d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = dropout

        nn.init.xavier_uniform_(self.s_qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_qkv_weight.data[i])

    def forward(
        self,
        s_packed: Tensor,
        ns_dense: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        s_to_packed: Tensor,
        ns_to_packed: Tensor,
        # Optional pyramid args. When provided, runs asymmetric Q (smaller,
        # drop-front gather of s_packed) vs K/V (full s_packed). When
        # absent, falls back to the symmetric self-attention path.
        cu_seqlens_q: Tensor | None = None,
        max_seqlen_q: int | None = None,
        s_q_to_packed: Tensor | None = None,
        ns_q_to_packed: Tensor | None = None,
        s_q_gather: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        B = ns_dense.shape[0]
        H, D_head = self.n_heads, self.d_head
        drop_p = self.attn_drop if self.training else 0.0
        pyramid = cu_seqlens_q is not None

        # Project (S packed, NS per-position dense)
        s_qkv = self.s_qkv(s_packed)                                        # [N_S_in, 3D]
        ns_qkv = torch.einsum("btn,tmn->btm", ns_dense, self.ns_qkv_weight)  # [B, n_ns, 3D]

        if not pyramid:
            # ---- symmetric path (current behavior) ----
            total_attn = s_qkv.shape[0] + B * self.n_ns
            attn_qkv = s_qkv.new_zeros(total_attn, 3 * self.d_model)
            attn_qkv.index_copy_(0, s_to_packed, s_qkv)
            attn_qkv.index_copy_(0, ns_to_packed, ns_qkv.reshape(B * self.n_ns, 3 * self.d_model))
            attn_qkv = attn_qkv.view(total_attn, 3, H, D_head)
            q, k, v = attn_qkv.unbind(1)
            out_attn = _flash_attn_varlen(
                q, k, v, cu_seqlens, cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                causal=True, dropout_p=drop_p,
            )
            out_flat = out_attn.reshape(total_attn, self.d_model)
            s_out_flat = out_flat.index_select(0, s_to_packed)
            ns_out_flat = out_flat.index_select(0, ns_to_packed)
            ns_out = ns_out_flat.view(B, self.n_ns, self.d_model)
            return self.out_proj(s_out_flat), self.out_proj(ns_out)

        # ---- pyramid path: asymmetric Q (smaller) vs K/V (full) ----
        # ``s_qkv`` last-dim layout is [Q | K | V] each of width D — slice as
        # views (no copy). NS analogously.
        D = self.d_model
        s_q_full = s_qkv[:, :D]                                          # [N_S_in, D] view
        s_kv     = s_qkv[:, D:]                                          # [N_S_in, 2D] view
        ns_q_full = ns_qkv[..., :D]                                      # [B, n_ns, D] view
        ns_kv     = ns_qkv[..., D:]                                      # [B, n_ns, 2D] view

        # Q side (smaller): drop-front gather then interleave with NS-Q.
        s_q = s_q_full.index_select(0, s_q_gather)                       # [N_S_q, D]
        n_attn_q = s_q.shape[0] + B * self.n_ns
        q_buf = s_q.new_zeros(n_attn_q, D)
        q_buf.index_copy_(0, s_q_to_packed, s_q)
        q_buf.index_copy_(0, ns_q_to_packed, ns_q_full.reshape(B * self.n_ns, D))
        q = q_buf.view(n_attn_q, H, D_head)

        # KV side (full): single combined buffer [N_attn_k, 2D] -> .view -> .unbind.
        # Autograd saves one tensor instead of two.
        n_attn_k = s_kv.shape[0] + B * self.n_ns
        kv_buf = s_kv.new_zeros(n_attn_k, 2 * D)
        kv_buf.index_copy_(0, s_to_packed, s_kv)
        kv_buf.index_copy_(0, ns_to_packed, ns_kv.reshape(B * self.n_ns, 2 * D))
        k, v = kv_buf.view(n_attn_k, 2, H, D_head).unbind(1)

        out_attn = _flash_attn_varlen(
            q, k, v, cu_seqlens_q, cu_seqlens,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen,
            causal=True, dropout_p=drop_p,
        )                                                                # [n_attn_q, H, D_head]

        out_flat = out_attn.reshape(n_attn_q, D)
        s_out_flat = out_flat.index_select(0, s_q_to_packed)             # [N_S_q, D]
        ns_out_flat = out_flat.index_select(0, ns_q_to_packed)
        ns_out = ns_out_flat.view(B, self.n_ns, D)
        return self.out_proj(s_out_flat), self.out_proj(ns_out)


class MixedFFNPacked(nn.Module):
    """Packed-input variant of MixedFFN.

    S uses shared Linear (works on packed [N_S, D]); NS uses per-position
    weights via einsum on dense [B, n_ns, D]. No interleaving needed —
    FFN is element-wise per token.
    """

    def __init__(self, d_model: int, ffn_dim: int, n_ns_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.n_ns = n_ns_tokens
        self.d_model = d_model

        self.s_w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.s_w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.ns_w1 = nn.Parameter(torch.empty(n_ns_tokens, ffn_dim, d_model))
        self.ns_w2 = nn.Parameter(torch.empty(n_ns_tokens, d_model, ffn_dim))
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.s_w1.weight)
        nn.init.xavier_uniform_(self.s_w2.weight)
        for i in range(self.n_ns):
            nn.init.xavier_uniform_(self.ns_w1.data[i])
            nn.init.xavier_uniform_(self.ns_w2.data[i])

    def forward(
        self, s_packed: Tensor, ns_dense: Tensor,
    ) -> tuple[Tensor, Tensor]:
        s_out = self.s_w2(self.dropout(F.gelu(self.s_w1(s_packed))))
        ns_h = F.gelu(torch.einsum("btn,tmn->btm", ns_dense, self.ns_w1))
        ns_out = torch.einsum("btm,tnm->btn", self.dropout(ns_h), self.ns_w2)
        return s_out, ns_out


class OneTransPackedBlock(nn.Module):
    """Pre-norm causal Transformer block running on packed-S + dense-NS streams."""

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int,
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MixedAttentionPacked(d_model, n_heads, n_ns_tokens, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MixedFFNPacked(d_model, d_model * ffn_mult, n_ns_tokens, dropout)

    def forward(
        self,
        s: Tensor,
        ns: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        s_to_packed: Tensor,
        ns_to_packed: Tensor,
        # Pyramid (optional); when present S queries are a strict subset of
        # the input and the S residual stream shrinks accordingly.
        cu_seqlens_q: Tensor | None = None,
        max_seqlen_q: int | None = None,
        s_q_to_packed: Tensor | None = None,
        ns_q_to_packed: Tensor | None = None,
        s_q_gather: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pyramid = cu_seqlens_q is not None
        # Attention
        s_n = self.norm1(s)
        ns_n = self.norm1(ns)
        s_a, ns_a = self.attn(
            s_n, ns_n, cu_seqlens, max_seqlen, s_to_packed, ns_to_packed,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
            s_q_to_packed=s_q_to_packed, ns_q_to_packed=ns_q_to_packed,
            s_q_gather=s_q_gather,
        )
        # Residual: in pyramid mode the S-stream shrinks (drop-front gather of
        # the input), so the skip connection is the gathered slice. NS keeps
        # full shape every layer.
        if pyramid:
            s = s.index_select(0, s_q_gather) + s_a
        else:
            s = s + s_a
        ns = ns + ns_a
        # FFN (element-wise, runs on whatever size s is now)
        s_n = self.norm2(s)
        ns_n = self.norm2(ns)
        s_f, ns_f = self.ffn(s_n, ns_n)
        s = s + s_f
        ns = ns + ns_f
        return s, ns


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class SequentialTokenizer(nn.Module):
    """Projects per-position history embeddings to S-tokens."""

    def __init__(self, raw_dim: int, d_model: int, max_len: int = 1024,
                 use_pos_embed: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(max_len, d_model) if use_pos_embed else None

    def forward(self, raw: Tensor) -> Tensor:
        """raw: [B, L, raw_dim] -> [B, L, d_model]"""
        tokens = self.proj(raw)
        if self.pos_emb is not None:
            L = tokens.shape[1]
            tokens = tokens + self.pos_emb(torch.arange(L, device=tokens.device))
        return tokens

    def forward_packed(
        self, raw_packed: Tensor, pos_indices: Tensor | None = None,
    ) -> Tensor:
        """Packed-input variant — same weights, bit-equivalent to ``forward``.

        Lets the jagged backbone apply the tokenizer Linear directly to packed
        ``[N_real, raw_dim]`` so the dense ``[B, L, d_model]`` tokenizer output
        is never materialized.

        ``pos_indices`` carries each packed token's position-within-its-pool
        (the same index the dense path uses via ``torch.arange(L)``); required
        when ``pos_emb`` is enabled. The same vocabulary slot in
        ``self.pos_emb`` is read so outputs match the dense path bitwise.
        """
        tokens = self.proj(raw_packed)
        if self.pos_emb is not None:
            assert pos_indices is not None, \
                "pos_indices required when pos_emb is enabled"
            tokens = tokens + self.pos_emb(pos_indices)
        return tokens


class AutoSplitTokenizer(nn.Module):
    """Projects concatenated NS features into n_ns_tokens tokens."""

    def __init__(self, ns_input_dim: int, d_model: int, n_ns_tokens: int):
        super().__init__()
        out_dim = d_model * n_ns_tokens
        self.proj = nn.Sequential(
            nn.Linear(ns_input_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.n_ns_tokens = n_ns_tokens
        self.d_model = d_model

    def forward(self, ns_features: Tensor) -> Tensor:
        """ns_features: [B, ns_input_dim] -> [B, n_ns_tokens, d_model]"""
        return self.proj(ns_features).view(-1, self.n_ns_tokens, self.d_model)


class OneTransModel(BaseModel):
    """OneTrans: schema-driven Transformer for ranking.

    All architecture decisions are driven by ``Config``:

    - **Embedding tables**: from ``config.model.embedding_tables``
    - **Sequence groups**: from ``config.feature.sequence_groups`` — each group gets
      its own ``SequentialTokenizer``
    - **Scalar features**: from ``config.feature.scalar_features`` — concatenated
      into the NS-token input
    - **Dense features**: from ``config.feature.dense_features`` — projected to D
      (``project=True``, activation from ``df.activation``) or concatenated
      raw (``project=False``)
    - **Task heads**: from ``config.task_names``

    Pipeline mode (``_pipeline_mode=True``): forward() passes a pre-built KJT
    directly to ``emb.ec``, giving FX a traceable getitem→call_module path so
    TrainPipelineSparseDist can pipeline the embedding all-to-all.
    """

    def __init__(
        self,
        config: Config,
        device: torch.device | None = None,
        meta_device: bool = False,
    ):
        super().__init__()
        self._pipeline_mode = False
        self.config = config
        mc = config.model
        fc = config.feature
        ot = mc.transformer

        if device is None:
            device = torch.device("cpu")
        emb_device = torch.device("meta") if meta_device else device
        self.tasks = list(config.task_names)
        D = mc.embedding_dim

        # Embeddings
        self.emb = TorchRecEmbeddings(
            mc.resolved_embedding_tables(),
            device=emb_device,
            embedding_init=mc.embedding_init,
            scalar_feature_names=set(fc.scalar_feature_names),
        )

        # Per-group sequential tokenizers
        self.seq_tokenizers = nn.ModuleDict({
            group_name: SequentialTokenizer(
                len(feats) * D, ot.d_model, max_len=2048, use_pos_embed=ot.pos_embed,
            ).to(device)
            for group_name, feats in fc.sequence_groups.items()
        })

        # Dense feature projections (project=True → Linear+activation, else raw concat)
        self.dense_projs = nn.ModuleDict()
        ns_raw_dim = len(fc.scalar_feature_names) * D
        for df in fc.dense_features:
            if df.project:
                act: nn.Module = nn.ReLU(inplace=True) if df.activation == "relu" else nn.GELU()
                self.dense_projs[df.name] = nn.Sequential(
                    nn.Linear(df.dim, D, device=device), act,
                )
                ns_raw_dim += D
            else:
                ns_raw_dim += df.dim

        self.ns_tokenizer = AutoSplitTokenizer(
            ns_raw_dim, ot.d_model, ot.n_ns_tokens,
        ).to(device)

        # Transformer stack: dense (default) or fully-packed
        # (train.use_jagged_attention=True). The flag lives under TrainConfig
        # because it's an execution strategy, not a model-arch choice.
        if self.config.train.use_jagged_attention:
            self.blocks = nn.ModuleList([
                OneTransPackedBlock(ot.d_model, ot.n_heads, ot.n_ns_tokens, ot.ffn_mult, ot.dropout)
                for _ in range(ot.n_layers)
            ]).to(device)
        else:
            self.blocks = nn.ModuleList([
                OneTransBlock(ot.d_model, ot.n_heads, ot.n_ns_tokens, ot.ffn_mult, ot.dropout,
                             attention_impl=self.config.train.attention_impl)
                for _ in range(ot.n_layers)
            ]).to(device)
        self._compile_blocks = config.train.torch_compile
        self._compile_backend = config.train.torch_compile_backend
        self.final_norm = RMSNorm(ot.d_model).to(device)

        # Head bridge
        head_input_dim = ot.n_ns_tokens * ot.d_model
        self.head_proj = nn.Sequential(
            nn.Linear(head_input_dim, ot.d_model, device=device),
            nn.GELU(),
            nn.Dropout(ot.dropout),
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(ot.d_model, 1, device=device)
            for task in self.tasks
        })

        # Contrastive heads: item representation uses only item-side scalars
        # (excludes user_id_feature) to avoid leaking user identity
        self._item_scalar_names = [sf.name for sf in fc.item_scalar_features]
        self.contrastive_user_proj = nn.Linear(ot.d_model, ot.d_model, device=device)
        self.contrastive_item_proj = nn.Sequential(
            nn.Linear(len(self._item_scalar_names) * D, ot.d_model, device=device),
            nn.ReLU(inplace=True),
        )

    def apply_compile(self) -> None:
        """Compile transformer blocks with torch.compile (call after DMP wrapping)."""
        if not self._compile_blocks:
            return
        logging.getLogger(__name__).info(
            f"Compiling {len(self.blocks)} OneTransBlocks with backend={self._compile_backend}")
        for i, block in enumerate(self.blocks):
            self.blocks[i] = torch.compile(
                block, fullgraph=False, dynamic=True, backend=self._compile_backend)

    def get_num_flops_per_sample(self) -> float:
        """Estimate FLOPs per sample for fwd+bwd (×3 matmul, ×2 FMA).

        Follows TorchTitan convention: embedding lookups excluded (memory-bound),
        all linear/einsum/attention ops counted with factor 6 for matmuls
        (3× fwd+bwd, 2× FMA) and 12 for attention (6× matmul directions, 2× FMA).
        Causal attention sparsity is not counted (same as PaLM/TorchTitan).
        """
        ot = self.config.model.transformer
        fc = self.config.feature
        D = ot.d_model
        H = ot.n_heads
        d_head = D // H
        n_ns = ot.n_ns_tokens
        ffn_dim = D * ot.ffn_mult
        emb_dim = self.config.model.embedding_dim

        L_hist = self.config.data.history_length
        n_groups = len(fc.sequence_groups)
        L_S = L_hist * n_groups
        schedule = pyramid_schedule(L_S, n_ns, ot.n_layers) if ot.use_pyramid else [L_S] * ot.n_layers

        flops = 0

        # Sequential tokenizers: Linear(raw_dim, D) + Linear(D, D) per group
        for group_name, feats in fc.sequence_groups.items():
            raw_dim = len(feats) * emb_dim
            flops += 6 * L_hist * (raw_dim * D + D * D)

        # Dense feature projections
        for df in fc.dense_features:
            if df.project:
                flops += 6 * df.dim * emb_dim

        # NS tokenizer: Linear(ns_raw_dim, D*n_ns) + Linear(D*n_ns, D*n_ns)
        ns_raw_dim = getattr(self.ns_tokenizer.proj[0], 'in_features', 0)
        ns_out_dim = n_ns * D
        if ns_raw_dim > 0:
            flops += 6 * (ns_raw_dim * ns_out_dim + ns_out_dim * ns_out_dim)

        # Transformer blocks (per layer with pyramid schedule)
        kv_len = L_S + n_ns
        for layer_idx in range(ot.n_layers):
            q_s = schedule[layer_idx]
            q_len = q_s + n_ns
            cur_n_s = kv_len - n_ns

            # S-token QKV: Linear(D, 3D) on cur_n_s tokens
            flops += 6 * cur_n_s * D * 3 * D
            # NS-token QKV: einsum [n_ns, 3D, D]
            flops += 6 * n_ns * 3 * D * D
            # Causal attention: Q*K^T + attn*V (full, not discounting causal sparsity)
            flops += 12 * H * d_head * q_len * kv_len
            # Output projection: Linear(D, D) on q_len tokens
            flops += 6 * q_len * D * D

            # FFN: S-tokens
            n_s_out = q_len - n_ns
            flops += 6 * n_s_out * D * ffn_dim * 2
            # FFN: NS-tokens (two einsums)
            flops += 6 * n_ns * D * ffn_dim * 2

            kv_len = q_len

        # Head projection: Linear(n_ns*D, D)
        flops += 6 * n_ns * D * D
        # Task heads: Linear(D, 1) per task
        flops += 6 * len(self.tasks) * D

        # Contrastive heads (only when contrastive_weight > 0)
        if self.config.train.contrastive_weight > 0:
            # contrastive_user_proj: Linear(D, D)
            flops += 6 * D * D
            # contrastive_item_proj: Linear(n_item_scalars*emb_dim, D)
            n_item_scalars = len(self._item_scalar_names)
            flops += 6 * n_item_scalars * emb_dim * D
            # cross_scores: user_emb @ item_emb.T  [B, D] @ [D, B] -> [B, B]
            # per-sample amortized: 2 * D * B (fwd only, ×3 for bwd = 6)
            batch_size = self.config.train.batch_size // (
                self.config.distributed.num_nodes * self.config.distributed.gpus_per_node)
            flops += 6 * D * batch_size

        return float(flops)

    def estimate_jagged_flops_per_sample(self, batch: dict[str, Tensor]) -> Tensor:
        """Per-sample fwd+bwd FLOPs for THIS batch's actual sequence lengths.

        Mirrors ``get_num_flops_per_sample`` but substitutes:

        - **S-token projections / FFN** scale by ``mean(s_real_lens) / L_S``
          (Path B saves these padded compute; non-Path-B jagged still pays
          the dense tokenizer cost — the formula honors that via
          ``pack_tokenizer``).
        - **Attention** scales by ``mean((s_real_i + n_ns)^2) / (L_S + n_ns)^2``
          per layer (the per-user attention cost is quadratic in sequence
          length, so longer-tail users dominate).
        - **Out projection** scales linearly with mean attn length.
        - Dense feature projections, NS tokenizer, NS FFN, head, and task
          heads stay constant per sample (always dense).

        Returns a 0-d tensor on the batch's device — caller is expected to
        ``.item()`` it (typically once per logging interval).
        """
        ot = self.config.model.transformer
        fc = self.config.feature
        D = ot.d_model
        H = ot.n_heads
        d_head = D // H
        n_ns = ot.n_ns_tokens
        ffn_dim = D * ot.ffn_mult
        emb_dim = self.config.model.embedding_dim
        L_hist = self.config.data.history_length
        n_groups = len(fc.sequence_groups)
        L_S = L_hist * n_groups

        n_lp = batch["hist_lp_len"].float()
        n_like = batch["hist_like_len"].float()
        n_skip = batch["hist_skip_len"].float()
        s_real = n_lp + n_like + n_skip                  # [B]
        attn_real = s_real + n_ns                        # [B]
        avg_s_real = s_real.mean()                       # scalar tensor
        avg_attn_real = attn_real.mean()
        avg_attn_real_sq = (attn_real * attn_real).mean()

        flops = avg_s_real.new_zeros(())

        # Per-pool sequential tokenizer Linear (raw_dim->D->D).
        # When pack_tokenizer=True, the Linear runs only on real tokens, so
        # FLOPs scale with ``avg_s_real / n_groups``. Otherwise it runs on
        # the dense ``L_hist`` tokens per pool.
        per_pool_tokens = (avg_s_real / n_groups) if self.config.train.pack_tokenizer else float(L_hist)
        for group_name, feats in fc.sequence_groups.items():
            raw_dim = len(feats) * emb_dim
            flops = flops + 6 * per_pool_tokens * (raw_dim * D + D * D)

        # Dense feature projections (always dense, fixed per sample).
        for df in fc.dense_features:
            if df.project:
                flops = flops + 6 * df.dim * emb_dim

        # NS tokenizer (always dense).
        ns_raw_dim = getattr(self.ns_tokenizer.proj[0], "in_features", 0)
        ns_out_dim = n_ns * D
        if ns_raw_dim > 0:
            flops = flops + 6 * (ns_raw_dim * ns_out_dim + ns_out_dim * ns_out_dim)

        # Transformer blocks. With pyramid, per-layer Q-side per-user count
        # = min(schedule[l], real_len). Attention compute is Q×K → uses
        # cur_q_per_user × prev_attn_real per user; FFN uses cur_q_per_user.
        if ot.use_pyramid:
            schedule = pyramid_schedule(L_S, n_ns, ot.n_layers)
        else:
            schedule = [L_S] * ot.n_layers
        prev_attn_real = attn_real                            # [B] = layer 0 K-side
        for l in range(ot.n_layers):
            cur_s_q_per_user = torch.minimum(
                s_real, s_real.new_tensor(float(schedule[l])),
            )                                                  # [B]
            cur_attn_q_per_user = cur_s_q_per_user + n_ns      # [B]
            avg_cur_s_q = cur_s_q_per_user.mean()
            avg_cur_attn_q = cur_attn_q_per_user.mean()
            # mean over batch of (cur_q_i * prev_kv_i), the per-user attention work
            avg_attn_qxk = (cur_attn_q_per_user * prev_attn_real).mean()

            # S-token QKV: projects only real-K tokens (full input to layer)
            flops = flops + 6 * prev_attn_real.mean() * D * 3 * D
            # NS-token QKV: dense per sample
            flops = flops + 6 * n_ns * 3 * D * D
            # Causal attention (undiscounted, matches dense formula convention):
            # per-user cur_q × prev_kv × 12 H d_head
            flops = flops + 12 * H * d_head * avg_attn_qxk
            # Output projection on Q-side outputs only
            flops = flops + 6 * avg_cur_attn_q * D * D
            # FFN S: runs on Q-side S only
            flops = flops + 6 * avg_cur_s_q * D * ffn_dim * 2
            # FFN NS: dense (always n_ns)
            flops = flops + 6 * n_ns * D * ffn_dim * 2

            prev_attn_real = cur_attn_q_per_user               # next layer's K side

        # Head + task heads (constant)
        flops = flops + 6 * n_ns * D * D
        flops = flops + 6 * len(self.tasks) * D

        # Contrastive heads (only when enabled)
        if self.config.train.contrastive_weight > 0:
            flops = flops + 6 * D * D
            n_item_scalars = len(self._item_scalar_names)
            flops = flops + 6 * n_item_scalars * emb_dim * D
            batch_size = self.config.train.batch_size // (
                self.config.distributed.num_nodes * self.config.distributed.gpus_per_node)
            flops = flops + 6 * D * batch_size

        return flops

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Schema-driven TorchRec lookup: maps batch keys to EC feature names."""
        feat_to_key = self.config.feature_to_batch_key()
        lookup_dict = {}
        for feat_name in self.config.feature.all_ec_feature_names():
            batch_key = feat_to_key.get(feat_name, feat_name)
            if batch_key in batch:
                lookup_dict[feat_name] = batch[batch_key]
        return self.emb(lookup_dict)

    def _lookup_pipeline(self, batch: dict) -> dict[str, Tensor]:
        """Embedding lookup via pre-built KJT (pipeline mode).

        Gives FX a clean ``batch["unpooled_kjt"] → emb.ec`` path so that
        TrainPipelineSparseDist can pipeline the embedding all-to-all.
        """
        ec_out = self.emb.ec(batch["unpooled_kjt"])
        first_ec_feat = self.config.feature.scalar_feature_names[0]
        first_batch_key = self.config.feature_to_batch_key().get(first_ec_feat, first_ec_feat)
        B = batch[first_batch_key].shape[0]
        embs: dict[str, Tensor] = {}
        for feat in self.emb._unpooled_features:
            jt = ec_out[feat]
            emb = jt.values()
            D = self.emb._feature_dims[feat]
            if feat in self.emb._scalar_features:
                embs[feat] = emb.view(B, D)
            else:
                embs[feat] = emb.view(B, -1, D)
        return embs

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def _build_pool_raw(
        self, embs: dict[str, Tensor], group_name: str,
    ) -> Tensor:
        """Concatenate embeddings for one sequence group: [B, L, N*D]."""
        feats = self.config.feature.sequence_groups[group_name]
        return torch.cat([embs[f] for f in feats], dim=-1)

    def _build_ns_raw(
        self, embs: dict[str, Tensor], batch: dict[str, Tensor],
    ) -> Tensor:
        """Build NS raw vector: scalar embeddings + dense features."""
        parts = [embs[f] for f in self.config.feature.scalar_feature_names]
        for df in self.config.feature.dense_features:
            if df.name in self.dense_projs:
                parts.append(self.dense_projs[df.name](batch[df.name]))
            else:
                parts.append(batch[df.name])
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def _backbone(
        self, embs: dict[str, Tensor], batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """Run the full transformer and return (h, s_repr).

        h:      [B, d_model] head-projected NS-token representation.
        s_repr: [B, d_model] mean-pooled S-token representation (user history
                only, no candidate item info -- used for contrastive loss).
        """
        ot = self.config.model.transformer
        first_ec = self.config.feature.scalar_feature_names[0]
        first_key = self.config.feature_to_batch_key().get(first_ec, first_ec)
        B = batch[first_key].shape[0]
        L_NS = ot.n_ns_tokens

        tc = self.config.train
        # Path B: pack each pool BEFORE the tokenizer Linear so the dense
        # [B, L_S, d_model] tokenizer output never exists. Same weights, same
        # outputs as the dense tokenizer path; saves ~2-6 GB HBM at hist=500/
        # 1000 with no recomputation.
        if tc.use_jagged_attention and tc.pack_tokenizer:
            return self._backbone_jagged_packed(embs, batch, B, L_NS)

        # S-token construction: per-group tokenization
        pool_tokens = []
        for group_name in self.config.feature.sequence_groups:
            raw = self._build_pool_raw(embs, group_name)
            pool_tokens.append(self.seq_tokenizers[group_name](raw))
        s_tokens = torch.cat(pool_tokens, dim=1)
        L_S = s_tokens.shape[1]

        # NS-token construction
        ns_raw = self._build_ns_raw(embs, batch)
        ns_tokens = self.ns_tokenizer(ns_raw)

        if tc.use_jagged_attention:
            return self._backbone_jagged(s_tokens, ns_tokens, batch, B, L_S, L_NS)

        # Dense path (default)
        x = torch.cat([s_tokens, ns_tokens], dim=1)

        # Pyramid schedule
        if ot.use_pyramid:
            schedule = pyramid_schedule(L_S, L_NS, ot.n_layers)
        else:
            schedule = [L_S] * ot.n_layers

        # Transformer blocks
        for layer_idx, block in enumerate(self.blocks):
            q_s = schedule[layer_idx]
            q_len = q_s + L_NS
            kv_len = x.shape[1]
            cur_n_s = kv_len - L_NS

            mask = build_pyramid_mask(q_len, kv_len, x.device)
            x = block(x, mask=mask, q_len=q_len, n_s_tokens=cur_n_s)

        x = self.final_norm(x)

        # S-token user representation
        n_s_final = x.shape[1] - L_NS
        s_repr = x[:, :n_s_final].mean(dim=1)

        # NS-token head projection
        ns_out = x[:, -L_NS:]
        h = self.head_proj(ns_out.reshape(B, -1))

        return h, s_repr

    def _backbone_jagged(
        self,
        s_tokens: Tensor,
        ns_tokens: Tensor,
        batch: dict[str, Tensor],
        B: int,
        L_S: int,
        L_NS: int,
    ) -> tuple[Tensor, Tensor]:
        """Phase B2 packed transformer path.

        Pack ONCE before the block loop, run all layers fully packed, unpack
        ONCE after for s_repr. NS stays dense throughout because its weights
        are per-position (n_ns separate matrices indexed by slot).
        """
        ot = self.config.model.transformer
        device = s_tokens.device
        L_per_pool = L_S // len(self.config.feature.sequence_groups)
        n_lp = batch["hist_lp_len"].to(torch.int32)
        n_like = batch["hist_like_len"].to(torch.int32)
        n_skip = batch["hist_skip_len"].to(torch.int32)
        s_real_lens = n_lp + n_like + n_skip                 # [B] int32
        attn_real_lens = s_real_lens + L_NS                  # [B] int32 — per-user attn seq

        # Build s_real_mask [B, L_S] bool: True at real-token positions in s_tokens
        arange_pool = torch.arange(L_per_pool, device=device, dtype=torch.int32)
        lp_mask = arange_pool.unsqueeze(0) >= (L_per_pool - n_lp.unsqueeze(1))
        like_mask = arange_pool.unsqueeze(0) >= (L_per_pool - n_like.unsqueeze(1))
        skip_mask = arange_pool.unsqueeze(0) >= (L_per_pool - n_skip.unsqueeze(1))
        s_real_mask = torch.cat([lp_mask, like_mask, skip_mask], dim=1)  # [B, L_S]

        # Pack S tokens: nonzero gives positions in s_tokens.flatten(0,1), in
        # row-major (per-user contiguous, per-pool front-padding-skipped) order.
        s_indices_in_dense = s_real_mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int64)
        s_packed = s_tokens.reshape(B * L_S, ot.d_model).index_select(0, s_indices_in_dense)
        # ^ [total_S_real, D]

        # Build cu_seqlens for FA-varlen: one segment per user covering S_real_i + n_ns
        cu_seqlens = attn_real_lens.cumsum(0).to(torch.int32)
        zero = cu_seqlens.new_zeros(1)
        cu_seqlens = torch.cat([zero, cu_seqlens])           # [B+1]
        max_seqlen = L_S + L_NS

        # s_to_packed[k] = position in attn buffer of the k-th S-token in s_packed.
        # For user i, S tokens are at attn positions [cu_seqlens[i], cu_seqlens[i] + S_real_i).
        # User-id-per-token via searchsorted on cumulative S-real lengths.
        cu_s_real = s_real_lens.cumsum(0).to(torch.int32)
        zero32 = cu_s_real.new_zeros(1)
        cu_s_real = torch.cat([zero32, cu_s_real])           # [B+1]
        # arange over total_S_real WITHOUT Python int() cast (FX-friendly).
        # ones_like + cumsum - 1 generates [0, 1, 2, ..., N-1] from any [N]-shaped tensor.
        arange_s = torch.ones_like(s_indices_in_dense, dtype=torch.int64).cumsum(0) - 1
        # user_id_per_token: which user does each S-real token belong to.
        s_user = torch.searchsorted(cu_s_real, arange_s.to(torch.int32), right=True) - 1   # [total_S_real]
        s_user = s_user.to(torch.int64)
        s_local = arange_s - cu_s_real[s_user].to(torch.int64)                              # within-user offset
        s_to_packed = cu_seqlens[s_user].to(torch.int64) + s_local                          # [total_S_real]

        # ns_to_packed[k] = position in attn buffer of the k-th NS slot (k = i*L_NS + j)
        arange_ns = torch.arange(B * L_NS, device=device, dtype=torch.int64)
        ns_user = arange_ns // L_NS
        ns_local = arange_ns - ns_user * L_NS
        # NS goes after each user's S in attn buffer
        ns_to_packed = (
            cu_seqlens[1:][ns_user].to(torch.int64) - L_NS + ns_local
        )                                                                                   # [B*L_NS]

        # Pre-compute per-layer pyramid indices when pyramid is enabled
        # (parallel to _backbone_jagged_packed).
        use_pyramid = ot.use_pyramid
        pyramid_layers: list[dict] | None = None
        if use_pyramid:
            pyramid_layers = self._build_pyramid_jagged_indices(
                s_real_lens=s_real_lens,
                cu_s_real=cu_s_real,
                cu_seqlens=cu_seqlens,
                s_to_packed=s_to_packed,
                ns_to_packed=ns_to_packed,
                B=B, L_NS=L_NS, L_S=L_S,
                n_layers=ot.n_layers,
                zero32=zero32,
                device=device,
            )
            last = pyramid_layers[-1]
            final_s_user = last["s_q_user"]
            final_s_real_lens = last["q_per_user"]
        else:
            final_s_user = s_user
            final_s_real_lens = s_real_lens

        # Run packed transformer (optionally with gradient checkpointing).
        # Checkpointing recomputes each block's interleave + attention in
        # backward, freeing the per-layer attn_qkv / out_attn buffers (~25-35
        # GB at hist=500/1000) for ~25% extra step time.
        s = s_packed
        ns = ns_tokens
        use_ckpt = self.config.train.grad_checkpoint and self.training
        for layer_idx, block in enumerate(self.blocks):
            if pyramid_layers is not None:
                p = pyramid_layers[layer_idx]
                if use_ckpt:
                    s, ns = _ckpt.checkpoint(
                        block, s, ns,
                        p["cu_seqlens_k"], p["max_seqlen_k"],
                        p["s_k_to_packed"], p["ns_k_to_packed"],
                        p["cu_seqlens_q"], p["max_seqlen_q"],
                        p["s_q_to_packed"], p["ns_q_to_packed"],
                        p["s_q_gather"],
                        use_reentrant=False,
                    )
                else:
                    s, ns = block(
                        s, ns,
                        p["cu_seqlens_k"], p["max_seqlen_k"],
                        p["s_k_to_packed"], p["ns_k_to_packed"],
                        cu_seqlens_q=p["cu_seqlens_q"],
                        max_seqlen_q=p["max_seqlen_q"],
                        s_q_to_packed=p["s_q_to_packed"],
                        ns_q_to_packed=p["ns_q_to_packed"],
                        s_q_gather=p["s_q_gather"],
                    )
            else:
                if use_ckpt:
                    s, ns = _ckpt.checkpoint(
                        block, s, ns, cu_seqlens, max_seqlen,
                        s_to_packed, ns_to_packed, use_reentrant=False,
                    )
                else:
                    s, ns = block(s, ns, cu_seqlens, max_seqlen, s_to_packed, ns_to_packed)

        s = self.final_norm(s)
        ns = self.final_norm(ns)

        # s_repr: per-user mean of (final-layer) S tokens. Pyramid reduces
        # the count; use the matching ``final_*`` indices.
        s_repr_sum = s.new_zeros(B, ot.d_model)
        s_repr_sum.index_add_(0, final_s_user, s)
        s_repr = s_repr_sum / final_s_real_lens.to(s.dtype).clamp_min(1).unsqueeze(1)

        # NS head projection (NS already dense [B, L_NS, D])
        h = self.head_proj(ns.reshape(B, -1))
        # Stash per-step actual jagged FLOPs/sample estimate (detached, on
        # device — the trainer .item()s once per log interval).
        self._last_jagged_flops_per_sample = self.estimate_jagged_flops_per_sample(batch).detach()
        return h, s_repr

    def _backbone_jagged_packed(
        self,
        embs: dict[str, Tensor],
        batch: dict[str, Tensor],
        B: int,
        L_NS: int,
    ) -> tuple[Tensor, Tensor]:
        """Path B variant of _backbone_jagged.

        Pack each pool's raw embeddings BEFORE the per-pool tokenizer Linear,
        then run the existing OneTransPackedBlock loop. The dense
        ``[B, L_S, d_model]`` tokenizer output is never materialized; same
        weights and bit-equivalent outputs as the dense tokenizer path.
        No recomputation: pure forward-side restructuring.

        Memory savings vs ``_backbone_jagged``:
          - per-pool tokenizer dense output ``[B, L, d_model]`` (3 pools)
          - intermediate FFN hidden state ``[B, L, d_model]`` from GELU
          - the cat ``[B, 3L, d_model]`` of pool tokens
          ≈ ~2-6 GB at hist=500/1000.
        """
        ot = self.config.model.transformer
        device = next(self.parameters()).device
        L = self.config.data.history_length
        n_groups = len(self.config.feature.sequence_groups)

        # Per-pool real lengths (already populated by the dataset for jagged).
        n_lp = batch["hist_lp_len"].to(torch.int32)
        n_like = batch["hist_like_len"].to(torch.int32)
        n_skip = batch["hist_skip_len"].to(torch.int32)
        pool_lens = {"hist_lp": n_lp, "hist_like": n_like, "hist_skip": n_skip}
        s_real_lens = n_lp + n_like + n_skip                         # [B] int32
        attn_real_lens = s_real_lens + L_NS                          # [B] int32

        # Per-pool: build raw [B, L, raw_dim], pack to [N_real, raw_dim],
        # apply tokenizer's Linear+GELU+Linear+pos_emb on PACKED input.
        pool_packed: dict[str, Tensor] = {}
        pool_user_id: dict[str, Tensor] = {}
        pool_within: dict[str, Tensor] = {}
        arange_pool = torch.arange(L, device=device, dtype=torch.int32)
        for group_name in self.config.feature.sequence_groups:
            n_real = pool_lens[group_name]
            raw = self._build_pool_raw(embs, group_name)              # [B, L, raw_dim]
            # Front-padded: real tokens occupy positions [L - n_real, L) per user.
            real_mask = arange_pool.unsqueeze(0) >= (L - n_real.unsqueeze(1))  # [B, L]
            flat_idx = real_mask.flatten().nonzero(as_tuple=False).squeeze(1).to(torch.int64)
            raw_packed = raw.reshape(B * L, -1).index_select(0, flat_idx)     # [N, raw_dim]
            # Position-within-pool indices match what the dense path's
            # `torch.arange(L)` would have indexed pos_emb at.
            pos_in_pool = (flat_idx % L).to(torch.int64)
            tokens_packed = self.seq_tokenizers[group_name].forward_packed(raw_packed, pos_in_pool)
            user_id = (flat_idx // L).to(torch.int64)
            pool_packed[group_name] = tokens_packed
            pool_user_id[group_name] = user_id
            # Within-user offset: 0..n_real_i-1, in pack order.
            pool_within[group_name] = pos_in_pool - (L - n_real[user_id]).to(torch.int64)

        # Build cu_s_real (prefix-sum of per-user S real lens).
        cu_s_real = s_real_lens.cumsum(0).to(torch.int32)
        zero32 = cu_s_real.new_zeros(1)
        cu_s_real = torch.cat([zero32, cu_s_real])                  # [B+1]

        # Compute target indices in s_packed for each pool's tokens.
        # User i's S sequence layout: [lp_real_i, like_real_i, skip_real_i].
        user_lp = pool_user_id["hist_lp"]
        user_like = pool_user_id["hist_like"]
        user_skip = pool_user_id["hist_skip"]
        target_lp = (
            cu_s_real[user_lp].to(torch.int64) + pool_within["hist_lp"]
        )
        target_like = (
            cu_s_real[user_like].to(torch.int64)
            + n_lp[user_like].to(torch.int64)
            + pool_within["hist_like"]
        )
        target_skip = (
            cu_s_real[user_skip].to(torch.int64)
            + n_lp[user_skip].to(torch.int64)
            + n_like[user_skip].to(torch.int64)
            + pool_within["hist_skip"]
        )

        # Allocate s_packed and scatter pool tokens into user-interleaved order.
        total_S_real = (
            pool_packed["hist_lp"].shape[0]
            + pool_packed["hist_like"].shape[0]
            + pool_packed["hist_skip"].shape[0]
        )
        s_packed = pool_packed["hist_lp"].new_zeros(total_S_real, ot.d_model)
        s_packed.index_copy_(0, target_lp, pool_packed["hist_lp"])
        s_packed.index_copy_(0, target_like, pool_packed["hist_like"])
        s_packed.index_copy_(0, target_skip, pool_packed["hist_skip"])

        # NS tokens (dense, unchanged).
        ns_raw = self._build_ns_raw(embs, batch)
        ns_tokens = self.ns_tokenizer(ns_raw)

        # Build cu_seqlens for FA-varlen and the s_to_packed / ns_to_packed
        # gather indices used inside MixedAttentionPacked.
        cu_seqlens = attn_real_lens.cumsum(0).to(torch.int32)
        cu_seqlens = torch.cat([zero32, cu_seqlens])                # [B+1]
        max_seqlen = n_groups * L + L_NS                            # static upper bound

        # arange over total_S_real, FX-friendly (ones+cumsum on the packed shape).
        arange_s = torch.ones(s_packed.shape[0], device=device, dtype=torch.int64).cumsum(0) - 1
        s_user = torch.searchsorted(cu_s_real, arange_s.to(torch.int32), right=True) - 1
        s_user = s_user.to(torch.int64)
        s_local = arange_s - cu_s_real[s_user].to(torch.int64)
        s_to_packed = cu_seqlens[s_user].to(torch.int64) + s_local

        arange_ns = torch.arange(B * L_NS, device=device, dtype=torch.int64)
        ns_user = arange_ns // L_NS
        ns_local = arange_ns - ns_user * L_NS
        ns_to_packed = (
            cu_seqlens[1:][ns_user].to(torch.int64) - L_NS + ns_local
        )

        # Pre-compute per-layer pyramid indices when pyramid is enabled
        # alongside jagged. Layer 0 is full-attention (q == k); layers 1..N-1
        # progressively drop the front of the S-stream per user.
        use_pyramid = ot.use_pyramid
        pyramid_layers: list[dict] | None = None
        if use_pyramid:
            pyramid_layers = self._build_pyramid_jagged_indices(
                s_real_lens=s_real_lens,
                cu_s_real=cu_s_real,
                cu_seqlens=cu_seqlens,
                s_to_packed=s_to_packed,
                ns_to_packed=ns_to_packed,
                B=B, L_NS=L_NS, L_S=n_groups * L,
                n_layers=ot.n_layers,
                zero32=zero32,
                device=device,
            )
            # The last layer's Q-side per-user lengths; used for s_repr below.
            last = pyramid_layers[-1]
            final_s_user = last["s_q_user"]
            final_s_real_lens = last["q_per_user"]
        else:
            final_s_user = s_user
            final_s_real_lens = s_real_lens

        # Run packed transformer.
        s = s_packed
        ns = ns_tokens
        use_ckpt = self.config.train.grad_checkpoint and self.training
        for layer_idx, block in enumerate(self.blocks):
            if pyramid_layers is not None:
                p = pyramid_layers[layer_idx]
                if use_ckpt:
                    s, ns = _ckpt.checkpoint(
                        block, s, ns,
                        p["cu_seqlens_k"], p["max_seqlen_k"],
                        p["s_k_to_packed"], p["ns_k_to_packed"],
                        p["cu_seqlens_q"], p["max_seqlen_q"],
                        p["s_q_to_packed"], p["ns_q_to_packed"],
                        p["s_q_gather"],
                        use_reentrant=False,
                    )
                else:
                    s, ns = block(
                        s, ns,
                        p["cu_seqlens_k"], p["max_seqlen_k"],
                        p["s_k_to_packed"], p["ns_k_to_packed"],
                        cu_seqlens_q=p["cu_seqlens_q"],
                        max_seqlen_q=p["max_seqlen_q"],
                        s_q_to_packed=p["s_q_to_packed"],
                        ns_q_to_packed=p["ns_q_to_packed"],
                        s_q_gather=p["s_q_gather"],
                    )
            else:
                if use_ckpt:
                    s, ns = _ckpt.checkpoint(
                        block, s, ns, cu_seqlens, max_seqlen,
                        s_to_packed, ns_to_packed, use_reentrant=False,
                    )
                else:
                    s, ns = block(s, ns, cu_seqlens, max_seqlen, s_to_packed, ns_to_packed)

        s = self.final_norm(s)
        ns = self.final_norm(ns)

        # s_repr: per-user mean of (final-layer) S tokens. Pyramid reduces the
        # number of S tokens per user; use the matching ``final_*`` indices.
        s_repr_sum = s.new_zeros(B, ot.d_model)
        s_repr_sum.index_add_(0, final_s_user, s)
        s_repr = s_repr_sum / final_s_real_lens.to(s.dtype).clamp_min(1).unsqueeze(1)

        h = self.head_proj(ns.reshape(B, -1))
        self._last_jagged_flops_per_sample = self.estimate_jagged_flops_per_sample(batch).detach()
        return h, s_repr

    def _build_pyramid_jagged_indices(
        self,
        s_real_lens: Tensor,
        cu_s_real: Tensor,
        cu_seqlens: Tensor,
        s_to_packed: Tensor,
        ns_to_packed: Tensor,
        B: int,
        L_NS: int,
        L_S: int,
        n_layers: int,
        zero32: Tensor,
        device,
    ) -> list[dict]:
        """Build per-layer Q/K/V index sets for pyramid + jagged.

        Returns a list of dicts (one per layer) with the args MixedAttention
        Packed needs to do drop-front Q gather + asymmetric varlen attention.

        Layer 0 is full attention (q == k, identity gather). Subsequent
        layers each drop the front of every user's S-stream by
        ``q_per_user[l-1] - q_per_user[l]`` tokens.
        """
        # Schedule of S-query counts per layer (e.g. [1500, 1288, ..., 16]).
        schedule = pyramid_schedule(L_S, L_NS, n_layers)
        # Per-user query count per layer: q_per_user[l, i] = min(schedule[l], real_len[i]).
        sched_t = torch.tensor(schedule, device=device, dtype=torch.int32)  # [n_layers]
        q_per_user_all = torch.minimum(
            sched_t.unsqueeze(1), s_real_lens.unsqueeze(0)
        )                                                                    # [n_layers, B]

        layers: list[dict] = []
        # Initial K-side comes from the FULL packed S (= layer-0 input).
        prev_cu_s = cu_s_real
        prev_cu_seqlens = cu_seqlens
        prev_s_to_packed = s_to_packed
        prev_ns_to_packed = ns_to_packed
        prev_max_seqlen = L_S + L_NS

        for l in range(n_layers):
            q_per_user = q_per_user_all[l]                                   # [B] int32
            attn_q = q_per_user + L_NS                                       # [B] int32

            # Q-side cu_seqlens (S-real per user + n_ns).
            cu_s_q = q_per_user.cumsum(0).to(torch.int32)
            cu_s_q = torch.cat([zero32, cu_s_q])                             # [B+1]
            cu_seqlens_q = attn_q.cumsum(0).to(torch.int32)
            cu_seqlens_q = torch.cat([zero32, cu_seqlens_q])                 # [B+1]
            max_seqlen_q = schedule[l] + L_NS                                # static

            # arange_s_q over total Q-side S tokens; user_id and within-user offset.
            n_s_q = cu_s_q[-1]                                               # SymInt
            arange_s_q = torch.ones(n_s_q.to(torch.int64), device=device, dtype=torch.int64).cumsum(0) - 1
            s_q_user = torch.searchsorted(cu_s_q, arange_s_q.to(torch.int32), right=True) - 1
            s_q_user = s_q_user.to(torch.int64)
            s_q_local = arange_s_q - cu_s_q[s_q_user].to(torch.int64)
            s_q_to_packed = cu_seqlens_q[s_q_user].to(torch.int64) + s_q_local

            arange_ns_q = torch.arange(B * L_NS, device=device, dtype=torch.int64)
            ns_q_user = arange_ns_q // L_NS
            ns_q_local = arange_ns_q - ns_q_user * L_NS
            ns_q_to_packed = cu_seqlens_q[1:][ns_q_user].to(torch.int64) - L_NS + ns_q_local

            # s_q_gather: which positions in the previous layer's packed S
            # become the current layer's Q. For layer 0 it's identity (no
            # drop). For layer l>0, drop = q_per_user[l-1] - q_per_user[l]
            # tokens from the front of each user's segment in prev_cu_s.
            if l == 0:
                s_q_gather = arange_s_q
            else:
                prev_q_per_user = q_per_user_all[l - 1]
                drop = (prev_q_per_user - q_per_user).to(torch.int64)        # [B]
                s_q_gather = (
                    prev_cu_s[s_q_user].to(torch.int64) + drop[s_q_user] + s_q_local
                )

            layers.append({
                "cu_seqlens_q": cu_seqlens_q,
                "max_seqlen_q": max_seqlen_q,
                "s_q_to_packed": s_q_to_packed,
                "ns_q_to_packed": ns_q_to_packed,
                "s_q_gather": s_q_gather,
                "cu_seqlens_k": prev_cu_seqlens,
                "max_seqlen_k": prev_max_seqlen,
                "s_k_to_packed": prev_s_to_packed,
                "ns_k_to_packed": prev_ns_to_packed,
                # Cached for the next layer / s_repr.
                "s_q_user": s_q_user,
                "q_per_user": q_per_user,
            })

            # Roll forward: this layer's Q becomes next layer's K.
            prev_cu_s = cu_s_q
            prev_cu_seqlens = cu_seqlens_q
            prev_s_to_packed = s_q_to_packed
            prev_ns_to_packed = ns_q_to_packed
            prev_max_seqlen = max_seqlen_q

        return layers

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if self._pipeline_mode:  # concrete bool; FX traces only the active branch
            embs = self._lookup_pipeline(batch)
        else:
            embs = self._lookup_all(batch)
        h, s_repr = self._backbone(embs, batch)
        preds = {task: head(h).squeeze(-1) for task, head in self.heads.items()}
        return preds

    def forward_with_cross_scores(
        self, batch: dict[str, Tensor], cross_task: str = "",
    ) -> tuple[dict[str, Tensor], Tensor]:
        embs = self._lookup_all(batch)
        h, s_repr = self._backbone(embs, batch)
        preds = {task: head(h).squeeze(-1) for task, head in self.heads.items()}

        user_emb = self.contrastive_user_proj(s_repr)
        item_raw = torch.cat([embs[f] for f in self._item_scalar_names], dim=-1)
        item_emb = self.contrastive_item_proj(item_raw)

        cross_scores = user_emb @ item_emb.T
        return preds, cross_scores
