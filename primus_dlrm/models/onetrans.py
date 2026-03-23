"""OneTrans: unified Transformer for sequence modeling + feature interaction.

Uses split-history pools (listen+, like, skip) with per-behavior MLP tokenizers.
Supports optional counter features and in-batch contrastive loss.
All categorical embeddings backed by TorchRec EmbeddingCollection.
"""
from __future__ import annotations

import math

import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from primus_dlrm.config import ModelConfig
from primus_dlrm.models.base import BaseModel
from primus_dlrm.models.embedding import TableSpec, TorchRecEmbeddings


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


@torch.fx.wrap
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

    def __init__(self, d_model: int, n_heads: int, n_ns_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_ns = n_ns_tokens

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

    def forward(self, x: Tensor, n_s: int, mask: Tensor | None = None) -> Tensor:
        B, T, _ = x.shape
        s_part = x[:, :n_s]
        ns_part = x[:, n_s:]

        s_qkv = self.s_qkv(s_part).reshape(B, n_s, 3, self.n_heads, self.d_head)
        s_q, s_k, s_v = s_qkv.unbind(2)

        ns_qkv = torch.einsum("btn,tmn->btm", ns_part, self.ns_qkv_weight)
        ns_qkv = ns_qkv.reshape(B, self.n_ns, 3, self.n_heads, self.d_head)
        ns_q, ns_k, ns_v = ns_qkv.unbind(2)

        q = torch.cat([s_q, ns_q], dim=1).transpose(1, 2)
        k = torch.cat([s_k, ns_k], dim=1).transpose(1, 2)
        v = torch.cat([s_v, ns_v], dim=1).transpose(1, 2)

        if mask is not None:
            q_len = mask.shape[0]
            if q_len < T:
                q = q[:, :, T - q_len:]

        attn_mask = mask.unsqueeze(0).unsqueeze(0) if mask is not None else None
        drop_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_p)

        out = out.transpose(1, 2).reshape(B, out.shape[2], self.d_model)
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
                 ffn_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MixedAttention(d_model, n_heads, n_ns_tokens, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MixedFFN(d_model, d_model * ffn_mult, n_ns_tokens, dropout)

    def forward(self, x: Tensor, mask: Tensor, q_len: int, n_s_tokens: int) -> Tensor:
        kv_len = x.shape[1]
        n_ns = kv_len - n_s_tokens

        normed = self.norm1(x)
        attn_out = self.attn(normed, n_s=n_s_tokens, mask=mask)
        x_tail = x[:, kv_len - q_len:]
        z = attn_out + x_tail

        n_s_out = q_len - n_ns
        ffn_out = self.ffn(self.norm2(z), n_s=n_s_out)
        return z + ffn_out


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


# ---------------------------------------------------------------------------
# Table specs
# ---------------------------------------------------------------------------

def _onetrans_table_specs(
    num_users: int, num_items: int, num_artists: int, num_albums: int,
    embedding_dim: int,
) -> list[TableSpec]:
    """Table specs for OneTrans: all unpooled, shared tables for hist + candidate."""
    return [
        TableSpec("item", num_items, embedding_dim, "none",
                  ["item", "hist_lp_item", "hist_like_item", "hist_skip_item"]),
        TableSpec("artist", num_artists, embedding_dim, "none",
                  ["artist", "hist_lp_artist", "hist_like_artist", "hist_skip_artist"]),
        TableSpec("album", num_albums, embedding_dim, "none",
                  ["album", "hist_lp_album", "hist_like_album", "hist_skip_album"]),
        TableSpec("uid", num_users, embedding_dim, "none", ["uid"]),
    ]


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class OneTransModel(BaseModel):
    """OneTrans: unified Transformer for ranking.

    Uses split-history pools with per-behavior MLP tokenizers.
    All categorical embeddings backed by TorchRec EmbeddingCollection.

    Pipeline mode (``_pipeline_mode=True``): forward() passes a pre-built KJT
    directly to ``emb.ec``, giving FX a traceable getitem→call_module path so
    TrainPipelineSparseDist can pipeline the embedding all-to-all.
    """

    BEHAVIOR_POOLS = ("hist_lp", "hist_like", "hist_skip")

    def __init__(
        self,
        config: ModelConfig,
        num_users: int,
        num_items: int,
        num_artists: int,
        num_albums: int,
        audio_input_dim: int = 256,
        device: torch.device | None = None,
        tasks: list[str] | None = None,
        num_counter_windows: int = 0,
        meta_device: bool = False,
    ):
        super().__init__()
        self._pipeline_mode = False
        self.config = config
        self.tasks = tasks or ["listen_plus"]
        self.num_counter_windows = num_counter_windows
        ot = config.onetrans
        D = config.embedding_dim

        if device is None:
            device = torch.device("cpu")

        emb_device = torch.device("meta") if meta_device else device
        self.emb = TorchRecEmbeddings(
            _onetrans_table_specs(num_users, num_items, num_artists, num_albums, D),
            device=emb_device,
            embedding_init=config.embedding_init,
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_input_dim, D, device=device),
            nn.GELU(),
        )

        # Per-behavior sequential tokenizers (separate MLPs, shared embeddings)
        s_raw_dim = 3 * D  # item + artist + album per position
        self.seq_tokenizers = nn.ModuleDict({
            pool: SequentialTokenizer(
                s_raw_dim, ot.d_model, max_len=2048, use_pos_embed=ot.pos_embed,
            ).to(device)
            for pool in self.BEHAVIOR_POOLS
        })

        # NS-token tokenizer: base features + optional counters
        ns_raw_dim = 5 * D
        if num_counter_windows > 0:
            W = num_counter_windows
            self.cross_proj = nn.Sequential(
                nn.Linear(9 * W, D, device=device),
                nn.ReLU(inplace=True),
            )
            ns_raw_dim += 3 * W + 3 * W + D

        self.ns_tokenizer = AutoSplitTokenizer(
            ns_raw_dim, ot.d_model, ot.n_ns_tokens,
        ).to(device)

        # Transformer stack
        self.blocks = nn.ModuleList([
            OneTransBlock(ot.d_model, ot.n_heads, ot.n_ns_tokens, ot.ffn_mult, ot.dropout)
            for _ in range(ot.n_layers)
        ]).to(device)
        self.final_norm = RMSNorm(ot.d_model).to(device)

        # Head bridge: concat NS-tokens -> projection -> task heads
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

        # Lightweight contrastive head
        self.contrastive_user_proj = nn.Linear(ot.d_model, ot.d_model, device=device)
        self.contrastive_item_proj = nn.Sequential(
            nn.Linear(3 * D, ot.d_model, device=device),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _lookup_all(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Single batched TorchRec lookup for all features."""
        return self.emb({
            "uid": batch["uid"],
            "item": batch["item_id"],
            "artist": batch["artist_id"],
            "album": batch["album_id"],
            "hist_lp_item": batch["hist_lp_item_ids"],
            "hist_like_item": batch["hist_like_item_ids"],
            "hist_skip_item": batch["hist_skip_item_ids"],
            "hist_lp_artist": batch["hist_lp_artist_ids"],
            "hist_like_artist": batch["hist_like_artist_ids"],
            "hist_skip_artist": batch["hist_skip_artist_ids"],
            "hist_lp_album": batch["hist_lp_album_ids"],
            "hist_like_album": batch["hist_like_album_ids"],
            "hist_skip_album": batch["hist_skip_album_ids"],
        })

    def _lookup_pipeline(self, batch: dict) -> dict[str, Tensor]:
        """Embedding lookup via pre-built KJT (pipeline mode).

        Gives FX a clean ``batch["unpooled_kjt"] → emb.ec`` path so that
        TrainPipelineSparseDist can pipeline the embedding all-to-all.
        """
        ec_out = self.emb.ec(batch["unpooled_kjt"])
        B = batch["uid"].shape[0]
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
        self, embs: dict[str, Tensor], prefix: str,
    ) -> Tensor:
        """Build raw feature tensor for one behavior pool: [B, L, 3D]."""
        return torch.cat([
            embs[f"{prefix}_item"],
            embs[f"{prefix}_artist"],
            embs[f"{prefix}_album"],
        ], dim=-1)

    def _build_ns_raw(
        self, embs: dict[str, Tensor], batch: dict[str, Tensor],
    ) -> Tensor:
        """Build the NS raw feature vector, including optional counters."""
        parts = [
            embs["uid"],
            embs["item"],
            embs["artist"],
            embs["album"],
            self.audio_proj(batch["audio_embed"]),
        ]
        if self.num_counter_windows > 0:
            parts.append(batch["user_counters"])
            parts.append(batch["item_counters"])
            parts.append(self.cross_proj(batch["cross_counters"]))
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
        ot = self.config.onetrans
        B = batch["uid"].shape[0]
        L_NS = ot.n_ns_tokens

        # S-token construction: per-behavior tokenization
        pool_tokens = []
        for pool in self.BEHAVIOR_POOLS:
            raw = self._build_pool_raw(embs, pool)
            pool_tokens.append(self.seq_tokenizers[pool](raw))
        s_tokens = torch.cat(pool_tokens, dim=1)
        L_S = s_tokens.shape[1]

        # NS-token construction
        ns_raw = self._build_ns_raw(embs, batch)
        ns_tokens = self.ns_tokenizer(ns_raw)

        # Concatenate: [S-tokens ; NS-tokens]
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
        self, batch: dict[str, Tensor], cross_task: str = "listen_plus",
    ) -> tuple[dict[str, Tensor], Tensor]:
        # See BaseModel.forward_with_cross_scores for why this is separate.
        embs = self._lookup_all(batch)
        h, s_repr = self._backbone(embs, batch)
        preds = {task: head(h).squeeze(-1) for task, head in self.heads.items()}

        user_emb = self.contrastive_user_proj(s_repr)
        item_raw = torch.cat([
            embs["item"], embs["artist"], embs["album"],
        ], dim=-1)
        item_emb = self.contrastive_item_proj(item_raw)

        cross_scores = user_emb @ item_emb.T
        return preds, cross_scores
