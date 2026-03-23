"""Interaction modules: ConcatMLP, DotInteraction, DCNv2."""
from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


def _make_mlp(dims: list[int], dropout: float = 0.1, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class InteractionModule(nn.Module):
    """Base class for feature interaction modules."""

    @abstractmethod
    def forward(
        self,
        user_features: list[torch.Tensor],
        item_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            user_features: list of [B, D_i] tensors.
            item_features: list of [B, D_j] tensors.

        Returns:
            [B, D_out] interaction representation.
        """
        ...


class ConcatMLP(InteractionModule):
    """Concatenate all features and pass through MLP."""

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()
        self.mlp = _make_mlp([input_dim] + hidden_dims, dropout, final_activation=True)
        self.output_dim = hidden_dims[-1]

    def forward(
        self,
        user_features: list[torch.Tensor],
        item_features: list[torch.Tensor],
    ) -> torch.Tensor:
        x = torch.cat(user_features + item_features, dim=-1)
        return self.mlp(x)


class DotInteraction(InteractionModule):
    """DLRM-style pairwise dot product interaction.

    Projects all features to a common dimension before computing pairwise
    dot products, so it handles heterogeneous feature dims correctly.
    """

    def __init__(self, feature_dims: list[int], top_mlp_dims: list[int], dropout: float = 0.1):
        super().__init__()
        common_dim = max(feature_dims)
        self.projections = nn.ModuleList([
            nn.Linear(d, common_dim) if d != common_dim else nn.Identity()
            for d in feature_dims
        ])
        num_features = len(feature_dims)
        num_pairs = num_features * (num_features - 1) // 2
        mlp_input = num_pairs + num_features * common_dim
        self.top_mlp = _make_mlp([mlp_input] + top_mlp_dims, dropout, final_activation=True)
        self.output_dim = top_mlp_dims[-1]
        self.common_dim = common_dim

    def forward(
        self,
        user_features: list[torch.Tensor],
        item_features: list[torch.Tensor],
    ) -> torch.Tensor:
        all_features = user_features + item_features
        projected = [proj(f) for proj, f in zip(self.projections, all_features)]
        stacked = torch.stack(projected, dim=1)  # [B, K, common_dim]

        dots = torch.bmm(stacked, stacked.transpose(1, 2))  # [B, K, K]
        B, K, _ = dots.shape
        idx = torch.triu_indices(K, K, offset=1, device=dots.device)
        pairwise = dots[:, idx[0], idx[1]]  # [B, num_pairs]

        flat_features = torch.cat(projected, dim=-1)  # [B, K*common_dim]
        combined = torch.cat([pairwise, flat_features], dim=-1)
        return self.top_mlp(combined)


class DCNv2(InteractionModule):
    """Deep & Cross Network v2."""

    def __init__(
        self,
        input_dim: int,
        num_cross_layers: int = 3,
        deep_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cross_layers = num_cross_layers

        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01)
            for _ in range(num_cross_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_cross_layers)
        ])

        if deep_dims is None:
            deep_dims = [input_dim, input_dim]
        self.deep_mlp = _make_mlp([input_dim] + deep_dims, dropout, final_activation=True)

        self.combine = nn.Linear(input_dim + deep_dims[-1], deep_dims[-1])
        self.output_dim = deep_dims[-1]

    def forward(
        self,
        user_features: list[torch.Tensor],
        item_features: list[torch.Tensor],
    ) -> torch.Tensor:
        x0 = torch.cat(user_features + item_features, dim=-1)  # [B, D_total]

        xl = x0
        for w, b in zip(self.cross_weights, self.cross_biases):
            xl = x0 * (xl @ w + b) + xl  # [B, D_total]

        deep = self.deep_mlp(x0)
        combined = torch.cat([xl, deep], dim=-1)
        return self.combine(combined)


def build_interaction(
    interaction_type: str,
    user_dims: list[int],
    item_dims: list[int],
    top_mlp_dims: list[int],
    dcn_num_cross_layers: int = 3,
    dropout: float = 0.1,
) -> InteractionModule:
    """Factory for interaction modules."""
    total_dim = sum(user_dims) + sum(item_dims)

    if interaction_type == "concat_mlp":
        return ConcatMLP(total_dim, top_mlp_dims, dropout)
    elif interaction_type == "dot":
        feature_dims = user_dims + item_dims
        return DotInteraction(feature_dims, top_mlp_dims, dropout)
    elif interaction_type == "dcnv2":
        return DCNv2(total_dim, dcn_num_cross_layers, top_mlp_dims, dropout)
    else:
        raise ValueError(f"Unknown interaction type: {interaction_type}")
