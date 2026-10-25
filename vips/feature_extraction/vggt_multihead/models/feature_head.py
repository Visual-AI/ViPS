# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the VGGT license found in
# vips/feature_extraction/vggt_multihead/LICENSE.
# This file has been modified for the ViPS distribution.

import torch
import torch.nn as nn


# Model name -> output feature dimension (channel) of the last-layer GT feature.
FEATURE_HEAD_DIMS = {
    "vggt": 2048,
    "da3": 1536,
    "radio": 1280,
    "ta": 1024,
    "wan": 1536,
}


class FeatureMLPHead(nn.Module):
    """
    A small fully-connected head that maps the shared backbone feature
    (dim_in, from the VGGT aggregator tokens) to a target model's feature
    dimension (output_dim).

    Operates on *per-token channel* feature maps of shape [..., D_in] and
    produces [..., D_out]. The spatial tokens (15x20=300) are already
    flattened by the caller, so this head is simply per-pixel MLP.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        hidden_dim: int = None,
        n_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(256, dim_in)
        if n_layers == 1:
            layers = [nn.Linear(dim_in, dim_out)]
        elif n_layers >= 2:
            layers = [nn.Linear(dim_in, hidden_dim), self._act(activation)]
            for _ in range(n_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), self._act(activation)]
            layers += [nn.Linear(hidden_dim, dim_out)]
        else:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        self.mlp = nn.Sequential(*layers)
        self.dim_out = dim_out

    @staticmethod
    def _act(name: str):
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [*, D_in] -> [*, D_out]
        return self.mlp(x)


class FeatureHeads(nn.Module):
    """
    Container holding one FeatureMLPHead per target model (vggt/da3/radio/ta/wan).
    All heads share the same input feature (a single backbone feature), and each
    is supervised by its own model's last-layer GT feature.
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = None,
        n_layers: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.dim_in = dim_in
        self.heads = nn.ModuleDict(
            {
                name: FeatureMLPHead(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    hidden_dim=hidden_dim,
                    n_layers=n_layers,
                    activation=activation,
                )
                for name, dim_out in FEATURE_HEAD_DIMS.items()
            }
        )

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: shared backbone feature, shape [B, S, P, dim_in]
        Returns:
            dict of {name: [B, S, P, dim_out]}
        """
        return {name: head(feat) for name, head in self.heads.named_children()}