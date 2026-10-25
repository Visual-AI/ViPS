# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# vips/feature_extraction/vggt_multihead/LICENSE.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import Aggregator
from .feature_head import FeatureHeads


class VGGT(nn.Module):
    """VGGT aggregator with five distilled feature heads.

    This inference-only variant intentionally omits the camera, depth, point,
    and tracking heads because VIPS only consumes the distilled features.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        feature_grid_hw=(15, 20),
        feature_head_hidden_dim=4096,
        feature_head_n_layers=3,
        feature_head_activation="relu",
    ):
        super().__init__()
        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.feature_grid_hw = tuple(feature_grid_hw)
        self.feature_heads = FeatureHeads(
            dim_in=2 * embed_dim,
            hidden_dim=feature_head_hidden_dim,
            n_layers=feature_head_n_layers,
            activation=feature_head_activation,
        )

    def _extract_patch_tokens(self, aggregated_tokens, images, patch_start_idx):
        batch, frames, _, height, width = images.shape
        feature = aggregated_tokens[:, :, patch_start_idx:, :]
        feature = feature.view(
            batch,
            frames,
            height // self.aggregator.patch_size,
            width // self.aggregator.patch_size,
            -1,
        )
        feature = feature.permute(0, 1, 4, 2, 3).contiguous()
        grid_h, grid_w = self.feature_grid_hw
        feature = F.adaptive_avg_pool2d(
            feature.reshape(batch * frames, -1, feature.shape[-2], feature.shape[-1]),
            (grid_h, grid_w),
        )
        return feature.reshape(batch, frames, -1, grid_h * grid_w).permute(0, 1, 3, 2)

    def forward(self, images: torch.Tensor):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        aggregated_tokens, patch_start_idx = self.aggregator(images)
        shared_feature = self._extract_patch_tokens(
            aggregated_tokens[-1], images, patch_start_idx
        )
        predictions = {"backbone_feature": shared_feature}
        predictions.update(
            {
                f"features_{name}": feature
                for name, feature in self.feature_heads(shared_feature).items()
            }
        )
        return predictions
