from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FlattenedCollator:
    """The flattened Qwen3-VL collator used by ViPS training."""

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [0] + [length for instance in instances for length in instance["attention_mask"]]
        batch = {
            "input_ids": torch.cat([item["input_ids"] for item in instances], dim=1),
            "labels": torch.cat([item["labels"] for item in instances], dim=1),
            "attention_mask": torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0),
            "position_ids": torch.cat([item["position_ids"] for item in instances], dim=2),
            "pixel_values": None,
            "image_grid_thw": None,
            "pixel_values_videos": torch.cat(
                [item["pixel_values_videos"] for item in instances], dim=0
            ),
            "video_grid_thw": torch.cat(
                [item["video_grid_thw"] for item in instances], dim=0
            ),
            "question_positions": [item["question_position"] for item in instances],
        }
        features = [item["foundation_features"] for item in instances if "foundation_features" in item]
        if features:
            batch["foundation_features"] = features
        return batch
