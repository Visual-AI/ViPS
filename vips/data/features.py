from __future__ import annotations

from pathlib import Path

import torch

from vips.modeling.qwen3vl import MODEL_LAYERS


FEATURE_DIRECTORIES = {
    "vggt": "vggt_features",
    "da3": "da3_features",
    "ta": "ta_features",
    "radio": "radio_features",
    "wan": "wan_features",
}


def scene_from_video(video: str) -> tuple[str, str]:
    parts = video.replace("\\", "/").split("/")
    source = parts[0]
    return source, Path(parts[-1]).stem


def load_training_features(
    feature_root: Path,
    video: str,
    model_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    source, scene = scene_from_video(video)
    result = {}
    for model_name in model_names:
        layer = MODEL_LAYERS[model_name]
        filename = f"wan_t2v_{layer}.pt" if model_name == "wan" else f"{model_name}_{layer}.pt"
        path = feature_root / source / FEATURE_DIRECTORIES[model_name] / scene / filename
        result[model_name] = {
            f"feature_full_{layer}": torch.load(path, map_location="cpu", mmap=True)
        }
    return result
