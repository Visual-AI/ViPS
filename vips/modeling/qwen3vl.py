"""Configuration helpers for the vendored Qwen3-VL ZeroDistill model."""

from __future__ import annotations

from typing import Any

from transformers.models.qwen3_vl import modeling_qwen3_vl


MODEL_LAYERS = {
    "vggt": 23,
    "da3": 39,
    "ta": 24,
    "radio": 31,
    "wan": 29,
}


def configure_vips(model_config: Any, config: dict[str, Any]) -> Any:
    settings = config["vips"]
    text_config = model_config.text_config
    text_config.enable_zero_distill = bool(settings["enabled"])
    text_config.prior_model_list = ",".join(settings["prior_models"])
    text_config.target_layer_ids = ",".join(
        str(layer) for layer in settings["target_layers"]
    )
    return model_config


def install_vips_model() -> None:
    """Ensure the bundled Transformers fork with ZeroDistill is active."""

    if not hasattr(modeling_qwen3_vl, "ZeroDistillBlockV14"):
        raise RuntimeError(
            "VIPS requires the bundled Transformers fork. "
            "Run through scripts/train.sh or scripts/eval.sh."
        )
