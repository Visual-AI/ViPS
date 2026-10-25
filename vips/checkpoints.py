"""Checks for the final adapter and non-LoRA weights saved by ViPS training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


NON_LORA_WEIGHTS = "non_lora_trainables.bin"


def check_training_output_dir(output_dir: Path) -> None:
    """Allow logs in an output directory, but never reuse saved training weights."""
    patterns = (
        "checkpoint-*",
        "adapter_model*",
        "adapter_config.json",
        NON_LORA_WEIGHTS,
        "pytorch_model*",
        "model*.safetensors*",
        "trainer_state.json",
    )
    existing = sorted({path.name for pattern in patterns for path in output_dir.glob(pattern)})
    if existing:
        raise FileExistsError(
            f"Training output {output_dir} already contains: {', '.join(existing)}. "
            "Resuming training is not supported in this release. "
            "Choose a new directory with --output-dir; existing weights will not be overwritten."
        )


def check_eval_checkpoint(checkpoint: Path) -> None:
    """Reject incomplete local checkpoints before loading the base model."""
    missing = [
        name for name in ("adapter_config.json", NON_LORA_WEIGHTS)
        if not (checkpoint / name).is_file()
    ]
    if not any(
        (checkpoint / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        missing.append("adapter_model.safetensors (or adapter_model.bin)")
    if missing:
        raise FileNotFoundError(
            f"Incomplete checkpoint at {checkpoint}. Missing: {', '.join(missing)}. "
            "Use the final training output directory; intermediate checkpoint-* "
            "directories do not contain all weights required for evaluation."
        )


def load_non_lora_trainables(
    model: nn.Module, checkpoint: Path, *, tune_merger: bool
) -> None:
    """Validate and restore non-LoRA weights on a model after merging its adapter."""
    path = checkpoint / NON_LORA_WEIGHTS
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict):
        raise ValueError(f"Expected a parameter dictionary in {path}")

    state = {}
    for key, value in saved.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected named tensors in {path}; invalid entry: {key!r}")
        name = key.removeprefix("base_model.model.")
        if name in state:
            raise ValueError(f"Duplicate non-LoRA parameter in {path}: {name}")
        state[name] = value

    # These modules are trained outside PEFT's adapter state. Frozen backbone
    # parameters are intentionally absent from this partial state dictionary.
    required = {"lm_head.weight"}
    modules = ["model.language_model.zero_conv_dict", "model.language_model.zero_routers"]
    if tune_merger:
        modules.append("model.visual.merger")
    for prefix in modules:
        required.update(
            f"{prefix}.{name}" for name, _ in model.get_submodule(prefix).named_parameters()
        )

    model_state = model.state_dict()
    missing = sorted(required - state.keys())
    unexpected = sorted(state.keys() - model_state.keys())
    mismatched = [
        f"{name}: saved {tuple(state[name].shape)}, expected {tuple(model_state[name].shape)}"
        for name in sorted(state.keys() & model_state.keys())
        if state[name].shape != model_state[name].shape
    ]
    problems = []
    if missing:
        problems.append("Missing required parameters: " + ", ".join(missing))
    if unexpected:
        problems.append("Unexpected parameters: " + ", ".join(unexpected))
    if mismatched:
        problems.append("Shape mismatches: " + "; ".join(mismatched))
    if problems:
        raise ValueError(
            f"Invalid non-LoRA weights in {path}. Check the checkpoint and evaluation config.\n"
            + "\n".join(problems)
        )

    model.load_state_dict(state, strict=False)
