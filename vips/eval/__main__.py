from __future__ import annotations

import argparse
import os
import sys
import types

from vips.checkpoints import check_eval_checkpoint
from vips.config import load_config, repository_root, resolve_repository_path

# Importing the class registers it with newer lmms-eval releases.
from .lmms_model import VIPSQwen3VL


def _register_with_legacy_lmms_eval() -> None:
    """Expose the model to the static registry used by the bundled lmms-eval fork."""
    import lmms_eval.models as lmms_models

    if not hasattr(lmms_models, "AVAILABLE_MODELS"):
        return

    module_name = "lmms_eval.models.vips_qwen3vl"
    module = types.ModuleType(module_name)
    module.VIPSQwen3VL = VIPSQwen3VL
    sys.modules[module_name] = module
    lmms_models.AVAILABLE_MODELS["vips_qwen3vl"] = "VIPSQwen3VL"


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args, lmms_args = parser.parse_known_args()
    config = load_config(args.config)
    checkpoint = resolve_repository_path(
        args.checkpoint or config["training"]["output_dir"]
    )
    if not {"-h", "--help"}.intersection(lmms_args):
        check_eval_checkpoint(checkpoint)
    video_root = resolve_repository_path(config["evaluation"]["videos"])
    feature_root = resolve_repository_path(config["evaluation"]["features"])
    output_dir = resolve_repository_path(config["evaluation"]["output_dir"])
    task_dir = repository_root() / "vips" / "eval" / "tasks"
    os.environ["VIPS_VSIBENCH_VIDEO_ROOT"] = str(video_root)
    _register_with_legacy_lmms_eval()

    model_args = [
        f"pretrained={config['model']['name_or_path']}",
        f"checkpoint={checkpoint}",
        f"config={resolve_repository_path(args.config)}",
        f"video_root={video_root}",
        f"attn_implementation={config['model']['attention']}",
    ]
    if feature_root is not None:
        model_args.append(f"feature_root={feature_root}")

    sys.argv = [
        "lmms_eval",
        "--model", "vips_qwen3vl",
        "--model_args", ",".join(model_args),
        "--tasks", config["evaluation"]["task"],
        "--batch_size", str(config["evaluation"]["batch_size"]),
        "--include_path", str(task_dir),
        "--log_samples",
        "--log_samples_suffix", config["name"],
        "--output_path", str(output_dir),
        *lmms_args,
    ]
    from lmms_eval.__main__ import cli_evaluate

    cli_evaluate()


if __name__ == "__main__":
    main()
