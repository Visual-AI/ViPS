from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

import torch

from vips.config import load_config, repository_root, resolve_repository_path

from .preprocessing import load_video_frames, preprocess_pil_frames
from .vggt_multihead.models import VGGT
from .vggt_multihead.utils import load_and_preprocess_images


MODEL_LAYERS = {"vggt": 23, "da3": 39, "ta": 24, "radio": 31, "wan": 29}
TRAIN_DIRECTORIES = {
    "vggt": "vggt_features",
    "da3": "da3_features",
    "ta": "ta_features",
    "radio": "radio_features",
    "wan": "wan_features",
}
EVAL_DIRECTORIES = {
    "vggt": "vggt",
    "da3": "da3",
    "ta": "ta",
    "radio": "radio",
    "wan": "want2v",
}


def feature_filename(model_name: str) -> str:
    prefix = "wan_t2v" if model_name == "wan" else model_name
    return f"{prefix}_{MODEL_LAYERS[model_name]}.pt"


def training_scenes(root: Path) -> Iterator[tuple[str, str, Path]]:
    for subset in sorted(root.iterdir()):
        videos = subset / "videos"
        if not videos.is_dir():
            continue
        for scene_dir in sorted(videos.iterdir()):
            if not scene_dir.is_dir():
                continue
            frames = sorted(
                path
                for path in scene_dir.iterdir()
                if path.suffix in {".jpg", ".png"}
            )
            if len(frames) == 32:
                yield subset.name, scene_dir.name, scene_dir


def vsibench_scenes(root: Path) -> Iterator[tuple[str, str, Path]]:
    for video_path in sorted(root.rglob("*.mp4")):
        relative = video_path.relative_to(root)
        yield str(relative.parent), video_path.stem, video_path


def load_model(checkpoint: Path, device: torch.device) -> VGGT:
    model = VGGT(feature_head_hidden_dim=4096, feature_head_n_layers=3)
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint_data.get("model", checkpoint_data)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    return model.to(device=device).eval()


def save_features(
    predictions: dict[str, torch.Tensor],
    output_root: Path,
    scene: str,
    relative_directories: dict[str, Path],
) -> None:
    for model_name, relative_directory in relative_directories.items():
        output_dir = output_root / relative_directory / scene
        output_dir.mkdir(parents=True, exist_ok=True)
        feature = predictions[f"features_{model_name}"].float().cpu().contiguous()
        if feature.ndim == 3:
            feature = feature.unsqueeze(0)
        torch.save(feature.to(torch.bfloat16), output_dir / feature_filename(model_name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("split", choices=("train", "vsibench"), nargs="?", default="vsibench")
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument(
        "--checkpoint",
        default=str(repository_root() / "checkpoints" / "vggt_multihead.pt"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)

    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-MultiHead feature extraction requires a CUDA GPU")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    capability = torch.cuda.get_device_capability(device)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16

    section = config["data"] if args.split == "train" else config["evaluation"]
    default_input = section["videos"]
    default_output = section["features"]
    if default_output is None and args.output_root is None:
        raise ValueError(f"{config['name']} does not define a feature output directory")
    input_root = resolve_repository_path(args.input_root or default_input)
    output_root = resolve_repository_path(args.output_root or default_output)
    checkpoint = resolve_repository_path(args.checkpoint)
    model = load_model(checkpoint, device)

    scenes = list(training_scenes(input_root) if args.split == "train" else vsibench_scenes(input_root))
    scenes = scenes[rank::world_size]
    if args.limit is not None:
        scenes = scenes[: args.limit]
    print(f"rank {rank}/{world_size}: assigned {len(scenes)} scenes")

    failures = 0
    for index, (subset, scene, source) in enumerate(scenes, start=1):
        try:
            if args.split == "train":
                frame_paths = sorted(
                    path for path in source.iterdir() if path.suffix in {".jpg", ".png"}
                )
                images = load_and_preprocess_images([str(path) for path in frame_paths])
                relative_directories = {
                    name: Path(subset) / directory
                    for name, directory in TRAIN_DIRECTORIES.items()
                }
            else:
                images = preprocess_pil_frames(load_video_frames(source))
                relative_directories = {
                    name: Path(directory) / subset
                    for name, directory in EVAL_DIRECTORIES.items()
                }
            with torch.inference_mode():
                predictions = model(images.to(device=device, dtype=dtype).unsqueeze(0))
            save_features(predictions, output_root, scene, relative_directories)
        except Exception as error:
            failures += 1
            print(f"FAILED {subset}/{scene}: {error}", flush=True)
        if index % 10 == 0:
            print(f"rank {rank}: {index}/{len(scenes)}, failures={failures}", flush=True)

    print(f"rank {rank}: done, failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
