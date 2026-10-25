from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from vips.config import resolve_repository_path

from .features import load_training_features
from .rope import get_rope_index


IGNORE_INDEX = -100
ASSISTANT_TOKEN_ID = 77091
EOS_TOKEN_ID = 151645


@lru_cache(maxsize=1)
def _video_metadata(video_root: str) -> dict[str, dict[str, Any]]:
    with (Path(video_root) / "video_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {entry["video_path"]: entry for entry in metadata["videos"]}


def _frames(video_root: Path, video: str) -> list[Image.Image]:
    frame_dir = video_root / video.removesuffix(".mp4")
    return [Image.open(path) for path in sorted(frame_dir.glob("*.jpg"))]


def _messages(sample: dict[str, Any], video_root: Path) -> tuple[list[dict], list[dict]]:
    video = sample["video"]
    frames = _frames(video_root, video)
    metadata = _video_metadata(str(video_root))[video]
    media = [{"type": "video", "video": frames}]
    messages = []

    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        if role == "assistant":
            content = [{"type": "text", "text": turn["value"]}]
        else:
            content = []
            for segment in re.split(r"(<video>)", turn["value"]):
                if segment == "<video>":
                    content.append(media.pop(0))
                elif segment.strip():
                    content.append({"type": "text", "text": segment.strip()})
        messages.append({"role": role, "content": content})

    video_metadata = [{
        "total_num_frames": metadata["total_num_frames"],
        "fps": metadata["fps"],
        "duration": metadata["duration"],
        "frames_indices": metadata["indices"],
    }]
    return messages, video_metadata


def _labels(input_ids: torch.LongTensor) -> tuple[torch.LongTensor, int]:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    tokens = input_ids[0].tolist()
    question_position = 0
    position = 0
    while position < len(tokens):
        if tokens[position] == ASSISTANT_TOKEN_ID:
            question_position = position + 1
            answer_start = position + 2
            answer_end = answer_start
            while answer_end < len(tokens) and tokens[answer_end] != EOS_TOKEN_ID:
                answer_end += 1
            if answer_end < len(tokens):
                labels[0, answer_start : answer_end + 2] = input_ids[
                    0, answer_start : answer_end + 2
                ]
                position = answer_end
        position += 1
    return labels, question_position


class VSIBenchTrainingDataset(Dataset):
    def __init__(self, processor: Any, config: dict[str, Any]) -> None:
        self.processor = processor
        self.video_root = resolve_repository_path(config["data"]["videos"])
        self.feature_root = resolve_repository_path(config["data"]["features"])
        self.use_vips = bool(config["vips"]["enabled"])
        self.model_names = list(config["vips"]["prior_models"])

        annotation_root = resolve_repository_path(config["data"]["annotations"])
        self.samples = []
        for path in sorted(annotation_root.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                self.samples.extend(json.loads(line) for line in handle)

        image_processor = processor.image_processor
        image_processor.min_pixels = 784
        image_processor.max_pixels = 50176
        image_processor.size["shortest_edge"] = 784
        image_processor.size["longest_edge"] = 50176
        video_processor = processor.video_processor
        if hasattr(video_processor, "min_pixels"):
            video_processor.min_pixels = 4096
            video_processor.max_pixels = 25165824
        video_processor.min_frames = 4
        video_processor.max_frames = 8
        video_processor.fps = 2
        video_processor.size["shortest_edge"] = 4096
        video_processor.size["longest_edge"] = 25165824

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        messages, video_metadata = _messages(sample, self.video_root)
        result = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            video_metadata=video_metadata,
            do_sample_frames=False,
        )
        if isinstance(result["input_ids"], list):
            result["input_ids"] = torch.tensor(result["input_ids"]).unsqueeze(0)
        result["labels"], result["question_position"] = _labels(result["input_ids"])

        image_grid = result.get("image_grid_thw")
        video_grid = result.get("video_grid_thw")
        result["position_ids"] = get_rope_index(
            self.processor.image_processor.merge_size,
            result["input_ids"],
            image_grid_thw=image_grid,
            video_grid_thw=video_grid,
        )
        result["attention_mask"] = [result["input_ids"].shape[1]]

        if self.use_vips:
            result["foundation_features"] = load_training_features(
                self.feature_root, sample["video"], self.model_names
            )
        return result
