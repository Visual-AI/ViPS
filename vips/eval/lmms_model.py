from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import decord
import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from decord import VideoReader, cpu
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from loguru import logger
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.integrations import tensor_parallel as tensor_parallel

from vips.checkpoints import check_eval_checkpoint, load_non_lora_trainables
from vips.config import load_config
from vips.modeling import configure_vips, install_vips_model


for class_name in (
    "EmbeddingParallel",
    "ColwiseParallel",
    "RowwiseParallel",
    "ALL_PARALLEL_STYLES",
):
    if not hasattr(tensor_parallel, class_name):
        setattr(tensor_parallel, class_name, type(class_name, (), {}))


MODEL_LAYERS = {"vggt": 23, "da3": 39, "ta": 24, "radio": 31, "wan": 29}
MODEL_DIRECTORIES = {
    "vggt": "vggt",
    "da3": "da3",
    "ta": "ta",
    "radio": "radio",
    "wan": "want2v",
}


def _video_frames(video_path: Path, num_frames: int = 32) -> tuple[list[Image.Image], dict]:
    decord.bridge.set_bridge("torch")
    reader = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(reader.get_avg_fps())
    indices = np.linspace(0, len(reader) - 1, num_frames, dtype=int).tolist()
    frames = [Image.fromarray(frame) for frame in reader.get_batch(indices).numpy()]
    metadata = {
        "total_num_frames": len(reader),
        "fps": fps,
        "duration": len(reader) / fps if fps > 0 else None,
        "frames_indices": indices,
    }
    return frames, metadata


def _feature_filename(model_name: str) -> str:
    prefix = "wan_t2v" if model_name == "wan" else model_name
    return f"{prefix}_{MODEL_LAYERS[model_name]}.pt"


def _load_features(
    video_path: Path,
    video_root: Path,
    feature_root: Path,
    model_names: list[str],
) -> dict[str, dict[str, torch.Tensor]]:
    relative = video_path.resolve().relative_to(video_root.resolve())
    source, scene = relative.parent, relative.stem
    features = {}
    for model_name in model_names:
        path = (
            feature_root
            / MODEL_DIRECTORIES[model_name]
            / source
            / scene
            / _feature_filename(model_name)
        )
        features[model_name] = {
            f"feature_full_{MODEL_LAYERS[model_name]}": torch.load(
                path, map_location="cpu", mmap=True
            )
        }
    return features


def _move_to_device(values: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


@register_model("vips_qwen3vl")
class VIPSQwen3VL(lmms):
    def __init__(
        self,
        pretrained: str,
        checkpoint: str,
        config: str,
        video_root: str,
        feature_root: str = "",
        batch_size: int | str = 1,
        attn_implementation: str = "flash_attention_2",
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        check_eval_checkpoint(Path(checkpoint))
        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))]
        )
        # LMMS-Eval uses this object to gather requests and synchronize ranks.
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device("cuda", accelerator.local_process_index)
        else:
            self._device = torch.device(device)
        # Each evaluation process runs a complete model on its selected device.
        device_map = str(self._device)

        self.settings = load_config(config)
        self.video_root = Path(video_root)
        self.feature_root = Path(feature_root) if feature_root else None
        self.model_names = list(self.settings["vips"]["prior_models"])
        self.use_vips = bool(self.settings["vips"]["enabled"])

        install_vips_model()
        model_config = configure_vips(AutoConfig.from_pretrained(pretrained), self.settings)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained,
            config=model_config,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )
        model = model.cpu()
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
        load_non_lora_trainables(
            model, Path(checkpoint), tune_merger=self.settings["training"]["tune_merger"]
        )
        self._model = model.to(self._device).eval()
        self._processor = AutoProcessor.from_pretrained(pretrained)
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self._rank = accelerator.process_index
        self._world_size = accelerator.num_processes

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._processor.tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return getattr(self._config, "max_position_embeddings", 32768)

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        responses = []
        progress = tqdm(
            total=len(requests),
            disable=self.rank != 0,
            desc="VSIBench",
        )
        for request in requests:
            context, generation_kwargs, doc_to_visual, doc_id, task, split = request.args
            generation_kwargs = dict(generation_kwargs)
            # Keep the original EOS/token-limit stopping behavior. `until` is
            # an lmms-eval option, not a transformers.generate() argument.
            generation_kwargs.pop("until", None)
            # Evaluation defaults also apply when task/CLI options are omitted.
            generation_kwargs.setdefault("do_sample", False)
            generation_kwargs.setdefault("num_beams", 1)
            generation_kwargs.setdefault("max_new_tokens", 128)
            if generation_kwargs["do_sample"] is False:
                # Sampling-only parameters are unused by greedy decoding.
                generation_kwargs.update(temperature=None, top_p=None, top_k=None)
            if self.rank == 0 and not responses:
                effective, _ = self.model._prepare_generation_config(None, **generation_kwargs)
                logger.info(
                    "Decoding: mode={}, do_sample={}, num_beams={}, max_new_tokens={}, "
                    "temperature={}, top_p={}, top_k={}",
                    effective.get_generation_mode().value,
                    effective.do_sample,
                    effective.num_beams,
                    effective.max_new_tokens,
                    effective.temperature,
                    effective.top_p,
                    effective.top_k,
                )
            document = self.task_dict[task][split][doc_id]
            visuals = doc_to_visual(document)
            while len(visuals) == 1 and isinstance(visuals[0], list):
                visuals = visuals[0]
            if len(visuals) != 1:
                raise ValueError(f"Expected one video, received {len(visuals)}")
            video_path = Path(visuals[0])
            frames, metadata = _video_frames(video_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": context},
                ],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                video_metadata=[metadata],
                do_sample_frames=False,
            )
            inputs = _move_to_device(inputs, self.device)
            if self.use_vips:
                if self.feature_root is None:
                    raise ValueError("VIPS evaluation requires feature_root")
                inputs["foundation_features"] = [
                    _load_features(
                        video_path,
                        self.video_root,
                        self.feature_root,
                        self.model_names,
                    )
                ]
                inputs["question_positions"] = None
            with torch.inference_mode():
                generated = self.model.generate(**inputs, **generation_kwargs)
            generated = [
                output[len(input_ids) :]
                for input_ids, output in zip(inputs["input_ids"], generated)
            ]
            response = self._processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            responses.append(response)
            progress.update(1)
        progress.close()
        return responses
