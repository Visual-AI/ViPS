import os
import sys
from datetime import timedelta
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
)

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


def _load_video_frames(video_path: str, max_frames_num: int) -> List[Image.Image]:
    """Load uniformly sampled frames from a video using decord."""
    import decord
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).numpy()
    return [Image.fromarray(f) for f in spare_frames]


def _move_to_device(inputs, device):
    """Move tensor values to device; leave non-tensors (e.g. second_per_grid_ts) unchanged."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


@register_model("vlm_qwen25vl")
class VlmQwen25VL(lmms):
    """
    Qwen2.5-VL model adapted for VSIBench evaluation (baseline, no 3D features).
    Video loading uses decord (same as vlm_3drs.py) to avoid PyAV resource issues.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda:0",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: str = "flash_attention_2",
        max_frames_num: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        self.pretrained = pretrained
        self.max_frames_num = int(max_frames_num)
        self.attn_implementation = attn_implementation

        eval_logger.info(f"Loading Qwen2.5-VL model from {pretrained} ...")
        eval_logger.info(f"Device: {self._device}, device_map: {self.device_map}")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            device_map=self.device_map,
        )
        self._processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

        self._config = self._model.config
        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        self.accelerator = accelerator
        self._rank = accelerator.process_index if accelerator.num_processes > 1 else 0
        self._world_size = accelerator.num_processes if accelerator.num_processes > 1 else 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._config.max_position_embeddings if hasattr(self._config, "max_position_embeddings") else 32768

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for VlmQwen25VL baseline.")

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding (Qwen2.5-VL)",
        )

        _LOG_INTERVAL = 10  # print question & answer every N samples

        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([
            reg.args for reg in requests
        ]):
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)

            # Assert we have exactly one video
            assert len(visuals) == 1, f"Expected 1 video, got {len(visuals)}"
            video_path = visuals[0]

            # Step 1: Load video frames using decord (same as vlm_3drs.py)
            frames = _load_video_frames(video_path, self.max_frames_num)

            # Step 2: Build text prompt with tokenize=False (no video file loading)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": contexts},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Step 3: Process (text + already-loaded frames) together.
            # Passing PIL Image list as `videos` avoids PyAV file I/O.
            inputs = self._processor(
                text=[text],
                videos=[frames],
                padding=True,
                return_tensors="pt",
            )

            # Move tensors to device; keep non-tensors (second_per_grid_ts) on CPU
            inputs = _move_to_device(inputs, self.device)

            # Generate - use max_new_tokens=128 for consistency with reference
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                )

            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            outputs = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            outputs = outputs.strip()

            # Log question & answer periodically
            if idx % _LOG_INTERVAL == 0:
                eval_logger.info(f"[Sample {idx}] QUESTION: {contexts[:200]}{'...' if len(contexts) > 200 else ''}")
                eval_logger.info(f"[Sample {idx}] GT: {str(doc.get('ground_truth', 'N/A'))[:200]}")
                eval_logger.info(f"[Sample {idx}] MODEL: {outputs[:300]}{'...' if len(outputs) > 300 else ''}")

            res.append(outputs)
            pbar.update(1)

        pbar.close()
        return res
