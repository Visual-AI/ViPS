import os
import re
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
import decord
from decord import VideoReader, cpu
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from peft import PeftModel

# ---- Monkey-patch: add missing TP classes for PEFT compatibility in multi-GPU ----
import transformers.integrations.tensor_parallel as _tp_module
for _cls_name in ("EmbeddingParallel", "ColwiseParallel", "RowwiseParallel", "ALL_PARALLEL_STYLES"):
    if not hasattr(_tp_module, _cls_name):
        setattr(_tp_module, _cls_name, type(_cls_name, (), {}))
# -----------------------------------------------------------------------------------

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# ---- ZeroDistill: 3D foundation model constants ----
MODEL_ID_TO_LAYER_IDS = {
    'vggt': [23],
    'da3': [39],
    'ta': [24],
    'radio': [31],
    'wan': [29],
}
MODEL_NAME_TO_FEATURE_DIR = {
    'vggt': 'vggt_features', 'da3': 'da3_features', 'ta': 'ta_features',
    'radio': 'radio_features', 'wan': 'wan_features',
}
MODEL_NAME_TO_FILE_TEMPLATE = {
    'vggt': '{model}_{layer_id}.pt', 'da3': '{model}_{layer_id}.pt',
    'ta': '{model}_{layer_id}.pt', 'radio': '{model}_{layer_id}.pt',
    'wan': 'wan_t2v_{layer_id}.pt',
}
MODEL_NAME_TO_OUTPUT_FILE = {
    'vggt': 'vggt.pt', 'da3': 'da3.pt', 'ta': 'ta.pt',
    'radio': 'radio.pt', 'wan': 'wan.pt',
}
# Feature roots for this legacy wrapper; relative defaults assume the repository root.
FEATURE_DATA_ROOT = os.environ.get("VIPS_TRAIN_FEATURE_ROOT", "data/train/features")
# ---------------------------------------------------

# ---- Monkey-patch: make BaseVideoProcessor use decord when torchcodec unavailable ----
import transformers.video_processing_utils as _vpu
from transformers.utils.import_utils import is_decord_available, is_torchcodec_available

_original_fetch_videos = _vpu.BaseVideoProcessor.fetch_videos
def _patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    if is_torchcodec_available():
        backend = "torchcodec"
    elif is_decord_available():
        backend = "decord"
    else:
        backend = "torchvision"
    if isinstance(video_url_or_urls, list):
        from transformers.video_utils import load_video
        return list(zip(*[load_video(x, backend=backend, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
    else:
        from transformers.video_utils import load_video
        return load_video(video_url_or_urls, backend=backend, sample_indices_fn=sample_indices_fn)

_vpu.BaseVideoProcessor.fetch_videos = _patched_fetch_videos
# -----------------------------------------------------------------------------------


def is_moe_model(model_name_or_path: str) -> bool:
    name = Path(model_name_or_path.rstrip("/")).name.lower()
    return bool(re.search(r"(a\d+b|moe)", name))


def _load_video_frames_with_metadata(video_path: str, max_frames_num: int = 32) -> Tuple[List[Image.Image], dict]:
    """Load uniformly sampled frames and return PIL images with correct video metadata.

    The metadata includes the video's actual fps and the real frame indices,
    so Qwen3VLProcessor can compute accurate timestamps (unlike passing PIL
    frames without metadata, which defaults to fps=24 and fake indices).
    """
    decord.bridge.set_bridge("torch")
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    video_fps = float(vr.get_avg_fps())
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).numpy()
    pil_frames = [Image.fromarray(f) for f in spare_frames]

    video_metadata = {
        "total_num_frames": total_frame_num,
        "fps": video_fps,
        "duration": total_frame_num / video_fps if video_fps > 0 else None,
        "frames_indices": frame_idx,
    }

    return pil_frames, video_metadata


def _load_foundation_features(
    video_path: str,
    prior_model_names: List[str],
    vsibench_feature_root: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load 3D foundation model features from disk.

    Parses source/scene from video path (e.g. ".../vsibench/scannet/videos/scene0000_00.mp4"),
    then loads features from the appropriate feature root:
      - VSIBench: <vsibench_feature_root>/<model_dir>/<source>/<scene>/<file>.pt
      - Others:   FEATURE_DATA_ROOT/<source>/<feat_dir>/<scene>/<file>.pt
    The explicit root takes precedence over VIPS_VSIBENCH_FEATURE_ROOT and
    the repository-relative default.
    """
    VSIBENCH_FEATURE_ROOT = vsibench_feature_root or os.environ.get(
        "VIPS_VSIBENCH_FEATURE_ROOT", "data/vsibench/features"
    )

    parts = video_path.replace("\\", "/").split("/")
    is_vsibench = "vsibench" in parts
    is_vsibench = True # HACK: force vsibench for now, since we only evaluate on vsibench dataset   
    if is_vsibench:
        vsibench_idx = parts.index("vsibench")
        source = parts[vsibench_idx + 1] if vsibench_idx + 1 < len(parts) else parts[-2]
    else:
        try:
            videos_idx = parts.index("videos")
            source = parts[videos_idx - 1] if videos_idx >= 1 else parts[0]
        except ValueError:
            source = parts[-2] if len(parts) >= 2 else parts[0]

    scene = os.path.splitext(parts[-1])[0]

    result = {}
    for model_name in prior_model_names:
        template = MODEL_NAME_TO_FILE_TEMPLATE[model_name]
        merged_filename = MODEL_NAME_TO_OUTPUT_FILE.get(model_name)
        model_result = {}

        if is_vsibench:
            model_dir = "want2v" if model_name == "wan" else model_name
            base_feat_path = os.path.join(VSIBENCH_FEATURE_ROOT, model_dir, source, scene)
        else:
            feat_dir = MODEL_NAME_TO_FEATURE_DIR[model_name]
            base_feat_path = os.path.join(FEATURE_DATA_ROOT, source, feat_dir, scene)

        # Try merged file first (single stacked tensor)
        merged_path = os.path.join(base_feat_path, merged_filename) if merged_filename else None
        if merged_path and os.path.exists(merged_path):
            stacked = torch.load(merged_path, map_location="cpu", mmap=True)
            for i, layer_id in enumerate(MODEL_ID_TO_LAYER_IDS[model_name]):
                model_result[f"feature_full_{layer_id}"] = stacked[i]
        else:
            # Fallback: load each layer from individual files
            for layer_id in MODEL_ID_TO_LAYER_IDS[model_name]:
                filename = template.format(model=model_name, layer_id=layer_id)
                feat_path = os.path.join(base_feat_path, filename)
                if not os.path.exists(feat_path):
                    raise FileNotFoundError(f"Feature not found: {feat_path}")
                model_result[f"feature_full_{layer_id}"] = torch.load(
                    feat_path, map_location="cpu", mmap=True
                )

        if model_result:
            result[model_name] = model_result
    return result


def _move_to_device(inputs, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }


@register_model("vlm_qwen3vl")
class VlmQwen3VL(lmms):
    """
    Qwen3-VL model adapted for VSIBench evaluation.
    Uses decord for video loading + processor.apply_chat_template(tokenize=True).
    Pre-loads frames with correct fps metadata so Qwen3VLProcessor generates
    accurate timestamps (avoids PyAV file I/O issues).
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda:0",
        device_map: str = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        attn_implementation: str = "flash_attention_2",
        peft_model_path: Optional[str] = None,
        processor_pretrained: Optional[str] = None,
        vsibench_feature_root: Optional[str] = None,
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
        self.attn_implementation = attn_implementation
        self.peft_model_path = peft_model_path
        self.processor_pretrained = processor_pretrained or pretrained

        eval_logger.info(f"Loading Qwen3-VL model from {pretrained} ...")
        if peft_model_path:
            eval_logger.info(f"PEFT/LoRA adapter will be loaded from {peft_model_path}")
        if processor_pretrained:
            eval_logger.info(f"Processor/tokenizer will be loaded from {processor_pretrained}")
        eval_logger.info(f"Device: {self._device}, device_map: {self.device_map}")

        if is_moe_model(pretrained):
            eval_logger.info("Detected MoE variant, using Qwen3VLMoeForConditionalGeneration.")
            model_class = Qwen3VLMoeForConditionalGeneration
        else:
            eval_logger.info("Using Qwen3VLForConditionalGeneration.")
            model_class = Qwen3VLForConditionalGeneration

        # ── ZeroDistill: load config from checkpoint to detect settings ──
        # (must be done before model loading to create ZeroDistill layers)
        self.zero_distill_enabled = False
        self.zd_prior_model_names = []
        self.vsibench_feature_root = vsibench_feature_root or os.environ.get(
            "VIPS_VSIBENCH_FEATURE_ROOT", "data/vsibench/features"
        )
        if peft_model_path:
            try:
                from transformers import AutoConfig
                ckpt_config = AutoConfig.from_pretrained(peft_model_path)
                # handle Qwen3-VL config which has either `text_config` or direct attributes
                if hasattr(ckpt_config, "text_config"):
                    z_text_config = ckpt_config.text_config
                else:
                    z_text_config = ckpt_config
                if getattr(z_text_config, "enable_zero_distill", False):
                    self.zero_distill_enabled = True
                    prior_list = getattr(z_text_config, "prior_model_list", "")
                    self.zd_prior_model_names = [
                        m.strip() for m in prior_list.split(",") if m.strip()
                    ]
                    eval_logger.info(
                        f"[ZeroDistill] detected from checkpoint, models={self.zd_prior_model_names}, "
                        f"FEATURE_DATA_ROOT={FEATURE_DATA_ROOT}"
                    )
            except Exception as e:
                eval_logger.warning(f"Could not read checkpoint config: {e}")
        # ─────────────────────────────────────────────────────────────────
        # Load model with the correct config (has ZeroDistill layers if enabled)
        if self.zero_distill_enabled:
            eval_logger.info("Loading model with ZeroDistill-enabled config...")
            base_config = model_class.config_class.from_pretrained(pretrained)
            base_config.text_config.enable_zero_distill = True
            base_config.text_config.prior_model_list = ",".join(self.zd_prior_model_names)
            self._model = model_class.from_pretrained(
                pretrained,
                config=base_config,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map=self.device_map,
            )
        else:
            self._model = model_class.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                device_map=self.device_map,
            )

        # Load PEFT/LoRA adapter if peft_model_path is provided
        if peft_model_path:
            eval_logger.info(f"Loading PEFT adapter from {peft_model_path} ...")
            # Move to CPU first to avoid GPU OOM and TP sharding issues during merge
            self._model = self._model.cpu()
            self._model = PeftModel.from_pretrained(self._model, peft_model_path)
            self._model = self._model.merge_and_unload()
            # Load non-LoRA trainables (e.g. ZeroDistill conv/router weights)
            non_lora_path = os.path.join(peft_model_path, "non_lora_trainables.bin")
            if os.path.exists(non_lora_path):
                state_dict = torch.load(non_lora_path, map_location="cpu")
                # ── 去除 PEFT 包装前缀 ──
                # 训练时 save 是从 PeftModel.named_parameters() 保存的, key 有 base_model.model. 前缀
                # merge_and_unload() 之后模型已解包, 需要去掉此前缀
                clean_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k.replace("base_model.model.", "", 1) if k.startswith("base_model.model.") else k
                    clean_state_dict[new_k] = v
                # ──────────────────────────
                load_result = self._model.load_state_dict(clean_state_dict, strict=False)
                loaded_keys = [k for k in clean_state_dict.keys() if k not in load_result.unexpected_keys]
                eval_logger.info(f"Loaded non-LoRA trainables. Successfully loaded keys:\n{len(loaded_keys)} total")
                if load_result.unexpected_keys:
                    eval_logger.warning(f"Unexpected keys in non-LoRA trainables ({len(load_result.unexpected_keys)}):\n{list(load_result.unexpected_keys)[:5]}...")
                if load_result.missing_keys:
                    eval_logger.warning(f"Missing keys in model ({len(load_result.missing_keys)}):\n{list(load_result.missing_keys)[:5]}...")
            # Move merged model back to target device
            self._model = self._model.to(self._device)
            eval_logger.info("PEFT adapter loaded and merged successfully.")

        self._processor = AutoProcessor.from_pretrained(self.processor_pretrained)
        # # ── 设置与训练一致的 video processor size，确保 token 数量对齐 ──
        # if hasattr(self._processor, "video_processor") and self._processor.video_processor is not None:
        #     vp = self._processor.video_processor
        #     video_max_pixels = 1024 * 28 * 28   # = 802,816，与训练默认值一致
        #     video_min_pixels = 256 * 28 * 28   # = 200,704
        #     if hasattr(vp, "size") and isinstance(vp.size, dict):
        #         vp.size["shortest_edge"] = video_min_pixels
        #         vp.size["longest_edge"] = video_max_pixels
        #         eval_logger.info(
        #             f"[VideoProcessor] Set size.shortest_edge={video_min_pixels}, size.longest_edge={video_max_pixels}"
        #         )
        # # ──────────────────────────────────────────────────────────────
        self._tokenizer = AutoTokenizer.from_pretrained(self.processor_pretrained, trust_remote_code=True)
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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("loglikelihood is not implemented for VlmQwen3VL baseline.")

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding (Qwen3-VL)",
        )

        _LOG_INTERVAL = 10

        for idx, (contexts, gen_kwargs, doc_to_visual, doc_id, task, split) in enumerate([
            reg.args for reg in requests
        ]):
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)

            assert len(visuals) == 1, f"Expected 1 video, got {len(visuals)}"
            video_path = visuals[0]

            # Load frames with decord + construct correct video_metadata (actual fps
            # and frame indices), so Qwen3VLProcessor can compute accurate timestamps.
            frames, video_metadata = _load_video_frames_with_metadata(video_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": contexts},
                    ],
                }
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                video_metadata=[video_metadata],
                do_sample_frames=False,
            )
            inputs = _move_to_device(inputs, self.device)
            # ── ZeroDistill: load foundation features if enabled ──
            extra_kwargs = {}
            if self.zero_distill_enabled:
                foundation_features = _load_foundation_features(
                    video_path,
                    self.zd_prior_model_names,
                    self.vsibench_feature_root,
                )
                extra_kwargs["foundation_features"] = [foundation_features]
                extra_kwargs["question_positions"] = None
            # ──────────────────────────────────────────────────────

            # HACK: pass question_type to model for routing stats
            self._model.language_model._current_question_type = doc.get("question_type", "unknown")
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    **extra_kwargs,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            outputs = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            outputs = outputs.strip()
            # print(f"outputs: {outputs[:300]}{'...' if len(outputs) > 300 else ''}")
            if idx % _LOG_INTERVAL == 0:
                eval_logger.info(f"[Sample {idx}] QUESTION: {contexts[:200]}{'...' if len(contexts) > 200 else ''}")
                eval_logger.info(f"[Sample {idx}] GT: {str(doc.get('ground_truth', 'N/A'))[:200]}")
                eval_logger.info(f"[Sample {idx}] MODEL: {outputs[:300]}{'...' if len(outputs) > 300 else ''}")

            res.append(outputs)
            pbar.update(1)

        pbar.close()
        return res
