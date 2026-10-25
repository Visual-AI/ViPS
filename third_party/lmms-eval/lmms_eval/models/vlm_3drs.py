import math
import os
import sys
import copy
from datetime import timedelta
from typing import List, Optional, Tuple, Union
import re
import fasteners

import numpy as np
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoConfig
import time
from PIL import Image
import decord
from decord import VideoReader, cpu
from word2number import w2n

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Add the 3DRS root directory to sys.path so we can import llava
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path = ["../..", "../../llava"] + sys.path

try:
    from llava.constants import (
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import SeparatorStyle, conv_templates
    from llava.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
    )
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.video_utils import VideoProcessor, merge_video_dict
except ImportError as e:
    eval_logger.debug(f"LLaVA may not installed correctly. Please check. {e}")

def safe_word2num(s: str) -> str:
    """
    将英文数字单词转换为数字字符串。
    如果输入不是有效的英文数字，则原样返回。
    """
    try:
        # 转换为整数，再转为字符串
        return str(w2n.word_to_num(s))
    except ValueError:
        # 无法转换时返回原字符串
        print(f"safe_word2num warning: '{s}' is not a valid number word. Returning original string.")
        return s

def preprocess_qwen(sources, tokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant."):
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX]
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


@register_model("vlm_3drs")
class Vlm3Drs(lmms):
    """
    Vlm3Drs Model adapted for vsibench
    """

    def __init__(
        self,
        pretrained: str = "lmms-lab/VLM-3R-7B-Qwen2",
        model_base: str = None,
        lora_path: str = None,
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map="cuda:0",
        conv_template="qwen_1_5",
        use_cache=True,
        truncate_context=False,
        max_frames_num: int = 32,
        video_folder: str = "../../data",
        embodiedscan_folder: str = "../../data/embodiedscan",
        frame_sampling_strategy: str = "uniform",
        force_sample: bool = True,
        overwrite_cfg: bool = False,
        tie_weights: bool = True,
        extra_prompt: str = "",
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
        self.model_base = model_base
        self.lora_path = lora_path
        self.model_name = get_model_name_from_path(pretrained)
        if "lora" not in self.model_name.lower():
            self.model_name += "-lora"

        if "llava" not in self.model_name.lower():
            self.model_name = "llava-" + self.model_name
        
        if "qwen" not in self.model_name.lower():
            self.model_name += "-qwen"
            
        self.max_frames_num = int(max_frames_num)
        self.force_sample = force_sample
        self.frame_sampling_strategy = frame_sampling_strategy
        disable_torch_init()

        config = {}
        if self.lora_path is not None and self.lora_path != "None":
            config = AutoConfig.from_pretrained(self.lora_path)
            config = config.to_dict()
        elif overwrite_cfg:
            config.update({
                'tie_word_embeddings': False, 
                'use_cache': True, 
                "vocab_size": 151649
            })

        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            self.pretrained, self.model_base, self.model_name, overwrite_config=config, device_map=self.device_map
        )

        if self.lora_path is not None and self.lora_path != "None":
            from transformers import AutoTokenizer
            from peft import PeftModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
            self._model.resize_token_embeddings(len(self._tokenizer))

            self._model = PeftModel.from_pretrained(self._model, self.lora_path, adapter_name="lora")
            self._model = self._model.merge_and_unload()
            state_dict = torch.load(os.path.join(self.lora_path, 'non_lora_trainables.bin'))
            self._model.load_state_dict(state_dict, strict=False)

        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
            
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        self.extra_prompt = extra_prompt

        # # Set up Video Processor
        # self.video_processor = VideoProcessor(
        #     video_folder=video_folder,
        #     annotation_dir=embodiedscan_folder,
        #     frame_sampling_strategy=self.frame_sampling_strategy,
        #     feature_3d_model=self.model.config.feature_3d_model,
        #     source_layer_ids=self.model.config.source_layer_ids,
        # )

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._rank = self.accelerator.process_index
            self._word_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._word_size = 1

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
        return self._max_length

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
        return self._word_size

    def load_video(self, video_path, max_frames_num):
        decord.bridge.set_bridge("torch")
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).numpy()
        
        return [Image.fromarray(f).resize((384, 384)) for f in spare_frames]

    def load_3d_features_for_vsibench(self, video_path):
        video_dir_parts = video_path.split('/')
        subset = video_dir_parts[-2]
        scene = video_dir_parts[-1].split('.')[0]
        
        feature_3d_model_list = []
        if hasattr(self.model.config, 'feature_3d_model'):
            feature_3d_model_list = self.model.config.feature_3d_model.split()
        else:
            feature_3d_model_list = ['vggt']
            
        model_id_to_layer_ids = {
            'vggt': [10, 14, 17, 20, 23],
            'da3': [11, 18, 25, 32, 39],
            'ta': [4, 9, 14, 19, 24],
            'radio': [11, 16, 21, 26, 31],
            'want2v': [9, 14, 19, 24, 29],
        }        
        feature_dict = {}
        for model_id in feature_3d_model_list:
            feature_dir = os.path.join(
                os.environ.get("VIPS_VSIBENCH_FEATURE_ROOT", "data/vsibench/features"),
                model_id, subset, scene,
            )
            layers = model_id_to_layer_ids.get(model_id)
            for layer_id in layers:
                if model_id == 'want2v':
                    pt_path = os.path.join(feature_dir, f"wan_t2v_{layer_id}.pt")
                else:
                    pt_path = os.path.join(feature_dir, f"{model_id}_{layer_id}.pt")
                    
                if os.path.exists(pt_path):
                    feat = torch.load(pt_path, map_location='cpu').half()
                    feature_dict[f"{model_id}_feature_full_{layer_id}"] = feat
                else:
                    raise FileNotFoundError(f"Feature file not found: {pt_path}")
        return feature_dict

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
                
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)
            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Qwen process handling for loglikelihood
            extra_prompt_str = f"{self.extra_prompt}\n" if self.extra_prompt and not self.extra_prompt.endswith('\n') else self.extra_prompt
            original_qs_without_token = contexts.replace(DEFAULT_IMAGE_TOKEN, "")
            prompt_to_use = f'{DEFAULT_IMAGE_TOKEN}\n{extra_prompt_str}{original_qs_without_token}'
            conversation_context = [{"from": "human", "value": prompt_to_use}, {"from": "gpt", "value": None}]
            
            try:
                contxt_id = preprocess_qwen(conversation_context, self.tokenizer, has_image=True).cuda()
            except Exception as e:
                contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)

            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], continuation)
            prompt = conv.get_prompt()

            conversation_full = [{"from": "human", "value": prompt_to_use}, {"from": "gpt", "value": continuation}]
            try:
                input_ids = preprocess_qwen(conversation_full, self.tokenizer, has_image=True).cuda()
            except Exception as e:
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                
            videos = []
            features_list = []
            for visual in visuals:
                if visual is not None:
                    videos.extend(self.load_video(visual, self.max_frames_num))
                    features_list.append(self.load_3d_features_for_vsibench(visual))

            video_dict = {}
            if len(videos) > 0:
                image_tensors = self._image_processor.preprocess(videos, return_tensors="pt")["pixel_values"].to(self.device)
                if type(image_tensors) is list:
                    image_tensors = [img.half() for img in image_tensors]
                else:
                    image_tensors = image_tensors.half()
                    
                if len(features_list) > 0:
                    for k in features_list[0]:
                        video_dict[k] = torch.stack([f[k] for f in features_list if k in f]).to(self.device).half()
            else:
                image_tensors = None
                video_dict = None

            if video_dict is not None:
                # NOTE: at this moment vsibench does not provide world coordinate. So here all zero tensor is used as placeholder, which may cause train evaluation gap. Please replace it with actual world coordinate if needed. 
                video_dict['world_coords'] = torch.zeros((1, 32, 384, 384, 3), device=self.device).half()

                video_dict["box_input"] = torch.Tensor([]).to(self.device).half()
                
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            context_length = contxt_id.shape[1]
            labels[0, :context_length] = -100

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    images=image_tensors.unsqueeze(0) if image_tensors is not None else None,
                    modalities="video",
                    video_dict=video_dict
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, context_length :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, context_length : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            doc = self.task_dict[task][split][doc_id]
            visuals = [doc_to_visual(doc)]
            visuals = self.flatten(visuals)
            qs = contexts
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            # Prompt used in model_scanqa.py preprocessing
            # extra_prompt handled by vsibench pre_prompt probably
            extra_prompt_str = f"{self.extra_prompt}\n" if self.extra_prompt and not self.extra_prompt.endswith('\n') else self.extra_prompt
            original_qs_without_token = contexts.replace(DEFAULT_IMAGE_TOKEN, "")
            prompt_to_use = f'{DEFAULT_IMAGE_TOKEN}\n{extra_prompt_str}{original_qs_without_token}'
            
            conversation = [{"from": "human", "value": prompt_to_use}, {"from": "gpt", "value": None}]  
            input_ids = preprocess_qwen(conversation, self.tokenizer, has_image=True).cuda()

            videos = []
            features_list = []
            for visual in visuals:
                if visual is not None:
                    videos.extend(self.load_video(visual, self.max_frames_num))
                    features_list.append(self.load_3d_features_for_vsibench(visual))
            # import pdb; pdb.set_trace()
            video_dict = {}
            if len(videos) > 0:
                image_tensors = self._image_processor.preprocess(videos, return_tensors="pt")["pixel_values"].to(self.device)
                if type(image_tensors) is list:
                    image_tensors = [img.half() for img in image_tensors]
                else:
                    image_tensors = image_tensors.half()
                if len(features_list) > 0:
                    for k in features_list[0]:
                        video_dict[k] = torch.stack([f[k] for f in features_list if k in f]).to(self.device).half()
            else:
                image_tensors = None
                video_dict = None

            if video_dict is not None:
                # NOTE: at this moment vsibench does not provide world coordinate. So here all zero tensor is used as placeholder, which may cause train evaluation gap. Please replace it with actual world coordinate if needed. 
                video_dict['world_coords'] = torch.zeros((1, 32, 384, 384, 3), device=self.device).half()

                video_dict["box_input"] = torch.Tensor([]).to(self.device).half()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                        
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensors.unsqueeze(0) if image_tensors is not None else None,
                    modalities="video",
                    video_dict=video_dict,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                    # stopping_criteria=[stopping_criteria],
                    # pad_token_id=pad_token_id
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            # 如果是object_counting的任务，转化为数字输出，例如three->3
            # if doc['question_type'] in ['object_abs_distance', 'object_counting', 'object_size_estimation', 'room_size_estimation']:
            #     outputs = safe_word2num(outputs)
            # if doc['question_type'] not in ['object_counting']:
            #     import pdb; pdb.set_trace()
            res.append(outputs)
            pbar.update(1)
            
        pbar.close()
        return res
