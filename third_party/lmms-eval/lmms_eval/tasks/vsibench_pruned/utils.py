
import os
import re
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from collections import OrderedDict

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = "vsibench"
# cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

# Flag: use refined post-processing (matches reference eval).
# Set env var VSIBENCH_REFINED_EVAL=true to enable.
_USE_REFINED = os.environ.get("VSIBENCH_REFINED_EVAL", "").lower() in ("true", "1", "yes")
def fuzzy_matching_refined(pred):
    """For MCA: extract option letter (A-D), fallback to first word."""
    match = re.search(r'^[A-D]\.?$', pred.split(' ')[0].strip())
    if match:
        pred = match.group(0).rstrip('.').upper().strip()
        return pred
    return pred.strip()

def fuzzy_matching_original(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def fuzzy_matching(pred):
    if _USE_REFINED:
        return fuzzy_matching_refined(pred)
    return fuzzy_matching_original(pred)

def fuzzy_matching_num(pred):
    """For NA: convert number words / extract first number, return 'None' if none found."""
    pred = pred.strip().lower()
    number_words = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
        'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
        'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
        'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80', 'ninety': '90',
        'zero': '0', 'a': '1', 'an': '1',
    }
    for word, digit in number_words.items():
        if re.search(r'\b' + word + r'\b', pred):
            return digit
    number_match = re.search(r'(\d+(\.\d+)?)', pred)
    if number_match:
        return number_match.group(1)
    return "None"

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        if _USE_REFINED:
            predicted = doc['prediction'].replace("Answer:", "")
        else:
            predicted = doc['prediction']
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(predicted), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                if _USE_REFINED:
                    doc[key] = eval(value)(to_float(fuzzy_matching_num(doc['prediction'])), to_float(doc['ground_truth']))
                else:
                    doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)

    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]

        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")

    if 'object_rel_direction_easy_accuracy' in output:
        output['object_rel_direction_accuracy'] = sum([
            output.pop('object_rel_direction_easy_accuracy'),
            output.pop('object_rel_direction_medium_accuracy'),
            output.pop('object_rel_direction_hard_accuracy'),
        ]) / 3.

    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")

    results = OrderedDict()
    results["overall"] = output["overall"].item() * 100.
    
    for question_type in [
        "object_counting",
        "object_abs_distance",
        "object_size_estimation",
        "room_size_estimation",
        "object_rel_distance",
        "object_rel_direction",
        "route_planning",
        "obj_appearance_order",
    ]:
        for metric in [
            "accuracy",
            "MRA:.5:.95:.05",
        ]:
            key = f"{question_type}_{metric}"
            if key in output:
                results[key] = output[key].item() * 100.

    tabulated_keys = ", ".join([_ for _ in results.keys()])
    tabulated_results = ", ".join([f"{_:.3f}" for _ in results.values()])
    eval_logger.info(f"Tabulated results: {tabulated_keys}")
    eval_logger.info(f"Tabulated results: {tabulated_results}")

    results["tabulated_keys"] = tabulated_keys
    results["tabulated_results"] = tabulated_results

    return results
