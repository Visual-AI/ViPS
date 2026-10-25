from __future__ import annotations

import os
import re
from collections import OrderedDict

import datasets
import numpy as np
import pandas as pd
from loguru import logger


MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def doc_to_visual(document):
    root = os.environ["VIPS_VSIBENCH_VIDEO_ROOT"]
    path = os.path.join(root, document["dataset"], document["scene_name"] + ".mp4")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [path]


def doc_to_text(document, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt") or "These are frames of a video."
    question = document["question"]
    if document["question_type"] in NA_QUESTION_TYPES:
        post_prompt = kwargs.get("na_post_prompt") or "Please answer the question using a single word or phrase."
        return "\n".join((pre_prompt, question, post_prompt))
    if document["question_type"] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(document["options"])
        post_prompt = kwargs.get("mca_post_prompt") or "Answer with the option's letter from the given choices directly."
        return "\n".join((pre_prompt, question, options, post_prompt))
    raise ValueError(f"Unknown question type: {document['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS"):
        return dataset.shuffle(seed=42)
    return dataset


def _multiple_choice_answer(prediction: str) -> str:
    prediction = prediction.replace("Answer:", "")
    first = prediction.split(" ")[0].strip()
    match = re.search(r"^[A-D]\.?$", first)
    return match.group(0).rstrip(".").upper() if match else prediction.strip()


def _number_answer(prediction: str) -> str:
    prediction = prediction.strip().lower()
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
        "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
        "zero": "0", "a": "1", "an": "1",
    }
    for word, number in number_words.items():
        if re.search(rf"\b{word}\b", prediction):
            return number
    match = re.search(r"\d+(?:\.\d+)?", prediction)
    return match.group(0) if match else "None"


def _mean_relative_accuracy(prediction: float, target: float) -> float:
    # VSI-Bench's reference scorer evaluates ten thresholds: .50, .55, ..., .95.
    intervals = np.linspace(0.5, 0.95, 10)
    return (abs(prediction - target) / target <= 1 - intervals).mean()


def process_results(document, results):
    document["prediction"] = results[0]
    question_type = document["question_type"]
    if question_type in MCA_QUESTION_TYPES:
        document["accuracy"] = float(
            _multiple_choice_answer(results[0]).lower()
            == document["ground_truth"].lower()
        )
    elif question_type in NA_QUESTION_TYPES:
        try:
            prediction = float(_number_answer(results[0]))
            target = float(document["ground_truth"])
            document["MRA:.5:.95:.05"] = _mean_relative_accuracy(prediction, target)
        except (TypeError, ValueError):
            document["MRA:.5:.95:.05"] = 0.0
    else:
        raise ValueError(f"Unknown question type: {question_type}")
    return {"vsibench_score": document}


def aggregate_results(results):
    frame = pd.DataFrame(results)
    scores = {}
    for question_type, indices in frame.groupby("question_type").groups.items():
        metric = "accuracy" if question_type in MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        scores[f"{question_type}_{metric}"] = frame.iloc[indices][metric].mean()
    if "object_rel_direction_easy_accuracy" in scores:
        scores["object_rel_direction_accuracy"] = sum(
            scores.pop(f"object_rel_direction_{level}_accuracy")
            for level in ("easy", "medium", "hard")
        ) / 3
    scores["overall"] = sum(scores.values()) / len(scores)
    logger.info(f"Evaluation results: {scores}")

    ordered = OrderedDict(overall=float(scores["overall"]) * 100)
    for question_type in (
        "object_counting", "object_abs_distance", "object_size_estimation",
        "room_size_estimation", "object_rel_distance", "object_rel_direction",
        "route_planning", "obj_appearance_order",
    ):
        for metric in ("accuracy", "MRA:.5:.95:.05"):
            key = f"{question_type}_{metric}"
            if key in scores:
                ordered[key] = float(scores[key]) * 100
    ordered["tabulated_keys"] = ", ".join(ordered)
    ordered["tabulated_results"] = ", ".join(
        f"{value:.3f}" for value in ordered.values() if isinstance(value, float)
    )
    return ordered
