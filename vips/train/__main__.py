from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import tensor_parallel as tensor_parallel

from vips.checkpoints import check_training_output_dir
from vips.config import load_config, resolve_repository_path
from vips.data import FlattenedCollator, VSIBenchTrainingDataset
from vips.modeling import configure_vips, install_vips_model
from vips.modeling.flash_attention import install_flattened_attention


for class_name in (
    "EmbeddingParallel",
    "ColwiseParallel",
    "RowwiseParallel",
    "ALL_PARALLEL_STYLES",
):
    if not hasattr(tensor_parallel, class_name):
        setattr(tensor_parallel, class_name, type(class_name, (), {}))


ZERO2 = {
    "bf16": {"enabled": "auto"},
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
    },
}


def build_model(config: dict[str, Any]) -> Qwen3VLForConditionalGeneration:
    install_vips_model()
    install_flattened_attention()
    model_name = config["model"]["name_or_path"]
    model_config = configure_vips(AutoConfig.from_pretrained(model_name), config)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        config=model_config,
        attn_implementation=config["model"]["attention"],
        dtype=torch.bfloat16 if config["training"]["bf16"] else None,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    for parameter in model.parameters():
        parameter.requires_grad = False
    training = config["training"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=training["lora_rank"],
            lora_alpha=training["lora_alpha"],
            lora_dropout=training["lora_dropout"],
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.lm_head.weight.requires_grad = True
    if training["tune_merger"]:
        model.base_model.model.visual.merger.requires_grad_(True)

    language_model = model.base_model.model.model.language_model
    language_model.zero_conv_dict.requires_grad_(True)
    language_model.zero_routers.requires_grad_(True)
    return model


def save_training_output(trainer: Trainer, processor: Any, output_dir: Path) -> None:
    if not trainer.args.should_save:
        return
    non_lora = {
        name: parameter.detach().cpu()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad and "lora_" not in name
    }
    torch.save(non_lora, output_dir / "non_lora_trainables.bin")
    if trainer.deepspeed and torch.cuda.is_available():
        torch.cuda.synchronize()
    trainer.save_model(str(output_dir))
    trainer.model.config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def build_training_arguments(
    training: dict[str, Any], output_dir: Path
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        seed=training.get("seed", 42),
        data_seed=training.get("seed", 42),
        num_train_epochs=training["num_epochs"],
        per_device_train_batch_size=training["per_device_batch_size"],
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        bf16=training["bf16"],
        eval_strategy="no",
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        save_only_model=True,
        learning_rate=training["learning_rate"],
        optim=training["optimizer"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        max_grad_norm=training["max_grad_norm"],
        lr_scheduler_type=training["scheduler"],
        logging_steps=1,
        report_to=training.get("report_to") or "none",
        gradient_checkpointing=training["gradient_checkpointing"],
        dataloader_num_workers=training["dataloader_workers"],
        dataloader_prefetch_factor=training["dataloader_prefetch_factor"],
        deepspeed=ZERO2,
        remove_unused_columns=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start a new ViPS training run; checkpoint resume is not supported."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", help="Output directory without existing training weights")
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repository_path(training["output_dir"])
    )
    check_training_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Trainer sets its seed after receiving the model, so seed initialization here too.
    set_seed(training.get("seed", 42))

    processor = AutoProcessor.from_pretrained(config["model"]["name_or_path"])
    processor.tokenizer.model_max_length = training["max_length"]
    processor.tokenizer.padding_side = "right"
    dataset = VSIBenchTrainingDataset(processor, config)
    model = build_model(config)

    trainer = Trainer(
        model=model,
        processing_class=processor.tokenizer,
        train_dataset=dataset,
        data_collator=FlattenedCollator(),
        args=build_training_arguments(training, output_dir),
    )
    trainer.train()
    trainer.save_state()
    trainer.model.config.use_cache = True
    save_training_output(trainer, processor, output_dir)


if __name__ == "__main__":
    main()
