"""
src/training/train_qlora.py
============================
Main QLoRA fine-tuning script for the Turkish RAG Hallucination Verifier.
Uses TRL SFTTrainer with PEFT LoRA adapters.

Usage (via script):
    python scripts/03_train_qlora.py --config configs/train_qwen3b_qlora.yaml

Usage (direct):
    from src.training.train_qlora import train
    train("configs/train_qwen3b_qlora.yaml")
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTTrainer, SFTConfig

from src.training.format_chat_dataset import format_dataset_for_training
from src.training.lora_config import build_bnb_config, build_lora_config
from src.utils.io import load_yaml
from src.utils.seed import set_seed


def train(config_path: str) -> None:
    """
    Run QLoRA fine-tuning from a YAML config file.

    Args:
        config_path: Path to training config YAML.
    """
    config = load_yaml(config_path)
    exp = config.get("experiment", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    save_cfg = config.get("saving", {})
    log_cfg = config.get("logging", {})
    data_cfg = config.get("data", {})

    # --- Seed ---
    seed = exp.get("seed", 42)
    set_seed(seed)

    # --- W&B (optional) ---
    run_name = log_cfg.get("run_name", exp.get("name", "qlora_run"))
    report_to = log_cfg.get("report_to", "none")
    if report_to == "wandb":
        try:
            import wandb as _wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` "
                "or set report_to: none in your training config."
            )
        _wandb.init(
            project=os.environ.get("WANDB_PROJECT", "turkish-rag-verifier"),
            name=run_name,
            tags=exp.get("tags", []),
            config=config,
        )

    # --- Tokenizer ---
    model_name = model_cfg["name"]
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        revision=model_cfg.get("revision", "main"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for SFT

    # --- Datasets ---
    prompt_version = data_cfg.get("prompt_version", "v1.0")
    print("Formatting training dataset...")
    train_dataset = format_dataset_for_training(
        data_cfg["train_file"],
        tokenizer=tokenizer,
        prompt_version=prompt_version,
        split_filter="train",
    )
    print("Formatting validation dataset...")
    eval_dataset = format_dataset_for_training(
        data_cfg["validation_file"],
        tokenizer=tokenizer,
        prompt_version=prompt_version,
        split_filter="validation",
    )
    print(f"Train: {len(train_dataset)} | Validation: {len(eval_dataset)}")

    # --- Model ---
    bnb_config = build_bnb_config(config)
    print(f"Loading model: {model_name}")

    # Resolve torch_dtype from training config so bf16/fp16/fp32 is consistent
    # between TrainingArguments and model loading.
    if train_cfg.get("bf16", False):
        _torch_dtype = torch.bfloat16
    elif train_cfg.get("fp16", False):
        _torch_dtype = torch.float16
    else:
        _torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        revision=model_cfg.get("revision", "main"),
        dtype=_torch_dtype,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    # --- Prepare for k-bit training ---
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )

    # --- Apply LoRA ---
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- SFTConfig (TRL 1.x API: combines TrainingArguments + SFT-specific args) ---
    output_dir = save_cfg.get("output_dir", f"outputs/models/{run_name}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Compute warmup_steps from warmup_ratio (warmup_ratio deprecated in transformers 5.x)
    total_steps = (
        len(train_dataset)
        // (train_cfg.get("per_device_train_batch_size", 1) * train_cfg.get("gradient_accumulation_steps", 8))
        * train_cfg.get("num_train_epochs", 2)
    )
    warmup_steps = max(1, int(total_steps * train_cfg.get("warmup_ratio", 0.03)))

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=train_cfg.get("remove_unused_columns", False),
        logging_steps=log_cfg.get("logging_steps", 10),
        eval_strategy=log_cfg.get("eval_strategy", "epoch"),
        save_strategy=save_cfg.get("save_strategy", "epoch"),
        save_total_limit=save_cfg.get("save_total_limit", 3),
        load_best_model_at_end=save_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=save_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=save_cfg.get("greater_is_better", False),
        report_to=report_to,
        run_name=run_name,
        seed=seed,
        # SFT-specific (TRL 1.6.0: max_seq_length -> max_length)
        max_length=train_cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
    )

    # --- Trainer (TRL 1.x: tokenizer -> processing_class) ---
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # --- Train ---
    print(f"\nStarting training: {run_name}")
    print(f"Output dir: {output_dir}")
    trainer.train()

    # --- Save final adapter ---
    adapter_dir = Path(output_dir) / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapter saved to: {adapter_dir}")

    if report_to == "wandb":
        import wandb as _wandb  # already imported above; safe to re-import
        _wandb.finish()

    print("\nTraining complete.")
    print(f"\nAdapter saved to: {adapter_dir}")
    print(f"\nTo run inference (adapter + base model):")
    print(f"  python scripts/04_predict.py \\")
    print(f"    --model {adapter_dir} \\")
    print(f"    --base-model {model_name} \\")
    print(f"    --input data/splits/test.jsonl \\")
    print(f"    --run-name {run_name}")
    print(f"\nThen evaluate:")
    print(f"  python scripts/05_evaluate.py \\")
    print(f"    --gold data/splits/test.jsonl \\")
    print(f"    --pred outputs/predictions/{run_name}_test_predictions.jsonl")