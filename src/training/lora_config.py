"""
src/training/lora_config.py
============================
Loads LoRA and quantization configurations from YAML config files.
Builds PEFT LoraConfig and BitsAndBytesConfig objects.

Usage:
    from src.training.lora_config import build_lora_config, build_bnb_config
    lora_cfg = build_lora_config(config)
    bnb_cfg = build_bnb_config(config)
"""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig


def build_bnb_config(config: dict) -> BitsAndBytesConfig:
    """
    Build BitsAndBytesConfig for 4-bit QLoRA from YAML config.

    Args:
        config: Parsed YAML config dict (full config, not just quantization section).

    Returns:
        BitsAndBytesConfig instance.
    """
    q = config.get("quantization", {})

    compute_dtype_str = q.get("bnb_4bit_compute_dtype", "bfloat16")
    compute_dtype = getattr(torch, compute_dtype_str)

    return BitsAndBytesConfig(
        load_in_4bit=q.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(config: dict) -> LoraConfig:
    """
    Build PEFT LoraConfig from YAML config.

    Args:
        config: Parsed YAML config dict (full config, not just lora section).

    Returns:
        LoraConfig instance.
    """
    lora = config.get("lora", {})
    model = config.get("model", {})

    target_modules = model.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    return LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        bias=lora.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
    )


def check_target_modules(model_name: str) -> list[str]:
    """
    Inspect a model's named modules to find projection layers.
    Use this to verify target_modules for new model architectures.

    Args:
        model_name: HuggingFace model name.

    Returns:
        List of module names containing 'proj'.

    Example:
        check_target_modules("Qwen/Qwen3-4B-Instruct")
    """
    from transformers import AutoModelForCausalLM

    print(f"Loading {model_name} to inspect modules (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    proj_modules = sorted(set(
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if "proj" in name and "." in name
    ))

    print(f"\nProjection modules found in {model_name}:")
    for m in proj_modules:
        print(f"  {m}")

    del model
    return proj_modules