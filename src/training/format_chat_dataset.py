"""
src/training/format_chat_dataset.py
=====================================
Converts VerifierExample JSONL records into chat-format training examples
compatible with TRL's SFTTrainer.

Each example becomes:
  [system_prompt] + [user_prompt with question/context/answer/claim] + [assistant JSON output]

Usage:
    from src.training.format_chat_dataset import format_dataset_for_training
    dataset = format_dataset_for_training("data/splits/train.jsonl", prompt_version="v1.0")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset

from src.data.schema import VerifierExample
from src.utils.io import read_jsonl


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt_templates(prompt_version: str = "v1.0") -> tuple[str, str]:
    """
    Load system and user prompt templates for the given version.

    Returns:
        (system_prompt, user_template) tuple.
    """
    root = Path(__file__).resolve().parents[2]
    prompt_dir = root / "prompts" / prompt_version

    system_path = prompt_dir / "verifier_system.txt"
    user_path = prompt_dir / "verifier_user.txt"

    if not system_path.exists():
        raise FileNotFoundError(f"System prompt not found: {system_path}")
    if not user_path.exists():
        raise FileNotFoundError(f"User prompt not found: {user_path}")

    system_prompt = system_path.read_text(encoding="utf-8").strip()
    user_template = user_path.read_text(encoding="utf-8").strip()

    return system_prompt, user_template


# ---------------------------------------------------------------------------
# Single example formatting
# ---------------------------------------------------------------------------

def format_example(
    example: VerifierExample,
    system_prompt: str,
    user_template: str,
    tokenizer,
) -> dict:
    """
    Format a single VerifierExample into a chat-format dict.

    Args:
        example:       VerifierExample instance.
        system_prompt: System prompt string.
        user_template: User prompt template with {question}, {context}, {answer}, {claim}.
        tokenizer:     HuggingFace tokenizer (for apply_chat_template).

    Returns:
        dict with "text" key containing the formatted chat string.
    """
    user_content = user_template.format(
        question=example.question,
        context=example.context,
        answer=example.answer,
        claim=example.claim,
    )

    assistant_content = json.dumps(
        {
            "label": example.label,
            "hallucination_score": example.hallucination_score,
            "error_type": example.error_type,
            "evidence_span": example.evidence_span,
            "explanation": example.explanation,
            "decision": example.decision,
        },
        ensure_ascii=False,
        indent=None,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # Apply chat template (adds special tokens, BOS/EOS, etc.)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text,
        "id": example.id,
        "label": example.label,
        "split": example.split,
        "prompt_version": example.prompt_version or "v1.0",
    }


# ---------------------------------------------------------------------------
# Dataset formatting
# ---------------------------------------------------------------------------

def format_dataset_for_training(
    jsonl_path: str,
    tokenizer,
    prompt_version: str = "v1.0",
    split_filter: Optional[str] = None,
    max_examples: Optional[int] = None,
    strict: bool = True,
    max_skip_ratio: float = 0.05,
) -> Dataset:
    """
    Load a JSONL file and format all examples for SFT training.

    Args:
        jsonl_path:     Path to JSONL file.
        tokenizer:      HuggingFace tokenizer.
        prompt_version: Prompt version to use.
        split_filter:   If set, only include examples with this split value.
        max_examples:   If set, limit to this many examples (for testing).
        strict:         If True (default), raise ValueError when any example
                        fails schema validation or formatting. Set False only
                        for debugging; use max_skip_ratio to guard against
                        silent data loss.
        max_skip_ratio: When strict=False, raise ValueError if the fraction of
                        skipped examples exceeds this threshold (default 0.05 =
                        5%). Prevents silent training data corruption.

    Returns:
        HuggingFace Dataset with "text" column.

    Raises:
        ValueError: In strict mode, on the first invalid example. In non-strict
                    mode, if skipped examples exceed max_skip_ratio.
    """
    system_prompt, user_template = load_prompt_templates(prompt_version)

    formatted = []
    skipped_details: list[str] = []
    total_seen = 0

    for raw in read_jsonl(jsonl_path):
        total_seen += 1
        example_id = raw.get("id", f"row_{total_seen}")

        try:
            example = VerifierExample(**raw)
        except Exception as e:
            msg = f"Schema validation failed for example '{example_id}': {e}"
            if strict:
                raise ValueError(msg)
            skipped_details.append(msg)
            continue

        if split_filter and example.split != split_filter:
            continue

        try:
            formatted_ex = format_example(example, system_prompt, user_template, tokenizer)
            formatted.append(formatted_ex)
        except Exception as e:
            msg = f"Formatting failed for example '{example_id}': {e}"
            if strict:
                raise ValueError(msg)
            skipped_details.append(msg)
            continue

        if max_examples and len(formatted) >= max_examples:
            break

    n_skipped = len(skipped_details)
    if n_skipped > 0:
        for detail in skipped_details:
            print(f"[SKIP] {detail}")
        skip_ratio = n_skipped / max(total_seen, 1)
        print(
            f"Warning: skipped {n_skipped}/{total_seen} examples "
            f"({skip_ratio:.1%}) during formatting."
        )
        if skip_ratio > max_skip_ratio:
            raise ValueError(
                f"Too many skipped examples: {n_skipped}/{total_seen} "
                f"({skip_ratio:.1%}) exceeds max_skip_ratio={max_skip_ratio:.1%}. "
                f"Fix the data or pass strict=False with a higher max_skip_ratio "
                f"only for debugging."
            )

    print(f"Formatted {len(formatted)} examples from {jsonl_path}")

    return Dataset.from_list(formatted)


# ---------------------------------------------------------------------------
# Token length analysis
# ---------------------------------------------------------------------------

def analyze_token_lengths(
    jsonl_path: str,
    tokenizer,
    prompt_version: str = "v1.0",
    percentiles: list[int] = [50, 75, 90, 95, 99, 100],
) -> dict:
    """
    Analyze token lengths of formatted examples.
    Use this to choose max_seq_length before training.

    Returns:
        dict with percentile statistics.
    """
    import numpy as np

    dataset = format_dataset_for_training(jsonl_path, tokenizer, prompt_version)
    lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset]

    stats = {
        "n": len(lengths),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }
    for p in percentiles:
        stats[f"p{p}"] = int(np.percentile(lengths, p))

    return stats