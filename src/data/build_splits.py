"""
src/data/build_splits.py
========================
Splits a validated JSONL file into train / validation / test sets.
Ensures no context leakage between splits.
Saves checksums for reproducibility.

Usage:
    python scripts/02_build_splits.py \
        --input data/processed/pilot.jsonl \
        --output data/splits/ \
        --seed 42 \
        --val-ratio 0.10 \
        --test-ratio 0.17
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.data.schema import VerifierExample

console = Console()


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def build_splits(
    input_path: str,
    output_dir: str,
    seed: int = 42,
    val_ratio: float = 0.10,
    test_ratio: float = 0.17,
    version: str = "v1.0",
    strict: bool = True,
) -> dict:
    """
    Split a JSONL file into train/validation/test sets.

    Strategy:
    - Context-level split: all claims from the same context go to the same split
      (prevents leakage — this is the primary constraint)
    - NOTE: True label-stratification is NOT guaranteed because context-level
      grouping takes priority. With small datasets, some labels may be
      underrepresented in test. Check label_distribution in the returned stats
      and re-run with a different seed if the test distribution is too skewed.
    - Saves checksums for reproducibility

    Args:
        input_path:  Path to validated JSONL file.
        output_dir:  Directory to write train/validation/test.jsonl files.
        seed:        Random seed for reproducibility.
        val_ratio:   Fraction of data for validation.
        test_ratio:  Fraction of data for test.
        version:     Dataset version string (used in filenames).

    Returns:
        dict with split statistics.
    """
    random.seed(seed)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load all examples (strict=True raises on invalid lines) ---
    examples = _load_jsonl(input_path, strict=strict)
    console.print(f"Loaded {len(examples)} examples from {input_path}")

    # --- Group by context hash (prevent leakage) ---
    context_groups: dict[str, list[VerifierExample]] = defaultdict(list)
    for ex in examples:
        ctx_hash = hashlib.md5(ex.context.encode()).hexdigest()
        context_groups[ctx_hash].append(ex)

    context_keys = list(context_groups.keys())
    random.shuffle(context_keys)

    n_contexts = len(context_keys)
    n_test = max(1, int(n_contexts * test_ratio))
    n_val = max(1, int(n_contexts * val_ratio))
    n_train = n_contexts - n_test - n_val

    if n_train <= 0:
        raise ValueError(
            f"Not enough contexts for splitting. "
            f"n_contexts={n_contexts}, n_test={n_test}, n_val={n_val}"
        )

    test_keys = set(context_keys[:n_test])
    val_keys = set(context_keys[n_test:n_test + n_val])
    train_keys = set(context_keys[n_test + n_val:])

    # --- Assign examples to splits ---
    train_examples, val_examples, test_examples = [], [], []
    for ctx_hash, exs in context_groups.items():
        if ctx_hash in test_keys:
            for ex in exs:
                ex.split = "test"
            test_examples.extend(exs)
        elif ctx_hash in val_keys:
            for ex in exs:
                ex.split = "validation"
            val_examples.extend(exs)
        else:
            for ex in exs:
                ex.split = "train"
            train_examples.extend(exs)

    # --- Shuffle within splits ---
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    random.shuffle(test_examples)

    # --- Write splits ---
    splits = {
        "train": train_examples,
        "validation": val_examples,
        "test": test_examples,
    }

    checksums = {}
    stats = {}

    for split_name, split_examples in splits.items():
        out_file = output_dir / f"{split_name}.jsonl"
        _write_jsonl(split_examples, out_file)
        md5 = _compute_md5(out_file)
        checksums[split_name] = {
            "file": str(out_file),
            "n": len(split_examples),
            "md5": md5,
        }
        label_dist = defaultdict(int)
        for ex in split_examples:
            label_dist[ex.label] += 1
        stats[split_name] = {
            "n": len(split_examples),
            "label_distribution": dict(label_dist),
        }
        console.print(
            f"  [green]{split_name}[/green]: {len(split_examples)} examples -> {out_file}"
        )

    # --- Save checksums ---
    checksum_file = output_dir / "checksums.json"
    with open(checksum_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "version": version,
                "source_file": str(input_path),
                "source_md5": _compute_md5(input_path),
                "splits": checksums,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    console.print(f"\nChecksums saved to {checksum_file}")

    # --- Verify no leakage ---
    leakage = _verify_no_leakage(train_examples, val_examples, test_examples)
    if leakage:
        console.print(f"[red]LEAKAGE DETECTED: {leakage}[/red]")
    else:
        console.print("[green]OK No context leakage detected between splits.[/green]")

    return {
        "seed": seed,
        "version": version,
        "stats": stats,
        "checksums": checksums,
        "leakage": leakage,
    }


# ---------------------------------------------------------------------------
# Leakage verification
# ---------------------------------------------------------------------------

def _verify_no_leakage(
    train: list[VerifierExample],
    val: list[VerifierExample],
    test: list[VerifierExample],
) -> list[str]:
    """Verify no context appears in multiple splits. Returns list of issues."""
    def ctx_hashes(examples):
        return {hashlib.md5(ex.context.encode()).hexdigest() for ex in examples}

    train_h = ctx_hashes(train)
    val_h = ctx_hashes(val)
    test_h = ctx_hashes(test)

    issues = []
    if train_h & test_h:
        issues.append(f"{len(train_h & test_h)} contexts overlap between train and test")
    if train_h & val_h:
        issues.append(f"{len(train_h & val_h)} contexts overlap between train and validation")
    if val_h & test_h:
        issues.append(f"{len(val_h & test_h)} contexts overlap between validation and test")
    return issues


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, strict: bool = True) -> list[VerifierExample]:
    """
    Load and parse a JSONL file into VerifierExample objects.

    Args:
        path:   Path to JSONL file.
        strict: If True (default), raise ValueError on any invalid line.
                If False, warn and skip invalid lines (use only with --allow-invalid-skip).

    Raises:
        ValueError: If strict=True and any line fails validation.
    """
    examples = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(VerifierExample(**data))
            except Exception as e:
                msg = f"Line {line_num}: {e}"
                if strict:
                    errors.append(msg)
                else:
                    console.print(f"[yellow]Warning: skipping {msg}[/yellow]")

    if errors:
        raise ValueError(
            f"build_splits aborted: {len(errors)} invalid line(s) in {path}.\n"
            f"Run scripts/01_validate_data.py first to fix all errors.\n"
            f"First error: {errors[0]}\n"
            f"Use --allow-invalid-skip to skip invalid lines (not recommended)."
        )

    return examples


def _write_jsonl(examples: list[VerifierExample], path: Path) -> None:
    """Write VerifierExample objects to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def _compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()