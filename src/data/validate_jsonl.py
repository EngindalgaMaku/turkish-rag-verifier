"""
src/data/validate_jsonl.py
==========================
Validates a JSONL file against the VerifierExample schema.
Reports all errors with line numbers and field names.

Usage:
    python scripts/01_validate_data.py --input data/processed/pilot.jsonl
    # or directly:
    from src.data.validate_jsonl import validate_jsonl_file
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.data.schema import VerifierExample, VALID_LABELS

console = Console()


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_jsonl_file(
    input_path: str,
    config_path: Optional[str] = None,
    strict: bool = True,
    check_leakage: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Validate a JSONL file against the VerifierExample schema.

    Args:
        input_path:    Path to the JSONL file.
        config_path:   Path to labels.yaml (optional, uses default if None).
        strict:        If True, exit with error code 1 on any validation failure.
        check_leakage: If True, check for train/test context leakage.
        verbose:       If True, print detailed report.

    Returns:
        dict with keys: n_total, n_valid, n_errors, errors, label_distribution,
                        split_distribution, md5_hash, n_score_warnings,
                        score_warnings
    """
    input_path = Path(input_path)
    if not input_path.exists():
        console.print(f"[red]ERROR: File not found: {input_path}[/red]")
        sys.exit(1)

    errors: list[dict] = []
    score_warnings: list[dict] = []
    valid_examples: list[VerifierExample] = []
    ids_seen: set[str] = set()
    context_hashes: dict[str, list[int]] = defaultdict(list)  # hash → line numbers

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            # --- JSON parse ---
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_num,
                    "id": "?",
                    "field": "JSON",
                    "error": f"Invalid JSON: {e}",
                })
                continue

            example_id = data.get("id", f"line_{line_num}")

            # --- Duplicate ID check ---
            if example_id in ids_seen:
                errors.append({
                    "line": line_num,
                    "id": example_id,
                    "field": "id",
                    "error": f"Duplicate ID: '{example_id}'",
                })
            ids_seen.add(example_id)

            # --- Pydantic validation ---
            try:
                example = VerifierExample(**data)
                valid_examples.append(example)

                # Track context hashes for leakage detection
                ctx_hash = hashlib.md5(example.context.encode()).hexdigest()
                context_hashes[ctx_hash].append(line_num)

                # Collect score range warnings stored by model_validator
                for msg in getattr(example, "_score_range_warnings", []):
                    score_warnings.append({
                        "line": line_num,
                        "id": example_id,
                        "message": msg,
                    })

            except ValidationError as e:
                for err in e.errors():
                    field = " > ".join(str(loc) for loc in err["loc"])
                    errors.append({
                        "line": line_num,
                        "id": example_id,
                        "field": field,
                        "error": err["msg"],
                    })

    # --- Leakage check ---
    leakage_warnings: list[dict] = []
    if check_leakage and valid_examples:
        leakage_warnings = _check_leakage(valid_examples)

    # --- Statistics ---
    label_dist = Counter(e.label for e in valid_examples)
    split_dist = Counter(e.split for e in valid_examples)
    domain_dist = Counter(e.domain for e in valid_examples if e.domain)
    source_dist = Counter(e.source_type for e in valid_examples)

    # --- MD5 of file ---
    md5 = _compute_md5(input_path)

    result = {
        "n_total": len(ids_seen),
        "n_valid": len(valid_examples),
        "n_errors": len(errors),
        "errors": errors,
        "n_score_warnings": len(score_warnings),
        "score_warnings": score_warnings,
        "leakage_warnings": leakage_warnings,
        "label_distribution": dict(label_dist),
        "split_distribution": dict(split_dist),
        "domain_distribution": dict(domain_dist),
        "source_distribution": dict(source_dist),
        "md5_hash": md5,
        "file": str(input_path),
    }

    if verbose:
        _print_report(result)

    if strict and (errors or leakage_warnings):
        console.print(
            f"\n[red]Validation FAILED: {len(errors)} errors, "
            f"{len(leakage_warnings)} leakage warnings.[/red]"
        )
        sys.exit(1)

    return result


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def _check_leakage(examples: list[VerifierExample]) -> list[dict]:
    """
    Check if the same context appears in both train and test splits.
    Returns a list of leakage warnings.
    """
    split_contexts: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        ctx_hash = hashlib.md5(ex.context.encode()).hexdigest()
        split_contexts[ex.split].add(ctx_hash)

    warnings = []
    train_hashes = split_contexts.get("train", set())
    test_hashes = split_contexts.get("test", set())
    val_hashes = split_contexts.get("validation", set())

    train_test_overlap = train_hashes & test_hashes
    train_val_overlap = train_hashes & val_hashes

    if train_test_overlap:
        warnings.append({
            "type": "train_test_leakage",
            "n_overlapping_contexts": len(train_test_overlap),
            "message": f"{len(train_test_overlap)} context(s) appear in both train and test splits.",
        })

    if train_val_overlap:
        warnings.append({
            "type": "train_val_leakage",
            "n_overlapping_contexts": len(train_val_overlap),
            "message": f"{len(train_val_overlap)} context(s) appear in both train and validation splits.",
        })

    return warnings


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(result: dict) -> None:
    """Print a rich-formatted validation report."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Turkish RAG Verifier - Data Validation Report[/bold]")
    console.print("=" * 60)
    console.print(f"File:     {result['file']}")
    console.print(f"MD5:      {result['md5_hash']}")
    console.print(f"Total:    {result['n_total']}")
    console.print(f"Valid:    [green]{result['n_valid']}[/green]")
    console.print(f"Errors:   [red]{result['n_errors']}[/red]")
    n_sw = result.get("n_score_warnings", 0)
    sw_color = "yellow" if n_sw > 0 else "green"
    console.print(f"Score range warnings: [{sw_color}]{n_sw}[/{sw_color}]")

    # Label distribution
    console.print("\n[bold]Label Distribution:[/bold]")
    label_table = Table(show_header=True, header_style="bold cyan")
    label_table.add_column("Label")
    label_table.add_column("Count", justify="right")
    label_table.add_column("Pct", justify="right")
    total = result["n_valid"] or 1
    for label in sorted(VALID_LABELS):
        count = result["label_distribution"].get(label, 0)
        pct = f"{100 * count / total:.1f}%"
        label_table.add_row(label, str(count), pct)
    console.print(label_table)

    # Split distribution
    console.print("\n[bold]Split Distribution:[/bold]")
    for split, count in sorted(result["split_distribution"].items()):
        console.print(f"  {split}: {count}")

    # Errors
    if result["errors"]:
        console.print(f"\n[bold red]Validation Errors ({result['n_errors']}):[/bold red]")
        err_table = Table(show_header=True, header_style="bold red")
        err_table.add_column("Line", justify="right")
        err_table.add_column("ID")
        err_table.add_column("Field")
        err_table.add_column("Error")
        for err in result["errors"][:50]:  # Show first 50
            err_table.add_row(
                str(err["line"]), str(err["id"]), err["field"], err["error"]
            )
        console.print(err_table)
        if len(result["errors"]) > 50:
            console.print(f"  ... and {len(result['errors']) - 50} more errors.")

    # Score range warnings
    score_warnings = result.get("score_warnings", [])
    if score_warnings:
        console.print(
            f"\n[bold yellow]Score Range Warnings ({len(score_warnings)}) "
            f"- guideline only, not errors:[/bold yellow]"
        )
        sw_table = Table(show_header=True, header_style="bold yellow")
        sw_table.add_column("Line", justify="right")
        sw_table.add_column("ID")
        sw_table.add_column("Warning")
        for sw in score_warnings[:50]:  # Show first 50
            sw_table.add_row(str(sw["line"]), str(sw["id"]), sw["message"])
        console.print(sw_table)
        if len(score_warnings) > 50:
            console.print(f"  ... and {len(score_warnings) - 50} more score warnings.")

    # Leakage warnings
    if result["leakage_warnings"]:
        console.print("\n[bold yellow]Leakage Warnings:[/bold yellow]")
        for w in result["leakage_warnings"]:
            console.print(f"  [yellow]WARNING:[/yellow] {w['message']}")

    if result["n_errors"] == 0 and not result["leakage_warnings"]:
        if n_sw > 0:
            console.print(
                f"\n[bold green]OK Validation PASSED[/bold green] "
                f"[yellow](with {n_sw} score range guideline warning(s))[/yellow]"
            )
        else:
            console.print("\n[bold green]OK Validation PASSED[/bold green]")


# ---------------------------------------------------------------------------
# MD5 helper
# ---------------------------------------------------------------------------

def _compute_md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()