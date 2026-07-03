"""
src/utils/io.py
===============
JSONL I/O utilities with error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """
    Read a JSONL file line by line.
    Skips empty lines. Raises on malformed JSON.

    Args:
        path: Path to JSONL file.

    Yields:
        Parsed dict for each line.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {path}: {e}") from e


def write_jsonl(records: list[dict | Any], path: str | Path) -> None:
    """
    Write a list of dicts (or Pydantic models) to a JSONL file.

    Args:
        records: List of dicts or Pydantic models with .model_dump() method.
        path:    Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if hasattr(record, "model_dump"):
                line = record.model_dump_json()
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Save a dict to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)