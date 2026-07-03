"""
src/inference/parse_json_output.py
====================================
Robust JSON parser for verifier model outputs.
Implements a 3-tier fallback strategy:
  1. Direct JSON parse
  2. Extract JSON block from surrounding text
  3. Regex field extraction
  4. Return parse_error sentinel

Usage:
    from src.inference.parse_json_output import parse_verifier_output
    result = parse_verifier_output('{"label": "contradicted", "hallucination_score": 0.95, ...}')
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.data.schema import (
    VALID_DECISIONS,
    VALID_ERROR_TYPES,
    VALID_LABELS,
    VerifierOutput,
)


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_verifier_output(raw_output: str) -> VerifierOutput:
    """
    Parse the raw string output from the verifier model into a VerifierOutput.

    Fallback strategy:
    1. Direct JSON.loads()
    2. Extract first {...} block from text
    3. Regex field extraction
    4. Return parse_error sentinel

    Args:
        raw_output: Raw string from model generation.

    Returns:
        VerifierOutput instance (may have parse_error=True if parsing failed).
    """
    if not raw_output or not raw_output.strip():
        return _parse_error_sentinel(raw_output, "Empty output")

    # Tier 1: Direct JSON parse
    result = _try_direct_json(raw_output)
    if result is not None:
        return result

    # Tier 2: Extract JSON block
    result = _try_extract_json_block(raw_output)
    if result is not None:
        return result

    # Tier 3: Regex field extraction
    result = _try_regex_extraction(raw_output)
    if result is not None:
        return result

    # Tier 4: Parse error sentinel
    return _parse_error_sentinel(raw_output, "All parsing strategies failed")


# ---------------------------------------------------------------------------
# Tier 1: Direct JSON parse
# ---------------------------------------------------------------------------

def _try_direct_json(text: str) -> Optional[VerifierOutput]:
    """Try to parse the entire text as JSON."""
    try:
        data = json.loads(text.strip())
        return _dict_to_output(data, text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tier 2: Extract JSON block
# ---------------------------------------------------------------------------

def _try_extract_json_block(text: str) -> Optional[VerifierOutput]:
    """
    Find the first {...} block in the text and parse it.
    Handles cases where the model adds explanation text before/after JSON.
    """
    # Find all {...} blocks (non-greedy won't work for nested, use stack approach)
    start = text.find('{')
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                json_str = text[start:i + 1]
                try:
                    data = json.loads(json_str)
                    return _dict_to_output(data, text)
                except (json.JSONDecodeError, ValueError):
                    return None

    return None


# ---------------------------------------------------------------------------
# Tier 3: Regex field extraction
# ---------------------------------------------------------------------------

def _try_regex_extraction(text: str) -> Optional[VerifierOutput]:
    """
    Extract individual fields using regex patterns.
    Used when JSON is malformed but fields are present.
    """
    def extract_str_field(field: str) -> Optional[str]:
        pattern = rf'"{field}"\s*:\s*"([^"]*)"'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def extract_float_field(field: str) -> Optional[float]:
        pattern = rf'"{field}"\s*:\s*([0-9]*\.?[0-9]+)'
        match = re.search(pattern, text)
        try:
            return float(match.group(1)) if match else None
        except (ValueError, AttributeError):
            return None

    label = extract_str_field("label")
    if label not in VALID_LABELS:
        # Try to find label as a bare word
        for valid_label in VALID_LABELS:
            if valid_label in text:
                label = valid_label
                break
        else:
            return None  # Can't determine label → give up

    score = extract_float_field("hallucination_score")
    if score is None:
        # Use label default
        from src.data.schema import SCORE_RANGES
        score = SCORE_RANGES.get(label, {}).get("typical", 0.5)

    error_type = extract_str_field("error_type")
    if error_type not in VALID_ERROR_TYPES:
        error_type = "none" if label == "supported" else "unsupported_inference"

    evidence_span = extract_str_field("evidence_span") or ""
    explanation = extract_str_field("explanation") or "Açıklama üretilemedi."
    decision = extract_str_field("decision")
    if decision not in VALID_DECISIONS:
        from src.data.schema import LABEL_TO_DEFAULT_DECISION
        decision = LABEL_TO_DEFAULT_DECISION.get(label, "warn")

    try:
        return VerifierOutput(
            label=label,
            hallucination_score=max(0.0, min(1.0, score)),
            error_type=error_type,
            evidence_span=evidence_span,
            explanation=explanation,
            decision=decision,
            parse_error=True,  # Mark as regex-extracted
            raw_output=raw_output_truncated(text),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dict → VerifierOutput
# ---------------------------------------------------------------------------

def _dict_to_output(data: dict, raw_text: str) -> Optional[VerifierOutput]:
    """Convert a parsed dict to VerifierOutput with validation."""
    try:
        label = data.get("label", "")
        if label not in VALID_LABELS:
            return None

        score = float(data.get("hallucination_score", 0.5))
        score = max(0.0, min(1.0, score))

        error_type = data.get("error_type", "none")
        if error_type not in VALID_ERROR_TYPES:
            error_type = "none"

        decision = data.get("decision", "warn")
        if decision not in VALID_DECISIONS:
            from src.data.schema import LABEL_TO_DEFAULT_DECISION
            decision = LABEL_TO_DEFAULT_DECISION.get(label, "warn")

        return VerifierOutput(
            label=label,
            hallucination_score=score,
            error_type=error_type,
            evidence_span=data.get("evidence_span", ""),
            explanation=data.get("explanation", ""),
            decision=decision,
            parse_error=False,
            raw_output=None,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parse error sentinel
# ---------------------------------------------------------------------------

def _parse_error_sentinel(raw_output: str, reason: str) -> VerifierOutput:
    """Return a sentinel VerifierOutput indicating parse failure."""
    return VerifierOutput(
        label="insufficient_context",  # Safe default
        hallucination_score=0.5,
        error_type="none",
        evidence_span="",
        explanation=f"[PARSE ERROR: {reason}]",
        decision="warn",
        parse_error=True,
        raw_output=raw_output_truncated(raw_output),
    )


def raw_output_truncated(text: str, max_chars: int = 500) -> str:
    """Truncate raw output for storage."""
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


# ---------------------------------------------------------------------------
# Batch parsing with statistics
# ---------------------------------------------------------------------------

def parse_batch(raw_outputs: list[str]) -> tuple[list[VerifierOutput], dict]:
    """
    Parse a batch of raw outputs and return results with statistics.

    Returns:
        (results, stats) where stats includes json_validity_rate.
    """
    results = [parse_verifier_output(raw) for raw in raw_outputs]

    n_total = len(results)
    n_parse_errors = sum(1 for r in results if r.parse_error)
    n_valid = n_total - n_parse_errors

    stats = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_parse_errors": n_parse_errors,
        "json_validity_rate": n_valid / n_total if n_total > 0 else 0.0,
    }

    return results, stats