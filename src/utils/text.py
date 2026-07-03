"""
src/utils/text.py
=================
Turkish text processing utilities.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Turkish character normalization
# ---------------------------------------------------------------------------

TURKISH_LOWER_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")
TURKISH_UPPER_MAP = str.maketrans("iığüşöç", "İIĞÜŞÖÇ")


def turkish_lower(text: str) -> str:
    """Lowercase Turkish text correctly (handles İ → i, I → ı)."""
    return text.translate(TURKISH_LOWER_MAP).lower()


def turkish_upper(text: str) -> str:
    """Uppercase Turkish text correctly."""
    return text.translate(TURKISH_UPPER_MAP).upper()


def normalize_text(text: str) -> str:
    """
    Normalize Turkish text for comparison:
    - Unicode NFC normalization
    - Strip leading/trailing whitespace
    - Collapse multiple spaces
    """
    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


# ---------------------------------------------------------------------------
# Token counting estimate (without loading a tokenizer)
# ---------------------------------------------------------------------------

def estimate_token_count(text: str, lang_factor: float = 1.4) -> int:
    """
    Estimate token count for Turkish text without loading a tokenizer.
    Turkish is agglutinative: ~1.3–1.5x more tokens than English per word.

    Args:
        text:        Input text.
        lang_factor: Multiplier for Turkish (default 1.4).

    Returns:
        Estimated token count.
    """
    words = text.split()
    return int(len(words) * lang_factor)


# ---------------------------------------------------------------------------
# Evidence span utilities
# ---------------------------------------------------------------------------

def find_span_in_context(span: str, context: str) -> bool:
    """
    Check if an evidence span appears in the context (case-insensitive).

    Args:
        span:    Evidence span string.
        context: Full context string.

    Returns:
        True if span is found in context.
    """
    if not span or not context:
        return False
    return turkish_lower(span.strip()) in turkish_lower(context)


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute token-level F1 between predicted and ground truth spans.
    Used for evidence_span evaluation.

    Args:
        prediction:   Predicted evidence span.
        ground_truth: Ground truth evidence span.

    Returns:
        Token F1 score [0, 1].
    """
    pred_tokens = set(turkish_lower(prediction).split())
    gt_tokens = set(turkish_lower(ground_truth).split())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


# ---------------------------------------------------------------------------
# Text truncation
# ---------------------------------------------------------------------------

def truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to at most max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."