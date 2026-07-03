"""
src/data/context_builder.py
============================
Builds and manages multi-chunk context strings for the verifier.
Handles context truncation with priority-based strategy.

Usage:
    from src.data.context_builder import build_context, truncate_context
    context = build_context(["chunk1 text", "chunk2 text"])
    # → "[Kaynak 1]: chunk1 text\n\n---\n\n[Kaynak 2]: chunk2 text"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Separator between chunks in the combined context
CHUNK_SEPARATOR = "\n\n---\n\n"
CHUNK_PREFIX = "[Kaynak {i}]: "


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def build_context(chunks: list[str]) -> str:
    """
    Combine multiple retrieved chunks into a single context string.
    Each chunk is prefixed with [Kaynak N]: for source attribution.

    Args:
        chunks: List of retrieved text chunks.

    Returns:
        Combined context string.

    Example:
        >>> build_context(["Metin 1", "Metin 2"])
        "[Kaynak 1]: Metin 1\n\n---\n\n[Kaynak 2]: Metin 2"
    """
    if not chunks:
        return ""
    if len(chunks) == 1:
        return chunks[0]  # Single chunk: no prefix needed

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        prefix = CHUNK_PREFIX.format(i=i)
        parts.append(f"{prefix}{chunk.strip()}")

    return CHUNK_SEPARATOR.join(parts)


def parse_context_chunks(context: str) -> list[str]:
    """
    Parse a combined context string back into individual chunks.
    Inverse of build_context().

    Args:
        context: Combined context string (may or may not have [Kaynak N]: prefixes).

    Returns:
        List of chunk strings (without prefixes).
    """
    if CHUNK_SEPARATOR not in context:
        # Single chunk or no separator
        return [context.strip()]

    parts = context.split(CHUNK_SEPARATOR)
    chunks = []
    for part in parts:
        part = part.strip()
        # Remove [Kaynak N]: prefix if present
        import re
        part = re.sub(r'^\[Kaynak \d+\]:\s*', '', part)
        if part:
            chunks.append(part)
    return chunks


def get_source_index_from_span(evidence_span: str) -> Optional[int]:
    """
    Extract the source index from an evidence span that includes [Kaynak N]: prefix.

    Args:
        evidence_span: e.g. "[Kaynak 2]: 1955 yılında yayımlanan"

    Returns:
        0-indexed source index, or None if no prefix found.

    Example:
        >>> get_source_index_from_span("[Kaynak 2]: 1955 yılında")
        1  # 0-indexed
    """
    import re
    match = re.match(r'^\[Kaynak (\d+)\]:', evidence_span.strip())
    if match:
        return int(match.group(1)) - 1  # Convert to 0-indexed
    return None


# ---------------------------------------------------------------------------
# Context truncation
# ---------------------------------------------------------------------------

@dataclass
class TruncationResult:
    context: str
    was_truncated: bool
    original_chunk_count: int
    remaining_chunk_count: int
    removed_chunk_indices: list[int]  # 0-indexed


def truncate_context(
    chunks: list[str],
    tokenizer,
    max_context_tokens: int,
    question: str = "",
    answer: str = "",
    claim: str = "",
) -> TruncationResult:
    """
    Truncate context chunks to fit within token budget.

    Priority (never truncate):
    1. System prompt (handled externally)
    2. Claim
    3. Question
    4. Answer
    5. Context → remove last chunks first (whole chunks, never mid-chunk)
       At least 1 chunk must always remain.

    Args:
        chunks:             List of retrieved text chunks.
        tokenizer:          HuggingFace tokenizer for token counting.
        max_context_tokens: Maximum tokens allowed for context.
        question:           User question (for token budget calculation).
        answer:             Generated answer (for token budget calculation).
        claim:              Claim being verified (for token budget calculation).

    Returns:
        TruncationResult with the truncated context and metadata.
    """
    if not chunks:
        return TruncationResult(
            context="",
            was_truncated=False,
            original_chunk_count=0,
            remaining_chunk_count=0,
            removed_chunk_indices=[],
        )

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Count tokens used by non-context fields
    fixed_tokens = (
        count_tokens(question) +
        count_tokens(answer) +
        count_tokens(claim)
    )
    available_tokens = max_context_tokens - fixed_tokens

    if available_tokens <= 0:
        # Extreme case: keep only first chunk
        return TruncationResult(
            context=build_context([chunks[0]]),
            was_truncated=True,
            original_chunk_count=len(chunks),
            remaining_chunk_count=1,
            removed_chunk_indices=list(range(1, len(chunks))),
        )

    # Try to fit as many chunks as possible (from first to last)
    kept_chunks = []
    removed_indices = []
    token_budget = available_tokens

    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens <= token_budget:
            kept_chunks.append(chunk)
            token_budget -= chunk_tokens
        else:
            removed_indices.append(i)

    # Ensure at least 1 chunk
    if not kept_chunks:
        kept_chunks = [chunks[0]]
        removed_indices = list(range(1, len(chunks)))

    was_truncated = len(removed_indices) > 0

    return TruncationResult(
        context=build_context(kept_chunks),
        was_truncated=was_truncated,
        original_chunk_count=len(chunks),
        remaining_chunk_count=len(kept_chunks),
        removed_chunk_indices=removed_indices,
    )


# ---------------------------------------------------------------------------
# Wikipedia TR quality filter
# ---------------------------------------------------------------------------

def filter_wikipedia_article(text: str, min_words: int = 500) -> bool:
    """
    Filter Wikipedia TR articles for quality.
    Returns True if the article passes quality checks.

    Filters:
    - Minimum word count (stubs are too short)
    - Machine translation indicators
    - Disambiguation pages

    Args:
        text:      Article text.
        min_words: Minimum word count threshold.

    Returns:
        True if article passes quality filter.
    """
    import re

    if not text or not text.strip():
        return False

    words = text.split()
    if len(words) < min_words:
        return False

    # Stub indicators
    stub_patterns = [
        r'bu madde bir taslaktır',
        r'bu makale bir taslaktır',
        r'stub',
        r'bu kısa makale',
    ]
    text_lower = text.lower()
    for pattern in stub_patterns:
        if re.search(pattern, text_lower):
            return False

    # Disambiguation page indicators
    disambiguation_patterns = [
        r'anlam ayrımı',
        r'bu sayfa.*anlam',
        r'disambiguation',
    ]
    for pattern in disambiguation_patterns:
        if re.search(pattern, text_lower):
            return False

    # Machine translation indicators (common MT artifacts in Turkish Wikipedia)
    mt_patterns = [
        r'\bbu makale.*çevrilmiştir\b',
        r'\botomatik çeviri\b',
    ]
    for pattern in mt_patterns:
        if re.search(pattern, text_lower):
            return False

    return True