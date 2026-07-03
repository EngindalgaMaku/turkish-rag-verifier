"""
src/data/claim_extraction.py
=============================
Rule-based and LLM-based claim extraction from Turkish answers.

Phase 1 (pilot): Rule-based — each sentence = one claim.
Phase 2: LLM-based — atomic, independent, verifiable claims.

Usage:
    from src.data.claim_extraction import extract_claims_rule_based
    claims = extract_claims_rule_based("İnce Memed 1955'te yayımlandı. Yaşar Kemal yazdı.")
    # → ["İnce Memed 1955'te yayımlandı.", "Yaşar Kemal yazdı."]
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Turkish abbreviations that contain a period but are NOT sentence endings
# ---------------------------------------------------------------------------
TURKISH_ABBREVIATIONS = [
    "Dr", "Prof", "Doç", "Yrd", "Öğr", "Gör", "Arş", "Uzm",
    "Mh", "Cad", "Sok", "Blv", "vb", "vs", "bkz", "örn",
    "mad", "No", "Sn", "Hz", "Haz", "Ağu", "Eyl", "Eki",
    "Kas", "Ara", "Oca", "Şub", "Mar", "Nis", "May", "Haz",
    "Tem", "s", "sf", "sy", "vol", "ed", "çev",
]

# Conjunctions that often join two independent claims
SPLIT_CONJUNCTIONS = [
    " ve ayrıca ",
    " bunun yanı sıra ",
    " öte yandan ",
    " bununla birlikte ",
    " aynı zamanda ",
    # Adversative conjunctions — strong claim boundary signals
    " ama metne göre ",
    " fakat metne göre ",
    " ancak metne göre ",
    " ama bağlama göre ",
    " fakat bağlama göre ",
    " ancak bağlama göre ",
    # Additional strong boundaries
    " iken ",
    " oysa ",
    " halbuki ",
    " ne var ki ",
]

# Patterns indicating multiple entities/versions in one sentence
# These trigger splitting even without explicit conjunctions
MULTI_ENTITY_PATTERNS = [
    # Multiple version numbers: HTTP/1.1 ... HTTP/2.0 ... HTTP/3.0
    r'(HTTP|SSL|TLS|TCP|UDP|IP|FTP|SMTP|DNS|REST|SOAP|gRPC|SQL|NoSQL|CSS|HTML|XML|JSON|YAML|API|SDK|OS|CPU|GPU|RAM|ROM|SSD|HDD|USB|WiFi|Bluetooth|IPv4|IPv6)\s*/?\s*[\d.]+.*?(HTTP|SSL|TLS|TCP|UDP|IP|FTP|SMTP|DNS|REST|SOAP|gRPC|SQL|NoSQL|CSS|HTML|XML|JSON|YAML|API|SDK|OS|CPU|GPU|RAM|ROM|SSD|HDD|USB|WiFi|Bluetooth|IPv4|IPv6)\s*/?\s*[\d.]+',
    # Multiple version numbers: v1.0 ... v2.0
    r'v[\d.]+.*?v[\d.]+',
    # Multiple years: 1923 ... 1938
    r'\b(1[0-9]{3}|20[0-9]{2})\b.*?\b(1[0-9]{3}|20[0-9]{2})\b',
]

# Minimum claim length in characters
MIN_CLAIM_LENGTH = 10

# Phrases that indicate non-verifiable content (opinions, uncertainty)
NON_VERIFIABLE_PATTERNS = [
    r"\bbence\b",
    r"\bsanırım\b",
    r"\bgaliba\b",
    r"\bolabilir\b",
    r"\bbelki\b",
    r"\bmuhtemelen\b",
    r"\btahmin\b",
]


# ---------------------------------------------------------------------------
# Phase 1: Rule-based extraction
# ---------------------------------------------------------------------------

def extract_claims_rule_based(answer: str) -> list[str]:
    """
    Extract claims from a Turkish answer using rule-based sentence splitting.

    Rules:
    1. Split on sentence boundaries (. ! ?)
    2. Protect Turkish abbreviations from false splits
    3. Split compound sentences joined by certain conjunctions
    4. Filter out very short or non-verifiable claims

    Args:
        answer: The full LLM-generated answer text.

    Returns:
        List of claim strings.
    """
    if not answer or not answer.strip():
        return []

    # Step 1: Protect abbreviations
    protected = _protect_abbreviations(answer)

    # Step 2: Split on sentence boundaries
    sentences = _split_sentences(protected)

    # Step 3: Restore abbreviations
    sentences = [_restore_abbreviations(s) for s in sentences]

    # Step 4: Split compound sentences
    all_claims = []
    for sent in sentences:
        sub_claims = _split_compound(sent)
        all_claims.extend(sub_claims)

    # Step 5: Clean and filter
    claims = []
    for claim in all_claims:
        claim = claim.strip()
        if _is_valid_claim(claim):
            claims.append(claim)

    return claims


def _protect_abbreviations(text: str) -> str:
    """Replace periods in abbreviations with a placeholder."""
    for abbr in TURKISH_ABBREVIATIONS:
        # Match abbreviation followed by period (word boundary)
        pattern = r'\b' + re.escape(abbr) + r'\.'
        text = re.sub(pattern, abbr + '<DOT>', text)
    return text


def _restore_abbreviations(text: str) -> str:
    """Restore abbreviation placeholders."""
    return text.replace('<DOT>', '.')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on . ! ? boundaries."""
    # Split after sentence-ending punctuation followed by whitespace or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_compound(sentence: str) -> list[str]:
    """
    Split a sentence that contains compound claims joined by conjunctions.
    Only splits if both parts are long enough to be independent claims.
    Also splits on plain 've' when both parts contain different verifiable entities.
    """
    # 1. Try strong conjunctions first
    for conj in SPLIT_CONJUNCTIONS:
        if conj in sentence.lower():
            idx = sentence.lower().find(conj)
            part1 = sentence[:idx].strip()
            part2 = sentence[idx + len(conj):].strip()
            if len(part1) >= MIN_CLAIM_LENGTH and len(part2) >= MIN_CLAIM_LENGTH:
                return [part1, part2]

    # 2. Try plain " ve " — only when multi-entity pattern detected
    if _has_multi_entity(sentence) and " ve " in sentence:
        idx = sentence.find(" ve ")
        part1 = sentence[:idx].strip()
        part2 = sentence[idx + 4:].strip()
        if len(part1) >= MIN_CLAIM_LENGTH and len(part2) >= MIN_CLAIM_LENGTH:
            return [part1, part2]

    return [sentence]


def _has_multi_entity(sentence: str) -> bool:
    """Return True if sentence contains multiple technology versions, years, or entities."""
    import re as _re
    for pattern in MULTI_ENTITY_PATTERNS:
        if _re.search(pattern, sentence, _re.IGNORECASE):
            return True
    return False


def _is_valid_claim(claim: str) -> bool:
    """Return True if the claim is worth verifying."""
    if len(claim) < MIN_CLAIM_LENGTH:
        return False
    # Filter out pure questions
    if claim.endswith('?'):
        return False
    # Filter out non-verifiable opinion markers
    claim_lower = claim.lower()
    for pattern in NON_VERIFIABLE_PATTERNS:
        if re.search(pattern, claim_lower):
            return False
    return True


# ---------------------------------------------------------------------------
# Phase 2: LLM-based extraction (stub — implement when pilot is complete)
# ---------------------------------------------------------------------------

def extract_claims_llm(
    answer: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    prompt_path: Optional[str] = None,
    max_claims: int = 10,
) -> list[str]:
    """
    Extract claims using an LLM.
    Produces more atomic and independent claims than the rule-based approach.

    NOTE: This is a stub. Implement after pilot is complete.
    The LLM should be called with the claim_extraction prompt template.

    Args:
        answer:      The full LLM-generated answer text.
        model_name:  HuggingFace model name.
        prompt_path: Path to claim extraction prompt template.
        max_claims:  Maximum number of claims to extract.

    Returns:
        List of claim strings.
    """
    raise NotImplementedError(
        "LLM-based claim extraction is not yet implemented. "
        "Use extract_claims_rule_based() for the pilot phase."
    )


# ---------------------------------------------------------------------------
# Error propagation analysis helper
# ---------------------------------------------------------------------------

def compare_claim_sets(
    ground_truth_claims: list[str],
    extracted_claims: list[str],
) -> dict:
    """
    Compare ground truth claims with automatically extracted claims.
    Used for error propagation analysis (scripts/06_error_analysis.py --mode propagation).

    Returns:
        dict with precision, recall, f1, and per-claim analysis.
    """
    if not ground_truth_claims:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "details": []}

    # Simple token-overlap matching
    def token_overlap(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    MATCH_THRESHOLD = 0.5

    matched_gt = set()
    matched_ex = set()

    for i, gt in enumerate(ground_truth_claims):
        for j, ex in enumerate(extracted_claims):
            if token_overlap(gt, ex) >= MATCH_THRESHOLD:
                matched_gt.add(i)
                matched_ex.add(j)

    precision = len(matched_ex) / len(extracted_claims) if extracted_claims else 0.0
    recall = len(matched_gt) / len(ground_truth_claims) if ground_truth_claims else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_ground_truth": len(ground_truth_claims),
        "n_extracted": len(extracted_claims),
        "n_matched": len(matched_gt),
    }