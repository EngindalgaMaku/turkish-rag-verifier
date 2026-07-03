"""
src/data/schema.py
==================
Pydantic data model for Turkish RAG Hallucination Verifier.
Loads all valid values from configs/labels.yaml.

Usage:
    from src.data.schema import VerifierExample, load_label_config
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_label_config(config_path: Optional[str] = None) -> dict:
    """Load labels.yaml from configs/ directory."""
    if config_path is None:
        # Default: look relative to project root
        root = Path(__file__).resolve().parents[2]
        config_path = root / "configs" / "labels.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load config once at module level
_CONFIG = load_label_config()

VALID_LABELS: set[str] = set(_CONFIG["labels"])
VALID_ERROR_TYPES: set[str] = set(_CONFIG["error_types"])
VALID_DECISIONS: set[str] = set(_CONFIG["decisions"])
VALID_SPLITS: set[str] = set(_CONFIG["splits"])
VALID_SOURCE_TYPES: set[str] = set(_CONFIG["source_types"])
VALID_DOMAINS: set[str] = set(_CONFIG["domains"])
SCORE_RANGES: dict = _CONFIG["score_ranges"]
LABEL_TO_DEFAULT_DECISION: dict = _CONFIG["label_to_default_decision"]


# ---------------------------------------------------------------------------
# Pydantic Model
# ---------------------------------------------------------------------------

class VerifierExample(BaseModel):
    """
    Single claim-level example for the Turkish RAG Hallucination Verifier.
    All required fields must be present. Optional fields default to None / "".
    """

    # --- Required fields ---
    id: str = Field(..., description="Unique example ID, format: trrag_XXXXXX")
    question: str = Field(..., min_length=5, description="User question")
    context: str = Field(..., min_length=10, description="Retrieved RAG context (may be multi-chunk)")
    answer: str = Field(..., min_length=1, description="Full LLM-generated answer")
    claim: str = Field(..., min_length=5, description="Single claim to verify")
    label: str = Field(..., description="Verification label")
    hallucination_score: float = Field(..., ge=0.0, le=1.0, description="Hallucination severity score [0,1]")
    error_type: str = Field(..., description="Error type (none for supported)")
    evidence_span: str = Field(default="", description="Verbatim span from context (required for supported/contradicted)")
    explanation: str = Field(..., min_length=10, description="Turkish explanation of the decision")
    decision: str = Field(..., description="Action decision")
    source_type: str = Field(..., description="Data source type")
    split: str = Field(..., description="Dataset split")

    # --- Recommended fields ---
    claim_index: Optional[int] = Field(default=0, ge=0, description="0-indexed position of claim in answer")
    claim_count_in_answer: Optional[int] = Field(default=1, ge=1, description="Total claims in answer")
    context_source_count: Optional[int] = Field(default=1, ge=1, description="Number of chunks in context")
    context_truncated: Optional[bool] = Field(default=False, description="Was context truncated?")
    evidence_source_index: Optional[int] = Field(default=0, ge=0, description="Which chunk the evidence came from")
    generator_model: Optional[str] = Field(default=None, description="Model that generated the answer")
    prompt_version: Optional[str] = Field(default="v1.0", description="Prompt version used")
    annotator_id: Optional[str] = Field(default=None, description="Annotator identifier")
    annotation_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Annotator confidence [0,1]")
    annotation_round: Optional[int] = Field(default=1, ge=1, description="Annotation round number")
    domain: Optional[str] = Field(default="other", description="Content domain")
    notes: Optional[str] = Field(default="", description="Free-text notes")

    # --- Optional Phase 2 fields ---
    web_evidence: Optional[str] = Field(default=None)
    web_support: Optional[str] = Field(default=None)
    llm_consistency: Optional[str] = Field(default=None)
    document_support: Optional[str] = Field(default=None)
    external_support: Optional[str] = Field(default=None)
    final_label: Optional[str] = Field(default=None)

    # ---------------------------------------------------------------------------
    # Field validators
    # ---------------------------------------------------------------------------

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"Invalid label '{v}'. Must be one of: {sorted(VALID_LABELS)}")
        return v

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        if v not in VALID_ERROR_TYPES:
            raise ValueError(f"Invalid error_type '{v}'. Must be one of: {sorted(VALID_ERROR_TYPES)}")
        return v

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        if v not in VALID_DECISIONS:
            raise ValueError(f"Invalid decision '{v}'. Must be one of: {sorted(VALID_DECISIONS)}")
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str) -> str:
        if v not in VALID_SPLITS:
            raise ValueError(f"Invalid split '{v}'. Must be one of: {sorted(VALID_SPLITS)}")
        return v

    @field_validator("source_type")
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        if v not in VALID_SOURCE_TYPES:
            raise ValueError(f"Invalid source_type '{v}'. Must be one of: {sorted(VALID_SOURCE_TYPES)}")
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_DOMAINS:
            raise ValueError(f"Invalid domain '{v}'. Must be one of: {sorted(VALID_DOMAINS)}")
        return v

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        if not v.startswith("trrag_"):
            raise ValueError(f"ID must start with 'trrag_', got: '{v}'")
        return v

    # ---------------------------------------------------------------------------
    # Cross-field validators
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_consistency(self) -> "VerifierExample":
        errors = []

        # Rule 1: supported → error_type must be none
        if self.label == "supported" and self.error_type != "none":
            errors.append(
                f"label='supported' requires error_type='none', got '{self.error_type}'"
            )

        # Rule 2: supported or contradicted → evidence_span must not be empty
        if self.label in {"supported", "contradicted"} and not self.evidence_span.strip():
            errors.append(
                f"label='{self.label}' requires a non-empty evidence_span"
            )

        # Rule 3: claim_index < claim_count_in_answer
        if (
            self.claim_index is not None
            and self.claim_count_in_answer is not None
            and self.claim_index >= self.claim_count_in_answer
        ):
            errors.append(
                f"claim_index ({self.claim_index}) must be < claim_count_in_answer ({self.claim_count_in_answer})"
            )

        # Rule 4: hallucination_score within label's range
        # This is a GUIDELINE check, not a hard error.
        # Score ranges are annotator guidance; real annotations may legitimately
        # fall outside the range (e.g. a very risky unsupported claim at 0.90).
        # Violations are stored as warnings in self._score_range_warnings for
        # reporting by validate_jsonl.py, but do NOT block parsing.
        score_range = SCORE_RANGES.get(self.label)
        score_warnings = []
        if score_range:
            lo, hi = score_range["min"], score_range["max"]
            if not (lo <= self.hallucination_score <= hi):
                score_warnings.append(
                    f"hallucination_score={self.hallucination_score} is outside the guideline range "
                    f"[{lo}, {hi}] for label='{self.label}' (guideline only, not an error)"
                )
        # Store warnings as a non-validated attribute for external inspection
        object.__setattr__(self, "_score_range_warnings", score_warnings)

        if errors:
            raise ValueError("; ".join(errors))

        return self

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def get_default_decision(self) -> str:
        """Return the default decision for this label."""
        return LABEL_TO_DEFAULT_DECISION.get(self.label, "warn")

    def is_hallucinated(self) -> bool:
        """Return True if the claim is considered hallucinated."""
        return self.label in {"contradicted", "unsupported"}

    def to_training_dict(self) -> dict:
        """Return only the fields needed for training."""
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "answer": self.answer,
            "claim": self.claim,
            "label": self.label,
            "hallucination_score": self.hallucination_score,
            "error_type": self.error_type,
            "evidence_span": self.evidence_span,
            "explanation": self.explanation,
            "decision": self.decision,
            "prompt_version": self.prompt_version,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# Model output schema (what the verifier produces at inference time)
# ---------------------------------------------------------------------------

class VerifierOutput(BaseModel):
    """
    Structured output produced by the verifier model for a single claim.
    This is what gets parsed from the model's JSON output.
    """
    label: str
    hallucination_score: float = Field(ge=0.0, le=1.0)
    error_type: str
    evidence_span: str = ""
    explanation: str
    decision: str

    # Set to True if the JSON was malformed and we used fallback parsing
    parse_error: bool = False
    raw_output: Optional[str] = None  # Original model output (for debugging)

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        if v not in VALID_LABELS and v != "parse_error":
            raise ValueError(f"Invalid label '{v}'")
        return v

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, v: str) -> str:
        if v not in VALID_ERROR_TYPES:
            return "none"  # Soft fallback for model output
        return v

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        if v not in VALID_DECISIONS:
            return "warn"  # Soft fallback for model output
        return v


# ---------------------------------------------------------------------------
# Aggregate result (answer-level summary across all claims)
# ---------------------------------------------------------------------------

class AggregateVerificationResult(BaseModel):
    """
    Answer-level summary produced after verifying all claims in an answer.
    """
    answer_id: str
    question: str
    answer: str
    claim_results: list[VerifierOutput]
    aggregate_decision: str
    aggregate_risk_score: float = Field(ge=0.0, le=1.0)
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_unsupported: int
    n_partially_supported: int
    n_insufficient_context: int
    processing_time_ms: Optional[float] = None
