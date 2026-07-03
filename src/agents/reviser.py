"""
src/agents/reviser.py
======================
Answer reviser for the Turkish RAG Hallucination Verifier pipeline.

Takes the original answer, per-claim verdicts, and the policy decision,
then produces a revised answer that removes or corrects hallucinated claims.

Architecture position:
    PolicyController (revise/reject) → Reviser → revised_answer

Revision strategies (v0 — rule-based):
    1. DROP:    Remove contradicted or unsupported claims entirely.
    2. HEDGE:   Wrap unsupported claims with uncertainty markers.
    3. REPLACE: Replace contradicted claims with evidence_span from context.
    4. TRUNCATE: Keep only supported claims.

Usage:
    from src.agents.reviser import Reviser
    from src.agents.policy_controller import ClaimVerdict, PolicyDecision

    reviser = Reviser(strategy="drop")
    revised = reviser.revise(
        original_answer="...",
        claim_verdicts=[...],
        policy_decision=policy_decision,
    )
    print(revised.revised_answer)
    print(revised.revision_log)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.agents.policy_controller import ClaimVerdict, PolicyDecision


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RevisionResult:
    """Result of a revision operation."""
    original_answer: str
    revised_answer: str
    strategy: str                           # drop | hedge | replace | truncate
    n_claims_removed: int = 0
    n_claims_hedged: int = 0
    n_claims_replaced: int = 0
    n_claims_kept: int = 0
    revision_log: List[str] = field(default_factory=list)   # Per-claim actions
    no_revision_needed: bool = False


# ---------------------------------------------------------------------------
# Reviser
# ---------------------------------------------------------------------------

class Reviser:
    """
    Rule-based answer reviser (v0).

    Strategies:
        "drop"     — Remove problematic claims. Safest, may lose information.
        "hedge"    — Add uncertainty markers to unsupported claims.
        "replace"  — Replace contradicted claims with evidence_span from context.
        "truncate" — Keep only fully supported claims.

    Args:
        strategy: Revision strategy. Default: "drop".
    """

    # Turkish uncertainty hedges
    HEDGE_PREFIX = "Bağlama göre doğrulanamayan bilgi: "
    HEDGE_SUFFIX = " (bu bilgi bağlamda desteklenmemektedir)"

    # Labels that trigger revision (hard problems — claim is wrong or absent)
    PROBLEMATIC_LABELS = {"contradicted", "unsupported"}
    # Labels that are kept but hedged (claim may be right, just not fully confirmed)
    HEDGE_LABELS = {"partially_supported"}
    # Labels that are kept as-is
    SAFE_LABELS = {"supported", "insufficient_context"}

    def __init__(self, strategy: str = "drop") -> None:
        valid_strategies = {"drop", "hedge", "replace", "truncate"}
        if strategy not in valid_strategies:
            raise ValueError(
                f"Unknown revision strategy: {strategy!r}. "
                f"Choose from: {valid_strategies}"
            )
        self.strategy = strategy

    def revise(
        self,
        original_answer: str,
        claim_verdicts: List[ClaimVerdict],
        policy_decision: Optional[PolicyDecision] = None,
    ) -> RevisionResult:
        """
        Revise the answer based on claim verdicts.

        Args:
            original_answer: The original LLM-generated answer.
            claim_verdicts:  Per-claim verifier outputs.
            policy_decision: Optional policy decision (used for context).

        Returns:
            RevisionResult with revised_answer and revision log.
        """
        # If no revision needed (accept decision)
        if policy_decision and policy_decision.final_decision == "accept":
            return RevisionResult(
                original_answer=original_answer,
                revised_answer=original_answer,
                strategy=self.strategy,
                n_claims_kept=len(claim_verdicts),
                no_revision_needed=True,
                revision_log=["No revision needed — all claims accepted."],
            )

        if not claim_verdicts:
            return RevisionResult(
                original_answer=original_answer,
                revised_answer=original_answer,
                strategy=self.strategy,
                revision_log=["No claims to revise."],
            )

        if self.strategy == "drop":
            return self._revise_drop(original_answer, claim_verdicts)
        elif self.strategy == "hedge":
            return self._revise_hedge(original_answer, claim_verdicts)
        elif self.strategy == "replace":
            return self._revise_replace(original_answer, claim_verdicts)
        elif self.strategy == "truncate":
            return self._revise_truncate(original_answer, claim_verdicts)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    # ------------------------------------------------------------------
    # Strategy: DROP — remove problematic claims
    # ------------------------------------------------------------------

    def _revise_drop(
        self,
        original_answer: str,
        claim_verdicts: List[ClaimVerdict],
    ) -> RevisionResult:
        """Remove contradicted/unsupported claims; hedge partially_supported ones."""
        kept_claims = []
        removed_claims = []
        n_hedged = 0
        log = []

        for v in claim_verdicts:
            if v.label in self.PROBLEMATIC_LABELS:
                removed_claims.append(v.claim)
                log.append(
                    f"DROPPED [{v.label}]: {v.claim[:80]}..."
                    if len(v.claim) > 80 else f"DROPPED [{v.label}]: {v.claim}"
                )
            elif v.label in self.HEDGE_LABELS:
                # Partially supported: keep the claim but add a soft hedge
                hedged = f"{v.claim} (bağlamda tam olarak doğrulanamadı)"
                kept_claims.append(hedged)
                n_hedged += 1
                log.append(
                    f"HEDGED [{v.label}]: {v.claim[:80]}..."
                    if len(v.claim) > 80 else f"HEDGED [{v.label}]: {v.claim}"
                )
            else:
                kept_claims.append(v.claim)
                log.append(
                    f"KEPT [{v.label}]: {v.claim[:80]}..."
                    if len(v.claim) > 80 else f"KEPT [{v.label}]: {v.claim}"
                )

        if not kept_claims:
            revised = (
                "Bu soruya verilen cevap bağlam tarafından desteklenmemektedir. "
                "Lütfen soruyu farklı bir şekilde sorun veya daha fazla bilgi sağlayın."
            )
            log.append("All claims removed — fallback message used.")
        else:
            revised = " ".join(kept_claims)

        return RevisionResult(
            original_answer=original_answer,
            revised_answer=revised,
            strategy="drop",
            n_claims_removed=len(removed_claims),
            n_claims_hedged=n_hedged,
            n_claims_kept=len(kept_claims),
            revision_log=log,
        )

    # ------------------------------------------------------------------
    # Strategy: HEDGE — add uncertainty markers
    # ------------------------------------------------------------------

    def _revise_hedge(
        self,
        original_answer: str,
        claim_verdicts: List[ClaimVerdict],
    ) -> RevisionResult:
        """Wrap unsupported claims with uncertainty language."""
        revised_parts = []
        n_hedged = 0
        n_kept = 0
        log = []

        for v in claim_verdicts:
            if v.label == "contradicted":
                # Contradicted claims are dropped even in hedge mode
                log.append(f"DROPPED [contradicted]: {v.claim[:60]}")
            elif v.label in ("unsupported", "partially_supported"):
                hedged = f"{v.claim} (bağlamda tam olarak doğrulanamadı)"
                revised_parts.append(hedged)
                n_hedged += 1
                log.append(f"HEDGED [{v.label}]: {v.claim[:60]}")
            else:
                revised_parts.append(v.claim)
                n_kept += 1
                log.append(f"KEPT [{v.label}]: {v.claim[:60]}")

        revised = " ".join(revised_parts) if revised_parts else (
            "Bu soruya verilen cevap bağlam tarafından desteklenmemektedir."
        )

        return RevisionResult(
            original_answer=original_answer,
            revised_answer=revised,
            strategy="hedge",
            n_claims_hedged=n_hedged,
            n_claims_kept=n_kept,
            revision_log=log,
        )

    # ------------------------------------------------------------------
    # Strategy: REPLACE — substitute contradicted claims with evidence
    # ------------------------------------------------------------------

    def _revise_replace(
        self,
        original_answer: str,
        claim_verdicts: List[ClaimVerdict],
    ) -> RevisionResult:
        """
        Replace contradicted claims with the evidence_span from context.
        Falls back to DROP for unsupported claims (no evidence to replace with).
        """
        revised_parts = []
        n_replaced = 0
        n_removed = 0
        n_kept = 0
        log = []

        for v in claim_verdicts:
            if v.label == "contradicted" and v.evidence_span:
                # Replace with the correct information from context
                revised_parts.append(v.evidence_span.strip())
                n_replaced += 1
                log.append(
                    f"REPLACED [contradicted]: '{v.claim[:50]}' → '{v.evidence_span[:50]}'"
                )
            elif v.label in self.PROBLEMATIC_LABELS:
                # No evidence to replace with → drop
                n_removed += 1
                log.append(f"DROPPED [{v.label}]: {v.claim[:60]}")
            else:
                revised_parts.append(v.claim)
                n_kept += 1
                log.append(f"KEPT [{v.label}]: {v.claim[:60]}")

        revised = " ".join(revised_parts) if revised_parts else (
            "Bu soruya verilen cevap bağlam tarafından desteklenmemektedir."
        )

        return RevisionResult(
            original_answer=original_answer,
            revised_answer=revised,
            strategy="replace",
            n_claims_replaced=n_replaced,
            n_claims_removed=n_removed,
            n_claims_kept=n_kept,
            revision_log=log,
        )

    # ------------------------------------------------------------------
    # Strategy: TRUNCATE — keep only fully supported claims
    # ------------------------------------------------------------------

    def _revise_truncate(
        self,
        original_answer: str,
        claim_verdicts: List[ClaimVerdict],
    ) -> RevisionResult:
        """Keep only claims with label == 'supported'."""
        supported_claims = [v.claim for v in claim_verdicts if v.label == "supported"]
        removed = [v for v in claim_verdicts if v.label != "supported"]
        log = []

        for v in claim_verdicts:
            if v.label == "supported":
                log.append(f"KEPT [supported]: {v.claim[:60]}")
            else:
                log.append(f"REMOVED [{v.label}]: {v.claim[:60]}")

        revised = " ".join(supported_claims) if supported_claims else (
            "Bu soruya verilen cevap bağlam tarafından desteklenmemektedir."
        )

        return RevisionResult(
            original_answer=original_answer,
            revised_answer=revised,
            strategy="truncate",
            n_claims_removed=len(removed),
            n_claims_kept=len(supported_claims),
            revision_log=log,
        )