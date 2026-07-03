"""
src/agents/policy_controller.py
================================
Controller / Agent Policy for the Turkish RAG Hallucination Verifier.

Takes per-claim verifier outputs and produces a final decision for the
whole answer: accept / warn / revise / reject / web_check.

Architecture position:
    Qwen verifier (per claim) → PolicyController → final decision + action

Decision logic (rule-based v0):
    - ALL claims supported                          → accept
    - Any claim contradicted                        → reject (or revise)
    - Any claim unsupported + no contradiction      → warn (or web_check)
    - Any claim insufficient_context                → web_check
    - Any claim partially_supported                 → warn
    - hallucination_score aggregate > threshold     → warn/reject

Usage:
    from src.agents.policy_controller import PolicyController, ClaimVerdict
    controller = PolicyController()
    result = controller.decide(claim_verdicts)
    print(result.final_decision)  # "accept" | "warn" | "revise" | "reject" | "web_check"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    """
    Verifier output for a single claim.
    Mirrors VerifierOutput from src/data/schema.py but decoupled for agent use.
    """
    claim: str
    label: str                          # supported | contradicted | unsupported | partially_supported | insufficient_context
    hallucination_score: float          # 0.0 (faithful) → 1.0 (hallucinated)
    error_type: Optional[str] = None    # entity_error | date_number_error | etc.
    evidence_span: Optional[str] = None # Quoted span from context
    explanation: Optional[str] = None   # Turkish explanation
    confidence: float = 1.0             # Model confidence (1 - uncertainty)


@dataclass
class PolicyDecision:
    """
    Final decision for the whole answer after evaluating all claim verdicts.
    """
    final_decision: str                         # accept | warn | revise | reject | web_check
    aggregate_hallucination_score: float        # Weighted average across claims
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_unsupported: int
    n_partially_supported: int
    n_insufficient_context: int
    flagged_claims: List[ClaimVerdict] = field(default_factory=list)   # Claims that triggered the decision
    action_message: str = ""                    # Human-readable explanation of the decision
    suggested_revision: Optional[str] = None    # If revise: what to change (filled by Reviser)
    retrieval_needed: bool = False              # If web_check: trigger additional retrieval


# ---------------------------------------------------------------------------
# Decision thresholds (configurable)
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # Aggregate hallucination score above this → at least warn
    "warn_score": 0.30,
    # Aggregate hallucination score above this → reject
    # (only used in extreme multi-claim cases; single-claim never rejects)
    "reject_score": 0.70,
    # Fraction of claims that must be supported for accept
    "accept_support_ratio": 1.0,
    # If any single claim has score above this → flag it
    "flag_claim_score": 0.50,
    # Minimum contradicted claims (with zero supported) to trigger reject
    # Set high to effectively disable reject for typical cases
    "reject_min_contradicted": 2,
}


# ---------------------------------------------------------------------------
# PolicyController
# ---------------------------------------------------------------------------

class PolicyController:
    """
    Rule-based policy controller (v0).

    Decision hierarchy (highest priority first):
    1. REJECT:     Any claim is contradicted (direct factual error)
    2. WEB_CHECK:  Any claim is insufficient_context (need more retrieval)
    3. REVISE:     Mix of supported + unsupported/partial (partial hallucination)
    4. WARN:       Any claim is unsupported or partially_supported
    5. ACCEPT:     All claims supported

    Override: aggregate_hallucination_score thresholds can escalate decisions.
    """

    def __init__(self, thresholds: Optional[dict] = None) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def decide(self, claim_verdicts: List[ClaimVerdict]) -> PolicyDecision:
        """
        Evaluate all claim verdicts and produce a final policy decision.

        Args:
            claim_verdicts: List of per-claim verifier outputs.

        Returns:
            PolicyDecision with final_decision and supporting metadata.
        """
        if not claim_verdicts:
            return PolicyDecision(
                final_decision="warn",
                aggregate_hallucination_score=0.5,
                n_claims=0,
                n_supported=0,
                n_contradicted=0,
                n_unsupported=0,
                n_partially_supported=0,
                n_insufficient_context=0,
                action_message="No claims extracted — cannot verify answer.",
            )

        # --- Count labels ---
        label_counts = {
            "supported": 0,
            "contradicted": 0,
            "unsupported": 0,
            "partially_supported": 0,
            "insufficient_context": 0,
        }
        for v in claim_verdicts:
            label = v.label if v.label in label_counts else "unsupported"
            label_counts[label] += 1

        n = len(claim_verdicts)

        # --- Aggregate hallucination score (weighted by claim count) ---
        agg_score = sum(v.hallucination_score for v in claim_verdicts) / n

        # --- Flagged claims (high individual score) ---
        flag_threshold = self.thresholds["flag_claim_score"]
        flagged = [v for v in claim_verdicts if v.hallucination_score >= flag_threshold]

        # --- Decision logic ---
        decision, message, retrieval_needed = self._apply_policy(
            label_counts=label_counts,
            agg_score=agg_score,
            n=n,
        )

        return PolicyDecision(
            final_decision=decision,
            aggregate_hallucination_score=round(agg_score, 4),
            n_claims=n,
            n_supported=label_counts["supported"],
            n_contradicted=label_counts["contradicted"],
            n_unsupported=label_counts["unsupported"],
            n_partially_supported=label_counts["partially_supported"],
            n_insufficient_context=label_counts["insufficient_context"],
            flagged_claims=flagged,
            action_message=message,
            retrieval_needed=retrieval_needed,
        )

    def _apply_policy(
        self,
        label_counts: dict,
        agg_score: float,
        n: int,
    ) -> tuple[str, str, bool]:
        """
        Core decision logic. Returns (decision, message, retrieval_needed).

        Policy table (v1 — fixed):
        ─────────────────────────────────────────────────────────────────
        Single-claim:
          supported              → accept
          partially_supported    → warn
          unsupported            → warn
          contradicted           → revise   (NOT reject — answer is fixable)
          insufficient_context   → insufficient_context

        Multi-claim:
          all supported                              → accept
          all insufficient_context                   → insufficient_context
          any contradicted                           → revise
          no contradicted, any partial/unsupported   → warn
          extreme: n>=2, n_supported==0, n_contradicted>=reject_min → reject

        Key changes vs v0:
          - Single-claim contradicted → revise (was: reject)
          - Single-claim unsupported  → warn   (was: revise)
          - reject is now reserved for extreme multi-claim all-wrong cases
        ─────────────────────────────────────────────────────────────────
        """
        n_contradicted = label_counts["contradicted"]
        n_unsupported  = label_counts["unsupported"]
        n_partial      = label_counts["partially_supported"]
        n_ic           = label_counts["insufficient_context"]
        n_supported    = label_counts["supported"]

        # ── Single-claim fast path ──────────────────────────────────────
        if n == 1:
            if n_supported == 1:
                return (
                    "accept",
                    "Single claim supported by context.",
                    False,
                )
            if n_partial == 1:
                return (
                    "warn",
                    "Single claim only partially supported by context.",
                    False,
                )
            if n_unsupported == 1:
                return (
                    "warn",
                    "Single claim not found in context. "
                    "Answer may contain unverified information.",
                    False,
                )
            if n_contradicted == 1:
                return (
                    "revise",
                    "Single claim contradicted by context. "
                    "Revise or remove the incorrect statement.",
                    False,
                )
            if n_ic == 1:
                return (
                    "insufficient_context",
                    "Cannot verify claim — insufficient context. "
                    "Additional retrieval needed.",
                    True,
                )

        # ── Multi-claim policy ──────────────────────────────────────────

        # All supported → accept
        if n_supported == n:
            return (
                "accept",
                f"All {n}/{n} claims supported by context. "
                f"Aggregate hallucination score: {agg_score:.2f}.",
                False,
            )

        # All insufficient_context → web_check
        if n_ic == n:
            return (
                "insufficient_context",
                f"All {n}/{n} claims cannot be verified with current context. "
                f"Additional retrieval needed.",
                True,
            )

        # Any insufficient_context (mixed) → web_check
        if n_ic > 0:
            return (
                "insufficient_context",
                f"{n_ic}/{n} claim(s) cannot be verified with current context. "
                f"Additional retrieval needed.",
                True,
            )

        # Any contradicted → revise
        # (reject only in extreme case: multiple contradicted, zero supported)
        if n_contradicted > 0:
            reject_min = self.thresholds.get("reject_min_contradicted", 2)
            if n_supported == 0 and n_contradicted >= reject_min:
                return (
                    "reject",
                    f"{n_contradicted}/{n} claim(s) directly contradicted, "
                    f"none supported. Answer is unreliable.",
                    False,
                )
            return (
                "revise",
                f"{n_contradicted}/{n} claim(s) contradicted by context. "
                f"Revise or remove the contradicted parts.",
                False,
            )

        # Partial or unsupported (no contradicted) → warn
        if n_partial > 0 or n_unsupported > 0:
            n_problematic = n_partial + n_unsupported
            return (
                "warn",
                f"{n_problematic}/{n} claim(s) not fully supported by context. "
                f"Answer may contain unverified information.",
                False,
            )

        # Score-based escalation (all labels supported but score is high)
        if agg_score >= self.thresholds["warn_score"]:
            return (
                "warn",
                f"Aggregate hallucination score {agg_score:.2f} above warn threshold. "
                f"Answer may contain uncertain content.",
                False,
            )

        # Fallback: accept
        return (
            "accept",
            f"All {n}/{n} claims supported by context. "
            f"Aggregate hallucination score: {agg_score:.2f}.",
            False,
        )

    def decision_to_emoji(self, decision: str) -> str:
        """Return a visual indicator for the decision."""
        return {
            "accept": "✓ ACCEPT",
            "warn": "⚠ WARN",
            "revise": "✎ REVISE",
            "reject": "✗ REJECT",
            "web_check": "🔍 WEB_CHECK",
        }.get(decision, "? UNKNOWN")