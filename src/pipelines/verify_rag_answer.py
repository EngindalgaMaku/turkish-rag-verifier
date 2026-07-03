"""
src/pipelines/verify_rag_answer.py
====================================
End-to-end Turkish RAG Hallucination Verification Pipeline.

Full pipeline:
    question + answer + context
        ↓
    ClaimExtractor      — split answer into atomic claims
        ↓
    EvidenceRetriever   — retrieve/confirm context per claim
        ↓
    Qwen Verifier       — label each claim vs context
        ↓
    PolicyController    — aggregate verdicts → final decision
        ↓
    Reviser             — revise answer if needed
        ↓
    PipelineResult      — structured output with all metadata

Usage:
    from src.pipelines.verify_rag_answer import RAGVerificationPipeline

    pipeline = RAGVerificationPipeline(
        verifier_model_path="outputs/models/qwen3b_qlora_pilot_v1/adapter",
        base_model_name="Qwen/Qwen2.5-3B-Instruct",
    )
    result = pipeline.run(
        question="Lozan Antlaşması ne zaman imzalandı?",
        answer="Lozan Antlaşması 1923'te imzalandı ve Osmanlı başkentini Ankara yaptı.",
        context="Lozan Antlaşması, 24 Temmuz 1923'te İsviçre'nin Lozan şehrinde imzalandı...",
    )
    print(result.final_decision)       # "revise"
    print(result.revised_answer)       # "Lozan Antlaşması 1923'te imzalandı."
    print(result.to_dict())            # Full structured output
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from src.agents.policy_controller import ClaimVerdict, PolicyController, PolicyDecision
from src.agents.evidence_retriever import EvidenceRetriever, RetrievalResult
from src.agents.reviser import Reviser, RevisionResult
from src.data.claim_extraction import extract_claims_rule_based


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClaimResult:
    """Full result for a single claim through the pipeline."""
    claim: str
    label: str
    hallucination_score: float
    error_type: Optional[str]
    evidence_span: Optional[str]
    explanation: Optional[str]
    retrieval_source: str
    retrieval_error: Optional[str] = None


@dataclass
class PipelineResult:
    """Full pipeline output for a question-answer pair."""
    question: str
    original_answer: str
    context: str
    claims: List[ClaimResult]
    final_decision: str                     # accept | warn | revise | reject | web_check
    aggregate_hallucination_score: float
    revised_answer: Optional[str]
    action_message: str
    revision_log: List[str] = field(default_factory=list)
    retrieval_needed: bool = False
    pipeline_latency_ms: float = 0.0
    n_claims: int = 0
    n_supported: int = 0
    n_contradicted: int = 0
    n_unsupported: int = 0
    n_partially_supported: int = 0
    n_insufficient_context: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        return d

    def summary(self) -> str:
        """Human-readable one-line summary."""
        decision_icons = {
            "accept": "✓",
            "warn": "⚠",
            "revise": "✎",
            "reject": "✗",
            "web_check": "🔍",
        }
        icon = decision_icons.get(self.final_decision, "?")
        return (
            f"{icon} {self.final_decision.upper()} | "
            f"score={self.aggregate_hallucination_score:.2f} | "
            f"{self.n_claims} claims: "
            f"{self.n_supported}✓ {self.n_contradicted}✗ "
            f"{self.n_unsupported}? {self.n_partially_supported}~ "
            f"{self.n_insufficient_context}∅"
        )


# ---------------------------------------------------------------------------
# RAGVerificationPipeline
# ---------------------------------------------------------------------------

class RAGVerificationPipeline:
    """
    End-to-end Turkish RAG hallucination verification pipeline.

    Components:
        - ClaimExtractor: rule-based sentence splitter (Phase 1)
        - EvidenceRetriever: passthrough / local / wikipedia / web
        - Qwen Verifier: fine-tuned QLoRA model
        - PolicyController: rule-based decision engine
        - Reviser: answer revision

    Args:
        verifier_model_path: Path to the fine-tuned adapter or merged model.
        base_model_name:     Base model name (required for adapter).
        retriever_mode:      "passthrough" | "local" | "wikipedia" | "web"
        retriever_corpus:    Document list for "local" mode.
        revision_strategy:   "drop" | "hedge" | "replace" | "truncate"
        policy_thresholds:   Override default PolicyController thresholds.
        prompt_version:      Verifier prompt version. Default: "v1.0".
        load_in_4bit:        Load verifier in 4-bit quantization. Default: True.
        lazy_load:           Delay model loading until first run(). Default: False.
    """

    def __init__(
        self,
        verifier_model_path: str,
        base_model_name: Optional[str] = None,
        retriever_mode: str = "passthrough",
        retriever_corpus: Optional[List[str]] = None,
        revision_strategy: str = "drop",
        policy_thresholds: Optional[dict] = None,
        prompt_version: str = "v1.0",
        load_in_4bit: bool = True,
        lazy_load: bool = False,
        use_hybrid_pipeline: bool = True,
        nli_threshold: float = 0.85,
    ) -> None:
        self.verifier_model_path = verifier_model_path
        self.base_model_name = base_model_name
        self.prompt_version = prompt_version
        self.load_in_4bit = load_in_4bit
        self.use_hybrid = use_hybrid_pipeline
        self.nli_threshold = nli_threshold

        # Initialize non-model components immediately
        self.retriever = EvidenceRetriever(
            mode=retriever_mode,
            corpus=retriever_corpus or [],
        )
        self.controller = PolicyController(thresholds=policy_thresholds)
        self.reviser = Reviser(strategy=revision_strategy)

        # Verifier is loaded lazily or immediately
        self._verifier = None
        self._nli_verifier = None
        if not lazy_load:
            self._load_verifier()

    def _load_verifier(self) -> None:
        """Load the Qwen verifier model."""
        from src.inference.verifier import Verifier
        print(f"Loading verifier: {self.verifier_model_path}")
        self._verifier = Verifier(
            model_path=self.verifier_model_path,
            base_model_name=self.base_model_name,
            prompt_version=self.prompt_version,
            load_in_4bit=self.load_in_4bit,
        )

    @property
    def verifier(self):
        if self._verifier is None:
            self._load_verifier()
        return self._verifier

    def run(
        self,
        question: str,
        answer: str,
        context: str,
        claims: Optional[List[str]] = None,
    ) -> PipelineResult:
        """
        Run the full verification pipeline on a question-answer pair.

        Args:
            question: The user's question.
            answer:   The LLM-generated answer to verify.
            context:  Retrieved context (document passage).
            claims:   Pre-extracted claims. If None, uses rule-based extraction.

        Returns:
            PipelineResult with full verification output.
        """
        start_time = time.time()

        # --- Step 1: Extract claims ---
        if claims is None:
            claims = extract_claims_rule_based(answer)

        if not claims:
            # Fallback: treat whole answer as one claim
            claims = [answer.strip()]

        # --- Step 2: Verify each claim ---
        claim_verdicts: List[ClaimVerdict] = []
        claim_results: List[ClaimResult] = []

        for claim in claims:
            # Step 2a: Retrieve evidence for this claim
            retrieval: RetrievalResult = self.retriever.retrieve(
                query=claim,
                context=context,
            )

            # Step 2b: Verify claim against retrieved context
            if retrieval.retrieval_error and not retrieval.context:
                # No context available → insufficient_context
                verdict = ClaimVerdict(
                    claim=claim,
                    label="insufficient_context",
                    hallucination_score=0.45,
                    error_type="retrieval_failure",
                    evidence_span=None,
                    explanation=f"Retrieval failed: {retrieval.retrieval_error}",
                )
            else:
                is_supported_by_nli = False
                nli_label = None
                nli_conf = 0.0
                
                if self.use_hybrid:
                    if self._nli_verifier is None:
                        from src.inference.zeroshot_nli_verifier import ZeroShotNLIVerifier
                        self._nli_verifier = ZeroShotNLIVerifier()
                    
                    is_supported_by_nli, nli_label, nli_conf = self._nli_verifier.verify(
                        context=retrieval.context,
                        claim=claim,
                        threshold=self.nli_threshold
                    )
                
                if is_supported_by_nli:
                    verdict = ClaimVerdict(
                        claim=claim,
                        label="supported",
                        hallucination_score=1.0 - nli_conf,
                        error_type="none",
                        evidence_span="",
                        explanation=f"Stage 1 (Zero-Shot NLI) verified as supported (confidence: {nli_conf:.2f}).",
                    )
                else:
                    verifier_output = self.verifier.verify(
                        question=question,
                        context=retrieval.context,
                        answer=answer,
                        claim=claim,
                    )
                    verdict = ClaimVerdict(
                        claim=claim,
                        label=verifier_output.label,
                        hallucination_score=verifier_output.hallucination_score,
                        error_type=verifier_output.error_type,
                        evidence_span=verifier_output.evidence_span,
                    explanation=verifier_output.explanation,
                )

            claim_verdicts.append(verdict)
            claim_results.append(ClaimResult(
                claim=claim,
                label=verdict.label,
                hallucination_score=verdict.hallucination_score,
                error_type=verdict.error_type,
                evidence_span=verdict.evidence_span,
                explanation=verdict.explanation,
                retrieval_source=retrieval.source,
                retrieval_error=retrieval.retrieval_error,
            ))

        # --- Step 3: Policy decision ---
        policy: PolicyDecision = self.controller.decide(claim_verdicts)

        # --- Step 4: Revise if needed ---
        revision: Optional[RevisionResult] = None
        revised_answer: Optional[str] = None
        revision_log: List[str] = []

        if policy.final_decision in ("revise", "reject"):
            revision = self.reviser.revise(
                original_answer=answer,
                claim_verdicts=claim_verdicts,
                policy_decision=policy,
            )
            revised_answer = revision.revised_answer
            revision_log = revision.revision_log
        elif policy.final_decision == "accept":
            revised_answer = answer  # No change needed

        # --- Assemble result ---
        elapsed_ms = (time.time() - start_time) * 1000

        return PipelineResult(
            question=question,
            original_answer=answer,
            context=context[:500] + "..." if len(context) > 500 else context,
            claims=claim_results,
            final_decision=policy.final_decision,
            aggregate_hallucination_score=policy.aggregate_hallucination_score,
            revised_answer=revised_answer,
            action_message=policy.action_message,
            revision_log=revision_log,
            retrieval_needed=policy.retrieval_needed,
            pipeline_latency_ms=round(elapsed_ms, 1),
            n_claims=policy.n_claims,
            n_supported=policy.n_supported,
            n_contradicted=policy.n_contradicted,
            n_unsupported=policy.n_unsupported,
            n_partially_supported=policy.n_partially_supported,
            n_insufficient_context=policy.n_insufficient_context,
        )

    def run_batch(
        self,
        examples: List[Dict[str, str]],
        show_progress: bool = True,
    ) -> List[PipelineResult]:
        """
        Run the pipeline on a batch of examples.

        Args:
            examples: List of dicts with keys: question, answer, context.
                      Optional key: claims (pre-extracted).
            show_progress: Print progress. Default: True.

        Returns:
            List of PipelineResult.
        """
        results = []
        n = len(examples)
        for i, ex in enumerate(examples):
            if show_progress:
                print(f"[{i+1}/{n}] Verifying: {ex.get('question', '')[:60]}...")
            result = self.run(
                question=ex["question"],
                answer=ex["answer"],
                context=ex["context"],
                claims=ex.get("claims"),
            )
            if show_progress:
                print(f"  → {result.summary()}")
            results.append(result)
        return results