"""
src/inference/verifier.py
==========================
Main verifier class. Loads a fine-tuned LoRA model and runs inference
on claim-level examples.

Usage:
    from src.inference.verifier import Verifier
    verifier = Verifier("outputs/models/qwen3b_qlora_pilot_v1")
    result = verifier.verify(question="...", context="...", answer="...", claim="...")
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.schema import AggregateVerificationResult, VerifierOutput
from src.inference.parse_json_output import parse_verifier_output
from src.training.format_chat_dataset import load_prompt_templates


# ---------------------------------------------------------------------------
# Lexical overlap helper
# ---------------------------------------------------------------------------

def _token_overlap_ratio(text_a: str, text_b: str, min_token_len: int = 3) -> float:
    """
    Compute the fraction of meaningful tokens in `text_a` that also appear in
    `text_b` (case-insensitive, punctuation-stripped).

    Only tokens with length >= min_token_len are considered to avoid noise from
    short function words.  Returns a value in [0.0, 1.0].
    """
    def tokenize(text: str) -> set:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return {t for t in text.split() if len(t) >= min_token_len}

    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


class Verifier:
    """
    Claim-level hallucination verifier using a fine-tuned LoRA model.
    """

    def __init__(
        self,
        model_path: str,
        base_model_name: Optional[str] = None,
        prompt_version: str = "v1.0",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Load the verifier model.

        Args:
            model_path:      Path to the saved model directory.
                             Can be: (a) merged model dir, (b) adapter dir with base model.
            base_model_name: Base model name (required if model_path is an adapter).
            prompt_version:  Prompt version to use for inference.
            device:          Device to use ("cuda", "cpu", "auto"). Default: auto.
            load_in_4bit:    Load in 4-bit quantization for inference.
            max_new_tokens:  Maximum tokens to generate.
            torch_dtype:     dtype for model weights. Default: bfloat16 on CUDA,
                             float32 on CPU. Pass torch.float16 for older GPUs
                             (V100, RTX 20xx) that do not support bfloat16.
        """
        self.model_path = Path(model_path)
        self.prompt_version = prompt_version
        self.max_new_tokens = max_new_tokens

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype: explicit > auto (bfloat16 on CUDA, float32 on CPU)
        if torch_dtype is not None:
            self._torch_dtype = torch_dtype
        elif self.device == "cuda":
            self._torch_dtype = torch.bfloat16
        else:
            self._torch_dtype = torch.float32

        print(f"Loading verifier from: {model_path}")
        print(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            quantization_config = bnb_config
        else:
            quantization_config = None

        # Check if this is an adapter or a merged model
        adapter_config = self.model_path / "adapter_config.json"
        is_adapter = adapter_config.exists()

        # GUARD: adapter detected but no base model provided → fail fast with clear message
        if is_adapter and not base_model_name:
            raise ValueError(
                f"Adapter model detected at '{self.model_path}' (adapter_config.json found), "
                f"but --base-model was not provided.\n"
                f"Please provide the base model name, e.g.:\n"
                f"  --base-model Qwen/Qwen2.5-3B-Instruct\n"
                f"Or point --model to the merged model directory instead of the adapter directory."
            )

        if is_adapter and base_model_name:
            # Load base model + adapter
            print(f"Loading base model: {base_model_name}")
            print(f"Loading LoRA adapter from: {self.model_path}")
            print(f"torch_dtype: {self._torch_dtype}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else self.device,
                trust_remote_code=True,
                torch_dtype=self._torch_dtype,
            )
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        else:
            # Load merged model directly
            print(f"Loading merged model from: {self.model_path}")
            print(f"torch_dtype: {self._torch_dtype}")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else self.device,
                trust_remote_code=True,
                torch_dtype=self._torch_dtype,
            )

        self.model.eval()

        # Load prompt templates
        self.system_prompt, self.user_template = load_prompt_templates(prompt_version)

        print("Verifier ready.")

    def verify(
        self,
        question: str,
        context: str,
        answer: str,
        claim: str,
    ) -> VerifierOutput:
        """
        Verify a single claim against the context.

        Args:
            question: User question.
            context:  Retrieved context (may be multi-chunk).
            answer:   Full LLM-generated answer.
            claim:    Single claim to verify.

        Returns:
            VerifierOutput with label, score, error_type, evidence_span, etc.
        """
        start_time = time.time()

        user_content = self.user_template.format(
            question=question,
            context=context,
            answer=answer,
            claim=claim,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template (generation prompt only — no assistant turn)
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the input)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        result = parse_verifier_output(raw_output)
        result.raw_output = raw_output[:200] if result.parse_error else None

        # --- Post-processing: parametric knowledge override guard ---
        # If model says "supported" but provides no evidence_span, it MAY have used
        # parametric knowledge instead of context-grounding.  We downgrade to
        # "partially_supported" (not "unsupported") because the claim could still be
        # correct — we just lack a quoted span to confirm it.
        #
        # We do NOT apply this guard to "contradicted": a contradiction without a
        # quoted span is still a meaningful signal (the model found a conflict).
        #
        # We preserve the original label/score/explanation so the UI can show what
        # the model actually said, and only add a note about the missing evidence.
        if (
            result.label == "supported"
            and not result.parse_error
            and (not result.evidence_span or result.evidence_span.strip() == "")
        ):
            result.__dict__["_original_label"] = result.label
            result.__dict__["_original_score"] = result.hallucination_score
            result.__dict__["_original_explanation"] = result.explanation
            result.label = "partially_supported"
            # Keep the model's own score but nudge it slightly upward to signal
            # reduced confidence; cap at 0.55 so it doesn't look like a hard failure.
            result.hallucination_score = min(
                max(result.hallucination_score, 0.35), 0.55
            )
            result.error_type = "missing_evidence_span"
            # Append a note to the existing explanation rather than replacing it.
            original_explanation = result.__dict__["_original_explanation"] or ""
            note = (
                "Bağlamda bu iddiayı destekleyen bir alıntı bulunamadı; "
                "claim kısmen destekleniyor olarak işaretlendi."
            )
            result.explanation = (
                f"{original_explanation}  [{note}]" if original_explanation else note
            )

        # --- Post-processing: span must come from context guard ---
        # If the model's evidence_span is not a substring of the context
        # (case-insensitive, after stripping punctuation), it likely hallucinated
        # the span from the answer or from parametric knowledge.
        # In that case we clear the span so the UI doesn't show misleading text.
        if result.evidence_span and result.evidence_span.strip():
            span_clean = result.evidence_span.strip().lower()
            ctx_clean = context.lower()
            # Allow partial match: at least 60% of span tokens must appear in context
            span_tokens = set(re.sub(r"[^\w\s]", " ", span_clean).split())
            ctx_tokens = set(re.sub(r"[^\w\s]", " ", ctx_clean).split())
            if span_tokens:
                overlap = len(span_tokens & ctx_tokens) / len(span_tokens)
                if overlap < 0.5:
                    result.__dict__["_hallucinated_span"] = result.evidence_span
                    result.evidence_span = None
                    result.error_type = result.error_type or "hallucinated_span"

        # --- Post-processing: placeholder evidence span guard ---
        # If model outputs a non-contextual placeholder like "soru'da belirtilen claim"
        # or other meta-references instead of an actual span from the context,
        # clear the evidence_span so the UI doesn't show misleading text.
        _PLACEHOLDER_PATTERNS = [
            "soru'da belirtilen claim",
            "soruda belirtilen claim",
            "claim'de belirtilen",
            "belirtilen claim",
            "yukarıdaki claim",
            "verilen claim",
        ]
        if result.evidence_span:
            span_lower = result.evidence_span.lower().strip()
            if any(p in span_lower for p in _PLACEHOLDER_PATTERNS):
                result.evidence_span = None
                result.error_type = result.error_type or "placeholder_span"

        # --- Post-processing: lexical overlap fallback ---
        # When the model labels a claim "unsupported" but the claim's key tokens
        # appear heavily in the context, the model likely failed at paraphrase
        # matching (a known weakness of small fine-tuned models).
        # In that case we downgrade to "partially_supported" so the pipeline
        # treats it as a soft warning rather than a hard failure.
        #
        # Threshold: ≥60% of the claim's meaningful tokens appear in the context.
        # We do NOT apply this to "contradicted" (the model found an explicit conflict).
        if (
            result.label == "unsupported"
            and not result.parse_error
        ):
            overlap = _token_overlap_ratio(claim, context)
            if overlap >= 0.60:
                result.__dict__["_original_label"] = result.label
                result.__dict__["_original_score"] = result.hallucination_score
                result.__dict__["_overlap_ratio"] = overlap
                result.label = "partially_supported"
                # Nudge score down to reflect that the context likely covers this
                result.hallucination_score = min(result.hallucination_score, 0.50)
                result.error_type = "paraphrase_mismatch"
                original_explanation = result.explanation or ""
                note = (
                    f"Claim'in anahtar kelimeleri bağlamda bulundu (örtüşme: {overlap:.0%}); "
                    "model parafraz eşleştirmede başarısız olmuş olabilir."
                )
                result.explanation = (
                    f"{original_explanation}  [{note}]" if original_explanation else note
                )

        # --- Post-processing: negation-supported guard (EXP-010 lesson) ---
        # When the model predicts "contradicted" or "partially_supported" for a claim
        # that contains a negation/exclusion keyword ("içermez", "değildir", etc.)
        # AND the evidence_span from the context contains the same negation keyword,
        # the model likely confused a negated-but-supported claim with a contradiction.
        # Upgrade to "supported" in this case.
        #
        # This guard targets T03/T04 type errors without retraining.
        _NEGATION_KEYWORDS = [
            "içermez", "içermiyor", "değildir", "değil",
            "gerektirmez", "gerektirmiyor", "yerine", "olmaz",
            "kullanmaz", "kullanmıyor", "dayanmaz", "dayanmıyor",
            "tutmaz", "tutmuyor", "sağlamaz", "sağlamıyor",
        ]
        if (
            result.label in ("contradicted", "partially_supported")
            and not result.parse_error
            and result.evidence_span
        ):
            claim_lower = claim.lower()
            span_lower = (result.evidence_span or "").lower()
            claim_has_negation = any(kw in claim_lower for kw in _NEGATION_KEYWORDS)
            span_has_negation  = any(kw in span_lower  for kw in _NEGATION_KEYWORDS)
            if claim_has_negation and span_has_negation:
                result.__dict__["_original_label"] = result.label
                result.__dict__["_negation_guard_triggered"] = True
                result.label = "supported"
                result.hallucination_score = min(result.hallucination_score, 0.15)
                result.error_type = "none"
                note = (
                    "Claim olumsuzlama içeriyor ve bağlamdaki kanıt aynı olumsuzlamayı "
                    "destekliyor; destekleniyor olarak düzeltildi."
                )
                result.explanation = (
                    f"{result.explanation}  [{note}]" if result.explanation else note
                )

        # --- Post-processing: contradicted→unsupported guard (EXP-010 lesson) ---
        # When the model predicts "contradicted" but the evidence_span does NOT
        # explicitly negate or contradict the claim (i.e. the span is absent or
        # the span token overlap with the claim is low), downgrade to "unsupported".
        # This prevents the model from over-using "contradicted" when the context
        # simply doesn't mention the claim.
        #
        # We only apply this when evidence_span is empty/missing, because a real
        # contradiction should always have a quoted span from the context.
        if (
            result.label == "contradicted"
            and not result.parse_error
            and (not result.evidence_span or result.evidence_span.strip() == "")
        ):
            result.__dict__["_original_label"] = result.label
            result.__dict__["_no_span_contradicted_guard"] = True
            result.label = "unsupported"
            result.hallucination_score = min(result.hallucination_score, 0.80)
            result.error_type = result.error_type or "unsupported_inference"
            note = (
                "Çelişki iddiası için bağlamdan kanıt metni bulunamadı; "
                "desteklenmiyor olarak düzeltildi."
            )
            result.explanation = (
                f"{result.explanation}  [{note}]" if result.explanation else note
            )

        elapsed_ms = (time.time() - start_time) * 1000
        # Store latency as attribute for batch reporting
        result.__dict__["_latency_ms"] = elapsed_ms

        return result

    def verify_answer(
        self,
        question: str,
        context: str,
        answer: str,
        claims: Optional[list[str]] = None,
    ) -> AggregateVerificationResult:
        """
        Verify all claims in an answer and produce an aggregate result.

        Args:
            question: User question.
            context:  Retrieved context.
            answer:   Full LLM-generated answer.
            claims:   Pre-extracted claims. If None, uses rule-based extraction.

        Returns:
            AggregateVerificationResult.
        """
        from src.data.claim_extraction import extract_claims_rule_based
        from collections import Counter

        if claims is None:
            claims = extract_claims_rule_based(answer)

        if not claims:
            claims = [answer]  # Fallback: treat whole answer as one claim

        claim_results = []
        total_start = time.time()

        for claim in claims:
            result = self.verify(question, context, answer, claim)
            claim_results.append(result)

        total_ms = (time.time() - total_start) * 1000

        # Aggregate
        label_counts = Counter(r.label for r in claim_results)
        risk_scores = [r.hallucination_score for r in claim_results]
        aggregate_risk = max(risk_scores) if risk_scores else 0.0

        # Aggregate decision: most severe
        severity_order = [
            "reject", "revise", "web_check", "warn",
            "insufficient_context", "accept"
        ]
        decisions = [r.decision for r in claim_results]
        aggregate_decision = "accept"
        for decision in severity_order:
            if decision in decisions:
                aggregate_decision = decision
                break

        return AggregateVerificationResult(
            answer_id=f"ans_{hash(answer) % 100000:05d}",
            question=question,
            answer=answer,
            claim_results=claim_results,
            aggregate_decision=aggregate_decision,
            aggregate_risk_score=aggregate_risk,
            n_claims=len(claim_results),
            n_supported=label_counts.get("supported", 0),
            n_contradicted=label_counts.get("contradicted", 0),
            n_unsupported=label_counts.get("unsupported", 0),
            n_partially_supported=label_counts.get("partially_supported", 0),
            n_insufficient_context=label_counts.get("insufficient_context", 0),
            processing_time_ms=total_ms,
        )