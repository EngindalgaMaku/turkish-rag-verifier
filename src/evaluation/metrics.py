"""
src/evaluation/metrics.py
==========================
All evaluation metrics for the Turkish RAG Hallucination Verifier.
Includes classification metrics, calibration (ECE), and evidence span metrics.

Usage:
    from src.evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(gold_labels, pred_labels, pred_scores, gold_spans, pred_spans)
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data.schema import VALID_LABELS
from src.utils.text import compute_token_f1, find_span_in_context


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    pred_scores: list[float],
    gold_spans: Optional[list[str]] = None,
    pred_spans: Optional[list[str]] = None,
    contexts: Optional[list[str]] = None,
    parse_errors: Optional[list[bool]] = None,
    latencies_ms: Optional[list[float]] = None,
    n_calibration_bins: int = 10,
) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        gold_labels:       Ground truth labels.
        pred_labels:       Predicted labels.
        pred_scores:       Predicted hallucination scores [0, 1].
        gold_spans:        Ground truth evidence spans (optional).
        pred_spans:        Predicted evidence spans (optional).
        contexts:          Context strings (for span-in-context check).
        parse_errors:      Boolean list indicating parse failures.
        latencies_ms:      Per-example latency in milliseconds.
        n_calibration_bins: Number of bins for ECE calculation.

    Returns:
        dict with all metrics.
    """
    assert len(gold_labels) == len(pred_labels), "Label lists must have same length"
    assert len(gold_labels) == len(pred_scores), "Score list must match label list length"

    label_list = sorted(VALID_LABELS)

    metrics = {}

    # --- Classification metrics ---
    metrics["accuracy"] = round(accuracy_score(gold_labels, pred_labels), 4)
    metrics["macro_f1"] = round(f1_score(gold_labels, pred_labels, average="macro", zero_division=0), 4)
    metrics["weighted_f1"] = round(f1_score(gold_labels, pred_labels, average="weighted", zero_division=0), 4)

    # Per-class metrics
    per_class = {}
    for label in label_list:
        gold_bin = [1 if g == label else 0 for g in gold_labels]
        pred_bin = [1 if p == label else 0 for p in pred_labels]
        per_class[label] = {
            "precision": round(precision_score(gold_bin, pred_bin, zero_division=0), 4),
            "recall": round(recall_score(gold_bin, pred_bin, zero_division=0), 4),
            "f1": round(f1_score(gold_bin, pred_bin, zero_division=0), 4),
            "support": int(sum(gold_bin)),
        }
    metrics["per_class"] = per_class

    # Key recall metrics (most important for hallucination detection)
    metrics["contradicted_recall"] = per_class.get("contradicted", {}).get("recall", 0.0)
    metrics["unsupported_recall"] = per_class.get("unsupported", {}).get("recall", 0.0)
    metrics["insufficient_context_recall"] = per_class.get("insufficient_context", {}).get("recall", 0.0)

    # Confusion matrix
    cm = confusion_matrix(gold_labels, pred_labels, labels=label_list)
    metrics["confusion_matrix"] = {
        "labels": label_list,
        "matrix": cm.tolist(),
    }

    # --- Calibration metrics (MANDATORY) ---
    calibration = compute_calibration_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        n_bins=n_calibration_bins,
    )
    metrics["calibration"] = calibration
    metrics["ECE"] = calibration["ECE"]
    metrics["MCE"] = calibration["MCE"]
    metrics["brier_score"] = calibration["brier_score"]

    # --- JSON validity rate ---
    if parse_errors is not None:
        n_total = len(parse_errors)
        n_errors = sum(parse_errors)
        metrics["json_validity_rate"] = round((n_total - n_errors) / n_total, 4) if n_total > 0 else 0.0
        metrics["n_parse_errors"] = n_errors
    else:
        metrics["json_validity_rate"] = None

    # --- Latency metrics ---
    if latencies_ms is not None and len(latencies_ms) > 0:
        metrics["latency"] = {
            "mean_ms": round(float(np.mean(latencies_ms)), 1),
            "median_ms": round(float(np.median(latencies_ms)), 1),
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
            "max_ms": round(float(np.max(latencies_ms)), 1),
        }

    # --- Evidence span metrics ---
    if gold_spans is not None and pred_spans is not None:
        span_metrics = compute_span_metrics(gold_spans, pred_spans, contexts)
        metrics["evidence_span"] = span_metrics

    # --- Summary ---
    metrics["n_examples"] = len(gold_labels)
    metrics["label_distribution"] = dict(Counter(gold_labels))
    metrics["pred_distribution"] = dict(Counter(pred_labels))

    return metrics


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_calibration_metrics(
    gold_labels: list[str],
    pred_labels: list[str],
    pred_scores: list[float],
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics: ECE, MCE, Brier score.

    For multi-class calibration, we treat hallucination_score as a binary
    probability: P(claim is hallucinated) = score.
    Ground truth: 1 if label in {contradicted, unsupported}, else 0.

    Args:
        gold_labels:  Ground truth labels.
        pred_labels:  Predicted labels.
        pred_scores:  Predicted hallucination scores [0, 1].
        n_bins:       Number of calibration bins.

    Returns:
        dict with ECE, MCE, brier_score, reliability_diagram data.
    """
    # Binary ground truth: is the claim NOT fully supported?
    # Includes all non-supported labels since score > 0 indicates some hallucination risk.
    # partially_supported (0.35) and insufficient_context (0.45) are "partially hallucinated".
    # We use a soft ground truth based on the canonical score mapping.
    LABEL_TO_GT_SCORE = {
        "supported": 0.05,
        "partially_supported": 0.35,
        "insufficient_context": 0.45,
        "unsupported": 0.75,
        "contradicted": 0.95,
    }
    # Use canonical scores as soft ground truth for calibration
    y_true = np.array([LABEL_TO_GT_SCORE.get(g, 0.5) for g in gold_labels])
    y_score = np.array(pred_scores)

    # Brier score
    brier_score = float(np.mean((y_score - y_true) ** 2))

    # ECE and MCE via binning
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    reliability_bins = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Last bin is inclusive on the right to capture score == 1.0
        if bin_upper == 1.0:
            in_bin = (y_score >= bin_lower) & (y_score <= bin_upper)
        else:
            in_bin = (y_score >= bin_lower) & (y_score < bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(y_score[in_bin]))
            calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)

            ece += calibration_error * prop_in_bin
            mce = max(mce, calibration_error)

            reliability_bins.append({
                "bin_lower": round(float(bin_lower), 2),
                "bin_upper": round(float(bin_upper), 2),
                "avg_confidence": round(avg_confidence_in_bin, 4),
                "accuracy": round(accuracy_in_bin, 4),
                "n": int(np.sum(in_bin)),
                "calibration_error": round(calibration_error, 4),
            })

    return {
        "ECE": round(ece, 4),
        "MCE": round(mce, 4),
        "brier_score": round(brier_score, 4),
        "n_bins": n_bins,
        "reliability_diagram": reliability_bins,
        "note": "Soft calibration: score vs canonical label score (supported=0.05, partially=0.35, insufficient=0.45, unsupported=0.75, contradicted=0.95)",
    }


# ---------------------------------------------------------------------------
# Evidence span metrics
# ---------------------------------------------------------------------------

def compute_span_metrics(
    gold_spans: list[str],
    pred_spans: list[str],
    contexts: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate evidence span quality.

    Metrics:
    - exact_match_rate: Exact string match (case-insensitive, stripped)
    - token_f1_mean: Mean token F1 between predicted and gold spans
    - span_in_context_rate: Fraction of predicted spans found in context

    Args:
        gold_spans:  Ground truth evidence spans.
        pred_spans:  Predicted evidence spans.
        contexts:    Context strings (for span_in_context check).

    Returns:
        dict with span metrics.
    """
    assert len(gold_spans) == len(pred_spans)

    # Only evaluate on examples where gold span is non-empty
    valid_indices = [i for i, g in enumerate(gold_spans) if g and g.strip()]

    if not valid_indices:
        return {"note": "No examples with non-empty gold spans"}

    exact_matches = 0
    token_f1_scores = []
    span_in_context_count = 0

    for i in valid_indices:
        gold = gold_spans[i].strip()
        pred = pred_spans[i].strip() if pred_spans[i] else ""

        # Exact match (case-insensitive)
        if gold.lower() == pred.lower():
            exact_matches += 1

        # Token F1
        f1 = compute_token_f1(pred, gold)
        token_f1_scores.append(f1)

        # Span in context
        if contexts is not None and i < len(contexts):
            if find_span_in_context(pred, contexts[i]):
                span_in_context_count += 1

    n_valid = len(valid_indices)

    result = {
        "n_evaluated": n_valid,
        "exact_match_rate": round(exact_matches / n_valid, 4),
        "token_f1_mean": round(float(np.mean(token_f1_scores)), 4),
        "token_f1_std": round(float(np.std(token_f1_scores)), 4),
    }

    if contexts is not None:
        result["span_in_context_rate"] = round(span_in_context_count / n_valid, 4)

    return result


# ---------------------------------------------------------------------------
# Formatted report
# ---------------------------------------------------------------------------

def format_metrics_report(metrics: dict, run_name: str = "") -> str:
    """
    Format metrics dict into a human-readable markdown report.

    Args:
        metrics:  Output of compute_all_metrics().
        run_name: Experiment run name.

    Returns:
        Markdown string.
    """
    lines = []
    lines.append(f"# Evaluation Report: {run_name}")
    lines.append(f"\nN examples: {metrics.get('n_examples', '?')}")
    lines.append(f"\n## Classification Metrics")
    lines.append(f"- Accuracy:    {metrics.get('accuracy', '?'):.4f}")
    lines.append(f"- Macro F1:    {metrics.get('macro_f1', '?'):.4f}  (primary metric)")
    lines.append(f"- Weighted F1: {metrics.get('weighted_f1', '?'):.4f}")
    lines.append(f"\n### Key Recall Metrics")
    lines.append(f"- Contradicted recall:         {metrics.get('contradicted_recall', '?'):.4f}")
    lines.append(f"- Unsupported recall:          {metrics.get('unsupported_recall', '?'):.4f}")
    lines.append(f"- Insufficient context recall: {metrics.get('insufficient_context_recall', '?'):.4f}")

    lines.append(f"\n### Per-Class Metrics")
    lines.append(f"| Label | Precision | Recall | F1 | Support |")
    lines.append(f"|-------|-----------|--------|----|---------|")
    for label, vals in metrics.get("per_class", {}).items():
        lines.append(
            f"| {label} | {vals['precision']:.4f} | {vals['recall']:.4f} | "
            f"{vals['f1']:.4f} | {vals['support']} |"
        )

    lines.append(f"\n## Calibration Metrics (MANDATORY)")
    cal = metrics.get("calibration", {})
    lines.append(f"- ECE:         {cal.get('ECE', '?'):.4f}  (target: < 0.15)")
    lines.append(f"- MCE:         {cal.get('MCE', '?'):.4f}")
    lines.append(f"- Brier Score: {cal.get('brier_score', '?'):.4f}")

    if metrics.get("json_validity_rate") is not None:
        lines.append(f"\n## JSON Validity")
        lines.append(f"- JSON validity rate: {metrics['json_validity_rate']:.4f}  (target: > 0.95)")
        lines.append(f"- Parse errors: {metrics.get('n_parse_errors', '?')}")

    if "latency" in metrics:
        lat = metrics["latency"]
        lines.append(f"\n## Latency")
        lines.append(f"- Mean: {lat['mean_ms']:.1f} ms")
        lines.append(f"- P95:  {lat['p95_ms']:.1f} ms")

    if "evidence_span" in metrics:
        span = metrics["evidence_span"]
        lines.append(f"\n## Evidence Span Quality")
        lines.append(f"- Exact match rate: {span.get('exact_match_rate', '?'):.4f}")
        lines.append(f"- Token F1 mean:    {span.get('token_f1_mean', '?'):.4f}")
        if "span_in_context_rate" in span:
            lines.append(f"- Span in context:  {span['span_in_context_rate']:.4f}")

    return "\n".join(lines)