"""
src/evaluation/evaluate_predictions.py
========================================
Loads gold labels and model predictions, computes all metrics,
saves results to outputs/metrics/ and outputs/reports/.

Usage:
    python scripts/05_evaluate.py \
        --gold data/splits/test.jsonl \
        --pred outputs/predictions/qwen3b_qlora_pilot_v1_test_predictions.jsonl \
        --run-name qwen3b_qlora_pilot_v1
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

from src.evaluation.metrics import compute_all_metrics, format_metrics_report
from src.utils.io import read_jsonl, save_json

console = Console()


def evaluate(
    gold_path: str,
    pred_path: str,
    run_name: Optional[str] = None,
    output_dir: str = "outputs",
    save_report: bool = True,
) -> dict:
    """
    Evaluate predictions against gold labels.

    Args:
        gold_path:   Path to gold JSONL file (test set).
        pred_path:   Path to predictions JSONL file.
        run_name:    Experiment run name (used in output filenames).
        output_dir:  Base output directory.
        save_report: If True, save metrics JSON and markdown report.

    Returns:
        dict with all metrics.
    """
    if run_name is None:
        run_name = Path(pred_path).stem

    # --- Load gold ---
    gold_records = {r["id"]: r for r in read_jsonl(gold_path)}
    console.print(f"Loaded {len(gold_records)} gold examples from {gold_path}")

    # --- Load predictions ---
    pred_records = {r["id"]: r for r in read_jsonl(pred_path)}
    console.print(f"Loaded {len(pred_records)} predictions from {pred_path}")

    # --- Align ---
    common_ids = sorted(set(gold_records.keys()) & set(pred_records.keys()))
    missing_in_pred = set(gold_records.keys()) - set(pred_records.keys())
    extra_in_pred = set(pred_records.keys()) - set(gold_records.keys())

    if missing_in_pred:
        console.print(f"[yellow]Warning: {len(missing_in_pred)} gold examples have no prediction.[/yellow]")
    if extra_in_pred:
        console.print(f"[yellow]Warning: {len(extra_in_pred)} predictions have no gold example.[/yellow]")

    console.print(f"Evaluating on {len(common_ids)} aligned examples.")

    # --- Extract aligned lists ---
    gold_labels, pred_labels, pred_scores = [], [], []
    gold_spans, pred_spans, contexts = [], [], []
    parse_errors, latencies = [], []

    for eid in common_ids:
        gold = gold_records[eid]
        pred = pred_records[eid]

        gold_labels.append(gold.get("label", ""))
        pred_labels.append(pred.get("predicted_label", pred.get("pred_label", pred.get("label", ""))))
        pred_scores.append(float(pred.get("confidence", pred.get("pred_hallucination_score", pred.get("hallucination_score", 0.5)))))

        gold_spans.append(gold.get("evidence_span", ""))

        # Evidence span — try multiple key names for backward compatibility:
        # 1. pred_evidence_span  (04_predict.py / 05_evaluate.py standard format)
        # 2. evidence_span       (direct field)
        # 3. evidence_spans[0]   (Colab notebook format — list, first element)
        # 4. raw_output JSON     (Colab notebook format — span embedded in raw_output string)
        pred_span = pred.get("pred_evidence_span") or pred.get("evidence_span") or ""
        if not pred_span:
            ev_list = pred.get("evidence_spans", [])
            if ev_list and isinstance(ev_list, list):
                pred_span = ev_list[0] if isinstance(ev_list[0], str) else ""
        if not pred_span:
            raw = pred.get("raw_output", "")
            if raw:
                try:
                    import json as _json
                    raw_parsed = _json.loads(raw)
                    pred_span = raw_parsed.get("evidence_span", "")
                except Exception:
                    pass
        pred_spans.append(pred_span)
        contexts.append(gold.get("context", ""))

        parse_errors.append(bool(pred.get("parse_error", False)))
        if "latency_ms" in pred:
            latencies.append(float(pred["latency_ms"]))

    # --- Compute metrics ---
    console.print("Computing metrics...")
    metrics = compute_all_metrics(
        gold_labels=gold_labels,
        pred_labels=pred_labels,
        pred_scores=pred_scores,
        gold_spans=gold_spans,
        pred_spans=pred_spans,
        contexts=contexts,
        parse_errors=parse_errors,
        latencies_ms=latencies if latencies else None,
    )

    metrics["run_name"] = run_name
    metrics["gold_file"] = str(gold_path)
    metrics["pred_file"] = str(pred_path)
    metrics["timestamp"] = datetime.now().isoformat()
    metrics["n_missing_predictions"] = len(missing_in_pred)

    # --- Print summary ---
    _print_summary(metrics)

    # --- Save ---
    if save_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f"{run_name}_metrics_{timestamp}.json"
        save_json(metrics, metrics_file)
        console.print(f"\nMetrics saved to: {metrics_file}")

        reports_dir = Path(output_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_file = reports_dir / f"{run_name}_report_{timestamp}.md"
        report_text = format_metrics_report(metrics, run_name)
        report_file.write_text(report_text, encoding="utf-8")
        console.print(f"Report saved to: {report_file}")

    return metrics


def _print_summary(metrics: dict) -> None:
    """Print a concise summary to console."""
    console.print("\n" + "=" * 50)
    console.print(f"[bold]Evaluation Summary: {metrics.get('run_name', '')}[/bold]")
    console.print("=" * 50)
    console.print(f"  Accuracy:              {metrics.get('accuracy', '?'):.4f}")
    console.print(f"  Macro F1:              [bold green]{metrics.get('macro_f1', '?'):.4f}[/bold green]  (primary)")
    console.print(f"  Contradicted Recall:   {metrics.get('contradicted_recall', '?'):.4f}")
    console.print(f"  Unsupported Recall:    {metrics.get('unsupported_recall', '?'):.4f}")
    console.print(f"  ECE:                   {metrics.get('ECE', '?'):.4f}  (target < 0.15)")
    if metrics.get("json_validity_rate") is not None:
        console.print(f"  JSON Validity Rate:    {metrics['json_validity_rate']:.4f}  (target > 0.95)")
    console.print("=" * 50)

    # Pilot success criteria check
    console.print("\n[bold]Pilot Success Criteria:[/bold]")
    macro_f1 = metrics.get("macro_f1", 0)
    json_val = metrics.get("json_validity_rate", 0) or 0
    cont_recall = metrics.get("contradicted_recall", 0)
    ece = metrics.get("ECE", 1.0)

    criteria = [
        ("Macro F1 > baseline (check manually)", True, "manual"),
        ("JSON validity > 0.95", json_val > 0.95, f"{json_val:.4f}"),
        ("Contradicted recall > 0.60", cont_recall > 0.60, f"{cont_recall:.4f}"),
        ("ECE < 0.15", ece < 0.15, f"{ece:.4f}"),
    ]

    for name, passed, value in criteria:
        status = "[green]OK[/green]" if passed is True else ("[red]FAIL[/red]" if passed is False else "[yellow]?[/yellow]")
        console.print(f"  {status} {name}: {value}")