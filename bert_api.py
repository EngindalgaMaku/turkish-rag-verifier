"""
bert_api.py — FastAPI REST API for XLM-R BERT Classifier (exp019)

Endpoints:
    GET  /health          — model durumu
    POST /classify        — tekli sınıflandırma
    POST /classify/batch  — çoklu sınıflandırma (max 50)

Usage:
    cd turkish-rag-verifier
    uvicorn bert_api:app --host 0.0.0.0 --port 8001 --reload

Environment variables:
    BERT_MODEL_DIR   — model dizini (default: outputs/bert_classifier/xlm_roberta_base_exp019_pilot_v7_weighted_ft)
    BERT_THRESHOLD   — kabul eşiği (default: 0.75)
    BERT_MAX_LENGTH  — tokenizer max_length (default: 512)
    BERT_BATCH_SIZE  — inference batch size (default: 32)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR   = os.getenv("BERT_MODEL_DIR",  "outputs/bert_classifier/xlm_roberta_base_exp019_pilot_v7_weighted_ft")
THRESHOLD   = float(os.getenv("BERT_THRESHOLD",  "0.75"))
MAX_LENGTH  = int(os.getenv("BERT_MAX_LENGTH",   "512"))
BATCH_SIZE  = int(os.getenv("BERT_BATCH_SIZE",   "32"))

LABELS = ["supported", "partially_supported", "unsupported", "contradicted", "insufficient_context"]
RISK_MAPPING = {
    "supported": "safe",
    "partially_supported": "unsafe",
    "unsupported": "unsafe",
    "contradicted": "unsafe",
    "insufficient_context": "unsafe",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Turkish RAG BERT Classifier API",
    description=(
        "XLM-R tabanlı Türkçe RAG hallucination sınıflandırıcısı. "
        "Bağlam, soru ve cevap alır; 5 sınıflı karar + güven skoru döner."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_id2label: dict = {}
_load_time: float = 0.0
_load_error: str = ""


def get_model():
    global _model, _tokenizer, _id2label, _load_time, _load_error
    if _model is not None:
        return _model, _tokenizer, _id2label

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # HF repo ID (org/model) veya local dizin desteklenir
        model_path = Path(MODEL_DIR)
        if model_path.exists():
            model_id = str(model_path)
        else:
            # HF repo ID olarak dene (org/model formatı)
            model_id = MODEL_DIR

        t0 = time.time()
        # Tokenizer: once model_id'den dene, yoksa base model'den al
        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            # Local model tokenizer dosyasi olmayabilir — base model'den al
            base_model = os.getenv("BERT_BASE_MODEL", "microsoft/mdeberta-v3-base")
            _tokenizer = AutoTokenizer.from_pretrained(base_model)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        _model.eval()

        raw = _model.config.id2label
        _id2label = {int(k): v for k, v in raw.items()}
        _load_time = round(time.time() - t0, 2)
        _load_error = ""
        return _model, _tokenizer, _id2label

    except Exception as exc:
        _load_error = str(exc)
        raise


@app.on_event("startup")
async def startup():
    try:
        get_model()
    except Exception:
        pass  # lazy load on first request


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    context: str  = Field(..., min_length=5,  description="RAG bağlamı")
    question: str = Field(..., min_length=3,  description="Kullanıcı sorusu")
    answer: str   = Field(..., min_length=3,  description="Doğrulanacak cevap")
    claim: Optional[str] = Field(None,        description="İddia (boşsa answer kullanılır)")
    gold_label: Optional[str] = Field(None,   description="Gerçek etiket (değerlendirme için)")

class ClassifyResult(BaseModel):
    predicted_label: str
    confidence: float
    risk: str
    decision: str
    probs: dict
    is_correct: Optional[bool]
    latency_ms: float

class BatchClassifyRequest(BaseModel):
    items: List[ClassifyRequest] = Field(..., max_length=50)

class BatchClassifyResult(BaseModel):
    count: int
    results: List[ClassifyResult]
    summary: dict
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_dir: str
    threshold: float
    load_time_s: float
    error: Optional[str]
    device: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_text(req: ClassifyRequest) -> str:
    claim = req.claim or req.answer
    return (
        f"Soru: {req.question}\n\n"
        f"Baglam:\n{req.context}\n\n"
        f"Cevap:\n{req.answer}\n\n"
        f"Iddia:\n{claim}"
    )


def run_inference(texts: list[str]) -> list[dict]:
    import math
    import numpy as np
    import torch

    model, tokenizer, id2label = get_model()
    device = next(model.parameters()).device

    def safe_float(v: float) -> float:
        """NaN/Inf değerleri 0.0 ile replace et."""
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return float(v)

    all_results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            # NaN guard: replace NaN logits with 0
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            probs_tensor = torch.softmax(logits, dim=-1).cpu()
            probs_list = probs_tensor.tolist()  # native Python float list

        for prob_vec in probs_list:
            # Safe conversion
            safe_vec = [safe_float(p) for p in prob_vec]
            # Renormalize if sum != 1 (due to NaN replacement)
            total = sum(safe_vec)
            if total > 0:
                safe_vec = [p / total for p in safe_vec]
            else:
                safe_vec = [1.0 / len(safe_vec)] * len(safe_vec)

            pred_idx = int(np.argmax(safe_vec))
            predicted_label = id2label[pred_idx]
            confidence = safe_vec[pred_idx]
            probs_dict = {id2label[j]: round(p, 6) for j, p in enumerate(safe_vec)}
            all_results.append({
                "predicted_label": predicted_label,
                "confidence": round(confidence, 6),
                "probs": probs_dict,
            })
    return all_results


def make_decision(label: str, confidence: float) -> str:
    if label == "supported" and confidence >= THRESHOLD:
        return "accept"
    return "block_or_review"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    import torch
    loaded = _model is not None
    device = "cuda" if (loaded and next(_model.parameters()).is_cuda) else "cpu"
    return HealthResponse(
        status="ok" if loaded else "model_not_loaded",
        model_loaded=loaded,
        model_dir=MODEL_DIR,
        threshold=THRESHOLD,
        load_time_s=_load_time,
        error=_load_error or None,
        device=device,
    )


@app.post("/classify", tags=["Classification"])
@app.post("/verify", tags=["Classification"])
async def classify(req: ClassifyRequest):
    """Tekli sınıflandırma."""
    try:
        t0 = time.time()
        text = build_text(req)
        results = run_inference([text])
        r = results[0]
        latency_ms = round((time.time() - t0) * 1000, 1)

        # Ensure Python native float (not numpy)
        confidence = float(r["confidence"]) if r["confidence"] is not None else 0.0
        probs = {k: round(float(v), 6) for k, v in r["probs"].items()}

        decision = make_decision(r["predicted_label"], confidence)
        risk = RISK_MAPPING.get(r["predicted_label"], "unsafe")

        is_correct = None
        if req.gold_label:
            is_correct = r["predicted_label"] == req.gold_label

        return JSONResponse(content={
            "predicted_label": r["predicted_label"],
            "confidence": round(confidence, 6),
            "risk": risk,
            "decision": decision,
            "probs": probs,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
        })
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")


@app.post("/classify/batch", response_model=BatchClassifyResult, tags=["Classification"])
async def classify_batch(req: BatchClassifyRequest):
    """Çoklu sınıflandırma (max 50 örnek)."""
    try:
        t0 = time.time()
        texts = [build_text(item) for item in req.items]
        raw_results = run_inference(texts)
        total_latency_ms = round((time.time() - t0) * 1000, 1)

        results = []
        label_counts: dict[str, int] = {lbl: 0 for lbl in LABELS}
        n_correct = 0
        n_with_gold = 0
        n_accept = 0
        n_block = 0

        for item, r in zip(req.items, raw_results):
            decision = make_decision(r["predicted_label"], r["confidence"])
            risk = RISK_MAPPING.get(r["predicted_label"], "unsafe")

            is_correct = None
            if item.gold_label:
                is_correct = r["predicted_label"] == item.gold_label
                n_with_gold += 1
                if is_correct:
                    n_correct += 1

            label_counts[r["predicted_label"]] = label_counts.get(r["predicted_label"], 0) + 1
            if decision == "accept":
                n_accept += 1
            else:
                n_block += 1

            results.append(ClassifyResult(
                predicted_label=r["predicted_label"],
                confidence=r["confidence"],
                risk=risk,
                decision=decision,
                probs=r["probs"],
                is_correct=is_correct,
                latency_ms=0.0,
            ))

        accuracy = round(n_correct / n_with_gold, 4) if n_with_gold > 0 else None
        summary = {
            "n_total": len(results),
            "n_accept": n_accept,
            "n_block": n_block,
            "label_counts": label_counts,
            "accuracy": accuracy,
            "n_with_gold_label": n_with_gold,
        }

        return BatchClassifyResult(
            count=len(results),
            results=results,
            summary=summary,
            total_latency_ms=total_latency_ms,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {exc}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "bert_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
        workers=1,
    )