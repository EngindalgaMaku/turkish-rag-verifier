"""
nli_api.py — FastAPI REST API for Score-based NLI Pipeline (EXP-038/043b)

Endpoints:
    GET  /health          — model durumu
    POST /score           — tekli NLI scoring + policy_v2 karar
    POST /score/batch     — çoklu scoring (max 50)

Pipeline:
    1. mDeBERTa NLI ile context-answer NLI skoru al
    2. Keyword overlap hesapla
    3. Policy v2 ile final_decision uret:
       - contradiction_score >= p_contra  -> revise
       - support_score >= p_high          -> accept
       - support_score >= p_mid           -> warn
       - keyword_overlap >= t_kw          -> warn
       - else                             -> insufficient_context

Usage:
    cd turkish-rag-verifier
    python -m uvicorn nli_api:app --host 0.0.0.0 --port 8002 --reload

Environment variables:
    NLI_MODEL      — HuggingFace model adı (default: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
    NLI_P_CONTRA   — contradiction threshold (default: 0.30)
    NLI_P_HIGH     — high support threshold (default: 0.80)
    NLI_P_MID      — mid support threshold (default: 0.30)
    NLI_T_KW       — keyword overlap threshold (default: 0.05)
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NLI_MODEL  = os.getenv("NLI_MODEL",    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
P_CONTRA   = float(os.getenv("NLI_P_CONTRA", "0.30"))
P_HIGH     = float(os.getenv("NLI_P_HIGH",   "0.80"))
P_MID      = float(os.getenv("NLI_P_MID",    "0.30"))
T_KW       = float(os.getenv("NLI_T_KW",     "0.05"))

TR_STOPWORDS = {
    "bir", "bu", "ve", "ile", "de", "da", "için", "olan", "olarak",
    "mi", "mu", "mı", "mü", "ki", "ne", "en", "çok", "daha",
    "ise", "ya", "veya", "hem", "ama", "ancak", "fakat", "lakin", "sadece",
    "yalnızca", "bile", "dahi", "gibi", "kadar", "sonra", "önce", "her",
    "tüm", "bütün", "hiç", "hiçbir", "bazı", "birçok", "çeşitli", "aynı",
    "farklı", "yeni", "eski", "büyük", "küçük", "iyi", "kötü", "doğru",
    "yanlış", "tam", "yarım", "az", "çok", "hep", "zaman", "yer", "şey",
    "var", "yok", "olur", "oldu", "olacak", "edilir", "edildi",
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Turkish RAG NLI Scorer API",
    description=(
        "mDeBERTa tabanlı zero-shot NLI + keyword overlap policy_v2. "
        "EXP-038/043b pipeline. Bağlam ve cevap alır; support/contradiction skoru + karar döner."
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
_ent_idx: int = 0
_con_idx: int = 2
_load_time: float = 0.0
_load_error: str = ""


def get_model():
    global _model, _tokenizer, _ent_idx, _con_idx, _load_time, _load_error
    if _model is not None:
        return _model, _tokenizer, _ent_idx, _con_idx

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        t0 = time.time()
        _tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        _model.eval()

        id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
        _ent_idx = next(i for i, l in id2label.items() if "entail" in l)
        _con_idx = next(i for i, l in id2label.items() if "contradict" in l)

        _load_time = round(time.time() - t0, 2)
        _load_error = ""
        return _model, _tokenizer, _ent_idx, _con_idx

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
# Helpers
# ---------------------------------------------------------------------------

def keyword_overlap(text: str, context: str, min_len: int = 4) -> float:
    def tokenize(t: str):
        t = t.lower()
        t = re.sub(r'[^\w\s]', ' ', t)
        tokens = set(t.split())
        return {tok for tok in tokens if len(tok) >= min_len and tok not in TR_STOPWORDS}
    text_kws = tokenize(text)
    ctx_kws = tokenize(context)
    if not text_kws:
        return 0.0
    return len(text_kws & ctx_kws) / len(text_kws)


def policy_v2(sup: float, con: float, kw: float) -> str:
    if con >= P_CONTRA:
        return "revise"
    elif sup >= P_HIGH:
        return "accept"
    elif sup >= P_MID:
        return "warn"
    elif kw >= T_KW:
        return "warn"
    else:
        return "insufficient_context"


def safe_float(v) -> float:
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except Exception:
        return 0.0


def run_nli(context: str, hypothesis: str) -> dict:
    import torch
    import torch.nn.functional as F

    model, tokenizer, ent_idx, con_idx = get_model()
    enc = tokenizer(
        context, hypothesis,
        truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(**enc).logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        probs = F.softmax(logits, dim=-1)[0].tolist()

    return {
        "support_score": round(safe_float(probs[ent_idx]), 6),
        "contradiction_score": round(safe_float(probs[con_idx]), 6),
        "neutral_score": round(safe_float(1.0 - probs[ent_idx] - probs[con_idx]), 6),
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ScoreRequest(BaseModel):
    context: str  = Field(..., min_length=5,  description="RAG bağlamı")
    answer: str   = Field(..., min_length=3,  description="Doğrulanacak cevap/iddia")
    question: Optional[str] = Field(None,     description="Kullanıcı sorusu (opsiyonel)")
    gold_label: Optional[str] = Field(None,   description="Gerçek karar (değerlendirme için)")

class ScoreResult(BaseModel):
    support_score: float
    contradiction_score: float
    neutral_score: float
    keyword_overlap: float
    decision: str
    is_correct: Optional[bool]
    latency_ms: float
    policy_params: dict

class BatchScoreRequest(BaseModel):
    items: List[ScoreRequest] = Field(..., max_length=50)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    policy: dict
    load_time_s: float
    error: Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    loaded = _model is not None
    return JSONResponse(content={
        "status": "ok" if loaded else "model_not_loaded",
        "model_loaded": loaded,
        "model_name": NLI_MODEL,
        "policy": {
            "p_contra": P_CONTRA,
            "p_high": P_HIGH,
            "p_mid": P_MID,
            "t_kw": T_KW,
        },
        "load_time_s": _load_time,
        "error": _load_error or None,
    })


@app.post("/score", tags=["Scoring"])
async def score(req: ScoreRequest):
    """Tekli NLI scoring + policy_v2 karar."""
    try:
        t0 = time.time()
        nli = run_nli(req.context, req.answer)
        kw = keyword_overlap(req.answer, req.context)
        decision = policy_v2(nli["support_score"], nli["contradiction_score"], kw)
        latency_ms = round((time.time() - t0) * 1000, 1)

        is_correct = None
        if req.gold_label:
            is_correct = decision == req.gold_label.lower().strip()

        return JSONResponse(content={
            "support_score": nli["support_score"],
            "contradiction_score": nli["contradiction_score"],
            "neutral_score": nli["neutral_score"],
            "keyword_overlap": round(kw, 4),
            "decision": decision,
            "is_correct": is_correct,
            "latency_ms": latency_ms,
            "policy_params": {
                "p_contra": P_CONTRA,
                "p_high": P_HIGH,
                "p_mid": P_MID,
                "t_kw": T_KW,
            },
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scoring error: {exc}")


@app.post("/score/batch", tags=["Scoring"])
async def score_batch(req: BatchScoreRequest):
    """Çoklu NLI scoring (max 50 örnek)."""
    try:
        t0 = time.time()
        results = []
        decision_counts = {"accept": 0, "warn": 0, "revise": 0, "insufficient_context": 0}
        n_correct = 0
        n_with_gold = 0

        for item in req.items:
            t1 = time.time()
            nli = run_nli(item.context, item.answer)
            kw = keyword_overlap(item.answer, item.context)
            decision = policy_v2(nli["support_score"], nli["contradiction_score"], kw)
            item_latency = round((time.time() - t1) * 1000, 1)

            is_correct = None
            if item.gold_label:
                is_correct = decision == item.gold_label.lower().strip()
                n_with_gold += 1
                if is_correct:
                    n_correct += 1

            decision_counts[decision] = decision_counts.get(decision, 0) + 1

            results.append({
                "support_score": nli["support_score"],
                "contradiction_score": nli["contradiction_score"],
                "neutral_score": nli["neutral_score"],
                "keyword_overlap": round(kw, 4),
                "decision": decision,
                "is_correct": is_correct,
                "latency_ms": item_latency,
            })

        total_latency_ms = round((time.time() - t0) * 1000, 1)
        accuracy = round(n_correct / n_with_gold, 4) if n_with_gold > 0 else None

        return JSONResponse(content={
            "count": len(results),
            "results": results,
            "summary": {
                "accuracy": accuracy,
                "n_with_gold": n_with_gold,
                "n_correct": n_correct,
                "decision_counts": decision_counts,
                "avg_support_score": round(sum(r["support_score"] for r in results) / len(results), 4),
                "avg_contradiction_score": round(sum(r["contradiction_score"] for r in results) / len(results), 4),
            },
            "total_latency_ms": total_latency_ms,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch scoring error: {exc}")