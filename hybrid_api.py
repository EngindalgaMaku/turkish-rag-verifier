# -*- coding: utf-8 -*-
"""
hybrid_api.py — EXP-048 Hibrit Hallucination Detection API

Port: 8003
Endpoints:
  GET  /health   — sistem durumu
  POST /detect   — tek örnek
  POST /batch    — çoklu (max 50)

Routing:
  ctx_words < 30  → NLI only  (mDeBERTa, port 8002)
  ctx_words 30-80 → Ensemble  (her ikisi paralel)
  ctx_words > 80  → BERT only (berturk_exp020, port 8001)
"""

from __future__ import annotations

import time
import asyncio
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hybrid_detector import detect, route, BERT_URL, NLI_URL

import httpx

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Hybrid Hallucination Detector",
    description="BERT (berturk_exp020) + NLI (mDeBERTa) hibrit pipeline — EXP-048",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Statik dosyalar — HTML demo arayuzleri
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent

# Ana sayfa ve kisayollar
@app.get("/", include_in_schema=False)
async def root():
    f = _HERE / "hybrid_demo.html"
    return FileResponse(f) if f.exists() else RedirectResponse("/docs")

@app.get("/demo", include_in_schema=False)
async def demo():
    f = _HERE / "hybrid_demo.html"
    return FileResponse(f) if f.exists() else RedirectResponse("/docs")

@app.get("/bert-demo", include_in_schema=False)
async def bert_demo_page():
    f = _HERE / "bert_demo.html"
    return FileResponse(f) if f.exists() else RedirectResponse("/demo")

@app.get("/nli-demo", include_in_schema=False)
async def nli_demo_page():
    f = _HERE / "nli_demo.html"
    return FileResponse(f) if f.exists() else RedirectResponse("/demo")

# Dogrudan dosya adi ile erisim: /hybrid_demo.html, /bert_demo.html, /nli_demo.html
@app.get("/{filename}.html", include_in_schema=False)
async def serve_html(filename: str):
    f = _HERE / f"{filename}.html"
    if f.exists():
        return FileResponse(f)
    return RedirectResponse("/demo")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    context:    str = Field(..., description="Kaynak metin (bağlam)")
    answer:     str = Field(..., description="RAG cevabı / doğrulanacak metin")
    question:   str = Field("", description="Kullanıcı sorusu (opsiyonel)")
    claim:      str = Field("", description="Spesifik iddia (opsiyonel, boşsa answer kullanılır)")
    gold_label: str = Field("", description="Gerçek etiket (değerlendirme için opsiyonel)")
    mode:       str = Field("auto", description="auto | bert | nli | ensemble")

class DetectResult(BaseModel):
    decision:           str
    confidence:         float
    hallucination_score: float
    explanation:        str
    routing:            str
    source:             str
    is_correct:         Optional[bool]
    context_words:      int
    latency_ms:         float
    bert:               Optional[dict]
    nli:                Optional[dict]

class BatchDetectRequest(BaseModel):
    items: List[DetectRequest] = Field(..., max_length=50)

class BatchDetectResult(BaseModel):
    count:          int
    results:        List[DetectResult]
    summary:        dict
    total_latency_ms: float

class HealthResponse(BaseModel):
    status:       str
    bert_ok:      bool
    nli_ok:       bool
    bert_url:     str
    nli_url:      str
    routing_thresholds: dict

# ---------------------------------------------------------------------------
# Health check helpers
# ---------------------------------------------------------------------------

async def check_backend(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{url}/health", timeout=3.0)
            return r.status_code == 200
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    bert_ok, nli_ok = await asyncio.gather(
        check_backend(BERT_URL),
        check_backend(NLI_URL),
    )
    status = "ok" if (bert_ok or nli_ok) else "degraded"
    return HealthResponse(
        status=status,
        bert_ok=bert_ok,
        nli_ok=nli_ok,
        bert_url=BERT_URL,
        nli_url=NLI_URL,
        routing_thresholds={
            "nli_max_words": 30,
            "bert_min_words": 80,
            "ensemble_range": "30-80",
        },
    )


@app.post("/detect", response_model=DetectResult)
async def detect_endpoint(req: DetectRequest):
    result = await detect(
        context=req.context,
        answer=req.answer,
        question=req.question,
        claim=req.claim,
        gold_label=req.gold_label,
        mode=req.mode,
    )
    return DetectResult(**result)


@app.post("/batch", response_model=BatchDetectResult)
async def batch_detect(req: BatchDetectRequest):
    t0 = time.time()

    tasks = [
        detect(
            context=item.context,
            answer=item.answer,
            question=item.question,
            claim=item.claim,
            gold_label=item.gold_label,
            mode=item.mode,
        )
        for item in req.items
    ]
    results = await asyncio.gather(*tasks)

    # Özet istatistikler
    decisions = [r["decision"] for r in results]
    routings  = [r["routing"]  for r in results]
    h_scores  = [r["hallucination_score"] for r in results]
    correct   = [r["is_correct"] for r in results if r["is_correct"] is not None]

    summary = {
        "decision_dist": {d: decisions.count(d) for d in set(decisions)},
        "routing_dist":  {r: routings.count(r)  for r in set(routings)},
        "avg_hallucination_score": round(sum(h_scores) / len(h_scores), 4) if h_scores else 0,
        "max_hallucination_score": round(max(h_scores), 4) if h_scores else 0,
        "accuracy": round(sum(correct) / len(correct), 4) if correct else None,
        "n_with_gold": len(correct),
    }

    total_ms = round((time.time() - t0) * 1000, 1)

    return BatchDetectResult(
        count=len(results),
        results=[DetectResult(**r) for r in results],
        summary=summary,
        total_latency_ms=total_ms,
    )


# ---------------------------------------------------------------------------
# BERT proxy endpoints — bert_demo.html için BERT API'yi dışarıya açar
# ---------------------------------------------------------------------------

@app.get("/bert/health")
async def bert_health_proxy():
    """BERT API health check proxy."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{BERT_URL}/health", timeout=5.0)
            return r.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}


@app.post("/bert/classify")
async def bert_classify_proxy(request: dict):
    """BERT API tekli sınıflandırma proxy."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{BERT_URL}/classify", json=request, timeout=30.0)
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))


@app.post("/bert/classify/batch")
async def bert_classify_batch_proxy(request: dict):
    """BERT API toplu sınıflandırma proxy."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{BERT_URL}/classify/batch", json=request, timeout=120.0)
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)