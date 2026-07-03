# -*- coding: utf-8 -*-
"""
hybrid_detector.py — EXP-048 Hibrit Hallucination Detection Pipeline

İki modeli birleştirir:
  - BERT (berturk_exp020, port 8001): Fine-tuned, uzun RAG bağlamları için
  - NLI  (mDeBERTa, port 8002):       Zero-shot, kısa/genel bağlamlar için

Routing:
  ctx_words < 30  → NLI only
  ctx_words 30-80 → Ensemble (her ikisi)
  ctx_words > 80  → BERT only

Kullanım:
  python hybrid_detector.py  # test modu
"""

import httpx
import asyncio
import time
from typing import Optional

BERT_URL = "http://localhost:8001"
NLI_URL  = "http://localhost:8002"

# Routing eşikleri
ROUTE_NLI_MAX   = 30   # < 30 kelime → NLI
ROUTE_BERT_MIN  = 80   # > 80 kelime → BERT
# 30-80 arası → ensemble

# Fusion eşikleri
BERT_CONTRA_THR = 0.85   # BERT contradiction confidence
NLI_CONTRA_THR  = 0.65   # NLI contradiction score
BERT_INSUF_THR  = 0.80   # BERT insufficient confidence
NLI_INSUF_SUP   = 0.10   # NLI support < bu → insuf adayı
NLI_INSUF_CON   = 0.20   # NLI contradiction < bu → insuf adayı
NLI_INSUF_KW    = 0.05   # NLI keyword_overlap < bu → insuf adayı
BERT_SUP_THR    = 0.80   # BERT supported confidence
NLI_SUP_THR     = 0.65   # NLI support score


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

BERT_TO_HYBRID = {
    "supported":            "accept",
    "partially_supported":  "warn",
    "unsupported":          "warn",
    "contradicted":         "revise",
    "insufficient_context": "insufficient_context",
}

BERT_HALLUCINATION_WEIGHT = {
    "contradicted":         1.0,
    "unsupported":          0.70,
    "partially_supported":  0.40,
    "insufficient_context": 0.25,
    "supported":            0.0,
}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route(context: str, mode: str = "auto") -> str:
    """
    mode: "auto" | "bert" | "nli" | "ensemble"
    """
    if mode != "auto":
        return mode
    words = len(context.split())
    if words < ROUTE_NLI_MAX:
        return "nli"
    elif words > ROUTE_BERT_MIN:
        return "bert"
    else:
        return "ensemble"


# ---------------------------------------------------------------------------
# Hallucination score
# ---------------------------------------------------------------------------

def bert_hallucination_score(bert: dict) -> float:
    label = bert.get("predicted_label", "supported")
    conf  = bert.get("confidence", 0.5)
    return BERT_HALLUCINATION_WEIGHT.get(label, 0.5) * conf


def nli_hallucination_score(nli: dict) -> float:
    con = nli.get("contradiction_score", 0.0)
    sup = nli.get("support_score", 0.0)
    return con * 0.75 + (1.0 - sup) * 0.25


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def fuse(bert: dict, nli: dict) -> dict:
    """
    Her iki model sonucunu birleştir → tek karar.
    """
    bert_label = bert.get("predicted_label", "supported")
    bert_conf  = bert.get("confidence", 0.5)
    nli_con    = nli.get("contradiction_score", 0.0)
    nli_sup    = nli.get("support_score", 0.0)
    nli_kw     = nli.get("keyword_overlap", 0.0)

    # 1. BERT güçlü contradiction sinyali
    if bert_label == "contradicted" and bert_conf >= BERT_CONTRA_THR:
        return {
            "decision": "revise",
            "source": "bert",
            "confidence": bert_conf,
            "reason": f"BERT contradicted ({bert_conf:.2f})"
        }

    # 2. NLI güçlü contradiction sinyali
    if nli_con >= NLI_CONTRA_THR:
        return {
            "decision": "revise",
            "source": "nli",
            "confidence": nli_con,
            "reason": f"NLI contradiction_score={nli_con:.2f}"
        }

    # 3. BERT insufficient
    if bert_label == "insufficient_context" and bert_conf >= BERT_INSUF_THR:
        return {
            "decision": "insufficient_context",
            "source": "bert",
            "confidence": bert_conf,
            "reason": f"BERT insufficient ({bert_conf:.2f})"
        }

    # 4. NLI insufficient (düşük sup + düşük con + düşük kw)
    if nli_sup < NLI_INSUF_SUP and nli_con < NLI_INSUF_CON and nli_kw < NLI_INSUF_KW:
        return {
            "decision": "insufficient_context",
            "source": "nli",
            "confidence": 1.0 - nli_sup,
            "reason": f"NLI: sup={nli_sup:.2f}, con={nli_con:.2f}, kw={nli_kw:.2f}"
        }

    # 5. Her iki model supported → accept
    if bert_label == "supported" and bert_conf >= BERT_SUP_THR and nli_sup >= NLI_SUP_THR:
        conf = (bert_conf + nli_sup) / 2
        return {
            "decision": "accept",
            "source": "ensemble",
            "confidence": conf,
            "reason": f"BERT supported ({bert_conf:.2f}) + NLI sup={nli_sup:.2f}"
        }

    # 6. BERT partially_supported veya unsupported → warn
    # Ama NLI guclu accept sinyali veriyorsa NLI'ye guvenir (BERT yanilabilir)
    if bert_label in ("partially_supported", "unsupported"):
        if bert_label == "unsupported" and nli_sup >= 0.70 and nli_con < 0.10:
            # NLI cok guclu destekliyor, BERT'i gec
            return {
                "decision": "accept",
                "source": "nli_override",
                "confidence": nli_sup,
                "reason": f"BERT unsupported ama NLI sup={nli_sup:.2f} >= 0.70, NLI kazandi"
            }
        return {
            "decision": "warn",
            "source": "bert",
            "confidence": bert_conf,
            "reason": f"BERT {bert_label} ({bert_conf:.2f})"
        }

    # 7. NLI warn sinyali
    nli_dec = nli.get("decision", "warn")
    if nli_dec == "warn":
        return {
            "decision": "warn",
            "source": "nli",
            "confidence": nli_sup,
            "reason": f"NLI warn: sup={nli_sup:.2f}"
        }

    # 8. Fallback — muhafazakar
    return {
        "decision": "warn",
        "source": "ensemble_fallback",
        "confidence": 0.5,
        "reason": "Modeller arasında çelişki, muhafazakar karar"
    }


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------

async def call_bert(client: httpx.AsyncClient, payload: dict) -> Optional[dict]:
    try:
        r = await client.post(f"{BERT_URL}/classify", json=payload, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


async def call_nli(client: httpx.AsyncClient, payload: dict) -> Optional[dict]:
    try:
        r = await client.post(f"{NLI_URL}/score", json=payload, timeout=15.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Main detect function
# ---------------------------------------------------------------------------

async def detect(
    context: str,
    answer: str,
    question: str = "",
    claim: str = "",
    gold_label: str = "",
    mode: str = "auto",
) -> dict:
    """
    Ana hallucination detection fonksiyonu.

    Returns:
        dict with keys:
          decision, confidence, hallucination_score, explanation,
          routing, bert, nli, context_words, latency_ms
    """
    t0 = time.time()
    ctx_words = len(context.split())
    routing = route(context, mode)

    # ---------------------------------------------------------------------------
    # Kural 0: Cevap "bilgi yok" ifadesi iceriyorsa erken insufficient_context
    # ---------------------------------------------------------------------------
    INSUF_PHRASES = [
        "baglamda bu soruyu yanitlayacak bilgi",
        "baglamda bilgi yer alm",
        "baglamda bu bilgi",
        "bu soruyu yanitlayacak bilgi",
        "yeterli bilgi bulunm",
        "bilgi yer alm",
        "bilgi bulunm",
        "cevap veremiyorum",
        "bilgi mevcut degil",
        "baglamda yer alm",
        "baglamda bu konuda",
        "baglamda herhangi bir bilgi",
    ]
    answer_lower = answer.lower()
    for phrase in INSUF_PHRASES:
        if phrase in answer_lower:
            latency_ms = round((time.time() - t0) * 1000, 1)
            is_correct = None
            if gold_label:
                is_correct = (gold_label in ("insufficient_context",))
            return {
                "decision": "insufficient_context",
                "confidence": 0.95,
                "hallucination_score": 0.20,
                "explanation": f"Cevap 'bilgi yok' ifadesi iceriyor: '{phrase}'",
                "routing": "keyword_rule",
                "source": "keyword_rule",
                "is_correct": is_correct,
                "bert": None,
                "nli": None,
                "context_words": ctx_words,
                "latency_ms": latency_ms,
            }

    bert_payload = {
        "context": context,
        "answer": answer,
        "question": question if question and len(question.strip()) >= 3 else "Bu iddia dogru mu?",
        "claim": claim or answer,
    }
    if gold_label:
        bert_payload["gold_label"] = gold_label

    # Soru yoksa cevabi soru olarak kullan — NLI daha iyi sonuc verir
    effective_question = question if question and len(question.strip()) >= 3 else answer
    nli_payload = {
        "context": context,
        "answer": answer,
        "question": effective_question,
    }
    if gold_label:
        nli_payload["gold_label"] = gold_label

    bert_result = None
    nli_result  = None

    async with httpx.AsyncClient() as client:
        if routing == "bert":
            bert_result = await call_bert(client, bert_payload)
        elif routing == "nli":
            nli_result = await call_nli(client, nli_payload)
        else:  # ensemble
            bert_result, nli_result = await asyncio.gather(
                call_bert(client, bert_payload),
                call_nli(client, nli_payload),
            )

    # Hata kontrolü
    bert_ok = bert_result and "error" not in bert_result
    nli_ok  = nli_result  and "error" not in nli_result

    # Karar
    if routing == "bert" and bert_ok:
        label = bert_result["predicted_label"]
        decision   = BERT_TO_HYBRID.get(label, "warn")
        confidence = bert_result["confidence"]
        h_score    = bert_hallucination_score(bert_result)
        source     = "bert"
        reason     = f"BERT: {label} ({confidence:.2f})"

    elif routing == "nli" and nli_ok:
        nli_dec_raw = nli_result["decision"]
        nli_con_raw = nli_result.get("contradiction_score", 0.0)
        nli_sup_raw = nli_result.get("support_score", 0.0)
        nli_kw_raw  = nli_result.get("keyword_overlap", 0.0)

        # warn + con >= 0.06 → revise (zayif contradiction sinyali yakalamak icin)
        if nli_dec_raw == "warn" and nli_con_raw >= 0.06 and nli_sup_raw < 0.20:
            decision = "revise"
            source   = "nli_upgraded"
            reason   = f"NLI warn->revise: con={nli_con_raw:.2f} >= 0.06"
        # warn + sup >= 0.50 → accept (NLI destekliyor ama esigin altinda)
        # con < 0.20 esigi: zayif contradiction sinyali gormezden gel
        elif nli_dec_raw == "warn" and nli_sup_raw >= 0.50 and nli_con_raw < 0.20:
            decision = "accept"
            source   = "nli_upgraded"
            reason   = f"NLI warn->accept: sup={nli_sup_raw:.2f} >= 0.50, con={nli_con_raw:.2f} < 0.20"
        # warn + kw >= 0.30 + con < 0.05 → accept
        # Keyword overlap yuksekse baglamda ayni kelimeler var, destekleniyor
        elif nli_dec_raw == "warn" and nli_kw_raw >= 0.30 and nli_con_raw < 0.05:
            decision = "accept"
            source   = "nli_kw_upgraded"
            reason   = f"NLI warn->accept: kw={nli_kw_raw:.2f} >= 0.30, con={nli_con_raw:.2f} < 0.05"
        else:
            decision = nli_dec_raw
            source   = "nli"
            reason   = (f"NLI: sup={nli_sup_raw:.2f}, con={nli_con_raw:.2f}")
        confidence = max(nli_sup_raw, nli_con_raw)
        h_score    = nli_hallucination_score(nli_result)

    elif routing == "ensemble" and bert_ok and nli_ok:
        fusion     = fuse(bert_result, nli_result)
        decision   = fusion["decision"]
        confidence = fusion["confidence"]
        source     = fusion["source"]
        reason     = fusion["reason"]
        h_score    = (bert_hallucination_score(bert_result) +
                      nli_hallucination_score(nli_result)) / 2

    elif bert_ok:
        # NLI başarısız, BERT'e fall back
        label      = bert_result["predicted_label"]
        decision   = BERT_TO_HYBRID.get(label, "warn")
        confidence = bert_result["confidence"]
        h_score    = bert_hallucination_score(bert_result)
        source     = "bert_fallback"
        reason     = f"NLI hatası, BERT fallback: {label}"

    elif nli_ok:
        # BERT başarısız, NLI'ye fall back
        decision   = nli_result["decision"]
        confidence = max(nli_result["support_score"], nli_result["contradiction_score"])
        h_score    = nli_hallucination_score(nli_result)
        source     = "nli_fallback"
        reason     = "BERT hatası, NLI fallback"

    else:
        decision   = "warn"
        confidence = 0.0
        h_score    = 0.5
        source     = "error"
        reason     = "Her iki model de başarısız"

    # is_correct
    is_correct = None
    if gold_label:
        # gold_label → kabul edilebilir hibrit karar seti
        # partially_supported: hem warn hem revise kabul (sinir etiket)
        # unsupported: insufficient_context da kabul (baglamda konu hic yoksa her ikisi mantikli)
        gold_accepted = {
            "supported":             {"accept"},
            "partially_supported":   {"warn", "revise"},
            "unsupported":           {"warn", "revise", "insufficient_context"},
            "contradicted":          {"revise"},
            "insufficient_context":  {"insufficient_context"},
            "accept":                {"accept"},
            "warn":                  {"warn"},
            "revise":                {"revise"},
        }.get(gold_label)
        is_correct = (decision in gold_accepted) if gold_accepted else None

    latency_ms = round((time.time() - t0) * 1000, 1)

    return {
        "decision": decision,
        "confidence": round(confidence, 4),
        "hallucination_score": round(h_score, 4),
        "explanation": reason,
        "routing": routing,
        "source": source,
        "is_correct": is_correct,
        "bert": bert_result,
        "nli": nli_result,
        "context_words": ctx_words,
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    tests = [
        {
            "name": "KISA: ali eve/carsiya",
            "context": "ali carsiya gitti",
            "answer": "ali eve gitti",
            "question": "ali nereye gitti?",
        },
        {
            "name": "KISA: RAM kalici/gecici",
            "context": "RAM gecici bellektir, elektrik kesilince kaybolur.",
            "answer": "RAM kalici bellektir.",
            "question": "RAM kalici mi?",
        },
        {
            "name": "UZUN: II. Dunya Savasi (contradicted)",
            "context": "II. Dunya Savasi, 1939-1945 yillari arasinda gerceklesen kuresel savas. Mihver ve Muttefik devletler arasinda yasanmistir. Savasin cikis nedenleri arasinda Almanya'nin yayilmaci politikasi ve Versay Antlasmasi'nin yarattigi gerilimler sayilabilir.",
            "answer": "Savas yalnizca tarafsiz devletler arasinda yasanmistir.",
            "question": "Baglamda savas hangi devlet gruplari arasinda yasanmistir?",
            "gold_label": "contradicted",
        },
        {
            "name": "UZUN: Fakir Baykurt (insufficient)",
            "context": "Fakir Baykurt, Turk romanci ve ogretmendir. Koy yasamini, koylulerin sorunlarini ve toplumsal esitsizlikleri konu alan eserleriyle bilinir. Yilanlarin Ocu, yazarin taninmis romanlarindan biridir.",
            "answer": "Baglamda bu soruyu yanitlayacak bilgi yer almiyor.",
            "question": "Baglamda Fakir Baykurt hangi yil dogmustur?",
            "gold_label": "insufficient_context",
        },
        {
            "name": "ORTA: Kucuk Menderes (supported)",
            "context": "Kucuk Menderes, Bati Anadolu'da yer alan akarsu sistemlerinden biridir. Ege Bolgesindeki ovalar ve tarim alanlariyla iliskilidir. Havzasi yerlesim ve uretim acisindan onem tasir.",
            "answer": "Kucuk Menderes ege bolgesindeki ovalar ve tarim alanlariyla iliskilidir.",
            "question": "Baglamda Kucuk Menderes icin hangi cikarim yapilabilir?",
            "gold_label": "supported",
        },
    ]

    async def run_tests():
        print(f"\n{'='*80}")
        print("EXP-048 Hibrit Hallucination Detector — Test")
        print(f"{'='*80}\n")
        print(f"{'Test':<40} {'Routing':<10} {'Karar':<22} {'H-Score':<10} {'Dogru?':<8} {'ms'}")
        print("-"*100)

        correct = 0
        total_with_gold = 0

        for t in tests:
            result = await detect(
                context=t["context"],
                answer=t["answer"],
                question=t.get("question", ""),
                gold_label=t.get("gold_label", ""),
            )
            ok = result.get("is_correct")
            if ok is not None:
                total_with_gold += 1
                if ok:
                    correct += 1
            mark = "✓" if ok else ("✗" if ok is False else "—")
            print(f"{t['name']:<40} {result['routing']:<10} "
                  f"{result['decision']:<22} {result['hallucination_score']:<10.3f} "
                  f"{mark:<8} {result['latency_ms']}")

        if total_with_gold > 0:
            print(f"\nSonuc: {correct}/{total_with_gold} dogru ({correct/total_with_gold*100:.0f}%)")

        print("\nDetayli sonuc (ilk test):")
        r = await detect(
            context=tests[0]["context"],
            answer=tests[0]["answer"],
            question=tests[0].get("question", ""),
        )
        # bert ve nli alanlarini cikar (cok uzun)
        r_short = {k: v for k, v in r.items() if k not in ("bert", "nli")}
        print(json.dumps(r_short, ensure_ascii=False, indent=2))

    asyncio.run(run_tests())