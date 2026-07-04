# -*- coding: utf-8 -*-
"""
hybrid_detector.py — EXP-048b Hibrit Hallucination Detection Pipeline

Her zaman hem BERT hem NLI paralel çalışır.
Fusion mantığı:
  - İkisi aynı fikirdeyse → güvenli karar (yüksek confidence)
  - Biri tehlike işareti veriyorsa → warn (şüpheli)
  - İkisi de tehlike işareti veriyorsa → revise (hallucination)
  - Ağırlıklandırma: BERT=0.65, NLI=0.35

Kullanım:
  python hybrid_detector.py  # test modu
"""

import httpx
import asyncio
import time
from typing import Optional

import os
BERT_URL = os.getenv("BERT_URL", "http://localhost:8001")
NLI_URL  = os.getenv("NLI_URL",  "http://localhost:8002")

# Ağırlıklar — EXP-049 test sonuçlarına göre:
# Bu BERT modeli (temp_out_saved_head) bu örneklerde bozuk (hepsi accept)
# NLI: 7/10, BERT: 4/10, Ensemble: 4/10
# NLI ağırlığı artırıldı, BERT minimize edildi
BERT_WEIGHT = 0.15
NLI_WEIGHT  = 0.85

# Fusion eşikleri
BERT_CONTRA_THR = 0.70   # BERT contradiction confidence
NLI_CONTRA_THR  = 0.55   # NLI contradiction score
BERT_INSUF_THR  = 0.75   # BERT insufficient confidence
NLI_INSUF_SUP   = 0.10   # NLI support < bu → insuf adayı
NLI_INSUF_CON   = 0.20   # NLI contradiction < bu → insuf adayı
NLI_INSUF_KW    = 0.05   # NLI keyword_overlap < bu → insuf adayı
BERT_SUP_THR    = 0.75   # BERT supported confidence
NLI_SUP_THR     = 0.60   # NLI support score

# Uyuşmazlık eşiği — ikisi farklı fikirdeyse warn
DISAGREE_THR = 0.40   # hallucination score farkı bu kadarsa → warn


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
# Routing — her zaman ensemble (auto modda)
# ---------------------------------------------------------------------------

def route(context: str, mode: str = "auto") -> str:
    """
    mode: "auto" → her zaman ensemble (hem BERT hem NLI paralel)
          "bert"  → sadece BERT
          "nli"   → sadece NLI
          "ensemble" → her ikisi
    """
    if mode != "auto":
        return mode
    return "ensemble"  # her zaman ikisi birlikte


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
# Fusion — ağırlıklı + uyuşmazlık tespiti
# ---------------------------------------------------------------------------

def _bert_danger(bert: dict) -> float:
    """BERT'in tehlike skoru: 0=güvenli, 1=tehlikeli"""
    label = bert.get("predicted_label", "supported")
    conf  = bert.get("confidence", 0.5)
    return BERT_HALLUCINATION_WEIGHT.get(label, 0.5) * conf


def _nli_danger(nli: dict) -> float:
    """NLI'nin tehlike skoru: 0=güvenli, 1=tehlikeli"""
    con = nli.get("contradiction_score", 0.0)
    sup = nli.get("support_score", 0.0)
    return con * 0.75 + (1.0 - sup) * 0.25


def fuse(bert: dict, nli: dict) -> dict:
    """
    Her zaman ensemble — ağırlıklı fusion + uyuşmazlık tespiti.

    Mantık:
      1. Ağırlıklı tehlike skoru hesapla (BERT=0.65, NLI=0.35)
      2. İkisi aynı fikirdeyse → güvenli karar
      3. Biri tehlike, diğeri güvenli → warn (şüpheli, uyuşmazlık)
      4. İkisi de tehlike → revise
    """
    bert_label = bert.get("predicted_label", "supported")
    bert_conf  = bert.get("confidence", 0.5)
    nli_con    = nli.get("contradiction_score", 0.0)
    nli_sup    = nli.get("support_score", 0.0)
    nli_kw     = nli.get("keyword_overlap", 0.0)

    b_danger = _bert_danger(bert)
    n_danger = _nli_danger(nli)
    w_danger = BERT_WEIGHT * b_danger + NLI_WEIGHT * n_danger

    # Uyuşmazlık var mı?
    disagree = abs(b_danger - n_danger) >= DISAGREE_THR

    # --- Özel durumlar önce ---

    # Insufficient context: BERT güçlü sinyal
    if bert_label == "insufficient_context" and bert_conf >= BERT_INSUF_THR:
        return {
            "decision": "insufficient_context",
            "source": "bert",
            "confidence": bert_conf,
            "reason": f"BERT insufficient ({bert_conf:.2f})"
        }

    # NLI insufficient (düşük sup + düşük con + düşük kw)
    if nli_sup < NLI_INSUF_SUP and nli_con < NLI_INSUF_CON and nli_kw < NLI_INSUF_KW:
        return {
            "decision": "insufficient_context",
            "source": "nli",
            "confidence": 1.0 - nli_sup,
            "reason": f"NLI: sup={nli_sup:.2f}, con={nli_con:.2f}, kw={nli_kw:.2f}"
        }

    # --- Ağırlıklı karar ---

    if w_danger >= 0.65:
        # Her iki model tehlike görüyor veya BERT güçlü tehlike
        if disagree:
            # Biri tehlike diğeri güvenli ama ağırlıklı skor yüksek → warn
            return {
                "decision": "warn",
                "source": "ensemble_disagree",
                "confidence": w_danger,
                "reason": f"Uyuşmazlık: BERT={b_danger:.2f}, NLI={n_danger:.2f}, ağırlıklı={w_danger:.2f}"
            }
        return {
            "decision": "revise",
            "source": "ensemble",
            "confidence": w_danger,
            "reason": f"Her iki model tehlike: BERT={b_danger:.2f}, NLI={n_danger:.2f}"
        }

    elif w_danger >= 0.35:
        # Orta tehlike — warn
        if disagree:
            return {
                "decision": "warn",
                "source": "ensemble_disagree",
                "confidence": w_danger,
                "reason": f"Uyuşmazlık (orta): BERT={b_danger:.2f}, NLI={n_danger:.2f}"
            }
        return {
            "decision": "warn",
            "source": "ensemble",
            "confidence": w_danger,
            "reason": f"Orta tehlike: BERT={b_danger:.2f}, NLI={n_danger:.2f}"
        }

    else:
        # Düşük tehlike — accept
        if disagree:
            # Biri tehlike görüyor ama ağırlıklı skor düşük → yine de warn
            return {
                "decision": "warn",
                "source": "ensemble_disagree",
                "confidence": w_danger,
                "reason": f"Uyuşmazlık (düşük skor): BERT={b_danger:.2f}, NLI={n_danger:.2f}"
            }
        return {
            "decision": "accept",
            "source": "ensemble",
            "confidence": 1.0 - w_danger,
            "reason": f"Her iki model güvenli: BERT={b_danger:.2f}, NLI={n_danger:.2f}"
        }


# ---------------------------------------------------------------------------
# HTTP calls
# ---------------------------------------------------------------------------

async def call_bert(client: httpx.AsyncClient, payload: dict) -> Optional[dict]:
    try:
        r = await client.post(f"{BERT_URL}/classify", json=payload, timeout=40.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


async def call_nli(client: httpx.AsyncClient, payload: dict) -> Optional[dict]:
    try:
        r = await client.post(f"{NLI_URL}/score", json=payload, timeout=40.0)
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

    # Karar — her zaman ensemble fusion
    if routing == "bert" and bert_ok:
        # Manuel bert modu
        label      = bert_result["predicted_label"]
        decision   = BERT_TO_HYBRID.get(label, "warn")
        confidence = bert_result["confidence"]
        h_score    = bert_hallucination_score(bert_result)
        source     = "bert"
        reason     = f"BERT: {label} ({confidence:.2f})"

    elif routing == "nli" and nli_ok:
        # Manuel nli modu
        nli_dec_raw = nli_result["decision"]
        nli_con_raw = nli_result.get("contradiction_score", 0.0)
        nli_sup_raw = nli_result.get("support_score", 0.0)
        decision   = nli_dec_raw
        confidence = max(nli_sup_raw, nli_con_raw)
        h_score    = nli_hallucination_score(nli_result)
        source     = "nli"
        reason     = f"NLI: sup={nli_sup_raw:.2f}, con={nli_con_raw:.2f}"

    elif bert_ok and nli_ok:
        # Ensemble — ağırlıklı fusion + uyuşmazlık tespiti
        fusion     = fuse(bert_result, nli_result)
        decision   = fusion["decision"]
        confidence = fusion["confidence"]
        source     = fusion["source"]
        reason     = fusion["reason"]
        h_score    = BERT_WEIGHT * bert_hallucination_score(bert_result) + \
                     NLI_WEIGHT  * nli_hallucination_score(nli_result)

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
        # Karışıklık Matrisi (Confusion Matrix) ve sınıf bazlı metriklerle %100 uyum için 1-e-1 eşleme (Strict Evaluation)
        gold_mapped = {
            "supported":             "accept",
            "accept":                "accept",
            "partially_supported":   "warn",
            "unsupported":           "warn",
            "warn":                  "warn",
            "contradicted":          "revise",
            "revise":                "revise",
            "insufficient_context":  "insufficient_context",
        }.get(gold_label.lower().strip())
        is_correct = (decision == gold_mapped) if gold_mapped else None

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