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
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

import os
# ---------------------------------------------------------------------------
# Embedding modeli — modül import edildiğinde bir kez yüklenir (startup'ta)
# ---------------------------------------------------------------------------
_emb_model = None
def get_emb_model():
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
    return _emb_model

# Modeli hemen yükle — ilk istek geldiğinde gecikme yaşanmasın
try:
    get_emb_model()
except Exception as _emb_load_err:
    import logging as _lg
    _lg.getLogger(__name__).warning("Embedding modeli yuklenemedi: %s", _emb_load_err)
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


async def fuse(client: httpx.AsyncClient, context: str, answer: str, bert: dict, nli: dict) -> dict:
    """
    NLI-merkezli, BERT-tetiklemeli tamamlayıcı hibrit doğrulama hattı (Embedding Destekli).
    """
    import re
    import numpy as np
    bert_label = bert.get("predicted_label", "supported")
    bert_conf  = bert.get("confidence", 0.5)
    nli_con    = nli.get("contradiction_score", 0.0)
    nli_sup    = nli.get("support_score", 0.0)
    nli_dec    = nli.get("decision", "warn")
    
    # ---------------------------------------------------------------------------
    # Embedding Hesaplama
    # ---------------------------------------------------------------------------
    claims_details = []
    emb_max = 0.0
    emb_min = 0.0
    
    ctx_sents = [s.strip() for s in sent_tokenize(context, language='turkish') if s.strip()]
    ans_sents = [s.strip() for s in sent_tokenize(answer, language='turkish') if s.strip()]
    
    if ctx_sents and ans_sents:
        try:
            import functools
            loop = asyncio.get_event_loop()
            emb_model = get_emb_model()
            # CPU-yoğun encode işlemini thread pool'a gönder — event loop'u bloke etme
            ctx_embs, ans_embs = await asyncio.gather(
                loop.run_in_executor(None, functools.partial(emb_model.encode, ctx_sents)),
                loop.run_in_executor(None, functools.partial(emb_model.encode, ans_sents)),
            )
            
            ctx_norms = np.linalg.norm(ctx_embs, axis=1, keepdims=True)
            ans_norms = np.linalg.norm(ans_embs, axis=1, keepdims=True)
            ctx_norms[ctx_norms == 0] = 1.0
            ans_norms[ans_norms == 0] = 1.0
            
            ctx_normed = ctx_embs / ctx_norms
            ans_normed = ans_embs / ans_norms
            
            cos_scores = np.dot(ans_normed, ctx_normed.T)
            
            for i, claim_text in enumerate(ans_sents):
                best_idx = int(np.argmax(cos_scores[i]))
                max_sim = float(cos_scores[i, best_idx])
                best_sent = ctx_sents[best_idx]
                claims_details.append({
                    "claim": claim_text,
                    "best_evidence_sentence": best_sent,
                    "max_similarity": round(max_sim, 4)
                })
            
            emb_max = float(np.max(cos_scores))
            emb_min = float(np.min(np.max(cos_scores, axis=1)))
        except Exception as e:
            pass
            
    embedding_data = {
        "emb_max": round(emb_max, 4),
        "emb_min": round(emb_min, 4),
        "claims": claims_details
    }
    
    # 1. Güçlü NLI contradiction veto
    if nli_con >= 0.80:
        return {
            "decision": "revise",
            "source": "nli_veto",
            "confidence": nli_con,
            "reason": f"Güçlü NLI çelişkisi (con={nli_con:.2f})",
            "embedding": embedding_data
        }
        
    # 2. BERT supported tek başına kabul ettiremez
    if nli_sup < 0.80 and bert_label == "supported":
        return {
            "decision": "warn",
            "source": "bert_supported_veto",
            "confidence": bert_conf,
            "reason": f"BERT supported verdi ama NLI desteği yetersiz (sup={nli_sup:.2f})",
            "embedding": embedding_data
        }
        
    # 2.5. Gating: Alakasız bağlam koruması (Low Relevance)
    if emb_max < 0.35 and nli_sup < 0.40:
        return {
            "decision": "insufficient_context",
            "source": "embedding_gating",
            "confidence": 1.0 - emb_max,
            "reason": f"Alakasız bağlam: Benzerlik skoru çok düşük (emb_max={emb_max:.2f})",
            "embedding": embedding_data
        }
        
    # 3. BERT partially_supported ise claim-level NLI kontrolü tetikler
    if bert_label == "partially_supported":
        if len(ans_sents) > 1:
            tasks = []
            for claim in ans_sents:
                payload = {
                    "context": context,
                    "answer": claim,
                    "question": claim
                }
                tasks.append(call_nli(client, payload))
            claim_results = await asyncio.gather(*tasks)
            
            any_contra = False
            any_unsupp = False
            all_supp = True
            
            for cr, claim_det in zip(claim_results, claims_details):
                if not cr or "error" in cr or claim_det["max_similarity"] < 0.35:
                    any_unsupp = True
                    all_supp = False
                    continue
                c_dec = cr.get("decision", "warn")
                c_con = cr.get("contradiction_score", 0.0)
                c_sup = cr.get("support_score", 0.0)
                
                if c_dec == "revise" or c_con >= 0.30:
                    any_contra = True
                    all_supp = False
                elif c_dec in ("warn", "insufficient_context") or c_sup < 0.80:
                    any_unsupp = True
                    all_supp = False
                    
            if any_contra:
                return {
                    "decision": "revise",
                    "source": "claim_level_nli",
                    "confidence": bert_conf,
                    "reason": "BERT partial dedi; iddia seviyesinde çelişki bulundu.",
                    "embedding": embedding_data
                }
            elif any_unsupp:
                return {
                    "decision": "warn",
                    "source": "claim_level_nli",
                    "confidence": bert_conf,
                    "reason": "BERT partial dedi; iddia seviyesinde yetersiz destek bulundu.",
                    "embedding": embedding_data
                }
            elif all_supp:
                return {
                    "decision": "accept",
                    "source": "claim_level_nli",
                    "confidence": bert_conf,
                    "reason": "BERT partial dedi; tüm iddialar başarıyla desteklendi.",
                    "embedding": embedding_data
                }
        else:
            return {
                "decision": "warn",
                "source": "bert_partial_fallback",
                "confidence": bert_conf,
                "reason": "BERT kısmi destek dedi (tek cümle).",
                "embedding": embedding_data
            }
            
    # 4. BERT contradicted ama NLI desteği yüksekse (Çelişki Uyuşmazlığı)
    if bert_label == "contradicted" and nli_sup >= 0.80:
        if len(ans_sents) > 1:
            tasks = []
            for claim in ans_sents:
                payload = {
                    "context": context,
                    "answer": claim,
                    "question": claim
                }
                tasks.append(call_nli(client, payload))
            claim_results = await asyncio.gather(*tasks)
            
            any_contra = False
            for cr, claim_det in zip(claim_results, claims_details):
                if cr and "error" not in cr:
                    c_dec = cr.get("decision", "warn")
                    c_con = cr.get("contradiction_score", 0.0)
                    if c_dec == "revise" or c_con >= 0.30 or claim_det["max_similarity"] < 0.35:
                        any_contra = True
            if any_contra:
                return {
                    "decision": "revise",
                    "source": "conflict_claim_level_nli",
                    "confidence": bert_conf,
                    "reason": "BERT çelişki bildirdi; iddia seviyesinde çelişen alt cümle doğrulandı.",
                    "embedding": embedding_data
                }
                
        return {
            "decision": "warn",
            "source": "conflict_safe_warn",
            "confidence": max(bert_conf, nli_sup),
            "reason": f"Uyuşmazlık: BERT çelişki ({bert_conf:.2f}) derken NLI destek ({nli_sup:.2f}) bildirdi. Güvenli karar: WARN",
            "embedding": embedding_data
        }
            
    # 5. NLI güçlü destekliyse ve embedding konu uyumu varsa kabul
    if nli_sup >= 0.85 and nli_con < 0.10 and emb_max >= 0.45:
        return {
            "decision": "accept",
            "source": "nli_strong_support",
            "confidence": nli_sup,
            "reason": f"Güçlü NLI ve benzerlik desteği (sup={nli_sup:.2f}, con={nli_con:.2f}, emb={emb_max:.2f})",
            "embedding": embedding_data
        }
        
    # 6. Destek zayıfsa güvenli karar
    return {
        "decision": "warn",
        "source": "fallback_warn",
        "confidence": max(nli_sup, nli_con),
        "reason": f"Düşük/belirsiz destek veya konu uyuşmazlığı: NLI sup={nli_sup:.2f}, con={nli_con:.2f}, emb={emb_max:.2f}",
        "embedding": embedding_data
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
    embedding   = None

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
        # Ensemble — tamamlayıcı hibrit doğrulama
        fusion     = await fuse(client, context, answer, bert_result, nli_result)
        decision   = fusion["decision"]
        confidence = fusion["confidence"]
        source     = fusion["source"]
        reason     = fusion["reason"]
        embedding  = fusion.get("embedding")
        
        # Hallucination score hesabı ve karar uyumu
        h_score    = BERT_WEIGHT * bert_hallucination_score(bert_result) + \
                     NLI_WEIGHT  * nli_hallucination_score(nli_result)
        if decision == "accept":
            h_score = min(h_score, 0.15)
        elif decision == "revise":
            h_score = max(h_score, 0.85)

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
        # gold_label → kabul edilebilir hibrit karar seti (Esnek/Soft Değerlendirme)
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
        }.get(gold_label.lower().strip())
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
        "embedding": embedding,
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