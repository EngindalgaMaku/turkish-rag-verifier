# Turkish RAG Hallucination Verifier — Deploy

Hibrit hallucination detection pipeline (CPU-only).

## Servisler

| Servis | Port | Aciklama |
|---|---|---|
| bert-api | 8001 | BERTurk EXP020 fine-tuned classifier |
| nli-api | 8002 | mDeBERTa zero-shot NLI |
| hybrid-api | 8003 | Routing + Fusion (public endpoint) |

## Hizli Baslat

```bash
# 1. BERT modelini HuggingFace'e yukle (bir kez)
python upload_bert_to_hf.py --repo KULLANICI/berturk-exp020 --token hf_xxx

# 2. Docker Compose ile baslat
BERT_HF_REPO=KULLANICI/berturk-exp020 HF_TOKEN=hf_xxx \
  docker compose -f docker-compose.hybrid.yml up -d

# 3. Test et
curl -X POST http://localhost:8003/detect \
  -H "Content-Type: application/json" \
  -d '{"context": "Python 1991 yilinda gelistirildi.", "answer": "Python 1991 yilinda cikti."}'
```

## Coolify Deploy

Bkz: [docs/COOLIFY_HYBRID_DEPLOY.md](../docs/COOLIFY_HYBRID_DEPLOY.md)

## Sistem Gereksinimleri

- RAM: 8GB (minimum)
- CPU: Herhangi modern islemci
- GPU: Gerekmiyor
- Disk: 5GB (model dosyalari)
