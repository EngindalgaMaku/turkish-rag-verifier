# -*- coding: utf-8 -*-
"""
BERT modelini (EXP020) HuggingFace Hub'a yukle.

Kullanim:
    pip install huggingface_hub
    python upload_bert_to_hf.py --repo YOUR_HF_USERNAME/berturk-rag-verifier-exp020

Argümanlar:
    --repo   : HuggingFace repo adi (ornek: kullanici/berturk-exp020)
    --model  : Yerel model klasoru (varsayilan: outputs/bert_classifier/berturk_exp020_pilot_v7_weighted_ft)
    --private: Repo'yu private yap (varsayilan: True)
    --token  : HF token (yoksa HF_TOKEN env degiskeninden okunur)
"""

import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
        help="HuggingFace repo adi, ornek: kullanici/berturk-exp020")
    parser.add_argument("--model", default=
        "outputs/bert_classifier/berturk_exp020_pilot_v7_weighted_ft",
        help="Yerel model klasoru")
    parser.add_argument("--private", action="store_true", default=True,
        help="Repo'yu private yap")
    parser.add_argument("--token", default=None,
        help="HuggingFace token (yoksa HF_TOKEN env'den okunur)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("HATA: HF_TOKEN env degiskeni veya --token argumani gerekli")
        print("Token al: https://huggingface.co/settings/tokens")
        return 1

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"HATA: Model klasoru bulunamadi: {model_path}")
        return 1

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("HATA: huggingface_hub yuklu degil")
        print("Yukle: pip install huggingface_hub")
        return 1

    api = HfApi(token=token)

    # Repo olustur (yoksa)
    print(f"Repo olusturuluyor: {args.repo} (private={args.private})")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            token=token,
            exist_ok=True
        )
        print("Repo hazir.")
    except Exception as e:
        print(f"Repo olusturma hatasi: {e}")
        return 1

    # Model dosyalarini yukle
    print(f"\nModel dosyalari yukleniyor: {model_path}")
    print("Bu islem birkaç dakika surebilir...\n")

    # Yuklenecek dosyalar
    extensions = {".json", ".bin", ".safetensors", ".txt", ".model", ".vocab"}
    files = [f for f in model_path.rglob("*") if f.is_file() and f.suffix in extensions]

    if not files:
        print(f"UYARI: {model_path} icinde model dosyasi bulunamadi")
        print("Beklenen dosyalar: config.json, pytorch_model.bin veya model.safetensors")
        return 1

    for f in files:
        rel = f.relative_to(model_path)
        print(f"  Yukleniyor: {rel} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=str(rel),
                repo_id=args.repo,
                repo_type="model",
                token=token
            )
        except Exception as e:
            print(f"  HATA: {rel} yuklenemedi: {e}")
            return 1

    # README ekle
    readme = f"""---
language: tr
license: mit
tags:
  - bert
  - turkish
  - hallucination-detection
  - rag
  - text-classification
---

# BERTurk RAG Hallucination Verifier — EXP020

Turkish RAG hallucination detection model.
Fine-tuned from `dbmdz/bert-base-turkish-cased` on pilot_v7 dataset (2836 examples).

## Labels
- `supported`
- `partially_supported`
- `unsupported`
- `contradicted`
- `insufficient_context`

## Validation Accuracy
98.2% on pilot_v7 validation split (synthetic data).

## Usage
See [turkish-rag-verifier](https://github.com/YOUR_USERNAME/turkish-rag-verifier) for API code.
"""
    try:
        api.upload_file(
            path_or_fileobj=readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            token=token
        )
    except Exception as e:
        print(f"README yuklenemedi (onemli degil): {e}")

    print(f"\nTamamlandi!")
    print(f"Model URL: https://huggingface.co/{args.repo}")
    print(f"\nDockerfile.hybrid icin BERT_HF_REPO={args.repo} olarak ayarla")
    return 0

if __name__ == "__main__":
    exit(main())