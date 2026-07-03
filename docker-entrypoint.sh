#!/bin/sh
# docker-entrypoint.sh — Servis baslatma ve model indirme

set -e

# BERT modeli runtime'da indirilecekse
if [ "$SERVICE" = "bert" ] && [ -n "$BERT_HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
    # Model zaten indirilmis mi kontrol et
    if [ ! -f "/app/models/bert/config.json" ]; then
        echo "BERT modeli indiriliyor: $BERT_HF_REPO"
        huggingface-cli download \
            "$BERT_HF_REPO" \
            --local-dir /app/models/bert \
            --token "$HF_TOKEN"
        echo "BERT modeli indirildi."
    else
        echo "BERT modeli zaten mevcut, atlanıyor."
    fi
fi

# Servisi baslat
if [ "$SERVICE" = "bert" ]; then
    echo "BERT API baslatiliyor (port 8001)..."
    exec uvicorn bert_api:app --host 0.0.0.0 --port 8001 --workers 1
elif [ "$SERVICE" = "nli" ]; then
    echo "NLI API baslatiliyor (port 8002)..."
    exec uvicorn nli_api:app --host 0.0.0.0 --port 8002 --workers 1
else
    echo "Hibrit API baslatiliyor (port 8003)..."
    exec uvicorn hybrid_api:app --host 0.0.0.0 --port 8003 --workers 1
fi