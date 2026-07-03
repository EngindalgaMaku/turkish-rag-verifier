import os
import sys
import time
import json
import random
import torch
import re
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print(" MODERNBERT-TR TRAINING ON PILOT_V5 DATASET")
print("==================================================")

# Paths
BASE_MODEL = "ytu-ce-cosmos/modernbert-tr-base-1k"
DATA_PATH = "data/synthetic/pilot_v5_2436_with_exp008.jsonl"
WIKI_60_PATH = "data/real_tests/squad2_500_tr.jsonl"
RAGTRUTH_136_PATH = "data/real_tests/ragtruth_136_tr.jsonl"
OUTPUT_DIR = "outputs/models/modernbert_tr_pilot_v5"

# Load data
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
print(f"Loaded {len(records)} records from pilot_v5.")

# Mapping 5-class to 3-class NLI
# supported -> 0, partially_supported/unsupported/insufficient_context -> 1 (neutral), contradicted -> 2
LABEL_MAP = {
    "supported": 0,
    "partially_supported": 1,
    "unsupported": 1,
    "insufficient_context": 1,
    "contradicted": 2
}
ID2LABEL = {0: "supported", 1: "neutral", 2: "contradicted"}

# Convert to HF Dataset format
def prepare_data_dict(recs):
    dict_data = {"premise": [], "hypothesis": [], "label": []}
    for r in recs:
        question = r.get("question", "").strip()
        context = r.get("context", "").strip()
        premise = f"Soru: {question}\nBağlam: {context}" if question else context
        
        hypothesis = r.get("claim", "").strip()
        label_str = r.get("label", "unsupported")
        
        dict_data["premise"].append(premise)
        dict_data["hypothesis"].append(hypothesis)
        dict_data["label"].append(LABEL_MAP.get(label_str, 1))
    return dict_data

train_recs, val_recs = train_test_split(records, test_size=0.1, random_state=42)
train_dataset = Dataset.from_dict(prepare_data_dict(train_recs))
val_dataset = Dataset.from_dict(prepare_data_dict(val_recs))

print(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=512
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL_MAP,
    trust_remote_code=True
).cuda()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4, # Run 4 epochs for better convergence on small dataset
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    logging_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print("\nTraining on pilot_v5 dataset...")
t_start = time.time()
trainer.train()
print(f"Training finished in {time.time() - t_start:.2f} seconds.")

# Save model
best_path = os.path.join(OUTPUT_DIR, "best")
trainer.save_model(best_path)
tokenizer.save_pretrained(best_path)
print(f"Saved best model to {best_path}")

# Load model in eval mode for evaluation
model.eval()

# ----------------------------------------------------------------------
# TEST 1: Handcrafted Wikipedia Benchmark (60 records)
# ----------------------------------------------------------------------
print("\n" + "="*50)
print(" TEST 1: Handcrafted Wikipedia (60 records)")
print("="*50)

eval_records_60 = []
with open(WIKI_60_PATH, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            eval_records_60.append(json.loads(line))

gold_bin_60 = []
pred_bin_60 = []

with torch.no_grad():
    for r in eval_records_60:
        inputs = tokenizer(r["context"], r["claim"], truncation=True, max_length=512, return_tensors="pt").to("cuda")
        logits = model(**inputs).logits
        pred_idx = torch.argmax(logits, dim=-1).item()
        
        # Binary mapping: 0 (supported) -> supported (1), 1 or 2 -> unsupported (0)
        pred_bin = 1 if pred_idx == 0 else 0
        gold_bin = 1 if r["gold_label"] == "supported" else 0
        
        gold_bin_60.append(gold_bin)
        pred_bin_60.append(pred_bin)

acc_60 = accuracy_score(gold_bin_60, pred_bin_60)
print(f"Accuracy: {acc_60*100:.2f}%")
print(classification_report(gold_bin_60, pred_bin_60, target_names=["unsupported", "supported"], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(gold_bin_60, pred_bin_60))

# ----------------------------------------------------------------------
# TEST 2: Full RAGTruth Turkish NLI Benchmark (136 records)
# ----------------------------------------------------------------------
print("\n" + "="*50)
print(" TEST 2: Full RAGTruth Turkish NLI (136 records)")
print("="*50)

eval_records_136 = []
with open(RAGTRUTH_136_PATH, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            eval_records_136.append(json.loads(line))

mapping_3_eval = {
    "supported": "supported",
    "partially_supported": "neutral",
    "unsupported": "neutral",
    "insufficient_context": "neutral",
    "contradicted": "contradicted"
}
id2label_map = {0: "supported", 1: "neutral", 2: "contradicted"}

gold_136 = []
pred_136 = []

with torch.no_grad():
    for r in eval_records_136:
        context = r.get("context", "")
        claim = r.get("claim", "")
        g_original = r.get("gold_label", "")
        
        if not g_original:
            continue
            
        g_mapped = mapping_3_eval.get(g_original, g_original)
        
        inputs = tokenizer(context, claim, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        logits = model(**inputs).logits
        pred_idx = torch.argmax(logits, dim=-1).item()
        pred_label = id2label_map[pred_idx]
        
        gold_136.append(g_mapped)
        pred_136.append(pred_label)

acc_136 = accuracy_score(gold_136, pred_136)
print(f"Accuracy: {acc_136*100:.2f}%")
print(classification_report(gold_136, pred_136, labels=["supported", "neutral", "contradicted"], zero_division=0))
print("Confusion Matrix:")
cm = confusion_matrix(gold_136, pred_136, labels=["supported", "neutral", "contradicted"])
print(f"{'':<15} {'supported':>12} {'neutral':>12} {'contradicted':>12}")
print(f"{'supported':<15} {cm[0][0]:>12} {cm[0][1]:>12} {cm[0][2]:>12}")
print(f"{'neutral':<15} {cm[1][0]:>12} {cm[1][1]:>12} {cm[1][2]:>12}")
print(f"{'contradicted':<15} {cm[2][0]:>12} {cm[2][1]:>12} {cm[2][2]:>12}")

# ----------------------------------------------------------------------
# TEST 3: RAGTruth Turkish Prose Only (35 records) - Threshold=0.0
# ----------------------------------------------------------------------
print("\n" + "="*50)
print(" TEST 3: RAGTruth Prose Only (35 records) - Threshold=0.0")
print("="*50)

# Prose subset from 136 records
prose_records = [r for r in eval_records_136 if not r["context"].strip().startswith("{")]

gold_prose = []
pred_prose = []

# Simple sentence splitter
def split_sentences(text):
    pattern = re.compile(r'[^.!?]+(?:[.!?]+)?')
    return [match.group(0).strip() for match in pattern.finditer(text) if match.group(0).strip()]

with torch.no_grad():
    for r in prose_records:
        context = r.get("context", "")
        question = r.get("question", "")
        claim = r.get("claim", "")
        gold_original = r.get("gold_label", "")
        
        gold_bin = "supported" if gold_original == "supported" else "unsupported"
        
        premise = f"Soru: {question}\nBağlam: {context}" if question else context
        sentences = split_sentences(claim)
        
        has_hallucination = False
        for sentence in sentences:
            inputs = tokenizer(premise, sentence, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            logits = model(**inputs).logits
            pred_idx = torch.argmax(logits, dim=-1).item()
            pred_nli_label = id2label_map[pred_idx]
            
            if pred_nli_label in ["neutral", "contradicted"]:
                has_hallucination = True
                break
                
        pred_bin = "unsupported" if has_hallucination else "supported"
        gold_prose.append(gold_bin)
        pred_prose.append(pred_bin)

acc_prose = accuracy_score(gold_prose, pred_prose)
print(f"Accuracy: {acc_prose*100:.2f}%")
print(classification_report(gold_prose, pred_prose, labels=["supported", "unsupported"], zero_division=0))
print("Confusion Matrix:")
cm_prose = confusion_matrix(gold_prose, pred_prose, labels=["supported", "unsupported"])
print(f"{'':<15} {'supported':>12} {'unsupported':>12}")
print(f"{'supported':<15} {cm_prose[0][0]:>12} {cm_prose[0][1]:>12}")
print(f"{'unsupported':<15} {cm_prose[1][0]:>12} {cm_prose[1][1]:>12}")
