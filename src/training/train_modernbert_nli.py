import os
import sys
import time
import json
import random
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print(" MODERNBERT-TR CLAIM-LEVEL NLI TRAINING")
print("==================================================")

# Config
BASE_MODEL = "ytu-ce-cosmos/modernbert-tr-base-1k"
TRAIN_DATA_PATH = "data/processed/ragtruth_tr_nli_train.jsonl"
VAL_DATA_PATH = "data/processed/ragtruth_tr_nli_val.jsonl"
OUTPUT_DIR = "outputs/models/modernbert_tr_nli"

# Downsample parameters for fast, stable laptop training
MAX_LENGTH = 512 # Keep it balanced for prompt + context + claim

# Label map
LABEL_MAP = {
    "supported": 0,
    "neutral": 1,
    "contradicted": 2
}
ID2LABEL = {0: "supported", 1: "neutral", 2: "contradicted"}

# 1. Load and sample data
def load_sampled_jsonl(path, sample_size):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    random.seed(42)
    if len(records) > sample_size:
        records = random.sample(records, sample_size)
    print(f"Loaded and sampled {len(records)} examples from {Path(path).name}.")
    return records

def load_balanced_jsonl(path, max_supported=12000):
    records_by_label = {"supported": [], "neutral": [], "contradicted": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                label = item.get("label", "neutral")
                if label in records_by_label:
                    records_by_label[label].append(item)
    
    random.seed(42)
    # Sample supported class to balance
    supported_sampled = records_by_label["supported"]
    if len(supported_sampled) > max_supported:
        supported_sampled = random.sample(supported_sampled, max_supported)
        
    # Combine all
    combined = supported_sampled + records_by_label["neutral"] + records_by_label["contradicted"]
    random.shuffle(combined)
    
    print(f"Loaded balanced dataset from {Path(path).name}:")
    print(f"  - supported: {len(supported_sampled)}")
    print(f"  - neutral: {len(records_by_label['neutral'])}")
    print(f"  - contradicted: {len(records_by_label['contradicted'])}")
    print(f"  - total: {len(combined)}")
    return combined

train_raw = load_balanced_jsonl(TRAIN_DATA_PATH, max_supported=12000)
val_raw = load_sampled_jsonl(VAL_DATA_PATH, 2000)

# Convert to HF Dataset
def convert_to_dataset(raw_data):
    data_dict = {
        "premise": [],
        "hypothesis": [],
        "label": []
    }
    for item in raw_data:
        # Premise combines question and context
        question = item.get("question", "").strip()
        context = item.get("context", "").strip()
        premise = f"Soru: {question}\nBağlam: {context}" if question else context
        
        hypothesis = item.get("claim", "").strip()
        label_str = item.get("label", "neutral")
        
        data_dict["premise"].append(premise)
        data_dict["hypothesis"].append(hypothesis)
        data_dict["label"].append(LABEL_MAP[label_str])
        
    return Dataset.from_dict(data_dict)

train_dataset = convert_to_dataset(train_raw)
val_dataset = convert_to_dataset(val_raw)

print(f"Train Dataset size: {len(train_dataset)}")
print(f"Val Dataset size: {len(val_dataset)}")

# 2. Tokenization
print(f"Loading tokenizer: {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    # Standard NLI input formatting: premise, hypothesis pairs
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# 3. Model Initialization
print(f"Loading base model: {BASE_MODEL}...")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL_MAP,
    trust_remote_code=True
)

# 4. Metrics calculation helper
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True, # Speed up training on RTX 4060
    logging_steps=100,
    report_to="none", # Disable wandb/tensorboard logging to avoid external dependencies
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# 6. Run Training
print("\nStarting ModernBERT NLI Fine-Tuning...")
t0 = time.time()
trainer.train()
print(f"Training completed in {time.time() - t0:.2f} seconds.")

# 7. Evaluate and print final classification report
print("\nRunning final validation evaluation...")
predictions = trainer.predict(tokenized_val)
preds = predictions.predictions.argmax(axis=1)
labels = predictions.label_ids

print("\n" + "="*50)
print(" FINAL CLASSIFICATION REPORT (VAL SET)")
print("="*50)
print(classification_report(labels, preds, target_names=["supported", "neutral", "contradicted"]))

# Save best model
print(f"Saving best model to: {OUTPUT_DIR}/best...")
trainer.save_model(os.path.join(OUTPUT_DIR, "best"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))
print("Done!")
