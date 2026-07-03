import json
import re
import sys
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print(" RAGTRUTH-TR SENTENCE-LEVEL NLI DATASET CREATOR")
print("==================================================")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "ragtruth_tr_nli_train.jsonl"
val_path = OUTPUT_DIR / "ragtruth_tr_nli_val.jsonl"

# 1. Custom sentence splitter with spans
def split_sentences_with_spans(text):
    spans = []
    # Match sentences ending with punctuation or end of string
    pattern = re.compile(r'[^.!?]+(?:[.!?]+)?')
    for match in pattern.finditer(text):
        start = match.start()
        end = match.end()
        sentence = text[start:end]
        stripped = sentence.strip()
        if not stripped:
            continue
        new_start = start + sentence.find(stripped)
        new_end = new_start + len(stripped)
        spans.append((new_start, new_end, stripped))
    return spans

# 2. Load dataset
print("Loading newmindai/RAGTruth-TR from Hugging Face...")
ds = load_dataset("newmindai/RAGTruth-TR", split="train")
print(f"Loaded {len(ds)} raw RAGTruth-TR records.")

# Convert to list of dicts
raw_records = list(ds)

# 3. Train/Val split at document/example level to prevent data leakage!
print("Splitting records into train/val (90% / 10%)...")
train_recs, val_recs = train_test_split(raw_records, test_size=0.1, random_state=42)
print(f"Train records: {len(train_recs)} | Val records: {len(val_recs)}")

# Helper to process records and write to file
def extract_sentences_and_write(records, output_file_path):
    print(f"Extracting sentences and writing to {output_file_path.name}...")
    written_count = 0
    with open(output_file_path, "w", encoding="utf-8") as out_f:
        for r in records:
            question = r.get("question", "").strip()
            context = r.get("context", "").strip()
            answer = r.get("answer", "")
            labels = r.get("labels", [])
            
            # Split answer into sentences with character spans
            sentences = split_sentences_with_spans(answer)
            
            for start, end, sentence_text in sentences:
                # Check overlap with hallucination spans
                is_hallucinated = False
                nli_label = "supported" # Default: supported (entailment)
                
                for span in labels:
                    s_start = span["start"]
                    s_end = span["end"]
                    s_label = span["label"]
                    
                    # Compute overlap
                    overlap = max(start, s_start) < min(end, s_end)
                    if overlap:
                        is_hallucinated = True
                        # If the span label is Conflict -> contradicted, else baseless -> neutral
                        if s_label == "Evident Conflict":
                            nli_label = "contradicted"
                        else:
                            nli_label = "neutral"
                        break # Once flagged as hallucinated, we don't need to check other spans
                
                # Write sentence-level record
                item = {
                    "question": question,
                    "context": context,
                    "claim": sentence_text,
                    "label": nli_label
                }
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written_count += 1
                
    print(f"Successfully wrote {written_count} sentence-level NLI records to {output_file_path.name}.")
    return written_count

# Process both splits
train_sentences = extract_sentences_and_write(train_recs, train_path)
val_sentences = extract_sentences_and_write(val_recs, val_path)

print("\n" + "="*50)
print(" DATASET CREATION COMPLETED!")
print("="*50)
print(f"Total Train Sentences: {train_sentences}")
print(f"Total Val Sentences: {val_sentences}")
