import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

# Paths
INPUT_PATH = Path("data/synthetic/pilot_v5_2436_with_exp008.jsonl")
OUTPUT_PATH = Path("data/synthetic/pilot_v5_cleaned.jsonl")

if not INPUT_PATH.exists():
    print(f"Error: {INPUT_PATH} not found.")
    exit(1)

# Compile basic fragment detectors
# Turkish postpositions or conjunctions that indicate a fragment if they end the claim
FRAG_ENDINGS = re.compile(
    r'\b(ve|veya|ile|hakkında|ilgili|ise|gibi|için|tarafından|olarak|adlı|denilen|adındaki|adıyla|olarak)\s*$',
    re.IGNORECASE
)

records = []
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Total initial records: {len(records)}")

cleaned_records = []
filtered_records = []

for r in records:
    claim = r.get("claim", "").strip()
    words = claim.split()
    
    # Filter 1: Min character length
    if len(claim) < 15:
        r["filter_reason"] = f"Length too short ({len(claim)} chars)"
        filtered_records.append(r)
        continue
        
    # Filter 2: Min word count
    if len(words) < 3:
        r["filter_reason"] = f"Word count too low ({len(words)} words)"
        filtered_records.append(r)
        continue
        
    # Filter 3: Unfinished endings (e.g. "... hakkında", "... ve")
    if FRAG_ENDINGS.search(claim):
        r["filter_reason"] = "Claim ends with a fragment/conjunction/postposition"
        filtered_records.append(r)
        continue
        
    # Filter 4: Check if the claim contains only non-alphanumeric chars
    if not any(c.isalnum() for c in claim):
        r["filter_reason"] = "No alphanumeric characters"
        filtered_records.append(r)
        continue

    cleaned_records.append(r)

print(f"Filtered out {len(filtered_records)} records due to fragmentation/quality issues.")
print(f"Remaining high-quality records: {len(cleaned_records)}")

# Print a few examples of filtered records
print("\n--- Filtered Examples ---")
for r in filtered_records[:10]:
    print(f"ID: {r['id']} | Reason: {r['filter_reason']} | Claim: '{r['claim']}'")

# Save cleaned dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for r in cleaned_records:
        # Remove temp key
        r.pop("filter_reason", None)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
