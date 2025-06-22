# scripts/preprocess_data.py

import json
import pandas as pd
import re
import os

# Ensure output folder exists
os.makedirs("../data", exist_ok=True)

# Load the CUAD dataset
with open(r"c:\Users\shash\OneDrive\Desktop\ML_La\Data\CUAD_v1.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract clauses from paragraphs
rows = []
for contract in raw_data["data"]:
    for paragraph in contract["paragraphs"]:
        context = paragraph["context"]
        labels = []
        for qa in paragraph["qas"]:
            if not qa["is_impossible"]:
                labels.append(qa["question"])  # using question text as label
        rows.append({
            "text": context,
            "labels": labels
        })

# Convert to DataFrame
df = pd.DataFrame(rows)
print(f"✅ Loaded {len(df)} clauses.")

# Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\;\:\-\(\)]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Save to CSV
df.to_csv(r"Data/cuad_cleaned.csv", index=False)
print("✅ Cleaned data saved to data/cuad_cleaned.csv")
