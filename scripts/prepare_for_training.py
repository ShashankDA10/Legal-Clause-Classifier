# scripts/prepare_for_training.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Load cleaned data
df = pd.read_csv(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\cuad_cleaned.csv")

# Drop rows with empty labels (optional)
df = df[df['labels'].notna() & (df['labels'] != '[]')]

# Convert stringified list to actual list
import ast
df["labels"] = df["labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Binarize labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

# Save the binarizer for later
with open(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\label_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

# Save X and y as separate files
df["clean_text"].to_csv(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\X_text.csv", index=False)
pd.DataFrame(y, columns=mlb.classes_).to_csv(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\y_labels.csv", index=False)

print("✅ Prepared data for training. Saved X_text.csv, y_labels.csv, and label_binarizer.pkl.")
