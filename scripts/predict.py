# scripts/predict.py

import joblib
import pandas as pd

# Load model and vectorizer
vectorizer = joblib.load(r"C:\Users\shash\OneDrive\Desktop\ML_La\Models\tfidf_vectorizer.pkl")
model = joblib.load(r"C:\Users\shash\OneDrive\Desktop\ML_La\Models\logistic_model.pkl")

# Input text (you can replace this with your own)
new_clauses = [
    "The party must maintain confidentiality of all proprietary information.",
    "This agreement may be terminated upon 30 days' notice by either party.",
    "This Agreement shall be governed by the laws of the State of California.",
    "Neither party shall be liable for delays due to force majeure events."
]

# Preprocess (minimal)
def clean(text):
    return text.lower().replace("\n", " ").strip()

cleaned_clauses = [clean(clause) for clause in new_clauses]

# Vectorize
X_vec = vectorizer.transform(cleaned_clauses)

# Predict
preds = model.predict(X_vec)

# Output
pred_labels = preds.tolist()
df_results = pd.DataFrame({
    "text": new_clauses,
    "predicted_labels": pred_labels
})

print(df_results)
