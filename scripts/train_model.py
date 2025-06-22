import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Load data
X = pd.read_csv(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\X_text.csv")["clean_text"].tolist()
y = pd.read_csv(r"C:\Users\shash\OneDrive\Desktop\ML_La\Data\y_labels.csv")

# Convert each row of label DataFrame to list of labels
y_list = y.apply(lambda row: [col for col, val in row.items() if val == 1], axis=1).tolist()

# Binarize labels with string class names
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(y_list)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear'))
clf.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = clf.predict(X_test_tfidf)
print("✅ Model Performance:\n")
print(classification_report(y_test, y_pred, zero_division=0, target_names=mlb.classes_))

# Save model, vectorizer, and label binarizer
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/classifier.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb, "models/label_binarizer.pkl")

print("✅ All files saved to 'models/' successfully.")
