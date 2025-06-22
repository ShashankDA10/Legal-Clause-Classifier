import streamlit as st
import joblib
import spacy
import re

# Load model and vectorizer
model = joblib.load("models/classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
mlb = joblib.load("models/label_binarizer.pkl")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Manual clause extraction logic
def extract_details_manually(text, labels):
    results = {}

    if "Agreement Date" in labels:
        match = re.search(r"entered into on (.+?),", text)
        results["Agreement Date"] = match.group(1) if match else "Not found"

    if "Effective Date" in labels:
        match = re.search(r"commence on (.+?) and", text)
        results["Effective Date"] = match.group(1) if match else "Not found"

    if "Expiration Date" in labels:
        match = re.search(r"terminate on (.+?) unless", text)
        results["Expiration Date"] = match.group(1) if match else "Not found"

    if "Document Name" in labels:
        match = re.search(r"^([A-Z ]+ AGREEMENT)", text)
        results["Document Name"] = match.group(1).title() if match else "Not found"

    if "Governing Law" in labels:
        match = re.search(r"laws of the State of (.+?)\.", text)
        results["Governing Law"] = match.group(1) if match else "Not found"

    if "Parties" in labels:
        match = re.findall(r"(Landlord|Tenant): ([A-Z][a-z]+ [A-Z][a-z]+)", text)
        results["Parties"] = ", ".join([m[1] for m in match]) if match else "Not found"

    if "Anti-Assignment" in labels:
        results["Anti-Assignment"] = (
            "Consent required from Landlord before assignment"
            if "without the prior written consent of the Landlord" in text
            else "Not found"
        )

    return results

# Streamlit UI
st.set_page_config(page_title="Clause Label Predictor", layout="centered")
st.title("📄 Legal Clause  Predictor")
user_input = st.text_area("Paste a clause or paragraph here:", height=250)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        X_input = vectorizer.transform([user_input])
        preds = model.predict(X_input)
        predicted_labels = mlb.inverse_transform(preds)

        if predicted_labels and predicted_labels[0]:
            labels = predicted_labels[0]
            extracted_info = extract_details_manually(user_input, labels)

            st.success("✅ Predicted Labels and Extracted Info:")
            for label in labels:
                st.markdown(
                    f"**Highlight the parts (if any) of this contract related to \"{label}\" that should be reviewed by a lawyer.** "
                    f"– Details: {extracted_info.get(label, 'Not found')}"
                )
        else:
            st.info("No labels were predicted for this input.")
