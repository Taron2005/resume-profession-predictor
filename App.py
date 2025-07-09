import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Load pre-trained models and vectorizers
logreg_tfidf = joblib.load("logistic_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

logreg_bert = joblib.load("logreg_bert.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load BERT tokenizer and model (for embedding)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


@st.cache(allow_output_mutation=True)
def encode_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return cls_embedding.cpu().numpy().reshape(1, -1)


st.title("Resume Job Role Classifier")

text_input = st.text_area("Paste resume text here:")

model_choice = st.selectbox("Select model:", ("TF-IDF + Logistic Regression", "BERT + Logistic Regression"))

if st.button("Predict"):
    if not text_input.strip():
        st.warning("Please enter resume text.")
    else:
        if model_choice == "TF-IDF + Logistic Regression":
            features = tfidf_vectorizer.transform([text_input])
            pred = logreg_tfidf.predict(features)[0]
        else:
            bert_vec = encode_bert(text_input)
            pred = logreg_bert.predict(bert_vec)[0]

        label = label_encoder.inverse_transform([pred])[0]
        st.success(f"Predicted Job Role: **{label}**")
