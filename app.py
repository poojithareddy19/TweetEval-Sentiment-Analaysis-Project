import streamlit as st
from src.utils import preprocess_tweet, LABELS, LR_DIR, LSTM_DIR, BERT_DIR
import joblib
import numpy as np
import os

st.set_page_config(page_title="TweetEval Sentiment", layout="centered")
st.title("TweetEval Sentiment — LR / LSTM / BERT")

MODEL_CHOICES = ["LogisticRegression", "LSTM", "BERT"]
model_choice = st.sidebar.selectbox("Select model", MODEL_CHOICES)

def load_lr():
    path = LR_DIR / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Expected model at {path}")
    return joblib.load(path)

def load_lstm():
    from tensorflow.keras.models import load_model
    tok_path = LSTM_DIR / "tokenizer.joblib"
    model_path = LSTM_DIR / "model_final.h5"
    if not tok_path.exists() or not model_path.exists():
        raise FileNotFoundError("LSTM tokenizer or model not found in models/lstm/")
    tokenizer = joblib.load(tok_path)
    model = load_model(str(model_path))
    return tokenizer, model

def load_bert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if not (BERT_DIR.exists()):
        raise FileNotFoundError("BERT folder not found: models/bert/")
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
    mdl.eval()
    return tok, mdl

st.write(f"Using model: **{model_choice}**")

text = st.text_area("Enter tweet text", height=120, value="I love this phone! 😍")

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a tweet")
        st.stop()

    t = preprocess_tweet(text)

    if model_choice == "LogisticRegression":
        pipe = load_lr()
        probs = pipe.predict_proba([t])[0]
        pred = int(np.argmax(probs))
    elif model_choice == "LSTM":
        tokenizer, mdl = load_lstm()
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = tokenizer.texts_to_sequences([t])
        seq = pad_sequences(seq, maxlen=80, padding='post', truncating='post')
        probs = mdl.predict(seq)[0]
        pred = int(np.argmax(probs))
    else: 
        tok, mdl = load_bert()
        import torch
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
        if torch.cuda.is_available():
            mdl.to("cuda")
            enc = {k:v.to("cuda") for k,v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        pred = int(np.argmax(probs))

    st.subheader(f"Predicted: {LABELS[pred].upper()} (confidence: {float(probs[pred]):.3f})")
    st.table({"class": [LABELS[i] for i in range(3)], "probability": [float(p) for p in probs]})


