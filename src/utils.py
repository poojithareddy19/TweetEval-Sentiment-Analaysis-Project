# src/utils.py
from pathlib import Path
import re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed folders (do not change these names)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LR_DIR = MODELS_DIR / "lr"
LSTM_DIR = MODELS_DIR / "lstm"
BERT_DIR = MODELS_DIR / "bert"

# Ensure directories exist (defensive - won't overwrite)
for p in (DATA_DIR, MODELS_DIR, LR_DIR, LSTM_DIR, BERT_DIR):
    p.mkdir(parents=True, exist_ok=True)

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

def preprocess_tweet(text: str) -> str:
    """Light, tweet-preserving cleaning (keeps emojis & hashtags)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    per_class = f1_score(y_true, y_pred, average=None, labels=[0,1,2]).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2]).tolist()
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_per_class": per_class, "confusion_matrix": cm}

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)