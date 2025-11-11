from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.utils import preprocess_tweet, compute_metrics, LR_DIR

LR_DIR.mkdir(parents=True, exist_ok=True)

def ds_to_df(ds, split):
    texts = [preprocess_tweet(t) for t in ds[split]["text"]]
    labels = ds[split]["label"]
    return pd.DataFrame({"text": texts, "label": labels})

def main():
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train = ds_to_df(ds, "train")
    val = ds_to_df(ds, "validation")
    test = ds_to_df(ds, "test")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
        ("clf", LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga', multi_class='multinomial'))
])
    
    params = {
        "tfidf__max_features": [10000],
        "clf__C": [1.0]
    }

    gs = GridSearchCV(pipe, params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(train["text"], train["label"])

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    val_preds = best.predict(val["text"])
    test_preds = best.predict(test["text"])
    print("Validation metrics:", compute_metrics(val["label"], val_preds))
    print("Test matrics:", compute_metrics(test["label"], test_preds))

    joblib.dump(best, LR_DIR / "pipeline.joblib")
    print("Saved pipeline to", LR_DIR / "pipeline.joblib")

if __name__ == "__main__":
    main()
