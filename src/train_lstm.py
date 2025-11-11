from pathlib import Path
import joblib
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import preprocess_tweet, LSTM_DIR

LSTM_DIR.mkdir(parents=True, exist_ok=True)

MAX_VOCAB = 30000
MAX_LEN = 80
EMBED_DIM = 100
BATCH = 128
EPOCHS = 8

def texts_and_labels(ds, split):
    texts = [preprocess_tweet(t) for t in ds[split]["text"]]
    labels = np.array(ds[split]["label"])
    return texts, labels

def build_model(vocab_size, embed_dim=EMBED_DIM, input_len=MAX_LEN):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=input_len),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_texts, train_labels = texts_and_labels(ds, "train")
    val_texts, val_labels = texts_and_labels(ds, "validation")
    test_texts, test_labels = texts_and_labels(ds, "test")

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    joblib.dump(tokenizer, LSTM_DIR / "tokenizer.joblib")

    def seqs(texts):
        s = tokenizer.texts_to_sequences(texts)
        return pad_sequences(s, maxlen=MAX_LEN, padding='post', truncating='post')

    X_train = seqs(train_texts)
    X_val = seqs(val_texts)
    X_test = seqs(test_texts)
    y_train = tf.keras.utils.to_categorical(train_labels, num_classes=3)
    y_val = tf.keras.utils.to_categorical(val_labels, num_classes=3)

    model = build_model(MAX_VOCAB)
    checkpoint = ModelCheckpoint(str(LSTM_DIR / "best.h5"), save_best_only=True, monitor='val_loss')
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH, callbacks=[checkpoint, es])
    model.save(str(LSTM_DIR / "model_final.h5"))

    # evaluate
    preds = np.argmax(model.predict(X_test), axis=1)
    from src.utils import compute_metrics
    print("Test metrics:", compute_metrics(test_labels, preds))

if __name__ == "__main__":
    main()
