# src/train.py

import os
import time
import json
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    TRAIN_FEATURE_FILE,
    VAL_FEATURE_FILE,
    TOKENIZER_PATH,
    MAX_CAPTION_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    MODEL_PATH,
    EMBEDDING_DIM,
    LSTM_UNITS
)

from model import build_model


# ==============================
# LOAD UTILITIES
# ==============================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_tokenizer():
    return load_pickle(TOKENIZER_PATH)


# ==============================
# CREATE TRAINING SEQUENCES
# ==============================
def create_sequences(tokenizer, max_len, captions, image_feature):
    X1, X2, y = [], [], []

    for caption in captions:
        seq = tokenizer.texts_to_sequences([caption])[0]

        for i in range(1, len(seq)):
            in_seq = pad_sequences(
                [seq[:i]], maxlen=max_len
            )[0]
            out_seq = seq[i]

            X1.append(image_feature)
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


def prepare_dataset(features, captions_dict, tokenizer):
    X1, X2, y = [], [], []

    for image_id, captions in captions_dict.items():
        if image_id not in features:
            continue

        img_feature = features[image_id]

        a, b, c = create_sequences(
            tokenizer,
            MAX_CAPTION_LENGTH,
            captions,
            img_feature
        )

        X1.extend(a)
        X2.extend(b)
        y.extend(c)

    return np.array(X1), np.array(X2), np.array(y)


# ==============================
# MAIN TRAINING
# ==============================
if __name__ == "__main__":
    print("[INFO] Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("[INFO] Loading COCO train captions...")
    train_captions = load_pickle("tokenizer/cleaned_train_captions.pkl")

    print("[INFO] Loading COCO validation captions...")
    val_captions = load_pickle("tokenizer/cleaned_val_captions.pkl")

    print("[INFO] Loading train image features...")
    train_features = load_pickle(TRAIN_FEATURE_FILE)

    print("[INFO] Loading validation image features...")
    val_features = load_pickle(VAL_FEATURE_FILE)

    print("[INFO] Preparing training data...")
    X1_train, X2_train, y_train = prepare_dataset(
        train_features,
        train_captions,
        tokenizer
    )

    print("[INFO] Preparing validation data...")
    X1_val, X2_val, y_val = prepare_dataset(
        val_features,
        val_captions,
        tokenizer
    )

    print(f"[INFO] Training samples: {len(X1_train)}")
    print(f"[INFO] Validation samples: {len(X1_val)}")

    print("[INFO] Building model...")
    model = build_model()

    print("[INFO] Starting training...")
    start = time.time()

    model.fit(
        [X1_train, X2_train],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X1_val, X2_val], y_val),
        verbose=1
    )

    train_time = time.time() - start
    train_time_min = round(train_time / 60, 2)

    print(f"[INFO] Training completed in {train_time_min} minutes")

    # ==============================
    # SAVE MODEL
    # ==============================
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved at: {MODEL_PATH}")

    # ==============================
    # SAVE MODEL CONFIG (FOR PREDICT)
    # ==============================
    model_config = {
        "dataset": "COCO 2017",
        "cnn": "ResNet50",
        "max_caption_length": MAX_CAPTION_LENGTH,
        "vocab_size": tokenizer.num_words,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS
    }

    with open(os.path.join(os.path.dirname(MODEL_PATH), "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # ==============================
    # SAVE TRAINING INFO (OPTIONAL)
    # ==============================
    training_info = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_time_minutes": train_time_min,
        "train_samples": int(len(X1_train)),
        "val_samples": int(len(X1_val))
    }

    with open(os.path.join(os.path.dirname(MODEL_PATH), "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=4)

    print("[INFO] Model metadata saved (JSON)")
    print("[DONE] Training pipeline completed successfully ✔")
