# src/train.py

import os
import json
import time
import pickle
import random
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

from config import (
    FEATURE_FILE, CAPTION_FILE,
    TRAIN_IMAGES, DEV_IMAGES, TEST_IMAGES,
    TOKENIZER_PATH,
    MAX_CAPTION_LENGTH, BATCH_SIZE, EPOCHS
)

from model import build_model


# ==============================
# LOAD UTILITIES
# ==============================
def load_list(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_features():
    with open(FEATURE_FILE, "rb") as f:
        return pickle.load(f)


def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)


def load_captions():
    from preprocess import load_captions, clean_captions
    captions = load_captions(CAPTION_FILE)
    clean_captions(captions)
    return captions


# ==============================
# DATA PREPARATION
# ==============================
def create_sequences(tokenizer, max_len, captions_list, image_feature):
    X1, X2, y = [], [], []

    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq = pad_sequences([seq[:i]], maxlen=max_len)[0]
            out_seq = seq[i]

            X1.append(image_feature)
            X2.append(in_seq)
            y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)


def prepare_dataset(image_ids, captions, features, tokenizer):
    X1, X2, y = [], [], []

    for img_id in image_ids:
        if img_id not in features:
            continue

        img_feat = features[img_id]
        caps = captions[img_id]

        a, b, c = create_sequences(
            tokenizer, MAX_CAPTION_LENGTH, caps, img_feat
        )

        X1.extend(a)
        X2.extend(b)
        y.extend(c)

    return np.array(X1), np.array(X2), np.array(y)


# ==============================
# EVALUATION
# ==============================
def generate_caption(model, image_feature, tokenizer):
    in_text = "<start>"

    for _ in range(MAX_CAPTION_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_CAPTION_LENGTH)

        yhat = model.predict([image_feature.reshape(1, -1), seq], verbose=0)
        word_id = np.argmax(yhat)
        word = tokenizer.index_word.get(word_id)

        if word is None:
            break

        in_text += " " + word
        if word == "<end>":
            break

    return in_text.replace("<start>", "").replace("<end>", "").strip()


def evaluate_bleu(model, test_ids, captions, features, tokenizer):
    actual, predicted = [], []

    for img_id in test_ids:
        yhat = generate_caption(model, features[img_id], tokenizer)
        refs = [c.split()[1:-1] for c in captions[img_id]]

        actual.append(refs)
        predicted.append(yhat.split())

    bleu1 = corpus_bleu(actual, predicted, weights=(1, 0, 0, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu4


# ==============================
# TRAINING FUNCTION
# ==============================
def train_experiment(train_pct):
    print(f"\n[INFO] Training with {train_pct}% of training data")

    features = load_features()
    tokenizer = load_tokenizer()
    captions = load_captions()

    train_ids = load_list(TRAIN_IMAGES)
    val_ids = load_list(DEV_IMAGES)
    test_ids = load_list(TEST_IMAGES)

    random.shuffle(train_ids)
    limit = int(len(train_ids) * train_pct / 100)
    train_ids = train_ids[:limit]

    X1_train, X2_train, y_train = prepare_dataset(
        train_ids, captions, features, tokenizer
    )

    X1_val, X2_val, y_val = prepare_dataset(
        val_ids, captions, features, tokenizer
    )

    model = build_model()

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

    bleu1, bleu4 = evaluate_bleu(
        model, test_ids, captions, features, tokenizer
    )

    # SAVE FINAL MODEL (ONLY ON 100%)
    if train_pct == 100:
        os.makedirs("models", exist_ok=True)
        model.save("models/final_caption_model.h5")
        print("[INFO] Final model saved ✔")

    return {
        "train_percentage": train_pct,
        "epochs": EPOCHS,
        "training_time_sec": round(train_time, 2),
        "bleu_1": round(bleu1, 4),
        "bleu_4": round(bleu4, 4)
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    training_percentages = [100]
    results = []

    for pct in training_percentages:
        metrics = train_experiment(pct)
        results.append(metrics)

        with open("results/metrics.json", "w") as f:
            json.dump(results, f, indent=4)

    print("\n[DONE] ALL TRAINING COMPLETED")
