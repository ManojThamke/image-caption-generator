# src/train.py

import os
import json
import time
import pickle
import random
import numpy as np
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

from config import (
    IMAGE_DIR, FEATURE_FILE, CAPTION_FILE,
    TRAIN_IMAGES, DEV_IMAGES, TEST_IMAGES,
    TOKENIZER_PATH, MODEL_PATH,
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
    vocab_size = len(tokenizer.word_index) + 1

    for caption in captions_list:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq = seq[:i]
            out_seq = seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]

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
        cap_list = captions[img_id]

        a, b, c = create_sequences(
            tokenizer, MAX_CAPTION_LENGTH, cap_list, img_feat
        )

        X1.extend(a)
        X2.extend(b)
        y.extend(c)

    return np.array(X1), np.array(X2), np.array(y)


# ==============================
# EVALUATION
# ==============================
def evaluate_bleu(model, test_ids, captions, features, tokenizer):
    actual, predicted = [], []

    index_word = {v: k for k, v in tokenizer.word_index.items()}

    for img_id in test_ids:
        yhat = generate_caption(model, features[img_id], tokenizer)
        actual_caps = [c.split()[1:-1] for c in captions[img_id]]
        predicted.append(yhat.split())
        actual.append(actual_caps)

    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu4


def generate_caption(model, image_feature, tokenizer):
    in_text = "<start>"
    for _ in range(MAX_CAPTION_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=MAX_CAPTION_LENGTH)
        yhat = model.predict([image_feature.reshape(1, -1), seq], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "<end>":
            break
    return in_text.replace("<start>", "").replace("<end>", "").strip()


# ==============================
# TRAINING LOOP
# ==============================
def train_experiment(train_pct):
    print(f"\n[INFO] Training with {train_pct}% of training data")

    features = load_features()
    tokenizer = load_tokenizer()
    captions = load_captions()

    train_ids = load_list(TRAIN_IMAGES)
    val_ids = load_list(DEV_IMAGES)
    test_ids = load_list(TEST_IMAGES)

    # Subsample training data
    random.shuffle(train_ids)
    limit = int(len(train_ids) * train_pct / 100)
    train_ids = train_ids[:limit]

    # Prepare datasets
    X1_train, X2_train, y_train = prepare_dataset(
        train_ids, captions, features, tokenizer
    )

    X1_val, X2_val, y_val = prepare_dataset(
        val_ids, captions, features, tokenizer
    )

    # Build model
    model = build_model()

    # Train
    start_time = time.time()
    history = model.fit(
        [X1_train, X2_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=([X1_val, X2_val], y_val),
        verbose=1
    )
    training_time = time.time() - start_time

    # Evaluate
    bleu1, bleu4 = evaluate_bleu(
        model, test_ids, captions, features, tokenizer
    )

    return {
        "train_percentage": train_pct,
        "epochs": EPOCHS,
        "training_time_sec": round(training_time, 2),
        "bleu_1": round(bleu1, 4),
        "bleu_4": round(bleu4, 4)
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    training_percentages = [40, 50, 60, 70, 80, 90, 100]
    results = []

    for pct in training_percentages:
        metrics = train_experiment(pct)
        results.append(metrics)

        with open("results/metrics.json", "w") as f:
            json.dump(results, f, indent=4)

    print("\n[DONE] All experiments completed")
    print("Results saved to results/metrics.json")
