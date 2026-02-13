# src/predict.py

import os
import json
import pickle
import numpy as np
import tensorflow as tf

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# ==============================
# LOAD CONFIG SAFELY
# ==============================
from config import (
    IMAGE_SIZE,
    MODEL_PATH,
    TOKENIZER_PATH
)

MODEL_DIR = os.path.dirname(MODEL_PATH)
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")


# ==============================
# LOAD MODEL, TOKENIZER, CNN
# ==============================
def load_resources():
    print("[INFO] Loading trained model, tokenizer & CNN...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found.")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("Tokenizer not found.")
    if not os.path.exists(MODEL_CONFIG_PATH):
        raise FileNotFoundError("model_config.json not found.")

    # Load model
    model = load_model(MODEL_PATH)

    # Load tokenizer
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    # Load model config (IMPORTANT)
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_config = json.load(f)

    # CNN for feature extraction
    cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    return model, tokenizer, model_config, cnn


model, tokenizer, model_config, cnn_model = load_resources()

MAX_CAPTION_LENGTH = model_config["max_caption_length"]

word_index = tokenizer.word_index
index_word = tokenizer.index_word


# ==============================
# SPECIAL TOKENS (COCO SAFE)
# ==============================
START_TOKEN = "<start>"
END_TOKEN = "<end>"

if START_TOKEN not in word_index or END_TOKEN not in word_index:
    raise ValueError("Start/End tokens not found in tokenizer.")


# ==============================
# IMAGE FEATURE EXTRACTION
# ==============================
def extract_image_features(image_path):
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature = cnn_model.predict(image, verbose=0)
    return feature[0]


# ==============================
# BEAM SEARCH CAPTION GENERATION
# ==============================
def generate_caption_beam(feature, beam_width=5, alpha=0.7):
    """
    Beam Search with length normalization
    """
    start_id = word_index[START_TOKEN]
    end_id = word_index[END_TOKEN]

    sequences = [[ [start_id], 0.0 ]]
    feature = feature.reshape(1, -1)

    for _ in range(MAX_CAPTION_LENGTH):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end_id:
                all_candidates.append([seq, score])
                continue

            padded_seq = pad_sequences(
                [seq],
                maxlen=MAX_CAPTION_LENGTH,
                padding="post"
            )

            preds = model.predict([feature, padded_seq], verbose=0)[0]
            top_k = np.argsort(preds)[-beam_width:]

            for word_id in top_k:
                prob = preds[word_id]
                candidate = [
                    seq + [word_id],
                    score + np.log(prob + 1e-10)
                ]
                all_candidates.append(candidate)

        # Length normalization
        def normalized_score(candidate):
            seq, raw_score = candidate
            length = len(seq)
            return raw_score / (length ** alpha)

        ordered = sorted(
            all_candidates,
            key=normalized_score,
            reverse=True
        )

        sequences = ordered[:beam_width]

        if all(seq[-1] == end_id for seq, _ in sequences):
            break

    # Decode captions
    captions = []
    for seq, _ in sequences:
        words = [
            index_word.get(i)
            for i in seq
            if i not in [start_id, end_id]
        ]
        caption = " ".join(words)
        captions.append(caption)

    return list(set(captions))


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("\n🧠 COCO Image Caption Generator")
    image_path = input("📷 Enter image path: ").strip().replace('"', '')

    if not os.path.exists(image_path):
        print("❌ Image not found.")
        exit()

    print("[INFO] Extracting image features...")
    features = extract_image_features(image_path)

    print("[INFO] Generating captions...")
    captions = generate_caption_beam(features, beam_width=5)

    print("\n✨ FINAL PREDICTIONS:")
    captions = [c for c in captions if len(c.split()) > 3]

    if captions:
        for i, cap in enumerate(captions[:3], 1):
            print(f"{i}. {cap.capitalize()}")
    else:
        print("No good captions generated. Try another image.")
