# src/predict.py

import os
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from config import IMAGE_SIZE, MAX_CAPTION_LENGTH, TOKENIZER_PATH

MODEL_PATH = "models/final_caption_model.h5"


# ==============================
# LOAD MODEL & TOKENIZER
# ==============================
print("[INFO] Loading model...")
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

index_word = tokenizer.index_word
word_index = tokenizer.word_index


# ==============================
# CNN FEATURE EXTRACTOR
# ==============================
cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


def extract_feature(img_path):
    image = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return cnn_model.predict(image, verbose=0)[0]


# ==============================
# CLEAN GREEDY + DIVERSITY SAMPLING
# ==============================
def generate_caption(feature, temperature=1.2):
    in_seq = []
    caption = []

    for _ in range(MAX_CAPTION_LENGTH):
        padded = pad_sequences([in_seq], maxlen=MAX_CAPTION_LENGTH)
        preds = model.predict([feature.reshape(1, -1), padded], verbose=0)[0]

        preds = np.log(preds + 1e-10) / temperature
        probs = np.exp(preds) / np.sum(np.exp(preds))

        word_id = np.random.choice(len(probs), p=probs)
        word = index_word.get(word_id)

        if word is None or word in caption:
            continue

        caption.append(word)
        in_seq.append(word_id)

        if len(caption) >= 12:
            break

    return " ".join(caption)


def generate_multiple_captions(feature, k=3):
    captions = set()
    while len(captions) < k:
        captions.add(generate_caption(feature))
    return list(captions)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    path = input("\nEnter image path: ").strip().strip('"').strip("'")
    path = os.path.normpath(path)

    if not os.path.exists(path):
        print("❌ Image not found")
        exit()

    print("[INFO] Extracting features...")
    feature = extract_feature(path)

    print("\n🖼️ Generated Captions:")
    captions = generate_multiple_captions(feature, k=3)

    for i, cap in enumerate(captions, 1):
        print(f"{i}. {cap}")
