# src/preprocess.py

import json
import os
import string
import pickle
from collections import defaultdict

from config import TRAIN_CAPTIONS, VAL_CAPTIONS, TOKENIZER_DIR


# ==============================
# LOAD COCO CAPTIONS (JSON)
# ==============================
def load_coco_captions(json_file):
    """
    Loads COCO captions JSON
    Returns:
        dict -> {image_id (int): [caption1, caption2, ...]}
    """
    mapping = defaultdict(list)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        caption = ann["caption"]
        mapping[image_id].append(caption)

    return mapping


# ==============================
# CLEAN CAPTIONS
# ==============================
def clean_captions(mapping):
    """
    Cleans captions:
    - lowercase
    - remove punctuation
    - remove non-alphabetic tokens
    - add <start> and <end>
    """
    table = str.maketrans("", "", string.punctuation)

    for image_id, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.split()
            caption = [word for word in caption if word.isalpha()]
            caption = " ".join(caption)

            captions[i] = "<start> " + caption + " <end>"


# ==============================
# SAVE CLEANED CAPTIONS
# ==============================
def save_captions(mapping, filename):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    path = os.path.join(TOKENIZER_DIR, filename)

    with open(path, "wb") as f:
        pickle.dump(mapping, f)

    print(f"[INFO] Saved: {path}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Loading COCO train captions...")
    train_captions = load_coco_captions(TRAIN_CAPTIONS)

    print("[INFO] Cleaning train captions...")
    clean_captions(train_captions)

    print("[INFO] Saving cleaned train captions...")
    save_captions(train_captions, "cleaned_train_captions.pkl")

    print("[INFO] Loading COCO validation captions...")
    val_captions = load_coco_captions(VAL_CAPTIONS)

    print("[INFO] Cleaning validation captions...")
    clean_captions(val_captions)

    print("[INFO] Saving cleaned validation captions...")
    save_captions(val_captions, "cleaned_val_captions.pkl")

    print("[DONE] COCO caption preprocessing completed successfully!")
