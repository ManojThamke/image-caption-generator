# src/tokenizer_builder.py

import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

from config import TOKENIZER_DIR, TOKENIZER_PATH, VOCAB_SIZE


# ==============================
# LOAD CLEANED TRAIN CAPTIONS
# ==============================
def load_cleaned_captions():
    """
    Expected format:
    {
        image_id (int): [
            "<start> a man riding a bike <end>",
            "<start> a person on a bicycle <end>",
            ...
        ]
    }
    """
    captions_path = os.path.join(TOKENIZER_DIR, "cleaned_train_captions.pkl")

    with open(captions_path, "rb") as f:
        captions = pickle.load(f)

    return captions


# ==============================
# CREATE TOKENIZER
# ==============================
def create_tokenizer(captions_dict):
    all_captions = []

    for caption_list in captions_dict.values():
        all_captions.extend(caption_list)

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~\t\n'
    )

    tokenizer.fit_on_texts(all_captions)

    return tokenizer, all_captions


# ==============================
# GET MAX CAPTION LENGTH
# ==============================
def get_max_caption_length(captions):
    return max(len(caption.split()) for caption in captions)


# ==============================
# SAVE TOKENIZER
# ==============================
def save_tokenizer(tokenizer):
    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    print(f"[INFO] Tokenizer saved at: {TOKENIZER_PATH}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Loading cleaned COCO train captions...")
    captions_dict = load_cleaned_captions()

    print("[INFO] Creating tokenizer from training captions only...")
    tokenizer, all_captions = create_tokenizer(captions_dict)

    print("[INFO] Saving tokenizer...")
    save_tokenizer(tokenizer)

    max_len = get_max_caption_length(all_captions)

    print("\n[DONE] Tokenizer created successfully!")
    print(f"Vocabulary Size (actual): {len(tokenizer.word_index) + 1}")
    print(f"Max Caption Length (observed): {max_len}")
    print("NOTE: MAX_CAPTION_LENGTH in config.py should be >= this value")
