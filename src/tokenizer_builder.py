# src/tokenizer_builder.py

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

from config import TOKENIZER_DIR, VOCAB_SIZE

# ==============================
# LOAD CLEANED CAPTIONS
# ==============================
def load_cleaned_captions():
    with open(f"{TOKENIZER_DIR}/cleaned_captions.pkl", "rb") as f:
        captions = pickle.load(f)
    return captions


# ==============================
# CREATE TOKENIZER
# ==============================
def create_tokenizer(captions_dict):
    all_captions = []

    for captions in captions_dict.values():
        for caption in captions:
            all_captions.append(caption)

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<unk>"
    )
    tokenizer.fit_on_texts(all_captions)

    return tokenizer, all_captions


# ==============================
# GET MAX CAPTION LENGTH
# ==============================
def max_caption_length(captions):
    return max(len(caption.split()) for caption in captions)


# ==============================
# SAVE TOKENIZER
# ==============================
def save_tokenizer(tokenizer):
    with open(f"{TOKENIZER_DIR}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Loading cleaned captions...")
    captions_dict = load_cleaned_captions()

    print("[INFO] Creating tokenizer...")
    tokenizer, all_captions = create_tokenizer(captions_dict)

    print("[INFO] Saving tokenizer...")
    save_tokenizer(tokenizer)

    max_len = max_caption_length(all_captions)

    print(f"[DONE] Tokenizer created")
    print(f"Vocabulary Size: {len(tokenizer.word_index) + 1}")
    print(f"Max Caption Length: {max_len}")
    print("Tokenizer saved successfully!")
    
