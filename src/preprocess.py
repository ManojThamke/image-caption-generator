# src/preprocess.py

import string
import pickle
from collections import defaultdict

from config import CAPTION_FILE, TOKENIZER_DIR

# ==============================
# LOAD CAPTIONS FILE
# ==============================
def load_captions(filename):
    """
    Loads captions from Flickr8k.token.txt
    Returns: dict -> {image_id: [caption1, caption2, ...]}
    """
    mapping = defaultdict(list)

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) < 1:
                continue

            image_caption = line.split('\t')
            image_id = image_caption[0].split('#')[0]
            caption = image_caption[1]

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
    - remove numbers
    - add <start> and <end> tokens
    """
    table = str.maketrans('', '', string.punctuation)

    for image_id, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.translate(table)
            caption = caption.replace('\d+', '')
            caption = ' '.join([word for word in caption.split() if len(word) > 1])

            captions[i] = '<start> ' + caption + ' <end>'


# ==============================
# SAVE CLEANED CAPTIONS
# ==============================
def save_captions(mapping, filename='cleaned_captions.pkl'):
    with open(f"{TOKENIZER_DIR}/{filename}", 'wb') as f:
        pickle.dump(mapping, f)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Loading captions...")
    captions = load_captions(CAPTION_FILE)

    print("[INFO] Cleaning captions...")
    clean_captions(captions)

    print("[INFO] Saving cleaned captions...")
    save_captions(captions)

    print("[DONE] Caption preprocessing completed successfully!")
