# src/config.py

import os

# ==============================
# BASE DIRECTORY
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# DATA PATHS
# ==============================
DATA_DIR = os.path.join(BASE_DIR, "data")

IMAGE_DIR = os.path.join(DATA_DIR, "images", "Flickr8k_Dataset")
CAPTION_FILE = os.path.join(DATA_DIR, "captions", "Flickr8k.token.txt")

SPLIT_DIR = os.path.join(DATA_DIR, "splits")
TRAIN_IMAGES = os.path.join(SPLIT_DIR, "Flickr_8k.trainImages.txt")
DEV_IMAGES   = os.path.join(SPLIT_DIR, "Flickr_8k.devImages.txt")
TEST_IMAGES  = os.path.join(SPLIT_DIR, "Flickr_8k.testImages.txt")

# ==============================
# SAVE PATHS
# ==============================
FEATURE_DIR = os.path.join(BASE_DIR, "features")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

FEATURE_FILE = os.path.join(FEATURE_DIR, "image_features.pkl")
MODEL_PATH   = os.path.join(MODEL_DIR, "caption_model.h5")
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

# ==============================
# MODEL PARAMETERS
# ==============================
IMAGE_SIZE = 224
EMBEDDING_DIM = 256
LSTM_UNITS = 256
BATCH_SIZE = 32
EPOCHS = 15

# ==============================
# TEXT PARAMETERS
# ==============================
MAX_CAPTION_LENGTH = 34   # Suitable for Flickr8k
VOCAB_SIZE = 8000
