# src/config.py

import os

# ==============================
# BASE DIRECTORY
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# DATA ROOT
# ==============================
DATA_DIR = os.path.join(BASE_DIR, "data")

# ==============================
# COCO IMAGE PATHS
# ==============================
COCO_DIR = os.path.join(DATA_DIR, "coco")

COCO_TRAIN_IMG = os.path.join(COCO_DIR, "images", "train2017")
COCO_VAL_IMG   = os.path.join(COCO_DIR, "images", "val2017")
COCO_TEST_IMG  = os.path.join(COCO_DIR, "images", "test2017")  # optional (not used)

# ==============================
# COCO ANNOTATION PATHS
# ==============================
ANNOTATION_DIR = os.path.join(COCO_DIR, "annotations")

TRAIN_CAPTIONS = os.path.join(ANNOTATION_DIR, "captions_train2017.json")
VAL_CAPTIONS   = os.path.join(ANNOTATION_DIR, "captions_val2017.json")

# ==============================
# SAVE DIRECTORIES
# ==============================
FEATURE_DIR   = os.path.join(BASE_DIR, "features")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

# ==============================
# FEATURE FILES
# ==============================
TRAIN_FEATURE_FILE = os.path.join(FEATURE_DIR, "coco_train_features.pkl")
VAL_FEATURE_FILE   = os.path.join(FEATURE_DIR, "coco_val_features.pkl")

# ==============================
# MODEL / TOKENIZER FILES
# ==============================
MODEL_PATH     = os.path.join(MODEL_DIR, "coco_caption_model.h5")
TOKENIZER_PATH = os.path.join(TOKENIZER_DIR, "tokenizer_coco.pkl")

# ==============================
# MODEL PARAMETERS
# ==============================
IMAGE_SIZE = 224          # ResNet50 input size
EMBEDDING_DIM = 256
LSTM_UNITS = 256

BATCH_SIZE = 64
EPOCHS = 7                # ✅ BEST choice for FULL COCO dataset

# ==============================
# TEXT PARAMETERS
# ==============================
MAX_CAPTION_LENGTH = 40   # COCO captions are longer
VOCAB_SIZE = 15000        # COCO vocabulary size
