# src/feature_extractor.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from config import (
    COCO_TRAIN_IMG,
    COCO_VAL_IMG,
    FEATURE_DIR,
    TRAIN_FEATURE_FILE,
    VAL_FEATURE_FILE,
    IMAGE_SIZE
)

# ==============================
# LOAD RESNET50 MODEL
# ==============================
def load_resnet_model():
    base_model = ResNet50(weights="imagenet")
    model = Model(
        inputs=base_model.input,
        outputs=base_model.layers[-2].output  # 2048-d vector
    )
    return model


# ==============================
# EXTRACT FEATURES FROM A FOLDER
# ==============================
def extract_features(image_dir):
    model = load_resnet_model()
    features = {}

    images = os.listdir(image_dir)
    print(f"[INFO] Processing {len(images)} images from {image_dir}")

    for img_name in tqdm(images):
        img_path = os.path.join(image_dir, img_name)

        # Extract image_id from filename
        # 000000581929.jpg → 581929
        image_id = int(img_name.split(".")[0])

        # Load and preprocess image
        image = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract features
        feature = model.predict(image, verbose=0)
        features[image_id] = feature.flatten()

    return features


# ==============================
# SAVE FEATURES
# ==============================
def save_features(features, file_path):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(features, f)
    print(f"[INFO] Features saved to {file_path}")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Starting COCO feature extraction...")

    # 🔹 Train features
    train_features = extract_features(COCO_TRAIN_IMG)
    save_features(train_features, TRAIN_FEATURE_FILE)

    # 🔹 Validation features
    val_features = extract_features(COCO_VAL_IMG)
    save_features(val_features, VAL_FEATURE_FILE)

    print("[DONE] COCO image feature extraction completed successfully!")
