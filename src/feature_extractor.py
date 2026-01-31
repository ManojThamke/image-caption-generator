# src/feature_extractor.py

import os
import pickle
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

from config import IMAGE_DIR, FEATURE_DIR, FEATURE_FILE, IMAGE_SIZE


# ==============================
# LOAD RESNET50 MODEL
# ==============================
def load_resnet_model():
    base_model = ResNet50(weights="imagenet")
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.layers[-2].output)
    return model


# ==============================
# EXTRACT FEATURES
# ==============================
def extract_features(directory):
    model = load_resnet_model()
    features = {}

    images = os.listdir(directory)

    print(f"[INFO] Total images found: {len(images)}")

    for img_name in tqdm(images):
        img_path = os.path.join(directory, img_name)

        # Load and preprocess image
        image = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract features
        feature = model.predict(image, verbose=0)
        features[img_name] = feature.flatten()

    return features


# ==============================
# SAVE FEATURES
# ==============================
def save_features(features):
    os.makedirs(FEATURE_DIR, exist_ok=True)
    with open(FEATURE_FILE, "wb") as f:
        pickle.dump(features, f)


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("[INFO] Starting image feature extraction...")
    features = extract_features(IMAGE_DIR)

    print("[INFO] Saving extracted features...")
    save_features(features)

    print("[DONE] Image feature extraction completed successfully!")
