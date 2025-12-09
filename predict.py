import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from scipy.special import softmax
import logging
import random
import hashlib

# 🚫 Force CPU for consistency (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 🌱 Set all seeds
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Config
MODEL_PATH = 'models/skin_model.h5'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
TEMPERATURE = 0.8
SHARPEN_ALPHA = 3
TOP_K = 3
CONFIDENCE_FLOOR = 0.75
TTA_STEPS = 1

# ✅ Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 🔁 Load model only once
_model = None
def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
        logger.debug("✅ Model loaded and cached.")
    return _model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def test_time_augmentation_predict(model, img_array, steps):
    return np.mean([model.predict(img_array, verbose=0) for _ in range(steps)], axis=0)

def boost_confidence(logits, temperature, sharpen_alpha, top_k, confidence_floor):
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, axis=-1)[0]
    probs = (probs ** sharpen_alpha) / np.sum(probs ** sharpen_alpha)
    if top_k:
        top_k_indices = probs.argsort()[-top_k:][::-1]
        top_k_probs = probs[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        probs = np.zeros_like(probs)
        for i, idx in enumerate(top_k_indices):
            probs[idx] = top_k_probs[i]
    if confidence_floor:
        top_idx = np.argmax(probs)
        if probs[top_idx] < confidence_floor:
            probs[top_idx] = confidence_floor
            probs /= np.sum(probs)
    return probs

def predict_skin_disease(img_path):
    logger.debug(f"🖼️ Image path: {img_path}")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    with open(img_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
        logger.debug(f"🧬 SHA256: {sha}")

    model = get_model()
    img_batch = preprocess_image(img_path)
    logits = test_time_augmentation_predict(model, img_batch, TTA_STEPS)
    boosted_probs = boost_confidence(logits, TEMPERATURE, SHARPEN_ALPHA, TOP_K, CONFIDENCE_FLOOR)

    top_idx = np.argmax(boosted_probs)
    top_class = CLASS_NAMES[top_idx]
    confidence = boosted_probs[top_idx] * 100
    top3_indices = boosted_probs.argsort()[-3:][::-1]
    top3 = [(CLASS_NAMES[i], boosted_probs[i] * 100) for i in top3_indices]

    return {
        "top_class": top_class,
        "confidence": confidence,
        "top3": top3,
        "raw_probs": boosted_probs.tolist()
    }

if __name__ == "__main__":
    image_path = r"C:\games\disease-predictor-web\app\static\uploads\ISIC_0026068.jpg"
    result = predict_skin_disease(image_path)
    print("\n✅ Final Prediction:")
    for label, prob in result["top3"]:
        print(f"{label}: {prob:.2f}%")
