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

# Config for brain scan model
BRAIN_MODEL_PATH = 'models/brain_model.h5'
BRAIN_IMG_SIZE = (224, 224)  # Adjust as per brain model input size
BRAIN_CLASS_NAMES = ['glioma', 'meningioma','notumor','pituitary']  # Example classes, adjust as needed

# ✅ Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 🔁 Load brain model only once
_brain_model = None
def get_brain_model():
    global _brain_model
    if _brain_model is None:
        _brain_model = load_model(BRAIN_MODEL_PATH)
        logger.debug("✅ Brain model loaded and cached.")
    return _brain_model

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
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

def predict_brain_scan(img_path):
    logger.debug(f"🖼️ Brain scan image path: {img_path}")

    if not os.path.exists(BRAIN_MODEL_PATH):
        raise FileNotFoundError(f"Brain model not found at {BRAIN_MODEL_PATH}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    with open(img_path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
        logger.debug(f"🧬 SHA256: {sha}")

    model = get_brain_model()
    img_batch = preprocess_image(img_path, BRAIN_IMG_SIZE)
    logits = test_time_augmentation_predict(model, img_batch, 1)
    boosted_probs = boost_confidence(logits, 0.8, 3, 3, 0.75)

    top_idx = np.argmax(boosted_probs)
    top_class = BRAIN_CLASS_NAMES[top_idx]
    confidence = boosted_probs[top_idx] * 100
    top3_indices = boosted_probs.argsort()[-3:][::-1]
    top3 = [(BRAIN_CLASS_NAMES[i], boosted_probs[i] * 100) for i in top3_indices]

    return {
        "top_class": top_class,
        "confidence": confidence,
        "top3": top3,
        "raw_probs": boosted_probs.tolist()
    }