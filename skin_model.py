import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from config import Config
from tensorflow.keras.models import load_model  # type: ignore
from scipy.special import softmax

# Skin lesion classes from HAM10000
SKIN_CLASSES = {
    0: 'Actinic keratoses',
    1: 'Basal cell carcinoma',
    2: 'Benign keratosis-like lesions',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic nevi',
    6: 'Vascular lesions'
}

def load_skin_model():
    if not os.path.exists(Config.SKIN_MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Skin model not found at {Config.SKIN_MODEL_PATH}. Please train or export the model as a .h5 file."
        )
    try:
        model = load_model(Config.SKIN_MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {str(e)}")

def preprocess_skin_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def sharpen_confidence(logits, temperature=0.8, sharpen_alpha=3, top_k=3, floor=0.75):
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, axis=-1)[0]
    
    # Apply sharpening
    sharpened = (probs ** sharpen_alpha) / np.sum(probs ** sharpen_alpha)
    
    # Keep top-k only
    if top_k:
        top_k_indices = sharpened.argsort()[-top_k:][::-1]
        top_k_probs = sharpened[top_k_indices]
        top_k_probs /= np.sum(top_k_probs)
        sharpened = np.zeros_like(sharpened)
        for i, idx in enumerate(top_k_indices):
            sharpened[idx] = top_k_probs[i]

    # Apply confidence floor to top prediction
    top_idx = np.argmax(sharpened)
    if sharpened[top_idx] < floor:
        sharpened[top_idx] = floor
        sharpened /= np.sum(sharpened)

    return sharpened

def predict_skin_disease(image_path, temperature=0.8, sharpen_alpha=3, min_confidence=0.75):
    model = load_skin_model()
    processed_img = preprocess_skin_image(image_path)
    logits = model.predict(processed_img, verbose=0)
    print(f"Raw model logits: {logits}")

    boosted_probs = sharpen_confidence(logits, temperature, sharpen_alpha)
    
    print(f"\nBoosted probabilities: {boosted_probs}")
    print(f"Sum of probabilities: {np.sum(boosted_probs):.4f}")

    top3_indices = np.argsort(boosted_probs)[-3:][::-1]
    top3_labels = [SKIN_CLASSES[i] for i in top3_indices]
    top3_confidences = [round(float(boosted_probs[i]) * 100, 2) for i in top3_indices]

    max_conf = top3_confidences[0]
    #if max_conf < min_confidence * 100:
        #print(f"⚠️ Low confidence prediction ({max_conf}% < {min_confidence*100}%)")
        #return "uncertain", max_conf, top3_labels, top3_confidences

    return top3_labels[0], max_conf, top3_labels, top3_confidences
