# app/image_analysis/utils.py

import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load model once at the top
model = load_model('models/skin_model.h5')
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # update with your actual class names

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100
    return pred_class, confidence
