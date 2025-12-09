import base64
from datetime import datetime
from flask import Blueprint, request, render_template, url_for, jsonify
import os
from app.image_analysis.skin_model import predict_skin_disease
import random, numpy as np, tensorflow as tf
import hashlib
from app.image_analysis.brain_predict import predict_brain_scan
from werkzeug.utils import secure_filename
from flask_login import login_required, current_user
from app.models import Prediction
from flask import current_app, flash, redirect

import base64
from io import BytesIO
from PIL import Image
import numpy as np
from flask import Blueprint, request, jsonify
import logging

# 🚫 Force CPU for consistency (optional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def sha256_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Blueprint definition
image_bp = Blueprint('image', __name__, template_folder='templates')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def decode_base64_image(base64_str):
    """
    Decode base64 image string and convert to numpy array.
    """
    try:
        # Remove the prefix (e.g. "data:image/jpeg;base64,")
        base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        img = Image.open(BytesIO(img_data))
        return np.array(img)
    except Exception as e:
        logging.error(f"Error decoding base64 image: {str(e)}")
        raise ValueError("Invalid base64 image data")
# Route for skin scan prediction
@image_bp.route('/skin-scan', methods=['GET', 'POST'])
@login_required
def skin_scan():
    if request.method == 'POST':
        print("POST request received")
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['file']
        patient_name = request.form.get('name', 'Unknown Patient')
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            print("File received")
            filename = secure_filename(file.filename)
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                prediction, confidence, top3_labels, top3_confidences = predict_skin_disease(filepath)
                if not isinstance(prediction, str) or not isinstance(confidence, (int, float)) or len(top3_labels) != 3 or len(top3_confidences) != 3:
                    raise ValueError("Invalid prediction return types")
                
                confidence_str = f"{confidence:.2f}%"
                
                from app import db
                try:
                    import logging
                    logging.basicConfig(level=logging.DEBUG)
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Saving skin scan prediction for user_id: {current_user.id}")
                    record = Prediction(
                        user_id=current_user.id,
                        model_used='SkinLesionCNN',
                        patient_name=patient_name,
                        symptoms='Skin lesion image analysis',
                        result=f"{prediction} (Confidence: {confidence_str})",
                        confidence=float(confidence)
                    )
                    db.session.add(record)
                    db.session.commit()
                    logger.debug("Skin scan prediction saved successfully")
                except Exception as db_error:
                    db.session.rollback()
                    logger.error(f"Database error while saving skin scan prediction: {str(db_error)}")
                    raise Exception(f"Database error: {str(db_error)}")
                
                return render_template('skin_scan.html',
                                    prediction=prediction,
                                    confidence=f"{confidence:.2f}%",
                                    image_url=url_for('static', filename=f'uploads/{filename}'))
            
            except Exception as e:
                print(f"Error during prediction: {e}")
                flash(f'Prediction failed: {str(e)}', 'danger')
                return redirect(request.url)

    return render_template('skin_scan.html')

@image_bp.route('/api/skin-predict', methods=['POST'])
def api_skin_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400

    file = request.files['file']
    patient_name = request.form.get('name', 'Unknown Patient')
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            prediction, confidence, top3_labels, top3_confidences = predict_skin_disease(filepath)
            if not isinstance(prediction, str) or not isinstance(confidence, (int, float)) or len(top3_labels) != 3 or len(top3_confidences) != 3:
                raise ValueError("Invalid prediction return types")

            # Save prediction to database
            from app import db
            try:
                record = Prediction(
                    user_id=current_user.id,
                    model_used='SkinLesionCNN',
                    patient_name=patient_name,
                    symptoms='Skin lesion image analysis',
                    result=f"{prediction} (Confidence: {confidence:.2f}%)",
                    confidence=float(confidence)
                )
                db.session.add(record)
                db.session.commit()
            except Exception as db_error:
                db.session.rollback()
                return jsonify({'error': f'Database error: {str(db_error)}'}), 500

            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence:.2f}%",
                'top3_labels': top3_labels,
                'top3_confidences': top3_confidences
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Invalid file type'}), 400

# --- Helper: Save file from request ---
def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath, filename

# --- Helper: Save base64 image ---
def save_base64_image(base64_data, prefix='image'):
    try:
        header, encoded = base64_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        return filepath, filename
    except Exception as e:
        raise ValueError(f"Failed to decode/save base64 image: {e}")

# --- Brain Scan Route ---
@image_bp.route('/brain-scan', methods=['GET', 'POST'])
@login_required
def brain_scan():
    if request.method == 'POST':
        file = request.files.get('file')
        patient_name = request.form.get('patient_name', 'Unknown Patient')
        if not file or file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if allowed_file(file.filename):
            try:
                filepath, filename = save_uploaded_file(file)
                return handle_brain_prediction(filepath, filename, patient_name=patient_name)
            except Exception as e:
                flash(f'Prediction failed: {str(e)}', 'danger')
                return redirect(request.url)

    return render_template('brain_scan.html')

# --- Brain Camera API ---
@image_bp.route('/api/brain-predict-camera', methods=['POST'])
def api_brain_predict():
    try:
        # Log the content type for debugging
        print(f"Received Content-Type: {request.content_type}")

        content_type = request.content_type.lower() if request.content_type else ''
        if content_type.startswith('application/json'):
            image_data = request.get_json().get('image')
            patient_name = request.get_json().get('patient_name', 'Unknown Patient')
            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400
            filepath, filename = save_base64_image(image_data, prefix='brain_cam')
        elif content_type.startswith('multipart/form-data'):
            # Check if file is present in request.files
            file = request.files.get('file')
            patient_name = request.form.get('patient_name', 'Unknown Patient')
            if file and allowed_file(file.filename):
                filepath, filename = save_uploaded_file(file)
            else:
                # Fallback: check for base64 image in form data
                image_data = request.form.get('image')
                if not image_data:
                    return jsonify({'error': 'No image data provided'}), 400
                filepath, filename = save_base64_image(image_data, prefix='brain_cam')
        else:
            return jsonify({'error': f'Unsupported Media Type: {request.content_type}'}), 415

        # Save prediction to database after getting prediction data
        prediction_data = predict_brain_scan(filepath)
        prediction = prediction_data.get('top_class')
        confidence = float(prediction_data.get('confidence'))  # Convert to native float
        confidence_percentage = f"{confidence:.2f}%"

        from app import db
        try:
            record = Prediction(
                user_id=current_user.id,
                model_used='BrainScanCNN',
                patient_name=patient_name,
                symptoms='Brain scan image analysis',
                result=f"{prediction} (Confidence: {confidence_percentage})",
                confidence=confidence
            )
            db.session.add(record)
            db.session.commit()
        except Exception as db_error:
            db.session.rollback()
            return jsonify({'error': f'Database error: {str(db_error)}'}), 500

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'confidence_percentage': confidence_percentage,
            'image_url': url_for('static', filename=f'uploads/{filename}')
        })

    except Exception as e:
        print(f"Camera prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def handle_brain_prediction(filepath, filename, api_mode=False, patient_name='Unknown Patient'):
    # Get prediction data
    prediction_data = predict_brain_scan(filepath)
    
    # Debugging: Log what is being returned from the model
    print(f"Prediction Data: {prediction_data}")  # Debugging line to log prediction data

    prediction = prediction_data.get('top_class')
    confidence = float(prediction_data.get('confidence'))  # Convert to native float
    top3_raw = prediction_data.get('top3')

    # Convert top3 probabilities to native float
    top3 = [(label, float(prob)) for label, prob in top3_raw]

    if not isinstance(prediction, str) or not isinstance(confidence, (int, float)) or not isinstance(top3, list) or len(top3) != 3:
        raise ValueError("Invalid prediction data from model")

    confidence_percentage = f"{confidence:.2f}%"  # Format this for the UI

    # Log formatted values to verify they are correct
    print(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Top3: {top3}")  # Debugging line

    if not api_mode:
        # Save prediction to the database (for non-API calls)
        from app import db
        try:
            record = Prediction(
                user_id=current_user.id,
                model_used='BrainScanCNN',
                patient_name=patient_name,
                symptoms='Brain scan image analysis',
                result=f"{prediction} (Confidence: {confidence_percentage})",
                confidence=confidence
            )
            db.session.add(record)
            db.session.commit()
        except Exception as db_error:
            db.session.rollback()
            raise Exception(f"Database error: {str(db_error)}")

        result = [{"class": label, "probability": prob / 100} for label, prob in top3]
        return render_template('brain_scan.html',
                               prediction=prediction,
                               confidence=confidence_percentage,
                               result=result,
                               image_url=url_for('static', filename=f'uploads/{filename}'))

    # API mode: Return raw confidence (numeric) for better flexibility in front-end handling
    return jsonify({
        'prediction': prediction,
        'confidence': confidence,  # Send numeric confidence here
        'confidence_percentage': confidence_percentage,  # Optionally, return formatted confidence string
        'top3_labels': [label for label, _ in top3],
        'top3_confidences': [prob for _, prob in top3],
        'image_url': url_for('static', filename=f'uploads/{filename}')
    })
@image_bp.route('/clear-history', methods=['POST'])
@login_required
def clear_history():
    try:
        Prediction.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash('Your history has been cleared.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('predict.history'))
@image_bp.route('/prediction/<int:prediction_id>', methods=['GET'])
@login_required
def prediction_details(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)
    return render_template('prediction_details.html', prediction=prediction)
