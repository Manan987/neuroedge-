from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# Global variable to store the model
model = None

# Class labels
CLASS_LABELS = {
    0: 'Glioma Tumor',
    1: 'Meningioma Tumor', 
    2: 'No Tumor',
    3: 'Pituitary Tumor'
}

# Update model loading with custom objects handling
from tensorflow.keras.models import load_model

def load_trained_model():
    try:
        # Load model with custom objects handling
        model = tf.keras.models.load_model('model/brain_tumor_model.h5', 
                                         compile=False)
        # Recompile with original parameters
        model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        print("\033[92mModel recompiled successfully!\033[0m")
        return model
    except Exception as e:
        print(f"\033[91mModel loading error: {e}\033[0m")
        return None

def preprocess_image(image_file):
    """Preprocess the uploaded image for prediction"""
    try:
        # Read image
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (224x224 for most CNN models)
        image = image.resize((224, 224))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'success': False
            })
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'success': False
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            })
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({
                'error': 'Invalid file type. Please upload an image file.',
                'success': False
            })
        
        # Preprocess image
        processed_image = preprocess_image(file)
        
        if processed_image is None:
            return jsonify({
                'error': 'Error processing image',
                'success': False
            })
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # Get class label
        predicted_label = CLASS_LABELS.get(predicted_class, 'Unknown')
        
        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence * 100, 2),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'success': False
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_status = 'loaded' if model is not None else 'not loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

if __name__ == '__main__':
    load_trained_model()
    import socket
    
    def find_available_port(start=5000, end=6000):
        for port in range(start, end+1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('0.0.0.0', port))
                    return port
                except OSError:
                    continue
            raise OSError(f"No ports available between {start}-{end}")
    
    # Update app.run()
    # OR better yet - use a fixed port
    app.run(debug=True, host='0.0.0.0', port=find_available_port(5000, 5050))