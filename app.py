# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load all models at startup
print("Loading models...")
svm_model = joblib.load('models/svm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
kmeans_model = joblib.load('models/kmeans_model.pkl')
cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
print("✓ All models loaded!")

def preprocess_image(image_bytes, img_size=128):
    """Preprocess uploaded image"""
    # Convert bytes to image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Resize
    img_array = cv2.resize(img_array, (img_size, img_size))
    
    # Normalize
    img_array = img_array / 255.0
    
    return img_array

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        model_name = request.form.get('model', 'cnn')
        
        # Read image
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)
        
        # Predict based on selected model
        if model_name == 'cnn':
            # CNN expects shape (1, height, width, channels)
            img_input = np.expand_dims(img_array, axis=0)
            prediction = cnn_model.predict(img_input)[0][0]
            confidence = prediction if prediction > 0.5 else 1 - prediction
            result = 'Dog' if prediction > 0.5 else 'Cat'
            
        else:
            # Traditional models expect flattened input
            img_flat = img_array.reshape(1, -1)
            
            if model_name == 'svm':
                prediction = svm_model.predict(img_flat)[0]
                proba = svm_model.decision_function(img_flat)[0]
                confidence = abs(proba) / 10  # Normalize
                
            elif model_name == 'rf':
                prediction = rf_model.predict(img_flat)[0]
                proba = rf_model.predict_proba(img_flat)[0]
                confidence = max(proba)
                
            elif model_name == 'lr':
                prediction = lr_model.predict(img_flat)[0]
                proba = lr_model.predict_proba(img_flat)[0]
                confidence = max(proba)
                
            elif model_name == 'kmeans':
                cluster = kmeans_model.predict(img_flat)[0]
                prediction = cluster
                confidence = 0.5  # K-means doesn't give confidence
            
            result = 'Dog' if prediction == 1 else 'Cat'
        
        return jsonify({
            'prediction': result,
            'confidence': float(confidence * 100),
            'model': model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, port=5000)