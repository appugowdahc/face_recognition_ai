import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

# Configuration
MODEL_PATH = "/home/appu/face_recognition_ai/saved_model/face_recognition_final.keras"
LABEL_MAP_PATH = "/home/appu/face_recognition_ai/label_map.json"
IMG_SIZE = 160

# Load model and label map
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    # Convert string keys to integers (JSON stores keys as strings)
    label_map = {int(k): v for k, v in label_map.items()}

def preprocess_image(image_path):
    """Preprocess image for model inference"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32')
    img = (img / 127.5) - 1.0  # Same normalization as training
    return np.expand_dims(img, axis=0)  # Add batch dimension

def predict_identity(image_path, confidence_threshold=0.7):
    """Predict identity with confidence threshold"""
    try:
        # Preprocess image
        processed_img = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_img)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        
        # Get label
        identity = label_map.get(str(predicted_idx), "Unknown")
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            print(identity,"identity")
            return "Unknown", confidence
        
        return identity, confidence
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0

# Example usage
if __name__ == "__main__":
    test_image = "/home/appu/face_recognition_ai/test_images/jd7.jpeg"
    identity, confidence = predict_identity(test_image)
    print(f"Predicted: {identity} with {confidence:.2%} confidence")