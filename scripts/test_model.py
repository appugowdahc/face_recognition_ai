import os,json
import cv2
import numpy as np
import tensorflow as tf
import pickle

# Constants
IMG_SIZE = 160
# MODEL_PATH = "/home/appu/face_recognition_ai/saved_model/face_recognition_augmented_model.keras"
MODEL_PATH = "/home/appu/face_recognition_ai/saved_model/face_recognition_model.keras"

with open("/home/appu/face_recognition_ai/label_map.json", "r") as f:
    LABEL_MAP = json.load(f)
TEST_IMAGE_PATH = "/home/appu/face_recognition_ai/test_images/jd7.jpeg"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess test image
img = cv2.imread(TEST_IMAGE_PATH)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)  # Model expects batch

# Predict
predictions = model.predict(img)
predicted_label = np.argmax(predictions[0])
confidence = predictions[0][predicted_label]

# Output result
print(f"Predicted Person: {LABEL_MAP[str(predicted_label)]} (Confidence: {confidence:.2f})")
