import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib   import Path

IMG_SIZE = 160
parent_dir = Path(__file__).parent.parent
# Constants
# Load model
model = tf.keras.models.load_model(f"{parent_dir}/saved_model/face_recognition_augmented_model.keras")

# Load label map
with open(f"{parent_dir}/label_map.json", "r") as f:
    label_map = json.load(f)

# Invert label_map to get class names
label_map = {int(k): v for k, v in label_map.items()}
# Path to the test image
test_image_path = f"{parent_dir}/test_images/jd4.jpeg"

# Load and preprocess
img = cv2.imread(test_image_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

# Result
print(f"Predicted Person: {label_map[predicted_class]} (Confidence: {confidence:.2f})")
