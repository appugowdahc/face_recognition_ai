import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path

IMG_SIZE = 160
parent_dir = Path(__file__).parent.parent

# Load model once at startup
model = tf.keras.models.load_model(f"{parent_dir}/saved_model/face_recognition_model.keras")

# Load label map
with open(f"{parent_dir}/label_map.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}


def predict_person(image_path: str):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {
        "person": label_map[predicted_class],
        "confidence": round(confidence, 2)
    }
test_img_path = f"{parent_dir}/test_images/brad_pitt1_crop.png"
print(predict_person(test_img_path))