import cv2
import numpy as np
import tensorflow as tf
import json
from pathlib import Path


def detect_and_crop_face_dnn(img):
    model_path = parent_dir / "face_detector"
    prototxt_path = str(model_path / "deploy.prototxt")
    weights_path = str(model_path / "res10_300x300_ssd_iter_140000.caffemodel")

    net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    confidence_threshold = 0.5
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure bounds are within the image
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            return img[y1:y2, x1:x2]

    raise ValueError("No face detected with sufficient confidence.")



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

    try:
        face_img = detect_and_crop_face_dnn(img)
    except ValueError as e:
        return {"error": str(e)}

    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    predictions = model.predict(face_img)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {
        "person": label_map[predicted_class],
        "confidence": round(confidence, 2)
    }

# def predict_person(image_path: str):
#     img = cv2.imread(image_path)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)

#     predictions = model.predict(img)
#     predicted_class = int(np.argmax(predictions[0]))
#     confidence = float(np.max(predictions[0]))

#     return {
#         "person": label_map[predicted_class],
#         "confidence": round(confidence, 2)
#     }
test_img_path = f"{parent_dir}/test_images/brad_pitt2.jpg"
print(predict_person(test_img_path))