# scripts/utils.py
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_image(img, target_size=(160, 160)):
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def get_embedding(model, face_img):
    preprocessed = preprocess_image(face_img)
    return model.predict(preprocessed)[0]

def detect_faces(frame, face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)


def recognize_face(model, known_embeddings, input_img, threshold=0.7):
    embedding = get_embedding(model, input_img)
    similarities = {
        name: cosine_similarity([embedding], [embed])[0][0]
        for name, embed in known_embeddings.items()
    }
    best_match = max(similarities, key=similarities.get)
    if similarities[best_match] >= threshold:
        return best_match
    return "Unknown"