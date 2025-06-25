import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from utils import get_embedding

IMG_SIZE = 160
DATASET_PATH = "/home/appu/face_recognition_ai/dataset"
MODEL_PATH = "/home/appu/face_recognition_ai/saved_model/face_recognition_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
known_embeddings = {}

try:
    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        embeddings = []
        for img_file in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, img_file))
            embeddings.append(get_embedding(embedding_model, img))
        known_embeddings[person] = np.mean(embeddings, axis=0)

    with open("/home/appu/face_recognition_ai/embeddings/known_faces_embeddings.pkl", "wb") as f:
        pickle.dump(known_embeddings, f)
except Exception as e:
    print(f"Error generating embeddings: {e}")