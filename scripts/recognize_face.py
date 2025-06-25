import cv2
import pickle
import tensorflow as tf
from utils import recognize_face

model = tf.keras.models.load_model("/home/appu/face_recognition_ai/saved_model/face_recognition_model.keras")
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

with open("/home/appu/face_recognition_ai/embeddings/known_faces_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

img = cv2.imread("/home/appu/face_recognition_ai/dataset/Johnny Depp/001_2288a4f6.jpg")
name = recognize_face(embedding_model, known_embeddings, img)
print(f"Recognized: {name}")