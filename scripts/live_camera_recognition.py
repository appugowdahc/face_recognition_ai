import cv2
import pickle
import tensorflow as tf
from utils import detect_faces, recognize_face

model = tf.keras.models.load_model("/home/appu/face_recognition_ai/saved_model/face_recognition_model.keras")
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

with open("/home/appu/face_recognition_ai/embeddings/known_faces_embeddings.pkl", "rb") as f:
    known_embeddings = pickle.load(f)

face_cascade_path = "/home/appu/face_recognition_ai/haarcascades/haarcascade_frontalface_default.xml"
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame, face_cascade_path)
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        name = recognize_face(embedding_model, known_embeddings, face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
