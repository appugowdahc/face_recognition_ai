import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU to avoid cuInit error

import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from pathlib import Path
import mtcnn
import urllib.request
import time
import sys

# ========================
# CONFIGURATION
# ========================
parent_dir = Path(__file__).parent.parent
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = f"{parent_dir}/dataset"
MODEL_SAVE_PATH = f"{parent_dir}/saved_model/facenet_model.keras"
FACENET_PATH = f"{parent_dir}/facenet_keras.h5"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ========================
# MODEL DOWNLOAD
# ========================
def download_with_progress(url, filename):
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192

            with open(filename, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = downloaded / total_size * 100
                    sys.stdout.write(f"\rDownloading: {progress:.1f}% ({downloaded}/{total_size} bytes)")
                    sys.stdout.flush()
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

# Download FaceNet model if missing
if not os.path.exists(FACENET_PATH):
    print("Downloading FaceNet model...")
    download_sources = [
        "https://github.com/nyoki-mtl/keras-facenet/raw/master/model/facenet_keras.h5",
    ]
    for url in download_sources:
        if download_with_progress(url, FACENET_PATH):
            break
    else:
        print("\n❌ Download failed. Please manually download 'facenet_keras.h5' and place it here:")
        print(FACENET_PATH)
        exit(1)

# ========================
# FACE DETECTOR (MTCNN)
# ========================
face_detector = mtcnn.MTCNN()

def align_face(img):
    try:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = face_detector.detect_faces(rgb_img)

        if not detections:
            return None

        main_face = max(detections, key=lambda x: x['confidence'])
        x, y, w, h = main_face['box']
        x, y = max(0, x), max(0, y)
        w, h = min(img.shape[1] - x, w), min(img.shape[0] - y, h)
        if w <= 10 or h <= 10:
            return None

        margin = 0.2
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        x_start = max(0, x - x_margin)
        y_start = max(0, y - y_margin)
        x_end = min(img.shape[1], x + w + x_margin)
        y_end = min(img.shape[0], y + h + y_margin)

        cropped = img[y_start:y_end, x_start:x_end]
        return cropped if cropped.size > 0 else None
    except Exception as e:
        print(f"Face alignment error: {e}")
        return None

# ========================
# LOAD FACENET MODEL
# ========================
def load_facenet():
    try:
        model = tf.keras.models.load_model(FACENET_PATH, compile=False)
        print("✅ FaceNet model loaded.")
        return Model(inputs=model.inputs, outputs=model.layers[-2].output)
    except Exception as e:
        print(f"\n❌ Failed to load FaceNet model: {e}")
        print("Try using TensorFlow 2.3–2.6 with Python 3.6–3.8 or use a model without Lambda layers.")
        exit(1)

print("Loading FaceNet model...")
facenet = load_facenet()
facenet.trainable = False

# ========================
# LOAD DATASET
# ========================
print("Loading and preprocessing dataset...")
data, labels = [], []
label_map = {}
label_counter = 0

for person in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    label_map[label_counter] = person
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        aligned = align_face(img)
        if aligned is None:
            continue
        aligned = cv2.resize(aligned, (IMG_SIZE, IMG_SIZE))
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        data.append(aligned)
        labels.append(label_counter)
    if any(labels.count(label_counter) for _ in range(1)):
        label_counter += 1
    else:
        del label_map[label_counter]

if not data:
    raise ValueError("No valid images found!")

data = np.array(data)
labels = np.array(labels)

print("\nClasses:", len(label_map))
print("Images:", len(data))
for i, name in label_map.items():
    print(f" - {name}: {(labels == i).sum()} images")

# Save sample images
os.makedirs(f"{parent_dir}/samples", exist_ok=True)
for i in range(min(10, len(data))):
    sample = (data[i] * 255).astype('uint8')
    cv2.imwrite(f"{parent_dir}/samples/sample_{i}.jpg", cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

# ========================
# EMBEDDING
# ========================
print("Generating embeddings...")
embeddings = facenet.predict(data, batch_size=32, verbose=1)

# ========================
# CLASSIFIER
# ========================
def build_classifier(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        BatchNormalization(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

le = LabelEncoder()
encoded_labels = le.fit_transform(labels)

if len(label_map) > 1:
    X_train, X_test, y_train, y_test = train_test_split(embeddings, encoded_labels, test_size=0.15, stratify=encoded_labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
else:
    X_train, X_val, X_test = np.split(embeddings, [int(.7*len(embeddings)), int(.85*len(embeddings))])
    y_train, y_val, y_test = np.split(encoded_labels, [int(.7*len(embeddings)), int(.85*len(embeddings))])

classifier = build_classifier(embeddings.shape[1], len(label_map))
classifier.summary()

# ========================
# TRAIN
# ========================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
]

print("Training classifier...")
classifier.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ========================
# EVALUATE
# ========================
print("Evaluating model...")
print("Train accuracy:", classifier.evaluate(X_train, y_train, verbose=0)[1])
print("Val accuracy:  ", classifier.evaluate(X_val, y_val, verbose=0)[1])
print("Test accuracy: ", classifier.evaluate(X_test, y_test, verbose=0)[1])

# ========================
# SAVE FINAL MODEL
# ========================
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
embedding_out = facenet(input_layer)
prediction = classifier(embedding_out)
combined = Model(inputs=input_layer, outputs=prediction)
combined.save(MODEL_SAVE_PATH)

with open(f"{parent_dir}/label_map.json", "w") as f:
    json.dump(label_map, f)

print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")
