# scripts/train_model.py
import os, json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Constants and paths
parent_dir = Path(__file__).parent.parent
IMG_SIZE = 224
DATASET_PATH = os.path.join(parent_dir, "dataset")
MODEL_PATH = os.path.join(parent_dir, "saved_model")
LABEL_MAP_PATH = os.path.join(parent_dir, "label_map.json")

# Load and preprocess dataset
data, labels, label_map = [], [], {}
for idx, person in enumerate(os.listdir(DATASET_PATH)):
    label_map[idx] = person
    person_path = os.path.join(DATASET_PATH, person)
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(idx)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Build model using EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
output = tf.keras.layers.Dense(len(label_map), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train with augmented data
batch_size = 64
model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=15,
    steps_per_epoch=len(X_train) // batch_size
)

# Save model and label map
os.makedirs(MODEL_PATH, exist_ok=True)
model.save(os.path.join(MODEL_PATH, "efficientnetB0_face_model.keras"))

with open(LABEL_MAP_PATH, "w") as f:
    json.dump(label_map, f)
