"""
this file will be using with fine tuning


"""

import os, json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
parent_dir = Path(__file__).parent.parent
IMG_SIZE = 160
DATASET_PATH = f"{parent_dir}/dataset"
BATCH_SIZE = 64
EPOCHS = 50

# Load and preprocess data
data, labels, label_map = [], [], {}
for idx, person in enumerate(sorted(os.listdir(DATASET_PATH))):
    label_map[idx] = person
    for img_file in os.listdir(os.path.join(DATASET_PATH, person)):
        img = cv2.imread(os.path.join(DATASET_PATH, person, img_file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(idx)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Build model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze for now

x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(len(label_map), activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initial training on augmented data
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    steps_per_epoch=len(X_train) // BATCH_SIZE
)

# Fine-tune model: unfreeze top 50 layers of base_model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune
model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    steps_per_epoch=len(X_train) // BATCH_SIZE
)

# Save model & label map
model.save(f"{parent_dir}/saved_model/face_recognition_final_model.keras")
with open(f"{parent_dir}/label_map.json", "w") as f:
    json.dump(label_map, f)

