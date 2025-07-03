import os
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

# ========================
# CONFIGURATION
# ========================
parent_dir = Path(__file__).parent.parent
IMG_SIZE = 160  # Increase to 224 for better accuracy
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = f"{parent_dir}/dataset"
MODEL_SAVE_PATH = f"{parent_dir}/saved_model/face_recognition_model.keras"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ========================
# FACE DETECTION FUNCTION
# ========================
def detect_face(img):
    """Detect and crop face from image"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # Fallback to entire image if no face detected
        return img
    
    (x, y, w, h) = faces[0]
    # Expand face area by 20%
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    x = max(0, x - margin_x)
    y = max(0, y - margin_y)
    w = min(img.shape[1] - x, w + 2 * margin_x)
    h = min(img.shape[0] - y, h + 2 * margin_y)
    
    return img[y:y+h, x:x+w]

# ========================
# DATA PREPARATION
# ========================
# Load and preprocess images
print("Loading and preprocessing images...")
image_groups = defaultdict(list)
label_map = {}
reverse_label_map = {}
label_counter = 0

for person in os.listdir(DATASET_PATH):
    if not os.path.isdir(os.path.join(DATASET_PATH, person)):
        continue
        
    label_map[label_counter] = person
    reverse_label_map[person] = label_counter
    person_dir = os.path.join(DATASET_PATH, person)
    
    for img_file in os.listdir(person_dir):
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(person_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Detect and crop face
        img = detect_face(img)
        if img.size == 0:
            continue
            
        # Resize and convert to RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Group by original image (before augmentation)
        base_name = img_file.split('_')[0]
        image_groups[(person, base_name)].append(img)
    
    label_counter += 1

# Create stratified split by original images
print("Creating train/test split...")
group_keys = list(image_groups.keys())
train_keys, test_keys = train_test_split(
    group_keys, 
    test_size=0.15,
    stratify=[key[0] for key in group_keys],
    random_state=42
)

# Build datasets
def build_dataset(keys):
    data, labels = [], []
    for key in keys:
        person, _ = key
        for img in image_groups[key]:
            data.append(img)
            labels.append(reverse_label_map[person])
    return np.array(data, dtype="float32") / 255.0, np.array(labels)

X_train, y_train = build_dataset(train_keys)
X_test, y_test = build_dataset(test_keys)

# Further split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.15, 
    stratify=y_train,
    random_state=42
)

# Diagnostic info
print("\n=== Dataset Information ===")
print(f"Classes: {len(label_map)}")
print(f"Train: {len(X_train)} images")
print(f"Validation: {len(X_val)} images")
print(f"Test: {len(X_test)} images")
print("Class distribution:")
for idx, name in label_map.items():
    print(f"  {name}: {np.sum(y_train == idx)} train, {np.sum(y_val == idx)} val, {np.sum(y_test == idx)} test")

# Visualize samples
plt.figure(figsize=(12, 8))
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.imshow(X_train[i])
    plt.title(f"{label_map[y_train[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{parent_dir}/data_samples.png")
print("Sample visualization saved to data_samples.png")

# ========================
# DATA AUGMENTATION
# ========================
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    channel_shift_range=50,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

# Calculate class weights
class_weights = {}
for cls in np.unique(y_train):
    weight = len(y_train) / (len(np.unique(y_train)) * np.sum(y_train == cls))
    class_weights[cls] = min(weight, 5.0)  # Cap at 5.0 to prevent overemphasis

# ========================
# MODEL ARCHITECTURE
# ========================
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model((IMG_SIZE, IMG_SIZE, 3), len(label_map))
model.summary()

# ========================
# CALLBACKS
# ========================
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-7),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy')
]

# ========================
# TRAINING
# ========================
print("\nTraining model...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    validation_data=val_datagen.flow(X_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# ========================
# EVALUATION
# ========================
print("\nEvaluating model...")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n=== Final Metrics ===")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ========================
# SAVING RESULTS
# ========================
model.save(MODEL_SAVE_PATH)
with open(f"{parent_dir}/label_map.json", "w") as f:    
    json.dump(label_map, f)

print(f"\nModel saved to {MODEL_SAVE_PATH}")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.savefig(f"{parent_dir}/training_history.png")
print("Training history plot saved")