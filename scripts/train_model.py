# scripts/train_model.py
import os,json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess_images import preprocess_image
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"


print("GPUs Available: ", tf.config.list_physical_devices('GPU'))


parent_dir = Path(__file__).parent.parent
IMG_SIZE = 160
DATASET_PATH = f"{parent_dir}/dataset"


##preprocess images 
# this code will create grayscale,brightened,blurred, contrasted, saturated images
for idx, person in enumerate(os.listdir(DATASET_PATH)):

    for img_file in os.listdir(os.path.join(DATASET_PATH, person)):
        preprocess_image(os.path.join(DATASET_PATH, person, img_file))


# Prepare data
data, labels, label_map = [], [], {}
for idx, person in enumerate(os.listdir(DATASET_PATH)):
    label_map[idx] = person
    for img_file in os.listdir(os.path.join(DATASET_PATH, person)):
        img = cv2.imread(os.path.join(DATASET_PATH, person, img_file))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(idx)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

#for data augmentation
x_train,y_train = data,labels
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_images = datagen.flow(x_train, y_train, batch_size=128)
print(augmented_images, "Augmented images ready for training.")

# exit("Data augmentation is not implemented in this script. Please implement it if needed.")

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
# Fit the data generator (optional but recommended for some transforms)
datagen.fit(X_train)


# Build model
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(len(label_map), activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train using augmented data
batch_size = 128
augmented_images = datagen.flow(X_train, y_train, batch_size=batch_size)

model.fit(
    augmented_images,
    validation_data=(X_test, y_test),
    epochs=10,
    steps_per_epoch=len(X_train) // batch_size
)



# Train and save
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
model.save(f"{parent_dir}/saved_model/face_recognition_model.keras")
with open(f"{parent_dir}/label_map.json", "w") as f:    
    json.dump(label_map, f)

# scripts/train_model.py
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

def train_model(log_callback=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

    parent_dir = Path(__file__).resolve().parent.parent
    IMG_SIZE = 160
    DATASET_PATH = parent_dir / "dataset"
    MODEL_PATH = parent_dir / "saved_model/face_recognition_model.keras"
    LABEL_PATH = parent_dir / "label_map.json"

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    log("🔍 Loading dataset...")
    data, labels, label_map = [], [], {}
    for idx, person in enumerate(os.listdir(DATASET_PATH)):
        label_map[idx] = person
        person_dir = DATASET_PATH / person
        for img_file in os.listdir(person_dir):
            img_path = person_dir / img_file
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    log(f"✅ Loaded {len(data)} images from {len(label_map)} classes.")

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output = tf.keras.layers.Dense(len(label_map), activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    log("🚀 Starting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=10,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log(
                    f"Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}"
                )
            )
        ]
    )

    model.save(MODEL_PATH)
    with open(LABEL_PATH, "w") as f:
        json.dump(label_map, f)

    log("✅ Training completed and model saved.")
    return {"status": "done", "classes": label_map, "samples": len(data)}

# if __name__ == "__main__":
#     # train_model()