import tensorflow as tf
import os
from pathlib import Path

def preprocess_image(image_path):
    # Read and decode image
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Extract filename and extension safely
    image_path_parts = Path(image_path)
    image_dir = image_path_parts.parent  # Get the directory of the image
    name = image_path_parts.stem  # Get the filename without extension
    ext = image_path_parts.suffix  # Get the file extension (e.g., ".


    # Define transformations and output names
    variations = {
        "grayscale": tf.image.rgb_to_grayscale(image),
        "brighter": tf.image.adjust_brightness(image, delta=0.3),
        "higher_contrast": tf.image.adjust_contrast(image, contrast_factor=2.0),
        "saturated": tf.image.adjust_saturation(image, saturation_factor=2.0),
    }

    # Convert grayscale to RGB before saving (for consistent format)
    if len(variations["grayscale"].shape) == 3 and variations["grayscale"].shape[-1] == 1:
        variations["grayscale"] = tf.image.grayscale_to_rgb(variations["grayscale"])

    # Save images
    for key, img in variations.items():
        output_path = f"{image_dir}/{name}_{key}_aug{ext}"
        # import pdb; pdb.set_trace()  # Debugging line to inspect variables
        tf.keras.preprocessing.image.save_img(output_path, img)
        print(f"Saved: {output_path}")

