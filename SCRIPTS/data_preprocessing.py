import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories
RAW_DATA_DIR = "DATA/raw_images/"
PROCESSED_DATA_DIR = "DATA/processed_images/"

# Create processed images directory if it doesn't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Image preprocessing parameters
IMG_SIZE = (224, 224)

# Function to preprocess images
def preprocess_and_augment():
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2,
                                 height_shift_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True)

    for celebrity in os.listdir(RAW_DATA_DIR):
        raw_path = os.path.join(RAW_DATA_DIR, celebrity)
        save_path = os.path.join(PROCESSED_DATA_DIR, celebrity)
        os.makedirs(save_path, exist_ok=True)

        for img_name in os.listdir(raw_path):
            img_path = os.path.join(raw_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.resize(image, IMG_SIZE)

            cv2.imwrite(os.path.join(save_path, img_name), image)

            # Generate augmented images
            img_array = np.expand_dims(image, axis=0)
            for i, batch in enumerate(datagen.flow(img_array, batch_size=1,
                                                   save_to_dir=save_path,
                                                   save_prefix="aug",
                                                   save_format="jpeg")):
                if i >= 5:  # Generate 5 augmented images per original
                    break

preprocess_and_augment()
