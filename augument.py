import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np

# Paths to your smaller classes
folders = {
    "hungry": "Pet_Dataset/hungry",
    "scared": "Pet_Dataset/scared",
    "surprised": "Pet_Dataset/surprised"
}

# Target count per class
TARGET_COUNT = 1000

# Augmentation setup
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

for class_name, folder_path in folders.items():
    images = os.listdir(folder_path)
    current_count = len(images)
    print(f"\nClass: {class_name}, Current: {current_count}")

    if current_count >= TARGET_COUNT:
        print(f"✅ Already enough images.")
        continue

    # Number of new images needed
    needed = TARGET_COUNT - current_count
    print(f"➡️ Generating {needed} new images...")

    img_files = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    i = 0
    while i < needed:
        img_name = np.random.choice(img_files)
        img_path = os.path.join(folder_path, img_name)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Generate and save augmented images
        for batch in datagen.flow(x, batch_size=1, save_to_dir=folder_path,
                                  save_prefix='aug', save_format='jpg'):
            i += 1
            if i >= needed:
                break

    print(f"✅ {class_name} folder now has ~{TARGET_COUNT} images.")
