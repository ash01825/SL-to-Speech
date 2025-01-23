import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import random
from multiprocessing import Pool, cpu_count

# Configure the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Root directory containing folders of frames (each folder represents a video)
root_dir = "/Users/ash/Desktop/SL-to-Speech/datasets/cusom_dataset/frames_sentence_level"
augmented_folder_count = 3  # Number of augmented versions to create for each folder

# Function to augment a single image
def augment_image(image_path, datagen, params):
    rotation, width_shift, height_shift, zoom, flip = params
    img = load_img(image_path)
    img_array = img_to_array(img)
    augmented_img = datagen.apply_transform(
        img_array, {
            'theta': rotation,
            'tx': width_shift,
            'ty': height_shift,
            'zx': zoom,
            'zy': zoom,
            'horizontal_flip': flip
        }
    )
    return augmented_img

# Function to augment all images in a folder
def augment_folder(args):
    input_folder, output_folder, datagen, original_folder_name = args

    # List all images in the folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    if not image_files:
        print(f"No images found in folder {input_folder}. Skipping...")
        return

    for aug_index in range(augmented_folder_count):
        # Random transformation parameters
        params = (
            random.randint(-15, 15),
            random.uniform(-0.1, 0.1),
            random.uniform(-0.1, 0.1),
            random.uniform(0.9, 1.1),
            random.choice([True, False])
        )

        # Create output folder
        output_aug_folder = os.path.join(output_folder, f"{original_folder_name}_aug_{aug_index + 1}")
        os.makedirs(output_aug_folder, exist_ok=True)

        # Process each image
        for i, image_file in enumerate(image_files):
            img_path = os.path.join(input_folder, image_file)
            augmented_img = augment_image(img_path, datagen, params)
            aug_img_path = os.path.join(output_aug_folder, f"aug_{i + 1}.jpg")
            save_img(aug_img_path, augmented_img)

        print(f"Augmented folder saved at: {output_aug_folder}")

# Prepare arguments for parallel processing
folders = [
    (os.path.join(root_dir, folder), root_dir, datagen, folder)
    for folder in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, folder))
]

# Process folders in parallel
if __name__ == "__main__":
    print("Starting parallel augmentation...")
    process_count = min(len(folders), cpu_count())  # Use half the CPU cores
    with Pool(processes=process_count) as pool:
        pool.map(augment_folder, folders)
    print("Augmentation completed.")