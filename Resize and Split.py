import os
from PIL import Image
import shutil
import math
import sys
import random
sys.path.append('../')
from config import (
    TEST_TRAIN_RATIO, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, RAW_DATA_FOLDER,
    MODEL_CONFIGS, DEVICE, TRANSFORMS, proportions
)

def resize_images_in_folder(input_folder, output_folder, resize_height, resize_width):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder for current processing '{input_folder}' does not exist. Skipping.")
        return 0, 0

    os.makedirs(output_folder, exist_ok=True)

    total_images_processed = 0
    total_items_skipped = 0 

    try:
        class_folders = sorted([f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))])
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found. Cannot list directories.")
        return 0, 0
    
    if not class_folders:
        print(f"Warning: No class subfolders found in '{input_folder}'.")
        return 0, 0

    for class_name in class_folders:
        class_input_path = os.path.join(input_folder, class_name)
        class_output_path = os.path.join(output_folder, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        images_in_current_class = 0
        items_skipped_in_current_class = 0

        try:
            image_filenames = sorted(os.listdir(class_input_path))
        except FileNotFoundError:
            print(f"Error: Class folder '{class_input_path}' not found. Skipping.")
            items_skipped_in_current_class += len(os.listdir(class_input_path) if os.path.exists(class_input_path) else [])
            continue

        for image_filename in image_filenames:
            if image_filename.startswith('.') or not any(image_filename.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                items_skipped_in_current_class += 1
                continue

            img_input_path = os.path.join(class_input_path, image_filename)
            
            if not os.path.isfile(img_input_path):
                items_skipped_in_current_class += 1
                continue
            
            try:
                with Image.open(img_input_path) as img:
                    resized_img = img.resize((resize_width, resize_height), Image.LANCZOS)
                    img_output_path = os.path.join(class_output_path, image_filename)
                    resized_img.save(img_output_path)
                images_in_current_class += 1

            except Exception:
                items_skipped_in_current_class += 1
        
        total_images_processed += images_in_current_class
        total_items_skipped += items_skipped_in_current_class

    return total_images_processed, total_items_skipped


base_input_folder = "../Datasets/3. Balanced/"
base_output_folder = "../Datasets/4. Resized/"

target_sizes = [224, 299]

print(f"Starting overall image resizing process for input: '{base_input_folder}'")
print(f"Output will be saved to: '{base_output_folder}'")

try:
    split_folders = [f for f in os.listdir(base_input_folder) if os.path.isdir(os.path.join(base_input_folder, f))]
    split_folders.sort()
except FileNotFoundError:
    print(f"Error: Base input folder '{base_input_folder}' not found. Please check the path.")
    split_folders = []

if not split_folders:
    print(f"No split folders (e.g., 'train', 'validate', 'test') found in '{base_input_folder}'. Exiting.")
else:
    print(f"\nFound {len(split_folders)} split folders: {', '.join(split_folders)}")

    for current_resize_size in target_sizes:
        print(f"\n{'='*50}")
        print(f"--- Resizing to {current_resize_size}x{current_resize_size} ---")
        print(f"{'='*50}")

        current_output_base_for_size = os.path.join(base_output_folder, str(current_resize_size))
        os.makedirs(current_output_base_for_size, exist_ok=True)
        print(f"Created output directory for size {current_resize_size}: {current_output_base_for_size}")
        
        if os.path.exists(current_output_base_for_size):
            for item in os.listdir(current_output_base_for_size):
                item_path = os.path.join(current_output_base_for_size, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        print(f"Cleared contents of {current_output_base_for_size} (if any).")

        total_processed_for_size = 0
        total_skipped_for_size = 0

        for split_name in split_folders:
            input_split_path = os.path.join(base_input_folder, split_name)
            output_split_path = os.path.join(current_output_base_for_size, split_name)

            print(f"\nProcessing split: '{split_name}' (Input: {input_split_path}, Output: {output_split_path})")
            
            processed, skipped = resize_images_in_folder(
                input_split_path,
                output_split_path,
                current_resize_size,
                current_resize_size
            )
            total_processed_for_size += processed
            total_skipped_for_size += skipped
            print(f"Split '{split_name}' completed. Processed {processed} images, Skipped {skipped} items.")

        print(f"\n--- Summary for {current_resize_size}x{current_resize_size} ---")
        print(f"Total images processed for this size: {total_processed_for_size}")
        print(f"Total items skipped for this size: {total_skipped_for_size}")

print(f"\n{'#'*60}")
print(f"Overall image resizing process finished for all target sizes.")
print(f"{'#'*60}")


input_folder = '../Datasets/1. Enhanched/'
output_folder = '../Datasets/2. Split/'

total_proportion = sum(proportions.values())
if not math.isclose(total_proportion, 1.0):
    print(f"Warning: Proportions do not sum to 1.0 (Current sum: {total_proportion}). This might lead to unexpected distribution.")

print("Configuration loaded successfully!")
print(f"Input Folder: {input_folder}")
print(f"Output Folder: {output_folder}")
print(f"Proportions: {proportions}")

os.makedirs(output_folder, exist_ok=True)
print(f"Created main output folder: {output_folder}")

for proportion_name in proportions.keys():
    proportion_path = os.path.join(output_folder, proportion_name)
    os.makedirs(proportion_path, exist_ok=True)
    print(f"Created subfolder: {proportion_path}")

print("\nOutput folder structure created.")

class_folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

if not class_folders:
    print(f"\nNo class folders found in '{input_folder}'. Please ensure your input folder contains subfolders for each class.")
else:
    print(f"\nFound {len(class_folders)} class(es) in '{input_folder}': {', '.join(class_folders)}")

    for class_name in class_folders:
        class_input_path = os.path.join(input_folder, class_name)
        images = [f for f in os.listdir(class_input_path) if os.path.isfile(os.path.join(class_input_path, f))]
        num_images = len(images)

        if num_images == 0:
            print(f"Skipping class '{class_name}': No images found.")
            continue

        print(f"\nProcessing class: '{class_name}' ({num_images} images)")

        for proportion_name in proportions.keys():
            class_output_path = os.path.join(output_folder, proportion_name, class_name)
            os.makedirs(class_output_path, exist_ok=True)

        random.shuffle(images)

        current_index = 0
        for proportion_name, ratio in proportions.items():
            num_to_copy = math.floor(num_images * ratio)
            if proportion_name == list(proportions.keys())[-1]:
                num_to_copy = num_images - current_index

            images_to_distribute = images[current_index : current_index + num_to_copy]
            target_class_path = os.path.join(output_folder, proportion_name, class_name)
            
            print(f"  Distributing {len(images_to_distribute)} images to '{proportion_name}' folder.")

            for image_file in images_to_distribute:
                source_path = os.path.join(class_input_path, image_file)
                destination_path = os.path.join(target_class_path, image_file)
                shutil.copy(source_path, destination_path)

            current_index += num_to_copy

    print("\nImage distribution complete!")
    print("Check your 'output_dataset' folder to see the results.")
