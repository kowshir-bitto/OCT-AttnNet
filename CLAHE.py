import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

input_folder_name = "../Datasets/0. Raw/"
output_folder_name = "../Datasets/1. Enhanched/"

os.makedirs(output_folder_name, exist_ok=True)

def apply_clahe(img_path):
    img_pil = Image.open(img_path).convert('RGB')
    img_cv = np.array(img_pil)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_img_cv = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_img_cv)

before_after_examples = {}

print(f"Processing images from '{input_folder_name}' and saving to '{output_folder_name}'...\n")

class_folders = [d for d in os.listdir(input_folder_name) if os.path.isdir(os.path.join(input_folder_name, d))]

if not class_folders:
    print(f"No class folders found in '{input_folder_name}'.")
    print("Please ensure your 'raw images' folder exists in the same directory as this script and contains subfolders, where each subfolder represents a class and contains images.")
else:
    for class_name in sorted(class_folders):
        class_input_path = os.path.join(input_folder_name, class_name)
        class_output_path = os.path.join(output_folder_name, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        first_image_processed = False
        image_files_in_class = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if not image_files_in_class:
            print(f"  Skipping class '{class_name}': No image files found.")
            continue
        print(f"  Processing class: {class_name}")
        for filename in image_files_in_class:
            img_path = os.path.join(class_input_path, filename)
            enhanced_image = apply_clahe(img_path)
            enhanced_image.save(os.path.join(class_output_path, filename))
            if not first_image_processed:
                original_img = Image.open(img_path).convert('RGB')
                before_after_examples[class_name] = (original_img, enhanced_image)
                first_image_processed = True

    print("\n--- Before and After CLAHE Examples for Each Class ---")
    if not before_after_examples:
        print("No examples to show. This might happen if no images were processed.")
    else:
        for class_name in sorted(before_after_examples.keys()):
            original_img, enhanced_img = before_after_examples[class_name]
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img)
            plt.title(f'{class_name} - Original')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(enhanced_img)
            plt.title(f'{class_name} - CLAHE Enhanced')
            plt.axis('off')
            plt.show()

print(f"\nAll CLAHE processed images are saved in the '{output_folder_name}' directory.")
