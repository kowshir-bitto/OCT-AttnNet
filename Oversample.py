import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

augmentation_ops = v2.Compose([
    v2.RandomHorizontalFlip(p=1.0),
    v2.RandomVerticalFlip(p=0.4),
    v2.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05), saturation=(0.95, 1.05), hue=(-0.03, 0.03)),
])

input_base_folder = '../Datasets/2. Split/'
output_base_folder = "../Datasets/3. Balanced/"

if os.path.exists(output_base_folder):
    shutil.rmtree(output_base_folder)
os.makedirs(output_base_folder, exist_ok=True)
print(f"Cleared and created main output folder: {output_base_folder}")

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

def apply_random_augmentations(image_pil):
    image_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])(image_pil)
    image_tensor = image_tensor.to(device)
    augmented_tensor = augmentation_ops(image_tensor)
    augmented_tensor = v2.ToDtype(torch.uint8, scale=True)(augmented_tensor.cpu())
    augmented_img_pil = T.ToPILImage()(augmented_tensor)
    return augmented_img_pil

def get_class_distribution(base_folder):
    distribution = {}
    class_paths = {}
    for class_name in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_name)
        if os.path.isdir(class_path):
            images_in_class = [f for f in os.listdir(class_path) if f.lower().endswith(IMAGE_EXTENSIONS)]
            distribution[class_name] = len(images_in_class)
            class_paths[class_name] = [os.path.join(class_path, f) for f in images_in_class]
    return distribution, class_paths

def plot_pie_chart(distribution, title):
    if not distribution:
        print(f"No data to plot for pie chart: {title}")
        return
    
    distribution = dict(sorted(distribution.items(), key=lambda item: item[1], reverse=True))
    labels = list(distribution.keys())
    sizes = list(distribution.values())
    total_images = sum(sizes)
    formatted_labels = [f"{label}: {count} ({(count / total_images) * 100:.1f}%)" for label, count in zip(labels, sizes)]
    colors = plt.cm.Paired(np.arange(len(labels)))
    startangle = 0
    fig, ax = plt.subplots(figsize=(13, 13))
    wedges, _ = ax.pie(sizes, startangle=startangle, colors=colors, radius=1.0)

    for i, (wedge, label) in enumerate(zip(wedges, formatted_labels)):
        angle = (wedge.theta2 + wedge.theta1) / 2
        angle_rad = np.deg2rad(angle)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        label_x = 1.4 * x
        label_y = 1.4 * y
        horizontalalignment = 'left' if x > 0 else 'right'
        ax.text(label_x, label_y, label, ha=horizontalalignment, va='center', fontsize=15)
        ax.plot([0.95 * x, label_x], [0.95 * y, label_y], color=colors[i], lw=1.5)

    ax.set(aspect='equal')
    plt.show()

print("Utility functions and GPU-aware augmentations defined.")

split_folders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]

if not split_folders:
    print(f"Error: No split folders (e.g., 'train', 'test') found in '{input_base_folder}'.")
else:
    print(f"\nFound {len(split_folders)} split folders: {', '.join(split_folders)}")

    for split_name in split_folders:
        print(f"\n{'='*50}")
        print(f"--- Processing Split: '{split_name}' ---")
        print(f"{'='*50}")

        current_input_folder = os.path.join(input_base_folder, split_name)
        current_output_folder = os.path.join(output_base_folder, split_name)
        
        os.makedirs(current_output_folder, exist_ok=True)
        print(f"Created output folder for '{split_name}': {current_output_folder}")

        original_distribution, original_image_paths = get_class_distribution(current_input_folder)

        if not original_distribution:
            print(f"Error: No class folders or images found in '{current_input_folder}'. Skipping this split.")
            continue

        total_original_images = sum(original_distribution.values())
        print(f"Initial Class Counts for '{split_name}':")
        for cls, count in original_distribution.items():
            percentage = (count / total_original_images) * 100 if total_original_images > 0 else 0
            print(f"   {cls}: {count} images ({percentage:.1f}%)")

        print(f"\n--- Displaying Initial Class Distribution for '{split_name}' ---")
        plot_pie_chart(original_distribution, f"Initial Class Distribution ({split_name})")

        print(f"\n--- Copying Original Images to Output Folder for '{split_name}' ---")
        for class_name, paths in original_image_paths.items():
            output_class_path = os.path.join(current_output_folder, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            for img_path in paths:
                try:
                    shutil.copy(img_path, output_class_path)
                except Exception as e:
                    print(f"    Error copying {img_path}: {e}")
        print(f"Original images copied for '{split_name}'.")

        print(f"\n--- Starting Image Augmentation for Minority Classes in '{split_name}' (Batching Enabled) ---")

        max_images = max(original_distribution.values()) if original_distribution else 0
        if max_images == 0:
            print(f"No classes found in '{split_name}' to augment.")
            continue

        augmented_examples_for_split = {}
        BATCH_SIZE = 32

        for class_name, current_count in original_distribution.items():
            if current_count < max_images:
                images_to_add = max_images - current_count
                print(f"  Class '{class_name}': Needs {images_to_add} augmented images to reach {max_images}.")
                
                class_original_paths = original_image_paths.get(class_name, [])
                if not class_original_paths:
                    print(f"    Warning: No original images found for class '{class_name}' in '{split_name}'. Cannot augment.")
                    continue

                output_class_path = os.path.join(current_output_folder, class_name)
                selected_images_for_augmentation = np.random.choice(class_original_paths, size=images_to_add, replace=True)
                
                example_original_img_pil = None
                example_augmented_img_pil = None

                for i in range(0, images_to_add, BATCH_SIZE):
                    batch_paths = selected_images_for_augmentation[i : i + BATCH_SIZE]
                    batch_pil_images = [Image.open(path).convert('RGB') for path in batch_paths]
                    image_tensors_cpu = [v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img) for img in batch_pil_images]
                    batch_tensor_gpu = torch.stack(image_tensors_cpu).to(device)
                    augmented_batch_gpu = augmentation_ops(batch_tensor_gpu)
                    augmented_batch_cpu = augmented_batch_gpu.cpu()
                    
                    for j, augmented_single_tensor_cpu in enumerate(augmented_batch_cpu):
                        original_img_idx = i + j
                        if original_img_idx >= images_to_add:
                            break
                        augmented_img_pil = T.ToPILImage()(v2.ToDtype(torch.uint8, scale=True)(augmented_single_tensor_cpu))
                        original_filename = os.path.basename(batch_paths[j])
                        name, ext = os.path.splitext(original_filename)
                        augmented_filename = f"{name}_aug_{i+j+1}{ext}"
                        augmented_img_pil.save(os.path.join(output_class_path, augmented_filename))

                        if example_original_img_pil is None:
                            example_original_img_pil = batch_pil_images[j]
                            example_augmented_img_pil = augmented_img_pil
                            augmented_examples_for_split[class_name] = (example_original_img_pil, example_augmented_img_pil)

                    del batch_pil_images, image_tensors_cpu, batch_tensor_gpu, augmented_batch_gpu, augmented_batch_cpu
                    torch.cuda.empty_cache()
            else:
                print(f"  Class '{class_name}': Already at max count ({max_images}). No augmentation needed.")

        print(f"\nImage augmentation complete for '{split_name}'.")

        print(f"\n--- Calculating Final Class Distribution for '{split_name}' ---")
        final_distribution, _ = get_class_distribution(current_output_folder)
        total_final_images = sum(final_distribution.values())

        print(f"Final Class Counts for '{split_name}':")
        for cls, count in final_distribution.items():
            percentage = (count / total_final_images) * 100 if total_final_images > 0 else 0
            print(f"   {cls}: {count} images ({percentage:.1f}%)")

        print(f"\n--- Displaying Final Class Distribution (After Augmentation) for '{split_name}' ---")
        plot_pie_chart(final_distribution, f"Final Class Distribution (After Augmentation) for {split_name}")

        print(f"\n--- Examples of Augmented Images (Original vs. Augmented) for '{split_name}' ---")
        if not augmented_examples_for_split:
            print(f"No augmented image examples to show for '{split_name}'. This might be because all classes were already balanced or no images were found.")
        else:
            for class_name in sorted(augmented_examples_for_split.keys()):
                original, augmented = augmented_examples_for_split[class_name]
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(original)
                plt.title(f'{class_name} - Original Image')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(augmented)
                plt.title(f'{class_name} - Augmented Version')
                plt.axis('off')
                plt.show()

print("\nAll splits processed. Check your 'output_dataset' folder structure.")