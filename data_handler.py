import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_folder, dymension=None):
    images = []
    labels = []
    label_to_index = {}
    index_counter = 0

    class_folders = sorted(os.listdir(data_folder))

    for label_name in class_folders:
        if label_name.startswith('.'):
            continue

        label_folder = os.path.join(data_folder, label_name)
        if not os.path.isdir(label_folder):
            continue

        label_to_index[label_name] = index_counter
        for filename in os.listdir(label_folder):
            img_path = os.path.join(label_folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                if dymension is not None:
                    img = img.resize((dymension, dymension))
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(index_counter)
            except Exception as e:
                print(f"Error loading image: {img_path} - {e}")
        index_counter += 1

    return np.array(images), np.array(labels), label_to_index

def split_data(X, y, test_train_ratio):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_train_ratio, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, y_train, X_test, y_test, transform, batch_size):
    train_dataset = ImageDataset(X_train, y_train, transform=transform)
    test_dataset = ImageDataset(X_test, y_test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def display_class_distribution(y, label_to_index, title):
    plt.figure(figsize=(8, 5))
    class_names = list(label_to_index.keys())
    class_counts = np.bincount(y)
    plt.bar(class_names, class_counts)
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=80, ha='right')
    plt.tight_layout()
    plt.show()

def display_images(images, labels, label_to_index, title, num_examples_per_class=1):
    plt.figure(figsize=(15, 8))
    unique_labels_indices = sorted(np.unique(labels))
    class_names = [k for k, v in sorted(label_to_index.items(), key=lambda item: item[1])]

    for i, class_idx in enumerate(unique_labels_indices):
        class_samples_indices = np.where(labels == class_idx)[0]
        for j in range(min(num_examples_per_class, len(class_samples_indices))):
            plt.subplot(num_examples_per_class, len(unique_labels_indices), j * len(unique_labels_indices) + i + 1)
            img_index = class_samples_indices[j]
            display_img = images[img_index]
            plt.imshow(display_img)
            if j == 0:
                plt.title(f"Class: {class_names[class_idx]}", fontsize=10)
            plt.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
