selected_model = "InceptionV3"
custom_title = 'BAM-ECA'

import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import sys
sys.path.append('../')
from config import (
    TEST_TRAIN_RATIO, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, DATA_FOLDER,
    MODEL_CONFIGS, DEVICE, TRANSFORMS
)
from Functions.data_handler import (
    load_data, split_data, create_dataloaders,
    display_class_distribution, display_images
)
from Functions.model_builder import build_model
from Functions.train import train_model
from Functions.evaluate import (
    evaluate_model_performance, plot_training_history, plot_confusion_matrix,
    save_results
)
from Functions.xai_explainer import (
    denormalize_image, preprocess_image_for_xai,
    explain_integrated_gradients, explain_gradcam,
    explain_gradcam_plus_plus, explain_shap
)

if selected_model not in MODEL_CONFIGS:
    raise ValueError(f"Model '{selected_model}' not found in MODEL_CONFIGS. Please add its configuration in config.py.")
model_config = MODEL_CONFIGS[selected_model]
current_dymension = model_config["input_size"]

print("\n--- Loading Data ---")
X, y, label_to_index = load_data(DATA_FOLDER + str(current_dymension) + "/mixed")
class_names = [k for k, v in sorted(label_to_index.items(), key=lambda item: item[1])]
num_classes = len(label_to_index)
print(f"Found {num_classes} classes: {class_names}")

print("Displaying original class distribution...")
display_class_distribution(y, label_to_index, "Class Distribution Before Splitting")

print("Splitting Data into Test Train ratio: " + str(TEST_TRAIN_RATIO))
X_train, X_test, y_train, y_test = split_data(X, y, TEST_TRAIN_RATIO)

print("Creating PyTorch DataLoaders...")
train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, TRANSFORMS, BATCH_SIZE)

print(f"\n--- Building Model: {selected_model} ---")
model = build_model(model_config, num_classes)
model.to(DEVICE)
print(f"Model will run on: {DEVICE}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\n--- Starting Training for {selected_model} ---")
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS
)

print(f"\n--- Evaluating Model: {selected_model} ---")
test_loss, test_accuracy, y_test_np, y_pred_labels, per_class_metrics = evaluate_model_performance(
    model, test_loader, criterion, num_classes, class_names
)

print("\n--- Generating Plots ---")
plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses)
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred_labels, target_names=class_names, zero_division=0))
plot_confusion_matrix(y_test_np, y_pred_labels, class_names)

print("\n--- Per-Class Metrics ---")
for class_name, metrics in per_class_metrics.items():
    print(f"Class: {class_name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value}")
    print("-" * 30)

model_save_file = custom_title + '-' + model_config["output_model_name"]
json_output_file = custom_title + '-' + model_config["output_json_name"] + '-' + custom_title

training_outputs = {
    'model_name': selected_model,
    'epochs': list(range(1, NUM_EPOCHS + 1)),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'classification_report': classification_report(y_test_np, y_pred_labels, target_names=class_names, output_dict=True, zero_division=0),
    'confusion_matrix': confusion_matrix(y_test_np, y_pred_labels).tolist(),
    'label_to_index': label_to_index,
    'index_to_label': {v: k for k, v in label_to_index.items()},
    'per_class_metrics': per_class_metrics
}
save_results(model, model_save_file, json_output_file, training_outputs)
print("\nProcess Completed!")

print("\n--- Generating XAI Explanations ---")
model.eval()
images_by_class = {class_idx: [] for class_idx in range(len(class_names))}
for i in range(len(X_test)):
    images_by_class[y_test[i]].append((X_test[i], y_test[i]))

for class_idx, images_in_class in images_by_class.items():
    if len(images_in_class) > 0:
        print(f"\n\n--- Explaining XAI for Class: {class_names[class_idx]} ---")
        original_image_np, true_label_idx = random.choice(images_in_class)
        input_tensor = preprocess_image_for_xai(original_image_np, TRANSFORMS)
        normalize_transform = TRANSFORMS.transforms[-1]
        mean_val = normalize_transform.mean
        std_val = normalize_transform.std
        denormalized_image_tensor = denormalize_image(input_tensor.cpu(), mean_val, std_val)
        denormalized_image_np = denormalized_image_tensor.squeeze(0).permute(1, 2, 0).numpy()

        with torch.no_grad():
            output = model(input_tensor.to(DEVICE))
            if isinstance(output, tuple):
                output = output[0]
            predicted_label_idx = torch.argmax(output).item()
        print(f"Model predicted class: {class_names[predicted_label_idx]}")

        print("Generating Integrated Gradients...")
        explain_integrated_gradients(model, input_tensor.to(DEVICE), predicted_label_idx, denormalized_image_np)

        print("Generating GradCAM explanation...")
        explain_gradcam(model, input_tensor.to(DEVICE), predicted_label_idx, denormalized_image_np, selected_model)

        print("Generating GradCAM++ explanation...")
        explain_gradcam_plus_plus(model, input_tensor.to(DEVICE), predicted_label_idx, denormalized_image_np, selected_model)
    else:
        print(f"\nNo test data available for class: {class_names[class_idx]}. Skipping XAI explanations for this class.")

print("\n--- XAI Explanation Generation Complete ---")
