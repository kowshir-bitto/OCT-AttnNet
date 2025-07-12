selected_model = "ResNet50"

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

if selected_model not in MODEL_CONFIGS:
    raise ValueError(f"Model '{selected_model}' not found in MODEL_CONFIGS.")
model_config = MODEL_CONFIGS[selected_model]
current_dymension = model_config["input_size"]

X, y, label_to_index = load_data(DATA_FOLDER, current_dymension)
class_names = [k for k, v in sorted(label_to_index.items(), key=lambda item: item[1])]
num_classes = len(label_to_index)
display_class_distribution(y, label_to_index, "Class Distribution Before Splitting")

X_train, X_test, y_train, y_test = split_data(X, y, TEST_TRAIN_RATIO)
train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, TRANSFORMS, BATCH_SIZE)

model = build_model(model_config, num_classes)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, test_loader, criterion, optimizer, NUM_EPOCHS
)

test_loss, test_accuracy, y_test_np, y_pred_labels, per_class_metrics = evaluate_model_performance(
    model, test_loader, criterion, num_classes, class_names
)

plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses)
classification_report(y_test_np, y_pred_labels, target_names=class_names, zero_division=0)
plot_confusion_matrix(y_test_np, y_pred_labels, class_names)

for class_name, metrics in per_class_metrics.items():
    for metric_name, value in metrics.items():
        pass

model_save_file = model_config["output_model_name"]
json_output_file = model_config["output_json_name"]

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
