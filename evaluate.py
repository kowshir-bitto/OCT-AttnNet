import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

def evaluate_model_performance(model, test_loader, criterion, num_classes, class_names):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs
            labels = labels
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    per_class_metrics = {}
    total_samples = len(y_true)

    for i, class_name in enumerate(class_names):
        TP = np.sum((y_true == i) & (y_pred == i))
        TN = np.sum((y_true != i) & (y_pred != i))
        FP = np.sum((y_true != i) & (y_pred == i))
        FN = np.sum((y_true == i) & (y_pred != i))
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        TNR = TN / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (TP + FN) if (TP + FN) > 0 else 0
        accuracy_per_class = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        per_class_metrics[class_name] = {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'TPR_Recall': float(f'{TPR:.4f}'),
            'FPR': float(f'{FPR:.4f}'),
            'TNR_Specificity': float(f'{TNR:.4f}'),
            'FNR': float(f'{FNR:.4f}'),
            'Accuracy': float(f'{accuracy_per_class:.4f}')
        }

    return test_loss, test_accuracy, y_true, y_pred, per_class_metrics

def plot_training_history(train_accuracies, val_accuracies, train_losses, val_losses):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def save_results(model, model_save_path, json_output_path, training_outputs):
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    with open(json_output_path, 'w') as f:
        json.dump(training_outputs, f, indent=4)
    print(f"Training outputs saved as JSON to: {json_output_path}")
