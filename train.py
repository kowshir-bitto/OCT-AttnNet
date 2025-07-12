import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in train_loader:
            inputs = inputs
            labels = labels

            optimizer.zero_grad()

            with torch.amp.autocast():
                outputs = model(inputs)

                if isinstance(outputs, tuple):
                    logits, aux_logits = outputs
                    loss = criterion(logits, labels) + 0.4 * criterion(aux_logits, labels)
                    _, predicted = torch.max(logits, 1)
                else:
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            with torch.amp.autocast():
                for val_inputs, val_labels in test_loader:
                    val_inputs = val_inputs
                    val_labels = val_labels
                    val_outputs = model(val_inputs)

                    if isinstance(val_outputs, tuple):
                        val_outputs = val_outputs[0]

                    val_loss = criterion(val_outputs, val_labels)

                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    _, val_predicted = torch.max(val_outputs, 1)
                    val_total_samples += val_labels.size(0)
                    val_correct_predictions += (val_predicted == val_labels).sum().item()

        epoch_val_loss = val_running_loss / len(test_loader.dataset)
        epoch_val_accuracy = val_correct_predictions / val_total_samples
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies
