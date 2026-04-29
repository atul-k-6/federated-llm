import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import json
from common.model import SimpleCNN

# Define the training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)

# Define the validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    per_class_correct = [0] * 10
    per_class_total = [0] * 10
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Per-class accuracy
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    per_class_correct[label] += 1
                per_class_total[label] += 1

    accuracy = 100 * correct / total
    per_class_accuracy = {
        f"class_{i}": 100 * per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0
        for i in range(10)
    }
    return val_loss / len(val_loader), accuracy, per_class_accuracy

# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset from `data_raw`
    data_path = "./data_raw"
    full_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

    # Split dataset into training, validation, and test sets
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)  # 70% for training
    val_size = int(0.2 * dataset_size)    # 20% for validation
    test_size = dataset_size - train_size - val_size  # Remaining 10% for testing
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create metrics directory
    metrics_dir = "./metrics/centralized"
    os.makedirs(metrics_dir, exist_ok=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(1, 101):  # Maximum of 100 epochs
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, per_class_accuracy = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

        # Save metrics to JSON
        metrics = {
            "round": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": val_accuracy,
            "per_class_accuracy": per_class_accuracy
        }
        with open(f"{metrics_dir}/round_{epoch}.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Test the model
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_accuracy, _ = validate(model, test_loader, criterion, device)
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()