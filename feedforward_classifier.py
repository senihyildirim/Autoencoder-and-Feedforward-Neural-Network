import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 1. Dataset Preparation
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 2. Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward(self, x):
        return self.net(x)

model = FeedforwardNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training
train_losses = []
train_accuracies = []

for epoch in range(50):
    model.train()
    total, correct, loss_total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = correct / total
    train_losses.append(loss_total / len(train_loader))
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1}/50 - Loss: {train_losses[-1]:.4f} - Accuracy: {acc:.4f}")

# 4. Evaluation
model.eval()
all_preds, all_labels = [], []
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

        # Misclassified örnekler
        for img, pred, true in zip(images.cpu(), preds.cpu(), labels):
            if pred != true:
                misclassified.append((img, pred.item(), true.item()))

accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"\nTest Accuracy: {accuracy:.4f}")

# 5. Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 6. Optional: Misclassified Samples
def show_misclassified(misclassified, num=10):
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    plt.figure(figsize=(15, 4))
    for i, (img, pred, true) in enumerate(misclassified[:num]):
        plt.subplot(1, num, i + 1)
        plt.imshow(img.squeeze(), cmap="gray")
        pred_label = class_names[pred]
        true_label = class_names[true]
        plt.title(f"Predicted: {pred_label}\nTruth: {true_label}", fontsize=9)
        plt.axis("off")
    plt.suptitle("Misclassified Examples", fontsize=12)
    plt.tight_layout()
    plt.show()


show_misclassified(misclassified)
