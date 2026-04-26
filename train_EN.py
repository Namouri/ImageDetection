import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils import load_dataset, get_device, evaluate_model, save_model
import matplotlib.pyplot as plt

# ====================== INSTÄLLNINGAR ======================
DATASET_PATH = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001

# ====================== LADDA DATA ======================
train_loader, val_loader, test_loader, class_names = load_dataset(
    DATASET_PATH,
    BATCH_SIZE
)

# ====================== MODELL ======================
device = get_device()

# Ladda förtränad EfficientNet-B0
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Byt ut sista lagret för våra klasser
num_classes = len(class_names)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(device)

print("\n===== EfficientNet-B0 =====")
print(f"Antal klasser: {num_classes}")
print(f"Klasser: {class_names}")

# ====================== TRÄNING ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.1
)

best_val_acc = 0.0
train_accs = []
val_accs = []
losses = []

for epoch in range(EPOCHS):
    # Träningsloop
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # Valideringsloop
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    losses.append(running_loss/len(train_loader))

    # Spara bästa modellen
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model, "efficientnet_best.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"Loss: {running_loss/len(train_loader):.4f} — "
          f"Train Acc: {train_acc:.2f}% — "
          f"Val Acc: {val_acc:.2f}%")
    scheduler.step()
    
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('EfficientNet - Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('EfficientNet - Loss')
plt.legend()

plt.tight_layout()
plt.savefig('efficientnet_training.png')
plt.close()
print("Graf sparad som efficientnet_training.png")

# ====================== UTVÄRDERING ======================
print(f"\nBästa valideringsnoggranhet: {best_val_acc:.2f}%")
evaluate_model(model, test_loader, class_names, device)