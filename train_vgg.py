import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from utils import load_dataset, get_device, evaluate_model, save_model

# ====================== INSTÄLLNINGAR ======================
DATASET_PATH = "dataset"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# ====================== LADDA DATA ======================
train_loader, val_loader, test_loader, class_names = load_dataset(
    DATASET_PATH,
    BATCH_SIZE
)

# ====================== MODELL ======================
device = get_device()

# Ladda förtränad VGG16
model = models.vgg16(pretrained=True)

# Frys alla lager utom sista
for param in model.parameters():
    param.requires_grad = False

# Byt ut sista lagret för våra klasser
num_classes = len(class_names)
model.classifier[6] = nn.Linear(4096, num_classes)

model = model.to(device)

print("\n===== VGG16 =====")
print(f"Antal klasser: {num_classes}")
print(f"Klasser: {class_names}")

# ====================== TRÄNING ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.classifier[6].parameters(),
    lr=LEARNING_RATE
)

best_val_acc = 0.0

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

    # Spara bästa modellen
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_model(model, "vgg16_best.pth")

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"Loss: {running_loss/len(train_loader):.4f} — "
          f"Train Acc: {train_acc:.2f}% — "
          f"Val Acc: {val_acc:.2f}%")

# ====================== UTVÄRDERING ======================
print(f"\nBästa valideringsnoggranhet: {best_val_acc:.2f}%")
evaluate_model(model, test_loader, class_names, device)