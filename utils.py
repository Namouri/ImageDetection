import torch
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from augmentation import get_train_transform, get_val_transform
import copy

# Ladda dataset


def load_dataset(dataset_path, batch_size=32):
    # funktionen laddar bilderna från mapparna och delar de i träning(80%)
    # validering (10%) och testing (10%)
    # 1. ladda hela dataset med augmentering
    full_dataset = datasets.ImageFolder(
        root=dataset_path, transform=get_train_transform()
    )

    print(f"Classes found: {full_dataset.classes}")
    print(f"Total number of images: {len(full_dataset)}")

    # 2. Dela upp datasetet
    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Använd val_transform för validering och test
    val_dataset_copy = copy.deepcopy(val_dataset)
    test_dataset_copy = copy.deepcopy(test_dataset)
    val_dataset_copy.dataset.transform = get_val_transform()
    test_dataset_copy.dataset.transform = get_val_transform()

    # Skapa dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_copy, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_copy, batch_size=batch_size, shuffle=False)

    print(f"Träning:   {train_size} bilder")
    print(f"Validering: {val_size} bilder")
    print(f"Test:       {test_size} bilder")

    return train_loader, val_loader, test_loader, full_dataset.classes


# ====================== VÄLJ ENHET ======================
def get_device():
    """
    Väljer automatiskt bästa tillgängliga hårdvara:
    - MPS  → Mac med Apple Silicon (snabbast på Mac)
    - CUDA → NVIDIA GPU (snabbast generellt)
    - CPU  → om inget annat finns
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Kör på: Mac MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Kör på: NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Kör på: CPU")
    return device


# ====================== UTVÄRDERING ======================
def evaluate_model(model, test_loader, class_names, device):
    """
    Utvärderar modellen på testdatan och skriver ut:
    - Precision, Recall, F1-score per klass
    - Confusion matrix
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n===== UTVÄRDERINGSRESULTAT =====")
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_predictions))

    return all_predictions, all_labels


# ====================== SPARA MODELL ======================
def save_model(model, filename):
    """
    Sparar modellens vikter till en fil
    """
    torch.save(model.state_dict(), filename)
    print(f"Modellen sparad som: {filename}")
