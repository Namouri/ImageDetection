import torch
import torch.nn as nn
import torchvision.models as models
from utils import load_dataset, get_device, evaluate_model
import time

# ====================== INSTÄLLNINGAR ======================
DATASET_PATH = "dataset"
BATCH_SIZE = 32

# ====================== LADDA DATA ======================
_, _, test_loader, class_names = load_dataset(DATASET_PATH, BATCH_SIZE)
device = get_device()
num_classes = len(class_names)

# ====================== HJÄLPFUNKTION ======================
def measure_inference_time(model, test_loader, device):
    """
    Mäter hur lång tid modellen tar på sig per bild i millisekunder
    """
    model.eval()
    start = time.time()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
    end = time.time()
    total_images = len(test_loader.dataset)
    ms_per_image = ((end - start) / total_images) * 1000
    return round(ms_per_image, 2)

# ====================== LADDA MODELLER ======================
print("\nLaddar modeller...")

# VGG16
vgg = models.vgg16()
vgg.classifier[6] = nn.Linear(4096, num_classes)
vgg.load_state_dict(torch.load("vgg16_best.pth", map_location=device))
vgg = vgg.to(device)
print("VGG16 laddad")

# EfficientNet-B0
eff = models.efficientnet_b0()
eff.classifier[1] = nn.Linear(eff.classifier[1].in_features, num_classes)
eff.load_state_dict(torch.load("efficientnet_best.pth", map_location=device))
eff = eff.to(device)
print("EfficientNet-B0 laddad")

# ResNet-18
res = models.resnet50()
res.fc = nn.Linear(res.fc.in_features, num_classes)
res.load_state_dict(torch.load("best_resnet_finetune.pth", map_location=device))
res = res.to(device)
print("ResNet-50 laddad")

# MobileNetV3
mob = models.mobilenet_v2()
mob.classifier[1] = nn.Linear(mob.classifier[1].in_features, num_classes)
mob.load_state_dict(torch.load("best_mobilenet_v2.pth", map_location=device))
mob = mob.to(device)
print("MobileNetV2 laddad")

# ====================== UTVÄRDERA ALLA MODELLER ======================
modeller = {
    "VGG16": vgg,
    "EfficientNet-B0": eff,
    "ResNet-50": res,
    "MobileNetV2": mob
}

resultat = {}

for namn, modell in modeller.items():
    print(f"\n{'='*40}")
    print(f"Utvärderar {namn}...")
    print(f"{'='*40}")

    # Utvärdera
    predictions, labels = evaluate_model(
        modell, test_loader, class_names, device
    )

    # Mät inferenstid
    ms = measure_inference_time(modell, test_loader, device)

    resultat[namn] = ms
    print(f"Inferenstid: {ms} ms per bild")

# ====================== SAMMANFATTNING ======================
print("\n")
print("="*60)
print("SAMMANFATTNING — JÄMFÖRELSE AV MODELLER")
print("="*60)
print(f"{'Modell':<20} {'Inferenstid (ms)':<20}")
print("-"*40)
for namn, ms in resultat.items():
    print(f"{namn:<20} {ms:<20}")
print("="*60)
print("\nSe ovan för precision, recall och F1 per modell och klass")