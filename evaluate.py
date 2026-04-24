"""
Evaluate your trained model on the val set.
Shows:
- Overall accuracy
- Per-class accuracy (how often it gets each class right)
- Confusion matrix (what it confuses with what)

Usage:
    python evaluate.py --data_dir .\dataset --model_path best_mobilenet_v2.pth
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────
# Val transform (same as training)
# ──────────────────────────────────────────────

def get_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────

def load_model(model_path, num_classes, device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


# ──────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────

def evaluate(model, val_loader, classes, device):
    num_classes = len(classes)

    # Matrix where [i][j] = how many times class i was predicted as class j
    confusion = np.zeros((num_classes, num_classes), dtype=int)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1)

            for true, pred in zip(labels.cpu(), preds.cpu()):
                confusion[true][pred] += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Overall accuracy
    total_correct = np.trace(confusion)   # diagonal = correct predictions
    total_images  = confusion.sum()
    overall_acc   = total_correct / total_images

    print("\n" + "="*55)
    print("  EVALUATION RESULTS")
    print("="*55)
    print(f"\n  Overall accuracy: {overall_acc*100:.2f}%")
    print(f"  ({total_correct} correct out of {total_images} val images)\n")

    # Per-class accuracy
    print("  Per-class accuracy:")
    print("  " + "-"*40)
    for i, class_name in enumerate(classes):
        class_total   = confusion[i].sum()
        class_correct = confusion[i][i]
        class_acc     = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name:15s}  {class_acc*100:6.2f}%  ({class_correct}/{class_total})")

    # Confusion matrix
    print("\n  Confusion Matrix:")
    print("  (rows = actual class, columns = predicted class)\n")

    # Header
    col_width = 12
    header = " " * 16 + "".join(f"{c:>{col_width}}" for c in classes)
    print("  " + header)
    print("  " + "-" * (16 + col_width * num_classes))

    for i, class_name in enumerate(classes):
        row = f"  {class_name:15s} "
        for j in range(num_classes):
            count = confusion[i][j]
            # Highlight correct predictions (diagonal)
            if i == j:
                row += f"[{count:>{col_width-2}}]"
            else:
                row += f"{count:>{col_width}}"
        print(row)

    # Explain what to look for
    print("\n  " + "="*55)
    print("  HOW TO READ THE CONFUSION MATRIX:")
    print("  " + "-"*55)
    print("  [X] = correct predictions (diagonal, you want these high)")
    print("   X  = wrong predictions (off-diagonal, you want these low)")
    print("\n  Example: if row 'damaged' has a high number under")
    print("  column 'ok', the model is confusing damaged with ok.")
    print("  " + "="*55 + "\n")

    return confusion, overall_acc


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tooltip classifier")
    p.add_argument("--data_dir",   default="./dataset")
    p.add_argument("--model_path", default="best_mobilenet_v2.pth")
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load val dataset
    val_dataset = datasets.ImageFolder(
        root=f"{args.data_dir}/val",
        transform=get_val_transform(args.img_size)
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    classes = val_dataset.classes
    print(f"\nClasses : {classes}")
    print(f"Val set : {len(val_dataset)} images")

    # Load model
    model = load_model(args.model_path, num_classes=len(classes), device=device)
    print(f"Model   : {args.model_path}")

    # Evaluate
    evaluate(model, val_loader, classes, device)


if __name__ == "__main__":
    main()