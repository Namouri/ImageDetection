import argparse
import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from augmentation import get_train_transform, get_val_transform

def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load MobileNetV2 pre-trained on ImageNet and replace the classifier head.
 
    freeze_backbone=True  → only the new head is trained (fast, good for small datasets)
    freeze_backbone=False → full fine-tuning (slower, better with more data)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

 
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    # MobileNetV2's classifier: Linear(1280 → num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
 
    return model



def build_dataloaders(data_dir: str, batch_size: int, img_size: int):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=get_train_transform(img_size)
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=get_val_transform(img_size)
    )
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
 
    print(f"\nClasses : {train_dataset.classes}")
    print(f"Train   : {len(train_dataset)} images")
    print(f"Val     : {len(val_dataset)} images\n")
 
    return train_loader, val_loader, train_dataset.classes
 
def train(model, train_loader, val_loader, criterion, optimizer,
          scheduler, device, epochs: int, save_path: str):
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
 
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}  " + "─" * 40)
 
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()
 
            running_loss = 0.0
            running_correct = 0
 
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
 
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
 
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
 
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
 
            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc  = running_correct / len(loader.dataset)
            print(f"  {phase:5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")
 
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)
                print(f"  ✔ New best val acc {best_acc:.4f} — model saved to {save_path}")
 
        if scheduler:
            scheduler.step()
 
    print(f"\nBest validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def predict(model, image_path: str, classes: list, device, img_size=224):
    """Run the model on a single image and return (class_name, confidence)."""
    from PIL import Image
 
    transform = get_val_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)   # (1, 3, H, W)
 
    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        idx    = probs.argmax().item()
 
    return classes[idx], probs[idx].item()
 
def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 for tooltip damage detection")
    p.add_argument("--data_dir",   default="./dataset",               help="Root folder with train/ and val/ sub-folders")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--img_size",   type=int,   default=224)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--save_path",  default="best_resnet_model.pth",  help="Where to save the best checkpoint")
    p.add_argument("--full_finetune", action="store_true",            help="Unfreeze backbone for full fine-tuning")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # Data
    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir, args.batch_size, args.img_size
    )
    num_classes = len(classes)
 
    # Model
    model = build_model(num_classes, freeze_backbone=not args.full_finetune)
    model = model.to(device)
 
    # Loss, optimiser, LR scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
 
    # Train
    model = train(model, train_loader, val_loader,
                  criterion, optimizer, scheduler,
                  device, args.epochs, args.save_path)
 
    # Quick smoke-test on a single image (optional)
    # label, conf = predict(model, "test.jpg", classes, device)
    # print(f"\nPrediction: {label}  ({conf*100:.1f}% confidence)")
 
 
if __name__ == "__main__":
    main()
