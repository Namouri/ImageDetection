"""
Plot training results for MobileNetV2 and ResNet50 side by side.
Shows accuracy, loss, and overfitting analysis for both models.

Usage:
    python plot_results.py

Make sure both JSON history files exist:
    best_mobilenet_v2_final_history.json
    best_resnet_final_history.json
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os


# ──────────────────────────────────────────────
# Load history files
# ──────────────────────────────────────────────

def load_history(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}\nMake sure you trained the model first.")
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────

def plot_results(mobilenet_history, resnet_history):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("MobileNetV2 vs ResNet50 — Tooltip Damage Classification",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    EPOCHS_TO_SHOW = 10

    mobilenet_history = {k: v[:EPOCHS_TO_SHOW] for k, v in mobilenet_history.items()}
    resnet_history    = {k: v[:EPOCHS_TO_SHOW] for k, v in resnet_history.items()}



    mobilenet_epochs = range(1, len(mobilenet_history["train_acc"]) + 1)
    resnet_epochs    = range(1, len(resnet_history["train_acc"]) + 1)

    # ── Colors ──────────────────────────────────
    TRAIN_COLOR = "#2196F3"   # blue
    VAL_COLOR   = "#FF5722"   # orange
    GAP_COLOR   = "#4CAF50"   # green for gap area

    # ────────────────────────────────────────────
    # Row 0: Accuracy plots
    # ────────────────────────────────────────────

    # MobileNet accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(mobilenet_epochs, mobilenet_history["train_acc"],
             color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="Train")
    ax1.plot(mobilenet_epochs, mobilenet_history["val_acc"],
             color=VAL_COLOR, linewidth=2, marker="s", markersize=4, label="Val")
    ax1.fill_between(mobilenet_epochs,
                     mobilenet_history["train_acc"],
                     mobilenet_history["val_acc"],
                     alpha=0.1, color=GAP_COLOR, label="Gap (overfitting risk)")
    ax1.set_title("MobileNetV2 — Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    # ResNet accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(resnet_epochs, resnet_history["train_acc"],
             color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="Train")
    ax2.plot(resnet_epochs, resnet_history["val_acc"],
             color=VAL_COLOR, linewidth=2, marker="s", markersize=4, label="Val")
    ax2.fill_between(resnet_epochs,
                     resnet_history["train_acc"],
                     resnet_history["val_acc"],
                     alpha=0.1, color=GAP_COLOR, label="Gap (overfitting risk)")
    ax2.set_title("ResNet50 — Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

    # Side by side best val accuracy bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    models_names = ["MobileNetV2", "ResNet50"]
    best_vals    = [max(mobilenet_history["val_acc"]), max(resnet_history["val_acc"])]
    bars = ax3.bar(models_names, [v * 100 for v in best_vals],
                   color=[TRAIN_COLOR, VAL_COLOR], width=0.4, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, best_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f"{val*100:.2f}%",
                 ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax3.set_title("Best Val Accuracy Comparison", fontweight="bold")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis="y")

    # ────────────────────────────────────────────
    # Row 1: Loss plots
    # ────────────────────────────────────────────

    # MobileNet loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(mobilenet_epochs, mobilenet_history["train_loss"],
             color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="Train")
    ax4.plot(mobilenet_epochs, mobilenet_history["val_loss"],
             color=VAL_COLOR, linewidth=2, marker="s", markersize=4, label="Val")
    ax4.set_title("MobileNetV2 — Loss", fontweight="bold")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ResNet loss
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(resnet_epochs, resnet_history["train_loss"],
             color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="Train")
    ax5.plot(resnet_epochs, resnet_history["val_loss"],
             color=VAL_COLOR, linewidth=2, marker="s", markersize=4, label="Val")
    ax5.set_title("ResNet50 — Loss", fontweight="bold")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Loss")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Overfitting gap over time
    ax6 = fig.add_subplot(gs[1, 2])
    mobilenet_gap = [t - v for t, v in zip(mobilenet_history["train_acc"],
                                            mobilenet_history["val_acc"])]
    resnet_gap    = [t - v for t, v in zip(resnet_history["train_acc"],
                                            resnet_history["val_acc"])]
    ax6.plot(mobilenet_epochs, [g * 100 for g in mobilenet_gap],
             color=TRAIN_COLOR, linewidth=2, marker="o", markersize=4, label="MobileNetV2")
    ax6.plot(resnet_epochs, [g * 100 for g in resnet_gap],
             color=VAL_COLOR, linewidth=2, marker="s", markersize=4, label="ResNet50")
    ax6.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax6.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="Overfitting threshold (5%)")
    ax6.fill_between(mobilenet_epochs, 0, [g * 100 for g in mobilenet_gap],
                     alpha=0.1, color=TRAIN_COLOR)
    ax6.set_title("Overfitting Gap (Train − Val Acc)", fontweight="bold")
    ax6.set_xlabel("Epoch")
    ax6.set_ylabel("Gap (%)")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.savefig("training_results.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Plot saved to training_results.png")
    plt.show()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main():
    mobilenet_path = "best_mobilenet_v2_final_history.json"
    resnet_path    = "best_resnet_final_history.json"

    print(f"Loading MobileNetV2 history from: {mobilenet_path}")
    mobilenet_history = load_history(mobilenet_path)

    print(f"Loading ResNet50 history from:    {resnet_path}")
    resnet_history = load_history(resnet_path)

    print("Generating plots...\n")
    plot_results(mobilenet_history, resnet_history)


if __name__ == "__main__":
    main()