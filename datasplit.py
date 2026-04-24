# split_data.py
import os, shutil, random

# ── EDIT THESE ──────────────────────────────────────────
sources = {
    "ok":         "C:/Users/noura/Downloads/raghad_nour_2026-04-14/newintakta",
    "damaged":    "C:/Users/noura/Downloads/Raghad-Nour-2026-04-17/camera2_images",
    "avlossnade": "C:/Users/noura/Downloads/raghad_nour_2026-04-14/newavlossnade",
}
output_dir = "./Dataset"   # where to create train/ and val/
val_split  = 0.2
# ────────────────────────────────────────────────────────

random.seed(42)

for class_name, src_folder in sources.items():
    images = [f for f in os.listdir(src_folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)

    n_val = int(len(images) * val_split)
    splits = {"val": images[:n_val], "Train": images[n_val:]}

    for split, files in splits.items():
        dest = os.path.join(output_dir, split, class_name)
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copy(os.path.join(src_folder, f), os.path.join(dest, f))
        print(f"{class_name} → {split}: {len(files)} images")