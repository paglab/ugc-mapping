#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# ---------------------------------------------------------------------
#      DeepLabV3+ Script for Ground-Cover Classification
# ---------------------------------------------------------------------

import os
import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from PIL import Image
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import rasterio

# --------------------------- portable paths -------------------------
REPO_ROOT   = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("DATASET",  REPO_ROOT / "Dataset" / "patches"))
OUTPUT_DIR  = Path(os.getenv("UNET_OUTPUT",   REPO_ROOT / "Output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_image_dir = DATASET_DIR / "images" / "train"
train_mask_dir  = DATASET_DIR / "masks"  / "train"
val_image_dir   = DATASET_DIR / "images" / "validation"
val_mask_dir    = DATASET_DIR / "masks"  / "validation"
test_image_dir  = DATASET_DIR / "images" / "test"
test_mask_dir   = DATASET_DIR / "masks"  / "test"

for d in (train_image_dir, train_mask_dir,
          val_image_dir,   val_mask_dir,
          test_image_dir,  test_mask_dir):
    if not d.exists():
        raise FileNotFoundError(f"Expected directory missing: {d}")


# ------------------------------ transforms --------------------------
class ToTensorNoScaling:
    def __call__(self, pic):
        return torch.tensor(np.array(pic), dtype=torch.long)

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet stats
])

mask_transforms = transforms.Compose([
    ToTensorNoScaling()  # Convert mask to tensor without scaling
])

# ------------------------------- dataset ----------------------------
class PatchDataset(Dataset):
    """
    Loads paired image and mask patches from specified directories.
    """
    def __init__(self, image_dir, mask_dir,
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        # Ensure images and masks match in count (and sorted order)
        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list  = sorted(os.listdir(mask_dir))
        if len(self.image_list) != len(self.mask_list):
            raise ValueError(
                f"Mismatch: {len(self.image_list)} images vs. {len(self.mask_list)} masks\n"
                f"  images: {image_dir}\n  masks:  {mask_dir}"
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path  = os.path.join(self.mask_dir, self.mask_list[idx])
        
        # --- load image GeoTIFF ---
        with rasterio.open(image_path) as src:
            img_arr = src.read()                        
            img_arr = np.transpose(img_arr, (1, 2, 0))  
            image   = Image.fromarray(img_arr.astype(np.uint8))

        # --- load mask GeoTIFF ---
        with rasterio.open(mask_path) as src:
            msk_arr = src.read(1)                       
            mask    = Image.fromarray(msk_arr.astype(np.uint8))

        # --- apply transforms ---
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask  = self.mask_transform(mask)

        return image, mask
    
# ------------------------------ loaders -----------------------------
train_loader = DataLoader(PatchDataset(train_image_dir, train_mask_dir,
                                       image_transforms, mask_transforms),
                          batch_size=16, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(PatchDataset(val_image_dir,   val_mask_dir,
                                       image_transforms, mask_transforms),
                          batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(PatchDataset(test_image_dir,  test_mask_dir,
                                       image_transforms, mask_transforms),
                          batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    

# ---------------------- model / loss / optimiser --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=9).to(device)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # ignoring background index 0
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Drop LR by ×0.1 if val_loss hasn’t improved for 5 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # we want to minimise val_loss
    factor=0.1,       # new_lr = old_lr * factor
    patience=5,       # wait 5 epochs with no improvement
    verbose=True,     # print a message whenever LR is reduced
)
scaler = GradScaler()

# ------------------------------ training ----------------------------
best_val_loss, num_epochs = float('inf'), 25
best_model_path = OUTPUT_DIR / "best_model_DeepLabV3Plus.pth"
epoch_stats, start_train = [], time.time()


for epoch in range(num_epochs):
    epoch_start_time = time.time()
    print(f"\nEpoch {epoch + 1} of {num_epochs}")
    model.train()
    running_loss = 0.0

    # Training Phase
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Epoch {epoch + 1}, Batch {batch_idx + 1} started.")
        try:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        except Exception as e:
            print(f"Error moving data to device at Training Batch {batch_idx + 1}: {e}")
            continue

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    print(f"Training Loss (Epoch {epoch + 1}): {train_loss:.4f}")

    # Validation Phase with Aggregated Loss Calculation
    print("Starting validation...")
    model.eval()
    total_valid_loss = 0.0
    total_valid_pixels = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            print(f"Validation Batch {batch_idx + 1} started.")
            try:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            except Exception as e:
                print(f"Error moving data to device at Validation Batch {batch_idx + 1}: {e}")
                continue

            with autocast():
                outputs = model(images)
                # Resize outputs to match the mask size
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            
            # Compute per-pixel loss without reduction
            per_pixel_loss = F.cross_entropy(outputs, masks, ignore_index=0, reduction='none')
            # Create mask for valid (non-background) pixels
            valid_mask = (masks != 0)
            valid_count = valid_mask.sum().item()
            if valid_count > 0:
                batch_loss = per_pixel_loss[valid_mask].sum().item()
                total_valid_loss += batch_loss
                total_valid_pixels += valid_count

            # Collect predictions and labels for accuracy metrics
            preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
            labels = masks.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels)

            print(f"Validation Batch {batch_idx + 1} completed.")

    # Compute aggregated validation loss over all valid pixels
    if total_valid_pixels > 0:
        val_loss = total_valid_loss / total_valid_pixels
    else:
        val_loss = 0.0
    print(f"Validation Loss (Epoch {epoch + 1}): {val_loss:.4f}")

    # Calculate Accuracy and Kappa (excluding background)
    filtered_preds = [p for (p, l) in zip(all_preds, all_labels) if l != 0]
    filtered_labels = [l for l in all_labels if l != 0]
    if len(filtered_labels) > 0:
        val_acc = accuracy_score(filtered_labels, filtered_preds)
        val_kappa = cohen_kappa_score(filtered_labels, filtered_preds)
        print(f"Validation Accuracy (excl. background): {val_acc:.4f}")
        print(f"Validation Kappa (excl. background): {val_kappa:.4f}")
    else:
        val_acc, val_kappa = 0.0, 0.0

    # Save Best Model if validation loss improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch + 1} with val_loss={val_loss:.4f} to {best_model_path}")
    else:
        print(f"Validation loss did not improve at epoch {epoch + 1}. Current val_loss={val_loss:.4f}, Best val_loss={best_val_loss:.4f}")

    # Tell the scheduler whether things improved
    scheduler.step(val_loss)
    
    # ─── Log learning rate ──────────────────────────────────────────
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} finished in {epoch_time:.2f} seconds")
    current_lr = optimizer.param_groups[0]['lr']
    epoch_stats.append((epoch+1, train_loss, val_loss, val_acc, val_kappa, current_lr))

training_total_time = time.time() - start_train
print(f"\nTotal Training Time: {training_total_time:.2f} seconds")

# ------------ save epoch log ----------------
loss_file_path = os.path.join(OUTPUT_DIR, 'Loss_DeepLabV3Plus.txt')

with open(loss_file_path, 'w') as f:
    f.write("Epoch\tTraining Loss\tValidation Loss\tAccuracy\tKappa\t\tLR\n")
    for epoch_num, train_loss, val_loss, val_acc, val_kappa, lr in epoch_stats:
        f.write(f"{epoch_num}\t{train_loss:.4f}\t\t{val_loss:.4f}\t\t"
                f"{val_acc:.4f}\t\t{val_kappa:.4f}\t\t{lr:.1e}\n")

print(f"Epoch-wise metrics saved to: {loss_file_path}")

# --------------------------- testing -------------------------------
print("\nStarting test evaluation...")
# Check if the best model file exists; if not, save the current model state as the best model.
if not os.path.exists(best_model_path):
    torch.save(model.state_dict(), best_model_path)
    print(f"Best model not found. Saved current model state to {best_model_path}")
else:
    # Load the best model for evaluation.
    model.load_state_dict(torch.load(best_model_path))
    print(f"Loaded best model from {best_model_path} for evaluation.")

model.eval()
test_preds, test_labels = [], []
test_start = time.time()

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        print(f"Test Batch {batch_idx + 1} started.")
        try:
            images, masks = images.to(device), masks.to(device)
        except Exception as e:
            print(f"Error moving data to device at Test Batch {batch_idx + 1}: {e}")
            continue
        
        outputs = model(images)
        # Resize outputs to match the mask size
        outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
        preds = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
        labels = masks.cpu().numpy().flatten()
        test_preds.extend(preds)
        test_labels.extend(labels)
        print(f"Test Batch {batch_idx + 1} completed.")

test_time = time.time() - test_start

# ----------------------Overall Metrics (Excluding Background)--------------------
test_labels_no_bg  = [l for l in test_labels if l != 0]
test_preds_no_bg   = [p for (p, l) in zip(test_preds, test_labels) if l != 0]

if len(test_labels_no_bg) > 0:
    overall_accuracy = accuracy_score(test_labels_no_bg, test_preds_no_bg)
    overall_kappa    = cohen_kappa_score(test_labels_no_bg, test_preds_no_bg)
    overall_f1       = f1_score(test_labels_no_bg, test_preds_no_bg, average='weighted')
else:
    overall_accuracy = 0.0
    overall_kappa    = 0.0
    overall_f1       = 0.0

print(f"Test Accuracy (Excl. Background): {overall_accuracy:.4f}")
print(f"Test Cohen's Kappa (Excl. Background): {overall_kappa:.4f}")
print(f"Test Weighted F1-Score (Excl. Background): {overall_f1:.4f}")
print(f"Test Evaluation Time: {test_time:.2f} seconds")

# ----------------------Confusion Matrix -------------------------
conf_matrix = confusion_matrix(test_labels, test_preds)
conf_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.txt')
np.savetxt(conf_matrix_path, conf_matrix, delimiter=',', fmt='%d')
print(f"Confusion matrix saved to: {conf_matrix_path}")

# ----------------------Per-Class Metrics -------------------------
n_classes = 9
dice_scores = []
iou_scores = []
per_class_f1 = f1_score(test_labels, test_preds, average=None, labels=range(n_classes))

for i in range(n_classes):
    TP = conf_matrix[i, i]
    FP = np.sum(conf_matrix[:, i]) - TP
    FN = np.sum(conf_matrix[i, :]) - TP
    denom_dice = (2 * TP + FP + FN)
    denom_iou = (TP + FP + FN)
    dice = (2.0 * TP / denom_dice) if denom_dice > 0 else 0
    iou = (TP / denom_iou) if denom_iou > 0 else 0
    dice_scores.append(dice)
    iou_scores.append(iou)

# -----------------Save Evaluation Metrics and Computational Times-----------------------
total_comp_time = training_total_time + test_time

metrics_str  = "Overall Metrics (Excluding Background):\n"
metrics_str += f"Accuracy: {overall_accuracy:.4f}\n"
metrics_str += f"Cohen's Kappa: {overall_kappa:.4f}\n"
metrics_str += f"Weighted F1-Score: {overall_f1:.4f}\n\n"
metrics_str += "Per-Class Metrics (0..8):\n"
metrics_str += "Class\tF1-Score\tDice Coefficient\tIoU (Jaccard Index)\n"
for i in range(n_classes):
    metrics_str += f"{i}\t{per_class_f1[i]:.4f}\t\t{dice_scores[i]:.4f}\t\t{iou_scores[i]:.4f}\n"
metrics_str += "\nComputational Time Metrics (in seconds):\n"
metrics_str += f"Total Training Time: {training_total_time:.2f}\n"
metrics_str += f"Test Evaluation Time: {test_time:.2f}\n"
metrics_str += f"Total Computational Time: {total_comp_time:.2f}\n"

metrics_file_path = os.path.join(OUTPUT_DIR, "evaluation_metrics_Deeplabv.txt")
with open(metrics_file_path, 'w') as f:
    f.write(metrics_str)

print(f"Evaluation metrics saved to: {metrics_file_path}")

gc.collect()
print("\nDone! Training & evaluation completed successfully.")
