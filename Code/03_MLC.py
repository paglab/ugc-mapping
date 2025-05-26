#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------
#      MLC Script using Quadratic Discriminant Analysis for Ground-Cover Classification
# ---------------------------------------------------------------------

import os
import gc
import time
import numpy as np
import tifffile
import psutil
from pathlib import Path

# Sklearn tools
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, f1_score
import joblib  

import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------- portable paths -------------------------
REPO_ROOT   = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv("DATASET", REPO_ROOT / "Dataset"))
OUTPUT_DIR  = Path(os.getenv("MLC_OUTPUT",  REPO_ROOT / "Output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATASET_DIR / "train"
VAL_PATH   = DATASET_DIR / "val"
TEST_PATH  = DATASET_DIR / "test"

# ------------------------Discover RGB images & their masks--------------------------------
def list_rgb_and_masks(folder: str):
    rgb = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".tif") and not f.lower().endswith("_labelled.tif")
    ])
    masks = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith("_labelled.tif")
    ])
    return rgb, masks

train_image_files, train_mask_files = list_rgb_and_masks(TRAIN_PATH)
val_image_files,   val_mask_files   = list_rgb_and_masks(VAL_PATH)
test_image_files,  test_mask_files  = list_rgb_and_masks(TEST_PATH)

# Optional: print a quick overview
print("Collected files→")
for split, imgs, msks in [("Train", train_image_files, train_mask_files),
                          ("Val",   val_image_files,   val_mask_files),
                          ("Test",  test_image_files,  test_mask_files)]:
    print(f"\n{split} images ({len(imgs)}):")
    for p in imgs:
        print("  ", os.path.basename(p))
    print(f"{split} masks   ({len(msks)}):")
    for p in msks:
        print("  ", os.path.basename(p))

n_classes = 8           # background = 0 ignored during patch extraction
MAX_PATCHES_PER_IMAGE = 10000

# -----------------------Utility Functions---------------------------------------
def print_memory_usage():
    mem = psutil.Process().memory_info().rss / 1024 ** 2
    print(f"Memory Usage: {mem:.2f} MB")

def get_label_file(image_file: str) -> str:
    """Return mask path matching *image_file* (expects suffix `_Labelled.tif`)."""
    base = os.path.splitext(image_file)[0]
    label_file = f"{base}_Labelled.tif"
    if os.path.exists(label_file):
        return label_file
    raise FileNotFoundError(f"No label file found for {image_file}")


def load_and_normalize_data(image_file):
    try:
        data = tifffile.imread(image_file)
        if data.ndim == 2:
            data = np.stack([data] * 3, axis=-1)
        elif data.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {data.shape}, expected 3 channels")
        
        label_file = get_label_file(image_file)
        labels = tifffile.imread(label_file)
        
        data_min, data_max = np.min(data), np.max(data)
        data = (data - data_min) / (data_max - data_min + 1e-8)
        data = data.astype(np.float32)
        labels = labels.astype(np.int32)
        return data, labels
    except Exception as e:
        print(f"Error loading file {image_file}: {e}")
        return None, None
    
# -----------------------Patch Extraction---------------------------------------
def extract_patches(image, label, kn, max_patches_per_image=None):
    patch_size = 2 * kn + 1
    padded_img = np.pad(image, ((kn, kn), (kn, kn), (0, 0)), mode='reflect')
    
    # 1) gather all valid coords
    coords = [(i,j) for i in range(label.shape[0])
                     for j in range(label.shape[1])
                     if label[i,j] != 0]

    # 2) if you want to limit to K patches, sample from coords
    if max_patches_per_image and len(coords) > max_patches_per_image:
        sampled_idxs = np.random.choice(len(coords),
                                        max_patches_per_image,
                                        replace=False)
        coords = [coords[i] for i in sampled_idxs]

    # 3) now build your patches & labels
    patches, labels = [], []
    for i,j in coords:
        patch = padded_img[i:i+patch_size, j:j+patch_size, :]
        patches.append(patch)
        labels.append(label[i,j] - 1)   
    return patches, labels

# ---------------------Custom "Dataset" to Gather Patches----------------------------------
class PatchDataset:
    def __init__(self, image_files, kn=5, max_patches_per_image=None):
        self.image_files = image_files
        self.kn = kn
        self.max_patches_per_image = max_patches_per_image

        self.all_patches = []
        self.all_labels = []
        total_patches = 0

        print(f"\n[Dataset Initialization] kn={kn}, max_patches_per_image={max_patches_per_image}")
        for idx, img_path in enumerate(self.image_files, start=1):
            img_array, lbl_array = load_and_normalize_data(img_path)
            if img_array is None or lbl_array is None:
                print(f"[WARNING] Skipping file due to error: {img_path}")
                continue

            patches, labels = extract_patches(img_array, lbl_array, self.kn, self.max_patches_per_image)
            self.all_patches.extend(patches)
            self.all_labels.extend(labels)
            total_patches += len(patches)

            print(f"[INFO] Finished extracting patches for {idx}/{len(self.image_files)}: {img_path}")
        
        print(f"[INFO] Dataset created with total patches: {total_patches}")
        print_memory_usage()

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        return self.all_patches[idx], self.all_labels[idx]

# -----------------------Build Datasets (Train, Val, Test)--------------------------------
print("[INFO] Building Datasets...")
kn = 5
start_build_time = time.time()

train_dataset = PatchDataset(train_image_files, kn, MAX_PATCHES_PER_IMAGE)
val_dataset   = PatchDataset(val_image_files,   kn, MAX_PATCHES_PER_IMAGE)
test_dataset  = PatchDataset(test_image_files,  kn, MAX_PATCHES_PER_IMAGE)

end_build_time = time.time()
print(f"\n[INFO] Finished building all datasets in {end_build_time - start_build_time:.2f} seconds.")

def dataset_to_Xy(patch_dataset):
    patches = patch_dataset.all_patches
    labels  = patch_dataset.all_labels
    X = np.array([p.reshape(-1) for p in patches], dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y

print("[INFO] Converting training patches into (X_train, y_train)...")
X_train, y_train = dataset_to_Xy(train_dataset)
print(f" - Done. Shapes: X_train={X_train.shape}, y_train={y_train.shape}")

print("[INFO] Converting validation patches into (X_val, y_val)...")
X_val, y_val = dataset_to_Xy(val_dataset)
print(f" - Done. Shapes: X_val={X_val.shape}, y_val={y_val.shape}")

print("[INFO] Converting test patches into (X_test, y_test)...")
X_test, y_test = dataset_to_Xy(test_dataset)
print(f" - Done. Shapes: X_test={X_test.shape}, y_test={y_test.shape}")

# ----------------Enhanced Hyperparameter Tuning-----------------------------
print("\n[INFO] Starting Hyperparameter Search using RandomizedSearchCV...")

# Combine training and validation data for tuning 
X_combined = np.concatenate([X_train, X_val], axis=0)
y_combined = np.concatenate([y_train, y_val], axis=0)

# Create predefined split: -1 for training, 0 for validation
train_fold = -1 * np.ones(len(y_train), dtype=int)
val_fold = 0 * np.ones(len(y_val), dtype=int)
test_fold = np.concatenate([train_fold, val_fold])
predef_split = PredefinedSplit(test_fold)

# Define an expanded parameter grid for QDA
uniform_priors = np.ones(n_classes) / n_classes  
param_dist = {
    "reg_param": np.linspace(0.0, 1.0, 21),   
    "priors":    [None, uniform_priors],
}

qda_base = QuadraticDiscriminantAnalysis(store_covariance=False)
random_search = RandomizedSearchCV(
    estimator=qda_base,
    param_distributions=param_dist,
    n_iter=50,           
    cv=predef_split,
    scoring="accuracy",
    n_jobs=-1,            
    random_state=42,
    verbose=1,
    refit=False,
)

start_rs_time = time.time()
random_search.fit(X_combined, y_combined)
end_rs_time   = time.time()
grid_search_time = end_rs_time - start_rs_time

print(f"[INFO] RandomizedSearchCV completed in {grid_search_time:.2f} s.")
print("[INFO] Best hyper-parameters:", random_search.best_params_)
print(f"[INFO] Best validation accuracy: {random_search.best_score_:.4f}")

# --------------Train Final Model with Best Hyperparameters--------------------------
print("[INFO] Training final QDA with best hyperparameters on X_train only...")

best_params = random_search.best_params_
best_mlc = QuadraticDiscriminantAnalysis(**best_params, store_covariance=False)

start_train_time = time.time()
best_mlc.fit(X_train, y_train)  # Removed sample_weight
end_train_time = time.time()
training_time = end_train_time - start_train_time

print(f"[INFO] Done training final MLC model in {training_time:.2f} seconds.")

# ---------------------Validation Evaluation--------------------------------------
print("\n[INFO] Validation Set Evaluation...")
start_val_eval = time.time()
y_val_pred = best_mlc.predict(X_val)
end_val_eval = time.time()

validation_time = end_val_eval - start_val_eval
val_cm = confusion_matrix(y_val, y_val_pred)
val_acc = accuracy_score(y_val, y_val_pred)
val_kappa = cohen_kappa_score(y_val, y_val_pred)

print(f" - Finished in {validation_time:.2f} seconds.")
print(" - Confusion Matrix (Validation):\n", val_cm)
print(f" - Accuracy: {val_acc:.4f}")
print(f" - Cohen's Kappa: {val_kappa:.4f}")

# --------------------------Test Evaluation---------------------------------------
print("\n[INFO] Test Set Evaluation...")
start_test_eval = time.time()
y_test_pred = best_mlc.predict(X_test)
end_test_eval = time.time()

test_time = end_test_eval - start_test_eval
test_cm = confusion_matrix(y_test, y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)
test_kappa = cohen_kappa_score(y_test, y_test_pred)

print(f" - Finished in {test_time:.2f} seconds.")
print(" - Confusion Matrix (Test):\n", test_cm)
print(f" - Accuracy: {test_acc:.4f}")
print(f" - Cohen's Kappa: {test_kappa:.4f}")

# ---------Save Model, Confusion Matrix, and Evaluation Metrics---------------------
print("\n[INFO] Saving model and evaluation outputs...")
model_path = os.path.join(OUTPUT_DIR, "best_mlc_model.pth")
joblib.dump(best_mlc, model_path)
print(f" - Best MLC model saved to: {model_path}")

conf_matrix_txt_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix.txt")
np.savetxt(conf_matrix_txt_path, test_cm, delimiter=',', fmt='%d')
print(f" - Confusion matrix saved as text file to {conf_matrix_txt_path}")

overall_accuracy = accuracy_score(y_test, y_test_pred)
overall_kappa = cohen_kappa_score(y_test, y_test_pred)
overall_f1 = f1_score(y_test, y_test_pred, average='weighted')
per_class_f1 = f1_score(y_test, y_test_pred, average=None)

dice_scores = []
iou_scores = []
for i in range(n_classes):
    TP = test_cm[i, i]
    FP = np.sum(test_cm[:, i]) - TP
    FN = np.sum(test_cm[i, :]) - TP
    denominator_dice = (2 * TP + FP + FN)
    denominator_iou = (TP + FP + FN)
    
    dice = (2 * TP / denominator_dice) if denominator_dice > 0 else 0
    iou = (TP / denominator_iou) if denominator_iou > 0 else 0
    
    dice_scores.append(dice)
    iou_scores.append(iou)

jaccard_indices = iou_scores.copy()

metrics_str = "Overall Metrics:\n"
metrics_str += f"Accuracy: {overall_accuracy:.4f}\n"
metrics_str += f"Cohen's Kappa: {overall_kappa:.4f}\n"
metrics_str += f"Weighted F1-Score: {overall_f1:.4f}\n\n"
metrics_str += "Per-Class Metrics:\n"
metrics_str += "Class\tF1-Score\tDice Coefficient\tIoU (Jaccard Index)\n"
for i in range(n_classes):
    metrics_str += f"{i}\t{per_class_f1[i]:.4f}\t\t{dice_scores[i]:.4f}\t\t\t{iou_scores[i]:.4f}\n"

metrics_str += "\nComputational Time Metrics (in seconds):\n"
metrics_str += f"RandomizedSearchCV Time: {grid_search_time:.2f}\n"
metrics_str += f"Training Time: {training_time:.2f}\n"
metrics_str += f"Validation Evaluation Time: {validation_time:.2f}\n"
metrics_str += f"Test Evaluation Time: {test_time:.2f}\n"
total_time = grid_search_time + training_time + validation_time + test_time
metrics_str += f"Total Computational Time: {total_time:.2f}\n"

metrics_file_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
with open(metrics_file_path, 'w') as f:
    f.write(metrics_str)
print(f" - Evaluation metrics saved to: {metrics_file_path}")

gc.collect()
print("\n[INFO] Done! Enhanced QDA-based MLC training & evaluation completed successfully.")