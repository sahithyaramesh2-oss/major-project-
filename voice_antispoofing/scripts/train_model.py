
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
CSV_FILE = RESULTS_DIR / "training_features.csv"
MODEL_FILE = RESULTS_DIR / "voice_antispoof_model.joblib"
METRICS_FILE = RESULTS_DIR / "validation_metrics.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load the extracted features
df = pd.read_csv(CSV_FILE)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# 2. Separate features and labels
X = df[[f"mfcc_{i}" for i in range(1, 14)]]
y = df["label"]

# 3. Create a stratified 80/20 holdout split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# 4. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 5. Obtain validation predictions and spoof scores
probabilities = model.predict_proba(X_val)
classes = list(model.classes_)
spoof_index = classes.index("spoof")
spoof_scores = probabilities[:, spoof_index]

# Threshold used for the reported FAR and FRR
threshold = 0.50

# Treat a recording as spoof when its spoof score reaches threshold
predictions = np.where(
    spoof_scores >= threshold, "spoof", "genuine"
)

print("\nValidation accuracy:", round(
    accuracy_score(y_val, predictions), 4
))

print("\nClassification report:")
report = classification_report(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
    output_dict=True,
    zero_division=0,
)
print(classification_report(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
    digits=4,
    zero_division=0,
))

# Confusion matrix: actual rows, predicted columns
cm = confusion_matrix(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
)
print("Confusion matrix (actual rows, predicted columns):")
print("Class order: ['genuine', 'spoof']")
print(cm)

# 6. Calculate FAR and FRR at threshold 0.50
# Spoof accepted as genuine = false acceptance
spoof_total = int((y_val == "spoof").sum())
genuine_total = int((y_val == "genuine").sum())

spoof_accepted_as_genuine = int(
    ((y_val == "spoof") & (predictions == "genuine")).sum()
)
genuine_rejected = int(
    ((y_val == "genuine") & (predictions == "spoof")).sum()
)

far = (
    spoof_accepted_as_genuine / spoof_total
    if spoof_total else 0.0
)
frr = (
    genuine_rejected / genuine_total
    if genuine_total else 0.0
)

print(f"\nDecision threshold: {threshold:.2f}")
print(
    f"FAR (spoof accepted as genuine): "
    f"{far:.4f} ({far * 100:.2f}%)"
)
print(
    f"FRR (genuine rejected as spoof): "
    f"{frr:.4f} ({frr * 100:.2f}%)"
)

# 7. Calculate ROC-AUC and approximate EER
roc_auc = roc_auc_score(
    (y_val == "spoof").astype(int),
    spoof_scores,
)

fpr, tpr, roc_thresholds = roc_curve(
    (y_val == "spoof").astype(int),
    spoof_scores,
)
fnr = 1 - tpr

# EER is estimated at the ROC point where FPR and FNR are closest
eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
eer = float((fpr[eer_index] + fnr[eer_index]) / 2)
eer_threshold = float(roc_thresholds[eer_index])

print(f"\nValidation ROC-AUC: {roc_auc:.4f}")
print(f"Approximate EER: {eer:.4f} ({eer * 100:.2f}%)")
print(f"Threshold near EER: {eer_threshold:.6f}")

# 8. Save the model
joblib.dump(model, MODEL_FILE)
print("\nModel saved to:", MODEL_FILE)

# 9. Save metrics for the research paper/dashboard
metrics = {
    "evaluation_type": "80/20 stratified holdout from training dataset",
    "dataset": "ASVspoof 2017 V2 training split",
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "threshold_for_far_frr": threshold,
    "accuracy": float(accuracy_score(y_val, predictions)),
    "roc_auc": float(roc_auc),
    "far_spoof_accepted_as_genuine": float(far),
    "frr_genuine_rejected_as_spoof": float(frr),
    "eer_approximate": eer,
    "eer_threshold_approximate": eer_threshold,
    "confusion_matrix_labels": ["genuine", "spoof"],
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
}

with open(METRICS_FILE, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("Validation metrics saved to:", METRICS_FILE)
print("\nEvaluation complete.")
