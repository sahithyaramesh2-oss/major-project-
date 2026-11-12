
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
MODEL_FILE = RESULTS_DIR / "voice_antispoof_svm_model.joblib"
METRICS_FILE = RESULTS_DIR / "svm_validation_metrics.json"

# 1. Load extracted MFCC features
df = pd.read_csv(CSV_FILE)

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# 2. Separate features and labels
X = df[[f"mfcc_{i}" for i in range(1, 14)]]
y = df["label"]

# 3. Use the SAME split as Random Forest
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# 4. Create StandardScaler + RBF SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )),
])

# 5. Train the SVM model
print("\nTraining SVM with RBF kernel...")
model.fit(X_train, y_train)

# 6. Obtain validation predictions and spoof scores
probabilities = model.predict_proba(X_val)

classes = list(model.classes_)
spoof_index = classes.index("spoof")
spoof_scores = probabilities[:, spoof_index]

threshold = 0.50

predictions = np.where(
    spoof_scores >= threshold,
    "spoof",
    "genuine",
)

# 7. Calculate accuracy and classification report
accuracy = accuracy_score(y_val, predictions)

print("\nSVM Validation Accuracy:", round(accuracy, 4))

report = classification_report(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
    output_dict=True,
    zero_division=0,
)

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
    digits=4,
    zero_division=0,
))

# 8. Confusion matrix
cm = confusion_matrix(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
)

print("\nConfusion Matrix (actual rows, predicted columns):")
print("Class order: ['genuine', 'spoof']")
print(cm)

# 9. Calculate FAR and FRR
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

print(f"\nDecision Threshold: {threshold:.2f}")
print(f"FAR: {far:.4f} ({far * 100:.2f}%)")
print(f"FRR: {frr:.4f} ({frr * 100:.2f}%)")

# 10. Calculate ROC-AUC and approximate EER
roc_auc = roc_auc_score(
    (y_val == "spoof").astype(int),
    spoof_scores,
)

fpr, tpr, roc_thresholds = roc_curve(
    (y_val == "spoof").astype(int),
    spoof_scores,
)

fnr = 1 - tpr

eer_index = int(np.nanargmin(np.abs(fpr - fnr)))

eer = float(
    (fpr[eer_index] + fnr[eer_index]) / 2
)

eer_threshold = float(roc_thresholds[eer_index])

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"Approximate EER: {eer:.4f} ({eer * 100:.2f}%)")
print(f"EER Threshold: {eer_threshold:.6f}")

# 11. Save SVM model separately
joblib.dump(model, MODEL_FILE)

print("\nSVM model saved to:", MODEL_FILE)

# 12. Save validation metrics separately
metrics = {
    "algorithm": "SVM with RBF kernel",
    "evaluation_type": "80/20 stratified holdout from training dataset",
    "dataset": "ASVspoof 2017 V2 training split",
    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "threshold_for_far_frr": threshold,
    "accuracy": float(accuracy),
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

print("SVM metrics saved to:", METRICS_FILE)
print("\nSVM Evaluation Complete!")