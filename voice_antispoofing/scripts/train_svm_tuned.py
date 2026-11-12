
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

CSV_FILE = RESULTS_DIR / "training_features.csv"

MODEL_FILE = MODELS_DIR / "voice_antispoof_svm_tuned_model.joblib"
METRICS_FILE = RESULTS_DIR / "svm_tuned_validation_metrics.json"
GRID_FILE = RESULTS_DIR / "svm_grid_search_results.csv"

# Create required directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Training features CSV not found: {CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

feature_columns = [
    f"mfcc_{i}" for i in range(1, 14)
]

required_columns = feature_columns + ["label"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[feature_columns]
y = df["label"]

print("Dataset shape:", df.shape)

print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. TRAIN-VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# --------------------------------------------------
# 4. CREATE SVM PIPELINE
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=False,
        random_state=42,
    )),
])

# --------------------------------------------------
# 5. HYPERPARAMETER GRID
# --------------------------------------------------

param_grid = {
    "svm__C": [0.1, 1, 10, 100],

    "svm__gamma": [
        "scale",
        0.001,
        0.01,
        0.1,
        1,
    ],
}

# --------------------------------------------------
# 6. STRATIFIED CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True,
)

# --------------------------------------------------
# 7. MODEL TRAINING AND TUNING
# --------------------------------------------------

print("\nStarting SVM hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation ROC-AUC:")
print(round(grid_search.best_score_, 4))

# Save grid search results
pd.DataFrame(
    grid_search.cv_results_
).to_csv(
    GRID_FILE,
    index=False,
)

# --------------------------------------------------
# 8. VALIDATION PREDICTIONS
# --------------------------------------------------

predictions = best_model.predict(X_val)

decision_scores = best_model.decision_function(X_val)

# Ensure positive decision scores represent spoof
svm_classes = best_model.named_steps["svm"].classes_

if svm_classes[1] != "spoof":
    decision_scores = -decision_scores

accuracy = accuracy_score(y_val, predictions)

print(
    "\nTuned SVM Validation Accuracy:",
    round(accuracy, 4)
)

# --------------------------------------------------
# 9. CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
    output_dict=True,
    zero_division=0,
)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        labels=["genuine", "spoof"],
        digits=4,
        zero_division=0,
    )
)

# --------------------------------------------------
# 10. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_val,
    predictions,
    labels=["genuine", "spoof"],
)

print("\nConfusion Matrix:")
print("Class order: ['genuine', 'spoof']")
print(cm)

# --------------------------------------------------
# 11. FALSE ACCEPTANCE RATE AND
#     FALSE REJECTION RATE
# --------------------------------------------------

spoof_total = int((y_val == "spoof").sum())
genuine_total = int((y_val == "genuine").sum())

false_accepts = int(
    (
        (y_val == "spoof")
        & (predictions == "genuine")
    ).sum()
)

false_rejects = int(
    (
        (y_val == "genuine")
        & (predictions == "spoof")
    ).sum()
)

far = (
    false_accepts / spoof_total
    if spoof_total else 0.0
)

frr = (
    false_rejects / genuine_total
    if genuine_total else 0.0
)

print(f"\nFAR: {far:.4f} ({far * 100:.2f}%)")
print(f"FRR: {frr:.4f} ({frr * 100:.2f}%)")

# --------------------------------------------------
# 12. ROC-AUC AND APPROXIMATE EER
# --------------------------------------------------

y_binary = (y_val == "spoof").astype(int)

roc_auc = roc_auc_score(
    y_binary,
    decision_scores,
)

fpr, tpr, roc_thresholds = roc_curve(
    y_binary,
    decision_scores,
)

fnr = 1 - tpr

eer_index = int(
    np.nanargmin(np.abs(fpr - fnr))
)

eer = float(
    (fpr[eer_index] + fnr[eer_index]) / 2
)

eer_threshold = float(
    roc_thresholds[eer_index]
)

print(f"\nValidation ROC-AUC: {roc_auc:.4f}")
print(f"Approximate EER: {eer:.4f} ({eer * 100:.2f}%)")
print(f"EER threshold: {eer_threshold:.6f}")

# --------------------------------------------------
# 13. SAVE TUNED MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_FILE,
)

print("\nTuned model saved to:", MODEL_FILE)

# --------------------------------------------------
# 14. SAVE VALIDATION METRICS
# --------------------------------------------------

metrics = {
    "algorithm": "Tuned SVM with RBF kernel",

    "evaluation_type": (
        "80/20 stratified holdout from training dataset"
    ),

    "dataset": "ASVspoof 2017 V2 training split",

    "training_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),

    "best_parameters": {
        "C": float(
            grid_search.best_params_["svm__C"]
        ),

        "gamma": grid_search.best_params_["svm__gamma"],

        "kernel": "rbf",
    },

    "cross_validation_folds": 5,

    "best_cross_validation_roc_auc": float(
        grid_search.best_score_
    ),

    "accuracy": float(accuracy),

    "roc_auc": float(roc_auc),

    "far_spoof_accepted_as_genuine": float(far),

    "frr_genuine_rejected_as_spoof": float(frr),

    "eer_approximate": eer,

    "eer_threshold_approximate": eer_threshold,

    "confusion_matrix_labels": [
        "genuine",
        "spoof",
    ],

    "confusion_matrix": cm.tolist(),

    "classification_report": report,
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )

print("\nMetrics saved to:", METRICS_FILE)

print("Grid search results saved to:", GRID_FILE)

print("\nSVM Hyperparameter Tuning Complete!")