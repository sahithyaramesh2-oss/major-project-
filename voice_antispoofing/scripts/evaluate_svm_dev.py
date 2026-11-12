
from pathlib import Path
import json

import joblib
import librosa
import numpy as np
import pandas as pd

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

DATASET_DIR = PROJECT_DIR / "dataset"
RESULTS_DIR = PROJECT_DIR / "results"
MODELS_DIR = PROJECT_DIR / "models"

AUDIO_DIR = DATASET_DIR / "ASVspoof2017_V2_dev"

PROTOCOL_FILE = (
    DATASET_DIR
    / "protocol_V2"
    / "ASVspoof2017_V2_dev.trl.txt"
)

MODEL_FILE = (
    MODELS_DIR
    / "voice_antispoof_svm_tuned_model.joblib"
)

METRICS_FILE = RESULTS_DIR / "svm_dev_metrics.json"

PREDICTIONS_FILE = (
    RESULTS_DIR / "svm_dev_predictions.csv"
)

# Selected operating threshold from development EER analysis
SELECTED_THRESHOLD = 0.077554

def main():

    # --------------------------------------------------
    # 2. CHECK REQUIRED FILES
    # --------------------------------------------------

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio folder not found: {AUDIO_DIR}"
        )

    if not PROTOCOL_FILE.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {PROTOCOL_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 3. READ PROTOCOL LABELS
    # --------------------------------------------------

    labels = {}

    with open(
        PROTOCOL_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            parts = line.split()

            if len(parts) >= 2:

                filename = parts[0]
                label = parts[1].lower()

                if label in ("genuine", "spoof"):
                    labels[filename] = label

    # --------------------------------------------------
    # 4. LOAD TRAINED SVM MODEL
    # --------------------------------------------------

    model = joblib.load(MODEL_FILE)

    print("Loaded tuned SVM model.")
    print("Model path:", MODEL_FILE)
    print("Selected threshold:", SELECTED_THRESHOLD)
    print("Development recordings:", len(labels))

    # --------------------------------------------------
    # 5. EXTRACT MFCC FEATURES
    # --------------------------------------------------

    filenames = []
    actual_labels = []
    feature_rows = []

    errors = 0

    for audio_path in sorted(AUDIO_DIR.glob("*.wav")):

        try:

            label = labels.get(audio_path.name)

            if label not in ("genuine", "spoof"):

                print(
                    "Missing/invalid label:",
                    audio_path.name,
                )

                errors += 1
                continue

            audio, sr = librosa.load(
                audio_path,
                sr=None,
                mono=True,
            )

            if len(audio) == 0:
                raise ValueError("Empty audio recording.")

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=13,
            )

            features = np.mean(
                mfcc,
                axis=1,
            )

            filenames.append(audio_path.name)
            actual_labels.append(label)
            feature_rows.append(features)

            if len(filenames) % 100 == 0:
                print("Processed:", len(filenames))

        except Exception as exc:

            print(
                "Error:",
                audio_path.name,
                exc,
            )

            errors += 1

    if not feature_rows:
        raise ValueError(
            "No development recordings were processed."
        )

    # --------------------------------------------------
    # 6. PREPARE FEATURE DATAFRAME
    # --------------------------------------------------

    X_dev = pd.DataFrame(
        feature_rows,
        columns=[
            f"mfcc_{i}"
            for i in range(1, 14)
        ],
    )

    y_true = np.array(actual_labels)

    print("\nFeature extraction complete.")
    print("Successfully processed:", len(filenames))
    print("Errors:", errors)

    # --------------------------------------------------
    # 7. GET SVM DECISION SCORES
    # --------------------------------------------------

    decision_scores = model.decision_function(
        X_dev
    )

    # Ensure positive decision scores represent spoof
    svm_classes = model.named_steps["svm"].classes_

    if svm_classes[1] != "spoof":
        decision_scores = -decision_scores

    # --------------------------------------------------
    # 8. APPLY SELECTED THRESHOLD
    # --------------------------------------------------

    predictions = np.where(
        decision_scores >= SELECTED_THRESHOLD,
        "spoof",
        "genuine",
    )

    # --------------------------------------------------
    # 9. CLASSIFICATION REPORT
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=["genuine", "spoof"],
        output_dict=True,
        zero_division=0,
    )

    print(
        "\nDevelopment Accuracy:",
        f"{accuracy:.4f}",
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
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
        y_true,
        predictions,
        labels=["genuine", "spoof"],
    )

    print("\nConfusion Matrix:")
    print("Class order: ['genuine', 'spoof']")
    print(cm)

    # --------------------------------------------------
    # 11. FAR AND FRR
    # --------------------------------------------------

    spoof_total = int(
        (y_true == "spoof").sum()
    )

    genuine_total = int(
        (y_true == "genuine").sum()
    )

    false_accepts = int(
        (
            (y_true == "spoof")
            & (predictions == "genuine")
        ).sum()
    )

    false_rejects = int(
        (
            (y_true == "genuine")
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

    print(
        f"\nSelected threshold: {SELECTED_THRESHOLD:.6f}"
    )

    print(
        f"FAR: {far:.4f} ({far * 100:.2f}%)"
    )

    print(
        f"FRR: {frr:.4f} ({frr * 100:.2f}%)"
    )

    # --------------------------------------------------
    # 12. ROC-AUC AND APPROXIMATE EER
    # --------------------------------------------------

    y_binary = (
        y_true == "spoof"
    ).astype(int)

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
        np.nanargmin(
            np.abs(fpr - fnr)
        )
    )

    eer = float(
        (fpr[eer_index] + fnr[eer_index]) / 2
    )

    eer_threshold = float(
        roc_thresholds[eer_index]
    )

    print(
        f"\nDevelopment ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"Approximate EER: {eer:.4f} ({eer * 100:.2f}%)"
    )

    print(
        f"EER threshold: {eer_threshold:.6f}"
    )

    # --------------------------------------------------
    # 13. SAVE INDIVIDUAL PREDICTIONS
    # --------------------------------------------------

    prediction_df = pd.DataFrame({

        "filename": filenames,

        "actual_label": actual_labels,

        "predicted_label": predictions,

        "spoof_decision_score": decision_scores,

    })

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    # --------------------------------------------------
    # 14. SAVE DEVELOPMENT METRICS
    # --------------------------------------------------

    metrics = {

        "algorithm": "Tuned SVM with RBF kernel",

        "evaluation_type": (
            "Independent development dataset evaluation"
        ),

        "dataset": (
            "ASVspoof 2017 V2 development split"
        ),

        "selected_operating_threshold": (
            SELECTED_THRESHOLD
        ),

        "threshold_selection_method": (
            "Equal Error Rate on development dataset"
        ),

        "total_recordings": len(filenames),

        "genuine_recordings": genuine_total,

        "spoof_recordings": spoof_total,

        "processing_errors": errors,

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

    # --------------------------------------------------
    # 15. COMPLETION
    # --------------------------------------------------

    print("\nMetrics saved to:", METRICS_FILE)

    print(
        "Predictions saved to:",
        PREDICTIONS_FILE,
    )

    print("\nDevelopment evaluation complete!")

if __name__ == "__main__":
    main()