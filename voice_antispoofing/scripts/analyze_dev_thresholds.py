
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve

PROJECT_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_FILE = (
    PROJECT_DIR / "results" / "svm_dev_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "results" / "svm_threshold_analysis.csv"
)

df = pd.read_csv(PREDICTIONS_FILE)

actual = df["actual_label"]
scores = df["spoof_decision_score"]

# Positive class: spoof
y_true = (actual == "spoof").astype(int)

# Find approximate EER threshold
fpr, tpr, thresholds = roc_curve(y_true, scores)
fnr = 1 - tpr

eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
eer_threshold = float(thresholds[eer_index])
eer = float((fpr[eer_index] + fnr[eer_index]) / 2)

# Compare default and alternative thresholds
candidate_thresholds = [
    0.0,
    0.025,
    0.05,
    eer_threshold,
    0.10,
    0.15,
]

results = []

for threshold in candidate_thresholds:

    predicted = np.where(
        scores >= threshold,
        "spoof",
        "genuine",
    )

    false_accepts = (
        (actual == "spoof") & (predicted == "genuine")
    ).sum()

    false_rejects = (
        (actual == "genuine") & (predicted == "spoof")
    ).sum()

    spoof_total = (actual == "spoof").sum()
    genuine_total = (actual == "genuine").sum()

    far = false_accepts / spoof_total
    frr = false_rejects / genuine_total

    accuracy = accuracy_score(actual, predicted)

    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "FAR": far,
        "FRR": frr,
        "average_error_rate": (far + frr) / 2,
    })

result_df = pd.DataFrame(results)
result_df = result_df.drop_duplicates(subset="threshold")
result_df = result_df.sort_values("threshold")

print("\nDevelopment Threshold Comparison")
print("--------------------------------")
print(result_df.to_string(
    index=False,
    formatters={
        "threshold": "{:.6f}".format,
        "accuracy": "{:.4f}".format,
        "FAR": "{:.4f}".format,
        "FRR": "{:.4f}".format,
        "average_error_rate": "{:.4f}".format,
    }
))

print(f"\nApproximate EER: {eer:.4f} ({eer * 100:.2f}%)")
print(f"EER threshold: {eer_threshold:.6f}")

result_df.to_csv(OUTPUT_FILE, index=False)

print("\nResults saved to:", OUTPUT_FILE)
