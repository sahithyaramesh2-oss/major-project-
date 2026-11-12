"""Test-set visualizations for the RBF SVM experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_curve


def _rates_at_thresholds(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = max(float(np.ptp(scores)) * 0.05, 1e-6)
    thresholds = np.r_[scores.min() - margin, np.unique(scores), scores.max() + margin]
    genuine = labels == 0
    attacks = labels == 1
    far = np.asarray([np.mean(scores[attacks] < value) for value in thresholds])
    frr = np.asarray([np.mean(scores[genuine] >= value) for value in thresholds])
    return thresholds, far, frr


def generate_svm_plots(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    results_dir: str | Path,
) -> None:
    """Write all requested SVM figures using independent out-of-fold test predictions."""
    plot_dir = Path(results_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for feature_set, group in predictions.groupby("feature_set", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        decisions = group["prediction"].to_numpy(dtype=int)
        stem = f"svm_rbf_{feature_set}"

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            labels,
            decisions,
            labels=[0, 1],
            display_labels=["Genuine/live", "Spoof/attack"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"Test confusion matrix: {feature_set.upper()} + RBF SVM")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}_confusion_matrix.png", dpi=180)
        plt.close(fig)

        if len(np.unique(labels)) == 2:
            false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(false_positive_rate, true_positive_rate, label="Out-of-fold test ROC")
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
            ax.set(
                title=f"Test ROC: {feature_set.upper()} + RBF SVM",
                xlabel="False positive rate",
                ylabel="True positive rate (spoof class)",
                xlim=(0, 1),
                ylim=(0, 1.02),
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{stem}_roc.png", dpi=180)
            plt.close(fig)

            precision, recall, _ = precision_recall_curve(labels, scores)
            average_precision = average_precision_score(labels, scores)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(recall, precision, label=f"AP={average_precision:.3f}")
            ax.set(
                title=f"Test precision-recall: {feature_set.upper()} + RBF SVM",
                xlabel="Recall (spoof class)",
                ylabel="Precision (spoof class)",
                xlim=(0, 1),
                ylim=(0, 1.02),
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{stem}_precision_recall.png", dpi=180)
            plt.close(fig)

            thresholds, far, frr = _rates_at_thresholds(labels, scores)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            ax.plot(thresholds, far, label="FAR / APCER")
            ax.plot(thresholds, frr, label="FRR / BPCER")
            ax.set(
                title=f"Test FAR vs FRR: {feature_set.upper()} + RBF SVM",
                xlabel="Spoof probability threshold",
                ylabel="Error rate",
                ylim=(-0.02, 1.02),
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{stem}_far_frr.png", dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.hist(scores[labels == 0], bins=25, alpha=0.7, density=True, label="Genuine/live")
        ax.hist(scores[labels == 1], bins=25, alpha=0.7, density=True, label="Spoof/attack")
        thresholds_used = group["threshold"].to_numpy(dtype=float)
        if len(thresholds_used):
            ax.axvline(float(np.mean(thresholds_used)), color="black", linestyle="--", label="Mean validation threshold")
        ax.set(
            title=f"Test score distribution: {feature_set.upper()} + RBF SVM",
            xlabel="Predicted spoof probability",
            ylabel="Density",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}_score_distribution.png", dpi=180)
        plt.close(fig)

    chart_metrics = [
        ("accuracy_mean", "Accuracy"),
        ("precision_mean", "Precision"),
        ("recall_mean", "Recall"),
        ("f1_mean", "F1"),
        ("far_mean", "FAR"),
        ("frr_mean", "FRR"),
        ("eer_mean", "EER"),
        ("roc_auc_mean", "ROC-AUC"),
        ("apcer_mean", "APCER"),
        ("bpcer_mean", "BPCER"),
        ("acer_mean", "ACER"),
    ]
    available = [(column, label) for column, label in chart_metrics if column in comparison]
    values = comparison.set_index("feature_set")[[column for column, _ in available]].rename(
        columns={column: label for column, label in available}
    ) * 100
    fig, ax = plt.subplots(figsize=(12, 6))
    values.T.plot.bar(ax=ax, width=0.82)
    ax.set(
        title="ROSE-Youtu test performance by feature representation (RBF SVM)",
        xlabel="Metric",
        ylabel="Rate / score (%)",
    )
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(title="Feature set")
    fig.tight_layout()
    fig.savefig(plot_dir / "svm_rbf_performance_comparison.png", dpi=180)
    plt.close(fig)
