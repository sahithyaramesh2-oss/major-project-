"""Reproducible plots for the inventory and video-level results."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .metrics import roc_points


def generate_plots(
    inventory: pd.DataFrame,
    predictions: pd.DataFrame | None,
    results_dir: str | Path,
    processed_dir: str | Path,
) -> None:
    output = Path(results_dir) / "plots"
    output.mkdir(parents=True, exist_ok=True)

    distribution = inventory["label"].map({0: "Genuine", 1: "Spoof"}).value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    distribution.reindex(["Genuine", "Spoof"], fill_value=0).plot.bar(ax=ax, color=["#3787b8", "#dc6b4a"])
    ax.set(title="Verified video class distribution", xlabel="Class", ylabel="Videos")
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=180)
    plt.close(fig)

    by_subject = inventory.groupby("subject_id").size().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    by_subject.plot.bar(ax=ax, color="#4c78a8")
    ax.set(title="Videos per subject", xlabel="Subject", ylabel="Videos")
    fig.tight_layout()
    fig.savefig(output / "videos_per_subject.png", dpi=180)
    plt.close(fig)

    frames_root = Path(processed_dir) / "frames"
    examples = []
    for label, class_name in ((0, "Genuine"), (1, "Attack")):
        rows = inventory[inventory["label"] == label]
        for row in rows.head(3).itertuples():
            folder = frames_root / str(row.subject_id) / Path(row.file_path).stem
            paths = sorted(folder.glob("*.png"))
            if paths:
                examples.append((class_name, row.attack_category, paths[len(paths) // 2]))
    if examples:
        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        for ax, item in zip(axes.flat, examples):
            name, category, image_path = item
            image = plt.imread(image_path)
            ax.imshow(image, cmap="gray")
            ax.set_title(f"{name}: {category}", fontsize=8)
            ax.axis("off")
        for ax in axes.flat[len(examples):]:
            ax.axis("off")
        fig.suptitle("Sampled frames (face crop when detected; otherwise full-frame fallback)")
        fig.tight_layout()
        fig.savefig(output / "sample_frames.png", dpi=180)
        plt.close(fig)

    if predictions is None or predictions.empty:
        return
    metric_frame = predictions.copy()
    grouped = metric_frame.groupby("model")
    derived = []
    for model, group in grouped:
        y = group["label"].to_numpy(dtype=int)
        thresholded = group["prediction"].to_numpy(dtype=int)
        genuine, attacks = y == 0, y == 1
        apcer = np.mean(thresholded[attacks] == 0) if attacks.any() else np.nan
        bpcer = np.mean(thresholded[genuine] == 1) if genuine.any() else np.nan
        derived.append({
            "model": model,
            "accuracy": np.mean(thresholded == y),
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": (apcer + bpcer) / 2 if genuine.any() and attacks.any() else np.nan,
        })
    metric_table = pd.DataFrame(derived).set_index("model")
    for metric in ("accuracy", "apcer", "bpcer", "acer"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        (metric_table[metric] * 100).sort_values().plot.bar(ax=ax, color="#4c78a8")
        ax.set(title=f"Video-level {metric.upper()} by model", xlabel="Model", ylabel=f"{metric.upper()} (%)")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        fig.savefig(output / f"{metric}_comparison.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    roc_count = 0
    for model, group in grouped:
        curve = roc_points(group["label"].to_numpy(), group["score"].to_numpy())
        if curve is not None:
            fpr, tpr, _ = curve
            ax.plot(fpr, tpr, label=model)
            roc_count += 1
    if roc_count:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set(title="Video-level ROC (out-of-fold predictions)", xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "roc_curves.png", dpi=180)
    plt.close(fig)

    model_names = sorted(predictions["model"].unique())
    columns = min(3, len(model_names))
    rows_count = int(np.ceil(len(model_names) / columns))
    fig, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 4 * rows_count), squeeze=False)
    for ax, model in zip(axes.flat, model_names):
        subset = predictions[predictions["model"] == model]
        ConfusionMatrixDisplay.from_predictions(
            subset["label"], subset["prediction"], labels=[0, 1],
            display_labels=["Genuine", "Spoof"], ax=ax, colorbar=False,
        )
        ax.set_title(model)
    for ax in axes.flat[len(model_names):]:
        ax.axis("off")
    fig.suptitle("Pooled out-of-fold video-level confusion matrices")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrices.png", dpi=180)
    plt.close(fig)

    eer_values = {}
    from .metrics import equal_error_rate
    for model, group in grouped:
        eer_values[model] = equal_error_rate(group["label"].to_numpy(), group["score"].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pd.Series(eer_values).sort_values().mul(100).plot.bar(ax=ax, color="#72a07d")
    ax.set(title="Video-level threshold-swept EER by model", xlabel="Model", ylabel="EER (%)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(output / "eer_comparison.png", dpi=180)
    plt.close(fig)