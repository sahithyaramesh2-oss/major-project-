"""Command-line entry point for dataset inspection and experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dataset import discover_videos, inventory_dataset, write_dataset_report
from .dashboard import generate_dashboard
from .features import build_feature_cache
from .svm_experiment import (
    inspect_client_protocol,
    run_svm_experiment,
    summarize_svm_results,
)
from .svm_plots import generate_svm_plots
from .video import extract_and_save_frames


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiment_config.json"
RESULTS = ROOT / "results"
SVM_RESULTS = RESULTS / "svm_rbf"


def _read_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _smoke_subset(inventory: pd.DataFrame, config: dict) -> pd.DataFrame:
    subjects = sorted(inventory["subject_id"].astype(str).unique(), key=int)
    needed = 4
    if len(subjects) < needed:
        raise ValueError("A grouped-CV smoke evaluation requires at least four subjects.")
    limit = min(int(config.get("smoke_test_subjects", needed)), len(subjects))
    selected_subjects = subjects[:max(needed, limit)]
    subset = inventory[inventory["subject_id"].astype(str).isin(selected_subjects)]
    selected = []
    per_class = int(config.get("smoke_test_videos_per_class_subject", 6))
    for _, group in subset.groupby("subject_id", sort=True):
        for label in (0, 1):
            selected.extend(group[group["label"] == label].sample(
                n=min(per_class, int((group["label"] == label).sum())),
                random_state=42,
            ).index.tolist())
    return subset.loc[sorted(selected)].reset_index(drop=True)


def run(phase: str, smoke_test: bool = False) -> None:
    config = _read_config()
    dataset_root = ROOT / config["dataset_root"]
    processed_dir = ROOT / "datasets" / "processed"
    if phase == "inspect":
        inventory = inventory_dataset(dataset_root, RESULTS)
        write_dataset_report(inventory, RESULTS / "dataset_inspection.md", config["source_url"])
        print(f"Inspected {len(inventory)} videos; inventory and verified labels written.")
        return

    inventory_file = RESULTS / "dataset_inventory.csv"
    discovered_paths = {
        path.relative_to(dataset_root.parent).as_posix()
        for path in discover_videos(dataset_root)
    }
    cached_paths = set()
    if inventory_file.exists():
        cached_paths = {
            Path(path).as_posix()
            for path in pd.read_csv(inventory_file, dtype={"subject_id": str})["file_path"].dropna()
        }
    if not inventory_file.exists() or discovered_paths != cached_paths:
        inventory = inventory_dataset(dataset_root, RESULTS)
        write_dataset_report(inventory, RESULTS / "dataset_inspection.md", config["source_url"])
    else:
        inventory = pd.read_csv(inventory_file, dtype={"subject_id": str})
    if phase == "dashboard":
        client_root = ROOT / config["client_dataset_root"] / "client"
        inspect_client_protocol(client_root, SVM_RESULTS)
        dashboard_path = generate_dashboard(inventory, RESULTS)
        print(f"Results dashboard created: {dashboard_path}")
        return
    if smoke_test:
        inventory = _smoke_subset(inventory, config)
        print(f"SMOKE TEST: using {len(inventory)} real videos from {inventory.subject_id.nunique()} subjects; do not report as scientific evaluation.")

    if phase == "plots":
        prediction_path = SVM_RESULTS / "csv" / "video_predictions.csv"
        fold_path = SVM_RESULTS / "csv" / "fold_metrics.csv"
        if not prediction_path.exists() or not fold_path.exists():
            raise FileNotFoundError("Run --phase train or --phase all before generating evaluation plots.")
        predictions = pd.read_csv(prediction_path)
        summary = pd.read_csv(SVM_RESULTS / "csv" / "model_comparison.csv")
        generate_svm_plots(predictions, summary, SVM_RESULTS)
        dashboard_path = generate_dashboard(inventory, RESULTS)
        print("SVM test plots and dashboard regenerated.")
        print(f"Results dashboard created: {dashboard_path}")
        return

    if phase in {"preprocess", "features", "train", "evaluate", "all"}:
        summaries = extract_and_save_frames(
            inventory.to_dict("records"),
            dataset_root,
            processed_dir,
            int(config["frames_per_video"]),
            int(config["image_size"]),
        )
        frame_dir = RESULTS / "csv"
        frame_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(frame_dir / "frame_extraction_summary.csv", index=False)
        print(f"Frame extraction summaries written for {len(summaries)} videos.")
        if phase == "preprocess":
            return

    if phase in {"features", "train", "evaluate", "all"}:
        cache_dir = processed_dir / ("smoke_test_features" if smoke_test else "features")
        features = build_feature_cache(
            inventory.to_dict("records"),
            processed_dir,
            cache_dir,
            int(config["lbp_points"]),
            int(config["lbp_radius"]),
            int(config["image_size"]),
            int(config["frames_per_video"]),
        )
        print(f"Feature vectors cached: LBP={len(features['lbp'])}, HOG={len(features['hog'])}.")
        if phase == "features":
            return

    if phase in {"train", "evaluate", "all"}:
        feature_configuration = {
            "lbp": {
                "method": "uniform LBP",
                "points": int(config["lbp_points"]),
                "radius": int(config["lbp_radius"]),
                "image_size": int(config["image_size"]),
            },
            "hog": {
                "method": "HOG",
                "orientations": 9,
                "pixels_per_cell": [16, 16],
                "cells_per_block": [2, 2],
                "block_norm": "L2-Hys",
                "image_size": int(config["image_size"]),
            },
            "combined": {
                "features": ["uniform LBP", "HOG"],
                "combination": "concatenation",
            },
        }
        client_root = ROOT / config["client_dataset_root"] / "client"
        inspect_client_protocol(client_root, SVM_RESULTS)
        folds, predictions, splits = run_svm_experiment(
            inventory,
            features,
            SVM_RESULTS,
            smoke_test=smoke_test,
            feature_configuration=feature_configuration,
        )
        summary = summarize_svm_results(folds, SVM_RESULTS)
        print(f"Completed {int((splits.status == 'completed').sum())} SVM model/fold evaluations.")
        if phase in {"train", "evaluate"}:
            generate_svm_plots(predictions, summary, SVM_RESULTS)
            dashboard_path = generate_dashboard(inventory, RESULTS)
            print(f"SVM results dashboard created: {dashboard_path}")
            return

    if phase == "all":
        generate_svm_plots(predictions, summary, SVM_RESULTS)
        dashboard_path = generate_dashboard(inventory, RESULTS)
        print("SVM test plots, comparison table, and split-aware dashboard written.")
        print(f"Results dashboard created: {dashboard_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ROSE-Youtu face presentation attack detection experiment")
    parser.add_argument(
        "--phase",
        choices=("inspect", "preprocess", "features", "train", "evaluate", "plots", "dashboard", "all"),
        default="all",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a small real-data pipeline check, not a scientific evaluation")
    args = parser.parse_args()
    run(args.phase, args.smoke_test)


if __name__ == "__main__":
    main()