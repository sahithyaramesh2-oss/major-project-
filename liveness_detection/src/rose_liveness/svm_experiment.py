"""Subject-disjoint, video-level RBF SVM experiments."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .experiment import _frame_feature_index
from .metrics import compute_metrics, validation_threshold


PARAM_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
}
FEATURE_CONFIG = {
    "lbp": {"method": "uniform LBP", "points": 8, "radius": 1},
    "hog": {
        "method": "HOG",
        "orientations": 9,
        "pixels_per_cell": [16, 16],
        "cells_per_block": [2, 2],
        "block_norm": "L2-Hys",
    },
    "combined": {"features": ["uniform LBP", "HOG"], "combination": "concatenation"},
}
CLASS_MAPPING = {"0": "genuine/live", "1": "spoof/attack"}
DATASET_NAME = "ROSE-Youtu face PAD"


def make_svm_pipeline() -> Pipeline:
    """Create the common scaled RBF SVM classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_tuned_svm(
    features: np.ndarray,
    labels: np.ndarray,
    subject_groups: np.ndarray,
) -> tuple[Pipeline, dict[str, float | str], float]:
    """Tune an SVM using grouped CV on training samples only."""
    groups = np.asarray(subject_groups).astype(str)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("SVM training data must contain genuine and attack samples.")
    group_count = len(np.unique(groups))
    if group_count < 2:
        raise ValueError("At least two training subjects are required for grouped tuning.")

    splitter = StratifiedGroupKFold(
        n_splits=min(3, group_count),
        shuffle=True,
        random_state=42,
    )
    search = GridSearchCV(
        estimator=make_svm_pipeline(),
        param_grid=PARAM_GRID,
        scoring="balanced_accuracy",
        cv=splitter,
        n_jobs=4,
        pre_dispatch=4,
        refit=True,
        error_score="raise",
    )
    search.fit(features, labels, groups=groups)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _video_feature_index(
    inventory: pd.DataFrame,
    frame_features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Average sampled-frame descriptors into one vector per video."""
    frame_rows, frame_matrix = _frame_feature_index(inventory, frame_features, feature_name)
    rows = []
    vectors = []
    for video_id, indices in frame_rows.groupby("video_id", sort=True).indices.items():
        index_array = np.asarray(indices, dtype=int)
        first = frame_rows.iloc[int(index_array[0])]
        rows.append(
            {
                "video_id": str(video_id),
                "subject_id": str(first["subject_id"]),
                "file_path": first["file_path"],
                "label": int(first["label"]),
                "attack_category": first["attack_category"],
                "n_frames": int(len(index_array)),
            }
        )
        vectors.append(frame_matrix[index_array].mean(axis=0, dtype=np.float64).astype(np.float32))
    if not vectors:
        raise RuntimeError(f"No usable video-level feature vectors found for {feature_name}.")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{feature_name} video features contain NaN or infinite values.")
    return pd.DataFrame(rows), matrix


def _spoof_scores(model: Pipeline, features: np.ndarray) -> np.ndarray:
    class_labels = list(model.named_steps["svm"].classes_)
    spoof_index = class_labels.index(1)
    return model.predict_proba(features)[:, spoof_index]


def predict_svm(
    model: Pipeline,
    features: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and spoof probabilities at a supplied validation threshold."""
    scores = _spoof_scores(model, features)
    return (scores >= threshold).astype(int), scores


def _subject_sort_key(subject_id: str) -> tuple[int, str]:
    return (0, f"{int(subject_id):08d}") if subject_id.isdigit() else (1, subject_id)


def _grouped_subject_splits(
    rows: pd.DataFrame,
    outer_folds: int = 5,
) -> list[tuple[list[str], list[str], list[str]]]:
    """Create test folds and separate grouped validation sets from subjects."""
    labels = rows["label"].to_numpy(dtype=int)
    groups = rows["subject_id"].astype(str).to_numpy()
    subject_count = len(np.unique(groups))
    if subject_count < 4:
        raise ValueError("At least four subjects are required for grouped train, validation, and test splits.")

    outer_splitter = StratifiedGroupKFold(
        n_splits=min(outer_folds, subject_count),
        shuffle=True,
        random_state=42,
    )
    placeholder_features = np.zeros((len(rows), 1), dtype=np.uint8)
    splits = []
    for fold_index, (train_validation_indices, test_indices) in enumerate(
        outer_splitter.split(placeholder_features, labels, groups=groups)
    ):
        train_validation_groups = groups[train_validation_indices]
        validation_folds = min(4, len(np.unique(train_validation_groups)))
        validation_splitter = StratifiedGroupKFold(
            n_splits=validation_folds,
            shuffle=True,
            random_state=42 + fold_index,
        )
        training_relative_indices, validation_relative_indices = next(
            validation_splitter.split(
                placeholder_features[train_validation_indices],
                labels[train_validation_indices],
                groups=train_validation_groups,
            )
        )
        training_groups = train_validation_groups[training_relative_indices]
        validation_groups = train_validation_groups[validation_relative_indices]
        test_groups = groups[test_indices]
        splits.append(
            (
                sorted(set(training_groups), key=_subject_sort_key),
                sorted(set(validation_groups), key=_subject_sort_key),
                sorted(set(test_groups), key=_subject_sort_key),
            )
        )
    return splits


def _modal_value(values: pd.Series) -> str:
    counts = Counter(str(value) for value in values)
    highest_count = max(counts.values())
    return sorted(value for value, count in counts.items() if count == highest_count)[0]


def run_svm_experiment(
    inventory: pd.DataFrame,
    frame_features: dict[str, dict[str, np.ndarray]],
    results_dir: str | Path,
    smoke_test: bool = False,
    feature_configuration: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate LBP, HOG, and combined RBF SVMs on held-out subjects."""
    output = Path(results_dir)
    active_feature_config = feature_configuration or FEATURE_CONFIG
    csv_dir = output / "csv"
    model_dir = output / "models"
    fold_model_dir = model_dir / "folds"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fold_model_dir.mkdir(parents=True, exist_ok=True)

    usable_inventory = inventory.loc[inventory["readable"].astype(bool)].reset_index(drop=True)
    if usable_inventory["subject_id"].astype(str).nunique() < 4:
        raise ValueError(
            "At least four readable subjects are required for independent train, "
            "validation, test, and grouped-CV partitions."
        )

    feature_data = {
        feature_name: _video_feature_index(usable_inventory, frame_features, feature_name)
        for feature_name in ("lbp", "hog", "combined")
    }
    video_id_sets = [
        set(rows["video_id"].astype(str))
        for rows, _ in feature_data.values()
    ]
    if any(video_ids != video_id_sets[0] for video_ids in video_id_sets[1:]):
        raise RuntimeError("LBP, HOG, and combined features do not cover the same videos.")
    subject_splits = _grouped_subject_splits(feature_data["lbp"][0])
    subjects = sorted(
        feature_data["lbp"][0]["subject_id"].astype(str).unique(),
        key=_subject_sort_key,
    )
    all_results: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    all_splits: list[dict] = []
    model_manifest: dict[str, dict] = {}

    for fold_index, (train_subjects, validation_subjects, test_subjects) in enumerate(
        subject_splits,
        start=1,
    ):
        for feature_name, (rows, matrix) in feature_data.items():
            train_mask = rows["subject_id"].isin(train_subjects).to_numpy()
            validation_mask = rows["subject_id"].isin(validation_subjects).to_numpy()
            test_mask = rows["subject_id"].isin(test_subjects).to_numpy()
            y_train = rows.loc[train_mask, "label"].to_numpy(dtype=int)
            y_validation = rows.loc[validation_mask, "label"].to_numpy(dtype=int)
            y_test = rows.loc[test_mask, "label"].to_numpy(dtype=int)
            model_name = f"SVM RBF + {feature_name.upper()}"

            if (
                set(np.unique(y_train)) != {0, 1}
                or set(np.unique(y_validation)) != {0, 1}
                or set(np.unique(y_test)) != {0, 1}
            ):
                all_splits.append(
                    {
                        "dataset_name": DATASET_NAME,
                        "outer_fold": fold_index,
                        "test_subjects": ",".join(test_subjects),
                        "validation_subjects": ",".join(validation_subjects),
                        "train_subjects": ",".join(train_subjects),
                        "model": model_name,
                        "status": "excluded_missing_class",
                    }
                )
                continue

            start = time.perf_counter()
            model, best_params, cv_score = fit_tuned_svm(
                matrix[train_mask],
                y_train,
                rows.loc[train_mask, "subject_id"].to_numpy(),
            )
            training_seconds = time.perf_counter() - start

            validation_scores = _spoof_scores(model, matrix[validation_mask])
            threshold = validation_threshold(y_validation, validation_scores)
            validation_metrics = compute_metrics(y_validation, validation_scores, threshold)

            test_rows = rows.loc[test_mask].reset_index(drop=True).copy()
            test_predictions, test_scores = predict_svm(
                model,
                matrix[test_mask],
                threshold,
            )
            test_metrics = compute_metrics(y_test, test_scores, threshold)
            test_metrics.update(
                {
                    "dataset_name": DATASET_NAME,
                    "model": model_name,
                    "feature_set": feature_name,
                    "outer_fold": fold_index,
                    "test_subjects": ",".join(test_subjects),
                    "validation_subjects": ",".join(validation_subjects),
                    "train_subjects": ",".join(train_subjects),
                    "validation_accuracy": validation_metrics["accuracy"],
                    "threshold": threshold,
                    "best_c": float(best_params["svm__C"]),
                    "best_gamma": str(best_params["svm__gamma"]),
                    "cv_balanced_accuracy": cv_score,
                    "training_seconds": training_seconds,
                    "n_test_videos": int(len(test_rows)),
                    "n_test_genuine": int(np.sum(y_test == 0)),
                    "n_test_spoof": int(np.sum(y_test == 1)),
                    "feature_dimensions": int(matrix.shape[1]),
                    "smoke_test": bool(smoke_test),
                }
            )
            all_results.append(test_metrics)
            test_rows["score"] = test_scores
            test_rows["prediction"] = test_predictions
            test_rows["model"] = model_name
            test_rows["feature_set"] = feature_name
            test_rows["threshold"] = threshold
            all_predictions.append(test_rows)
            all_splits.append(
                {
                    "dataset_name": DATASET_NAME,
                    "outer_fold": fold_index,
                    "test_subjects": ",".join(test_subjects),
                    "validation_subjects": ",".join(validation_subjects),
                    "train_subjects": ",".join(train_subjects),
                    "model": model_name,
                    "status": "completed",
                    "cv_subject_folds": min(3, len(train_subjects)),
                    "best_c": float(best_params["svm__C"]),
                    "best_gamma": str(best_params["svm__gamma"]),
                    "decision_threshold": threshold,
                    "cv_balanced_accuracy": cv_score,
                    "n_train_videos": int(train_mask.sum()),
                    "n_validation_videos": int(validation_mask.sum()),
                    "n_test_videos": int(test_mask.sum()),
                }
            )

            model_path = fold_model_dir / (
                f"svm_rbf_face_{feature_name}_test_fold_{fold_index}.pkl"
            )
            joblib.dump(model, model_path, compress=3)
            model_manifest[f"{feature_name}:test_fold_{fold_index}"] = {
                "model_file": f"folds/{model_path.name}",
                "dataset_name": DATASET_NAME,
                "classifier": "sklearn.svm.SVC(kernel='rbf')",
                "feature_set": feature_name,
                "feature_dimensions": int(matrix.shape[1]),
                "feature_configuration": active_feature_config[feature_name],
                "best_parameters": {
                    "C": float(best_params["svm__C"]),
                    "gamma": str(best_params["svm__gamma"]),
                },
                "decision_threshold": threshold,
                "score_class": 1,
                "score_interpretation": "probability of spoof/attack",
                "train_subjects": train_subjects,
                "validation_subjects": validation_subjects,
                "held_out_test_subjects": test_subjects,
            }

    results = pd.DataFrame(all_results)
    if results.empty:
        raise RuntimeError("No valid subject-disjoint SVM evaluation folds were available.")
    predictions = pd.concat(all_predictions, ignore_index=True)
    splits = pd.DataFrame(all_splits)
    results.to_csv(csv_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(csv_dir / "video_predictions.csv", index=False)
    splits.to_csv(csv_dir / "split_manifest.csv", index=False)
    (model_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    usable_rows = feature_data["lbp"][0]
    usable_videos = len(usable_rows)
    class_counts = usable_rows["label"].value_counts().to_dict()
    usable_video_ids = set(usable_rows["video_id"].astype(str))
    readable_video_ids = set(usable_inventory["video_id"].astype(str))
    missing_feature_ids = readable_video_ids - usable_video_ids
    exclusions = usable_inventory.loc[
        usable_inventory["video_id"].astype(str).isin(missing_feature_ids),
        ["video_id", "subject_id", "file_path", "label", "attack_category"],
    ].copy()
    exclusions["exclusion_reason"] = "No readable extracted frame feature was available."
    exclusions.to_csv(csv_dir / "feature_exclusions.csv", index=False)
    experiment_metadata = {
        "dataset_name": DATASET_NAME,
        "classifier": "Support Vector Machine with RBF kernel",
        "classifier_parameters": {
            "kernel": "rbf",
            "C_initial": 1.0,
            "gamma_initial": "scale",
            "class_weight": "balanced",
            "probability": True,
            "random_state": 42,
        },
        "parameter_grid": PARAM_GRID,
        "selection_metric": "balanced_accuracy",
        "cross_validation": "3-fold StratifiedGroupKFold grouped by training subject",
        "outer_evaluation": f"{len(subject_splits)}-fold StratifiedGroupKFold by subject; each subject held out once",
        "validation_split": "separate grouped validation partition from each outer training set",
        "threshold_selection": "minimum validation ACER; fixed for corresponding held-out test fold",
        "reported_metrics": "independent held-out test predictions only",
        "evaluation_unit": "video",
        "score_class": 1,
        "class_mapping": CLASS_MAPPING,
        "feature_configuration": active_feature_config,
        "subject_ids": subjects,
        "outer_folds": int(len(subject_splits)),
        "n_readable_videos": int(len(usable_inventory)),
        "n_usable_videos": int(usable_videos),
        "n_missing_feature_videos": int(len(missing_feature_ids)),
        "n_genuine_videos": int(class_counts.get(0, 0)),
        "n_spoof_videos": int(class_counts.get(1, 0)),
        "smoke_test": bool(smoke_test),
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(experiment_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, predictions, splits


def summarize_svm_results(
    results: pd.DataFrame,
    results_dir: str | Path,
) -> pd.DataFrame:
    """Write fold-mean comparison metrics and selected-parameter summaries."""
    metrics = [
        "validation_accuracy",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "far",
        "frr",
        "eer",
        "roc_auc",
        "apcer",
        "bpcer",
        "acer",
        "threshold",
        "training_seconds",
    ]
    summaries = []
    for (dataset_name, feature_set, model_name), group in results.groupby(
        ["dataset_name", "feature_set", "model"], sort=True
    ):
        row = {
            "dataset_name": dataset_name,
            "feature_set": feature_set,
            "model": model_name,
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        row["best_c"] = _modal_value(group["best_c"])
        row["best_gamma"] = _modal_value(group["best_gamma"])
        row["best_c_selections"] = json.dumps(
            {key: int(value) for key, value in Counter(group["best_c"].astype(str)).items()},
            sort_keys=True,
        )
        row["best_gamma_selections"] = json.dumps(
            {key: int(value) for key, value in Counter(group["best_gamma"].astype(str)).items()},
            sort_keys=True,
        )
        row["total_samples"] = int(group["n_test_videos"].sum())
        row["genuine_samples"] = int(group["n_test_genuine"].sum())
        row["spoof_samples"] = int(group["n_test_spoof"].sum())
        summaries.append(row)

    summary = pd.DataFrame(summaries).sort_values(
        ["acer_mean", "accuracy_mean"],
        ascending=[True, False],
        na_position="last",
    )
    csv_dir = Path(results_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_dir / "model_comparison.csv", index=False)
    metrics_dir = Path(results_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for feature_set, group in results.groupby("feature_set", sort=True):
        pooled_confusion = np.sum(
            np.asarray(group["confusion_matrix"].tolist(), dtype=int),
            axis=0,
        )
        report = {
            "dataset_name": DATASET_NAME,
            "feature_set": feature_set,
            "classifier": "Support Vector Machine with RBF kernel",
            "class_mapping": CLASS_MAPPING,
            "summary": summary.loc[summary["feature_set"] == feature_set]
            .iloc[0]
            .replace({np.nan: None})
            .to_dict(),
            "pooled_test_confusion_matrix_labels_genuine_attack": pooled_confusion.tolist(),
            "fold_test_metrics": group.replace({np.nan: None}).to_dict(orient="records"),
        }
        path = metrics_dir / f"face_{feature_set}_metrics.json"
        path.write_text(
            json.dumps(report, indent=2, allow_nan=False, default=str) + "\n",
            encoding="utf-8",
        )
    return summary


def inspect_client_protocol(
    client_dataset_root: str | Path,
    results_dir: str | Path,
) -> dict:
    """Record whether the auxiliary client split has usable supervised train/test data."""
    root = Path(client_dataset_root)
    split_details = {}
    for split_name in ("adaptation", "test"):
        list_path = root / f"{split_name}_list.txt"
        image_dir = root / "rgb" / split_name
        label_counts: Counter[str] = Counter()
        record_count = 0
        if list_path.is_file():
            with list_path.open(encoding="utf-8") as file:
                for line in file:
                    fields = line.split()
                    if len(fields) >= 3:
                        record_count += 1
                        label_counts[fields[-1]] += 1
        image_count = (
            sum(1 for path in image_dir.rglob("*.jpg") if path.is_file())
            if image_dir.is_dir()
            else 0
        )
        split_details[split_name] = {
            "annotation_records": record_count,
            "annotation_label_counts": dict(sorted(label_counts.items())),
            "available_rgb_images": image_count,
        }

    adaptation_labels = set(split_details["adaptation"]["annotation_label_counts"])
    test_record_count = split_details["test"]["annotation_records"]
    test_image_count = split_details["test"]["available_rgb_images"]
    reasons = []
    if adaptation_labels != {"0", "1"}:
        reasons.append("The adaptation list does not contain both bona fide and attack labels.")
    if test_image_count < test_record_count:
        reasons.append("The local RGB test images do not cover every annotated test record.")
    reason = " ".join(reasons) or None
    status = {
        "dataset_name": "ROSE-Youtu auxiliary client RGB split",
        "status": "not_evaluated" if reason else "eligible_for_separate_evaluation",
        "split_details": split_details,
        "reason_not_evaluated": reason,
        "note": (
            "This is an auxiliary processed split, not an additional modality. "
            "It is not mixed with original-video samples."
        ),
    }
    target = Path(results_dir) / "client_dataset_status.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status
