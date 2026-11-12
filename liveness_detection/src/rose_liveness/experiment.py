"""Subject-disjoint cross-validation and video-level model evaluation."""

from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .metrics import compute_metrics, validation_threshold

def _frame_feature_index(
    inventory: pd.DataFrame,
    frame_features: dict[str, dict[str, np.ndarray]],
    feature_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    records = []
    vectors = []
    selected = ("lbp", "hog") if feature_name == "combined" else (feature_name,)
    indexes = {feature: frame_features[feature] for feature in selected}

    feature_keys_by_video: dict[str, list[str]] = {}
    for key in indexes[selected[0]]:
        video_id = key.rsplit(":", 1)[0]
        feature_keys_by_video.setdefault(video_id, []).append(key)
    for row in inventory.itertuples(index=False):
        keys = sorted(feature_keys_by_video.get(str(row.video_id), []))
        for key in keys:
            if any(key not in indexes[name] for name in selected):
                continue
            vectors.append(np.concatenate([
                indexes[name][key] for name in selected
            ]).astype(np.float32))
            records.append({
                "video_id": str(row.video_id),
                "subject_id": str(row.subject_id),
                "file_path": row.file_path,
                "label": int(row.label),
                "attack_category": row.attack_category,
                "frame_index": key.rsplit(":", 1)[1],
            })
    if not vectors:
        raise RuntimeError(f"No usable frame-level feature vectors found for {feature_name}")
    return pd.DataFrame(records), np.vstack(vectors)


def _aggregate_video_scores(frame_rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    scored = frame_rows.copy()
    scored["score"] = scores
    metadata = ["subject_id", "file_path", "label", "attack_category"]
    videos = scored.groupby("video_id", as_index=False).agg(
        score=("score", "mean"),
        **{column: (column, "first") for column in metadata},
    )
    return videos


def _make_model(model_name: str):
    if model_name == "SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, kernel="rbf", probability=False, class_weight="balanced", random_state=42),
        )
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced_subsample",
        random_state=42, n_jobs=-1, min_samples_leaf=2,
    )


def _spoof_scores(model, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        spoof_index = classes.index(1)
        return model.predict_proba(matrix)[:, spoof_index]
    return np.asarray(model.decision_function(matrix), dtype=float).reshape(-1)


def run_subject_disjoint_experiment(
    inventory: pd.DataFrame,
    frame_features: dict[str, dict[str, np.ndarray]],
    results_dir: str | Path,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = Path(results_dir)
    (output / "csv").mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(parents=True, exist_ok=True)
    usable_inventory = inventory.loc[inventory["readable"].astype(bool)].reset_index(drop=True)
    if smoke_test:
        chosen = []
        for subject in sorted(usable_inventory["subject_id"].astype(str).unique()):
            group = usable_inventory[usable_inventory["subject_id"].astype(str) == subject]
            for label in (0, 1):
                samples = group[group["label"] == label]
                chosen.extend(samples.sample(
                    n=min(6, len(samples)), random_state=42
                ).index.tolist())
        usable_inventory = usable_inventory.loc[sorted(set(chosen))].reset_index(drop=True)
    subjects = sorted(usable_inventory["subject_id"].astype(str).unique(), key=int)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for held-out testing plus subject-disjoint validation.")

    all_results = []
    all_predictions = []
    all_splits = []
    model_to_feature = [
        ("LBP + SVM", "lbp", "SVM"),
        ("LBP + Random Forest", "lbp", "Random Forest"),
        ("HOG + SVM", "hog", "SVM"),
        ("HOG + Random Forest", "hog", "Random Forest"),
        ("LBP + HOG + SVM", "combined", "SVM"),
        ("LBP + HOG + Random Forest", "combined", "Random Forest"),
    ]
    settings = {}
    for model_name, feature_name, classifier_name in model_to_feature:
        settings[model_name] = {
            "feature_set": feature_name,
            "classifier": classifier_name,
            "parameters": _make_model(classifier_name).get_params(),
        }
    (output / "model_settings.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    feature_data = {
        feature_name: _frame_feature_index(usable_inventory, frame_features, feature_name)
        for feature_name in ("lbp", "hog", "combined")
    }
    for test_index, test_subject in enumerate(subjects):
        validation_subject = subjects[(subjects.index(test_subject) + 1) % len(subjects)]
        train_subjects = [subject for subject in subjects if subject not in {test_subject, validation_subject}]
        for model_name, feature_name, classifier_name in model_to_feature:
            rows, matrix = feature_data[feature_name]
            train_mask = rows["subject_id"].isin(train_subjects).to_numpy()
            validation_mask = rows["subject_id"].eq(validation_subject).to_numpy()
            test_mask = rows["subject_id"].eq(test_subject).to_numpy()
            y_train = rows.loc[train_mask, "label"].to_numpy()
            y_validation = rows.loc[validation_mask, "label"].to_numpy()
            y_test = rows.loc[test_mask, "label"].to_numpy()
            if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_validation)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
                all_splits.append({
                    "test_subject": test_subject,
                    "validation_subject": validation_subject,
                    "train_subjects": ",".join(train_subjects),
                    "model": model_name,
                    "status": "skipped_missing_class",
                })
                continue
            model = _make_model(classifier_name)
            start = time.perf_counter()
            model.fit(matrix[train_mask], y_train)
            training_seconds = time.perf_counter() - start
            validation_frame_scores = _spoof_scores(model, matrix[validation_mask])
            validation_videos = _aggregate_video_scores(
                rows.loc[validation_mask], validation_frame_scores
            )
            threshold = validation_threshold(
                validation_videos["label"].to_numpy(),
                validation_videos["score"].to_numpy(),
            )
            test_frame_scores = _spoof_scores(model, matrix[test_mask])
            test_videos = _aggregate_video_scores(rows.loc[test_mask], test_frame_scores)
            y_test_videos = test_videos["label"].to_numpy(dtype=int)
            test_scores = test_videos["score"].to_numpy(dtype=float)
            metrics = compute_metrics(y_test_videos, test_scores, threshold)
            metrics.update({
                "model": model_name,
                "feature_set": feature_name,
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "train_subjects": ",".join(train_subjects),
                "threshold": threshold,
                "training_seconds": training_seconds,
                "n_test_videos": int(len(test_videos)),
                "feature_dimensions": int(matrix.shape[1]),
            })
            all_results.append(metrics)
            test_videos["prediction"] = (test_scores >= threshold).astype(int)
            test_videos["model"] = model_name
            test_videos["threshold"] = threshold
            all_predictions.append(test_videos)
            all_splits.append({
                "test_subject": test_subject,
                "validation_subject": validation_subject,
                "train_subjects": ",".join(train_subjects),
                "model": model_name,
                "status": "completed",
            })
            if test_index == len(subjects) - 1:
                joblib.dump(model, output / "models" / f"{model_name.lower().replace(' ', '_').replace('+', 'plus')}.joblib")

    results = pd.DataFrame(all_results)
    if results.empty:
        raise RuntimeError("No valid subject-disjoint evaluation folds were available.")
    predictions = pd.concat(all_predictions, ignore_index=True)
    splits = pd.DataFrame(all_splits)
    results.to_csv(output / "csv" / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "csv" / "video_predictions.csv", index=False)
    splits.to_csv(output / "csv" / "split_manifest.csv", index=False)
    category_errors = []
    for (model_name, category), group in predictions[predictions["label"] == 1].groupby(["model", "attack_category"]):
        category_errors.append({
            "model": model_name,
            "attack_category": category,
            "videos": len(group),
            "apcer": float(np.mean(group["prediction"].to_numpy() == 0)),
        })
    pd.DataFrame(category_errors).to_csv(output / "csv" / "attack_category_apcer.csv", index=False)
    return results, predictions, splits


def summarize_results(results: pd.DataFrame, results_dir: str | Path) -> pd.DataFrame:
    columns = ["accuracy", "apcer", "bpcer", "acer", "eer", "precision", "recall", "f1", "roc_auc", "training_seconds"]
    summary = results.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["acer_mean", "accuracy_mean"], ascending=[True, False], na_position="last")
    summary.to_csv(Path(results_dir) / "csv" / "model_comparison.csv", index=False)
    markdown = summary.to_markdown(index=False, floatfmt=".4f")
    (Path(results_dir) / "tables").mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / "tables" / "model_comparison.md").write_text(
        "# Subject-disjoint preliminary model comparison\n\n"
        "Values are fold means and standard deviations across valid held-out subjects. "
        "The rank is by lowest mean ACER; threshold selection used a separate validation subject inside each fold.\n\n"
        + markdown + "\n",
        encoding="utf-8",
    )
    return summary