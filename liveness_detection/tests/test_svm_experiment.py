from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import joblib
from sklearn.model_selection import StratifiedGroupKFold

from rose_liveness.svm_experiment import PARAM_GRID, fit_tuned_svm, make_svm_pipeline, predict_svm


def test_svm_pipeline_uses_scaled_rbf_with_balanced_probability_scores():
    model = make_svm_pipeline()

    assert model.named_steps["scaler"].__class__.__name__ == "StandardScaler"
    svm = model.named_steps["svm"]
    assert svm.kernel == "rbf"
    assert svm.class_weight == "balanced"
    assert svm.probability is True
    assert svm.random_state == 42


def test_svm_grid_matches_requested_search_space():
    assert PARAM_GRID == {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1],
    }


def test_tuning_uses_training_only_and_subject_grouped_cv(tmp_path):
    rng = np.random.default_rng(42)
    subjects = np.repeat(np.array(["1", "2", "3", "4"]), 4)
    labels = np.tile(np.array([0, 0, 1, 1]), 4)
    features = rng.normal(size=(len(labels), 3))
    features[:, 0] += labels * 2.0

    model, parameters, _ = fit_tuned_svm(features, labels, subjects)

    assert model.named_steps["scaler"].mean_ == pytest.approx(features.mean(axis=0))
    assert parameters["svm__C"] in [0.1, 1, 10, 100]
    assert parameters["svm__gamma"] in ["scale", 0.001, 0.01, 0.1]
    assert model.predict_proba(features).shape == (len(labels), 2)

    model_path = tmp_path / "svm.pkl"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)
    predictions, scores = predict_svm(restored, features, threshold=0.5)
    assert predictions.shape == labels.shape
    assert scores.shape == labels.shape

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    for training_indices, validation_indices in splitter.split(features, labels, groups=subjects):
        assert not set(subjects[training_indices]) & set(subjects[validation_indices])
