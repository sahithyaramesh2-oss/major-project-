from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from rose_liveness.labels import parse_filename
from rose_liveness.metrics import compute_metrics, equal_error_rate, validation_threshold


@pytest.mark.parametrize(
    ("name", "expected_label", "category"),
    [
        ("Ps_NT_HW_wg_E_2_131.mp4", 1, "still_printed_paper"),
        ("Pq_NT_ZTE_wg_E_2_120.mp4", 1, "quivering_printed_paper"),
        ("G_NT_5s_g_E_3_1.mp4", 0, "genuine"),
        ("Mc_T_IP_g_E_7_67.mp4", 1, "cropped_paper_mask"),
    ],
)
def test_official_filename_labels(name, expected_label, category):
    parsed = parse_filename(name)
    assert parsed.label == expected_label
    assert parsed.attack_category == category


def test_rejects_unverified_prefix_and_bad_filename():
    with pytest.raises(ValueError):
        parse_filename("Unknown_NT_HW_wg_E_2_1.mp4")
    with pytest.raises(ValueError):
        parse_filename("Ps_NT_HW_2_1.mp4")


def test_known_binary_metrics():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = compute_metrics(y, scores, threshold=0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["apcer"] == pytest.approx(0.5)
    assert metrics["bpcer"] == pytest.approx(0.5)
    assert metrics["acer"] == pytest.approx(0.5)
    assert metrics["far"] == metrics["apcer"]
    assert metrics["frr"] == metrics["bpcer"]
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]


def test_perfect_eer_and_single_class():
    assert equal_error_rate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(0.0)
    assert np.isnan(equal_error_rate(np.array([0, 0]), np.array([0.1, 0.9])))


def test_validation_threshold_is_selected_from_supplied_validation_scores():
    threshold = validation_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert threshold == pytest.approx(0.8)