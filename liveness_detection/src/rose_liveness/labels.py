"""Verified ROSE-Youtu filename parsing and label conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LIVE_CODE = "G"
ATTACK_CATEGORIES = {
    "Ps": "still_printed_paper",
    "Pq": "quivering_printed_paper",
    "Vl": "lenovo_display_replay",
    "Vm": "mac_display_replay",
    "Mc": "cropped_paper_mask",
    "Mf": "uncropped_paper_mask",
    "Mu": "upper_cut_paper_mask",
    "Ml": "lower_cut_paper_mask",
}
KNOWN_LABEL_CODES = {LIVE_CODE, *ATTACK_CATEGORIES}
OFFICIAL_SOURCE = "https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/"


@dataclass(frozen=True)
class VideoLabel:
    video_id: str
    subject_id: str
    label: int
    attack_category: str
    presentation_code: str
    speaking: str
    device: str
    glasses: str


def parse_filename(path: str | Path) -> VideoLabel:
    """Parse L_S_D_X_E_p_N.mp4 using the official ROSE-Youtu convention."""
    file_path = Path(path)
    parts = file_path.stem.split("_")
    if len(parts) != 7:
        raise ValueError(f"Expected seven underscore-delimited fields: {file_path.name}")

    code, speaking, device, glasses = parts[:4]
    subject, index = parts[5:]
    if code not in KNOWN_LABEL_CODES:
        raise ValueError(f"Unknown ROSE-Youtu presentation code {code!r}: {file_path.name}")
    if speaking not in {"T", "NT"}:
        raise ValueError(f"Unknown speaking code {speaking!r}: {file_path.name}")
    if device not in {"HS", "HW", "IP", "5s", "ZTE"}:
        raise ValueError(f"Unknown recording device {device!r}: {file_path.name}")
    if glasses not in {"g", "wg"}:
        raise ValueError(f"Unknown glasses code {glasses!r}: {file_path.name}")
    if not subject.isdigit() or not index.isdigit():
        raise ValueError(f"Subject and video index must be numeric: {file_path.name}")
    if file_path.parent.name.isdigit() and file_path.parent.name != subject:
        raise ValueError(f"Subject folder does not match filename: {file_path}")

    label = 0 if code == LIVE_CODE else 1
    return VideoLabel(
        video_id=f"{subject}_{index}",
        subject_id=subject,
        label=label,
        attack_category="genuine" if label == 0 else ATTACK_CATEGORIES[code],
        presentation_code=code,
        speaking=speaking,
        device=device,
        glasses=glasses,
    )


def parse_annotation(path: str | Path) -> VideoLabel | None:
    """Return parsed verified labels, or None when a filename is unsupported."""
    try:
        return parse_filename(path)
    except ValueError:
        return None