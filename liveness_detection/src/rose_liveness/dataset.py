"""Dataset discovery and inventory generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from .labels import OFFICIAL_SOURCE, parse_filename

INVENTORY_COLUMNS = [
    "file_path", "filename", "subject_id", "video_id", "duration_seconds",
    "fps", "frame_count", "width", "height", "label", "attack_category",
    "presentation_code", "device", "speaking", "glasses", "readable",
    "decode_error",
]


def discover_videos(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4")


def inspect_video(path: Path, root: Path) -> dict:
    parsed = parse_filename(path)
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    readable = bool(capture.isOpened())
    decode_error = ""
    if readable:
        ok, frame = capture.read()
        readable = bool(ok and frame is not None and frame.size)
        if not readable:
            decode_error = "Could not decode the first frame"
    else:
        decode_error = "OpenCV could not open the video"
    capture.release()

    return {
        "file_path": path.relative_to(root.parent).as_posix(),
        "filename": path.name,
        "subject_id": parsed.subject_id,
        "video_id": parsed.video_id,
        "duration_seconds": frame_count / fps if fps > 0 else None,
        "fps": fps or None,
        "frame_count": frame_count or None,
        "width": width or None,
        "height": height or None,
        "label": parsed.label,
        "attack_category": parsed.attack_category,
        "presentation_code": parsed.presentation_code,
        "device": parsed.device,
        "speaking": parsed.speaking,
        "glasses": parsed.glasses,
        "readable": readable,
        "decode_error": decode_error,
    }


def inventory_dataset(dataset_root: str | Path, results_dir: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    videos = discover_videos(root)
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found under {root}")
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(tqdm(
            executor.map(partial(inspect_video, root=root), videos),
            total=len(videos),
            desc="Inspecting videos",
        ))
    inventory = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out / "dataset_inventory.csv", index=False)
    metadata_dir = root.parent / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    inventory[[
        "video_id", "subject_id", "file_path", "filename", "label",
        "attack_category", "presentation_code", "device", "speaking", "glasses",
    ]].to_csv(metadata_dir / "video_labels.csv", index=False)

    csv_dir = out / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    class_counts = inventory.groupby(["label", "attack_category"], dropna=False).size().rename("video_count").reset_index()
    class_counts["class_name"] = class_counts["label"].map({0: "genuine", 1: "spoof"})
    class_counts.to_csv(csv_dir / "class_distribution.csv", index=False)
    return inventory


def write_dataset_report(inventory: pd.DataFrame, report_path: str | Path, source: str = OFFICIAL_SOURCE) -> None:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    subjects = sorted(inventory["subject_id"].dropna().unique(), key=lambda value: int(value))
    subject_counts = inventory.groupby("subject_id").size().sort_index()
    class_counts = inventory["label"].value_counts().to_dict()
    required_training_subjects = {"2", "3", "4", "5", "6", "7", "9", "10", "11", "12"}
    missing_training_subjects = sorted(
        required_training_subjects.difference(subjects),
        key=lambda value: int(value),
    )
    attacks = inventory.loc[inventory.label == 1, "attack_category"].value_counts().sort_index()
    failed = inventory.loc[~inventory["readable"].astype(bool), ["file_path", "decode_error"]]
    lines = [
        "# Dataset inspection", "",
        f"- Official source and naming documentation: {source}",
        f"- Discovered MP4 videos: {len(inventory)}",
        f"- Subject folders represented: {len(subjects)} ({', '.join(subjects)})",
        f"- Verified genuine videos: {int(class_counts.get(0, 0))}",
        f"- Verified attack videos: {int(class_counts.get(1, 0))}",
        f"- Readable under sampled OpenCV decoding: {int(inventory['readable'].sum())}/{len(inventory)}",
        "- Labels use the documented first filename field; `G` is genuine and documented attack codes are spoof. No inferred labels are used.",
        "- The `datasets/client/` preprocessed data and its protocol lists were not included in this original-video experiment.",
        "", "## Videos by subject", "", "| Subject | Videos |", "|---|---:|",
    ]
    lines.extend(f"| {subject} | {int(count)} |" for subject, count in subject_counts.items())
    lines.extend(["", "## Verified attack categories", "", "| Category | Videos |", "|---|---:|"])
    lines.extend(f"| {category} | {int(count)} |" for category, count in attacks.items())
    lines.extend(["", "## Video properties", "", "| Property | Summary |", "|---|---|"])
    for column in ("duration_seconds", "fps", "frame_count", "width", "height"):
        numeric = pd.to_numeric(inventory[column], errors="coerce")
        lines.append(f"| {column} | min {numeric.min():.2f}; median {numeric.median():.2f}; max {numeric.max():.2f} |")
    lines.extend(["", "## Unreadable or incomplete videos", ""])
    if failed.empty:
        lines.append("No files failed the sampled decode checks.")
    else:
        lines.extend(f"- `{row.file_path}`: {row.decode_error}" for row in failed.itertuples())
    lines.extend([
        "", "## Coverage and limitations", "",
        "The official ROSE-Youtu page describes 4,225 total videos and a 20-subject public release. The local original-video folder is a partial download. "
        "The published intra-dataset training partition uses subjects 2, 3, 4, 5, 6, 7, 9, 10, 11, and 12; "
        f"missing required training subjects in this local inventory: {', '.join(missing_training_subjects) if missing_training_subjects else 'none'}. "
        "The complete published held-out partition is not verified from the local snapshot, so this report does not claim to reproduce the official protocol. "
        "Use the recorded subject-disjoint evaluation and consult the official dataset documentation for benchmark splits.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")