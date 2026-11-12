"""Deterministic frame sampling and face-region preprocessing."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sample_video_frames(
    path: str | Path,
    frames_per_video: int = 5,
    image_size: int = 128,
) -> tuple[list[np.ndarray], str, int]:
    """Uniformly sample frames; fall back to resized full frames if no face is found."""
    if frames_per_video < 1:
        raise ValueError("frames_per_video must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return [], "open_failed", 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        capture.release()
        return [], "empty_video", 0
    indices = np.linspace(0, frame_count - 1, min(frames_per_video, frame_count), dtype=int)
    cascade_root = getattr(getattr(cv2, "data", None), "haarcascades", None)
    cascade_path = Path(cascade_root) / "haarcascade_frontalface_default.xml" if cascade_root else None
    detector = None
    if cascade_path and cascade_path.is_file() and hasattr(cv2, "CascadeClassifier"):
        candidate = cv2.CascadeClassifier(str(cascade_path))
        if not candidate.empty():
            detector = candidate
    output: list[np.ndarray] = []
    face_count = 0
    for index in np.unique(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resize_factor = max(1.0, gray.shape[1] / 640.0)
        if resize_factor > 1:
            detection_image = cv2.resize(
                gray,
                (int(gray.shape[1] / resize_factor), int(gray.shape[0] / resize_factor)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            detection_image = gray
        detections = (
            detector.detectMultiScale(
                detection_image, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            if detector is not None
            else ()
        )
        if len(detections):
            x, y, width, height = max(detections, key=lambda box: int(box[2]) * int(box[3]))
            x, y = int(x * resize_factor), int(y * resize_factor)
            width, height = int(width * resize_factor), int(height * resize_factor)
            margin_x, margin_y = int(width * 0.12), int(height * 0.12)
            left, top = max(0, x - margin_x), max(0, y - margin_y)
            right = min(frame.shape[1], x + width + margin_x)
            bottom = min(frame.shape[0], y + height + margin_y)
            crop = gray[top:bottom, left:right]
            face_count += 1
        else:
            crop = gray
        if crop.size:
            output.append(cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA))
    capture.release()
    if detector is None:
        status = "detector_unavailable_full_frame_fallback"
    else:
        status = "face_crop" if face_count == len(output) and output else "mixed_or_full_frame_fallback"
    if not output:
        status = "decode_failed"
    return output, status, face_count


def extract_and_save_frames(
    videos: list[dict],
    dataset_root: str | Path,
    processed_dir: str | Path,
    frames_per_video: int,
    image_size: int,
) -> list[dict]:
    root = Path(dataset_root).parent
    output_root = Path(processed_dir) / "frames"

    def process_video(video: dict) -> dict:
        relative = Path(video["file_path"])
        frame_path = output_root / video["subject_id"] / relative.stem
        frame_path.mkdir(parents=True, exist_ok=True)
        existing = sorted(frame_path.glob("*.png"))
        metadata_path = frame_path / "metadata.json"
        expected_count = min(frames_per_video, int(video.get("frame_count") or frames_per_video))
        if existing and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {}
        cache_valid = (
            len(existing) == expected_count
            and metadata.get("frames_per_video") == frames_per_video
            and metadata.get("image_size") == image_size
        )
        if cache_valid:
            status = metadata["face_detection_status"]
            extracted = len(existing)
            face_detected = metadata["frames_face_detected"]
        else:
            for image_path in existing:
                image_path.unlink()
            source = root / relative
            frames, status, face_detected = sample_video_frames(source, frames_per_video, image_size)
            for index, frame in enumerate(frames):
                if not cv2.imwrite(str(frame_path / f"{index:03d}.png"), frame):
                    raise OSError(f"Could not save extracted frame for {source}")
            extracted = len(frames)
            metadata_path.write_text(json.dumps({
                "frames_per_video": frames_per_video,
                "image_size": image_size,
                "face_detection_status": status,
                "frames_face_detected": face_detected,
            }, indent=2) + "\n", encoding="utf-8")
        return {
            "video_id": video["video_id"],
            "subject_id": video["subject_id"],
            "file_path": video["file_path"],
            "label": video["label"],
            "frames_extracted": extracted,
            "face_detection_status": status,
            "frames_face_detected": face_detected,
            "frames_full_frame_fallback": extracted - face_detected if face_detected is not None else None,
        }

    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(
            executor.map(process_video, videos),
            total=len(videos),
            desc="Extracting sampled frames",
        ))