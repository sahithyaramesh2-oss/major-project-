"""Cached frame-wise LBP and HOG descriptors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.feature import hog, local_binary_pattern
from tqdm import tqdm


def extract_lbp(image: np.ndarray, points: int = 8, radius: int = 1) -> np.ndarray:
    lbp = local_binary_pattern(image, points, radius, method="uniform")
    histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(points + 3), range=(0, points + 2))
    values = histogram.astype(np.float32)
    total = values.sum()
    return values / total if total else values


def extract_hog(image: np.ndarray) -> np.ndarray:
    values = hog(
        image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True,
    )
    return values.astype(np.float32)


def build_feature_cache(
    inventory: list[dict],
    processed_dir: str | Path,
    cache_dir: str | Path,
    points: int = 8,
    radius: int = 1,
    image_size: int = 128,
    frames_per_video: int = 5,
) -> dict[str, dict[str, np.ndarray]]:
    root = Path(processed_dir) / "frames"
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    expected_keys = set()
    for row in inventory:
        frame_dir = root / str(row["subject_id"]) / Path(row["file_path"]).stem
        expected_keys.update(
            f'{row["video_id"]}:{image_path.stem}'
            for image_path in frame_dir.glob("*.png")
        )
    cached = {}
    if all((cache_path / f"{name}_features.npz").exists() for name in ("lbp", "hog")):
        cache_matches = True
        for name in ("lbp", "hog"):
            with np.load(cache_path / f"{name}_features.npz") as data:
                keys = data["keys"].astype(str)
                cache_matches = cache_matches and (
                    set(keys) == expected_keys
                    and int(data["points"]) == points
                    and int(data["radius"]) == radius
                    and int(data["image_size"]) == image_size
                    and int(data["frames_per_video"]) == frames_per_video
                )
        if cache_matches:
            for name in ("lbp", "hog"):
                with np.load(cache_path / f"{name}_features.npz") as data:
                    keys = data["keys"].astype(str)
                    cached[name] = dict(zip(keys, data["features"]))
            return cached
    result: dict[str, dict[str, np.ndarray]] = {"lbp": {}, "hog": {}}
    import cv2
    for row in tqdm(inventory, desc="Extracting LBP and HOG"):
        frame_dir = root / row["subject_id"] / Path(row["file_path"]).stem
        for image_path in sorted(frame_dir.glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            key = f'{row["video_id"]}:{image_path.stem}'
            result["lbp"][key] = extract_lbp(image, points, radius)
            result["hog"][key] = extract_hog(image)
    if not result["lbp"]:
        raise RuntimeError("No readable extracted frames were available for feature extraction.")
    for name, features in result.items():
        keys = sorted(features)
        matrix = np.vstack([features[key] for key in keys]).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError(f"{name} features contain NaN or infinite values")
        np.savez_compressed(
            cache_path / f"{name}_features.npz",
            keys=np.asarray(keys),
            features=matrix,
            points=points,
            radius=radius,
            image_size=image_size,
            frames_per_video=frames_per_video,
        )
    return result