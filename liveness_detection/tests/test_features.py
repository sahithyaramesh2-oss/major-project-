import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rose_liveness.features import build_feature_cache


def test_stale_feature_cache_is_recomputed_without_reading_old_payload(tmp_path):
    processed_dir = tmp_path / "processed"
    frame_dir = processed_dir / "frames" / "2" / "G_NT_5s_g_E_2_1"
    frame_dir.mkdir(parents=True)
    assert cv2.imwrite(str(frame_dir / "000.png"), np.full((32, 32), 127, dtype=np.uint8))

    cache_dir = processed_dir / "features"
    cache_dir.mkdir()
    stale_cache = {
        "keys": np.array(["old-video:000"]),
        "points": 8,
        "radius": 1,
        "image_size": 32,
        "frames_per_video": 1,
    }
    for name in ("lbp", "hog"):
        np.savez_compressed(cache_dir / f"{name}_features.npz", **stale_cache)

    inventory = [{
        "video_id": "2_1",
        "subject_id": "2",
        "file_path": "original/2/G_NT_5s_g_E_2_1.mp4",
    }]
    features = build_feature_cache(
        inventory,
        processed_dir,
        cache_dir,
        image_size=32,
        frames_per_video=1,
    )

    assert set(features) == {"lbp", "hog"}
    assert set(features["lbp"]) == {"2_1:000"}
    assert set(features["hog"]) == {"2_1:000"}
    assert np.isfinite(features["lbp"]["2_1:000"]).all()
    assert np.isfinite(features["hog"]["2_1:000"]).all()
