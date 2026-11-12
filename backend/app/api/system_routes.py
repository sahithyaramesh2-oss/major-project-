import os
from pathlib import Path
from typing import List, Dict, Any
import cv2
from fastapi import APIRouter, HTTPException, Response
from app.config import settings
from app.core.face_engine import face_engine
from app.database import engine

router = APIRouter(prefix="/system", tags=["System & Dataset"])

@router.get("/status")
def get_system_status() -> Dict[str, Any]:
    """Check health of the backend, MySQL connection, and InsightFace model."""
    db_ok = False
    db_type = "unknown"
    try:
        with engine.connect() as conn:
            db_ok = True
            db_type = "MySQL" if "mysql" in str(engine.url) else "SQLite"
    except Exception as e:
        db_type = f"Error: {e}"

    face_model_ok = face_engine.app is not None

    return {
        "status": "ONLINE" if (db_ok and face_model_ok) else "DEGRADED",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_type,
        "database_connected": db_ok,
        "face_model": "InsightFace ArcFace (buffalo_l)",
        "models_ready": face_model_ok,
        "similarity_threshold": settings.FACE_SIMILARITY_THRESHOLD,
        "encryption": "AES-256-GCM Authenticated Encryption"
    }

@router.get("/samples")
def list_available_samples() -> Dict[str, Any]:
    """List available real dataset samples from MSU_AVIS for interactive testing."""
    dataset_path = Path(settings.DATASET_PATH)
    gallery_face_path = dataset_path / "Gallery" / "face"
    probes_path = dataset_path / "Probe_clips"

    gallery_files = []
    if gallery_face_path.exists():
        for f in sorted(gallery_face_path.glob("*.mp4"))[:12]:
            gallery_files.append({
                "filename": f.name,
                "path": str(f),
                "subject_hint": f.stem.replace("G_F", "Subject ")
            })

    probe_samples = []
    if probes_path.exists():
        # Pick clips from S01_V1 (genuine for Subj 1), S02_V1 (impostor for Subj 1, genuine for Subj 2), etc.
        for session_dir in sorted(probes_path.glob("S*_V1"))[:6]:
            clips = list(sorted(session_dir.glob("*.mp4")))
            for clip in clips[:3]:
                probe_samples.append({
                    "session": session_dir.name,
                    "filename": clip.name,
                    "rel_path": f"{session_dir.name}/{clip.name}",
                    "subject": session_dir.name.split("_")[0]
                })

    return {
        "dataset_found": dataset_path.exists(),
        "gallery_faces": gallery_files,
        "probe_samples": probe_samples
    }

@router.get("/sample-frame/{category}/{subpath:path}")
def get_sample_frame(category: str, subpath: str):
    """
    Extract a middle frame from an MSU_AVIS video and return it as image/jpeg.
    category: 'gallery' or 'probe'
    """
    dataset_path = Path(settings.DATASET_PATH)
    if category == "gallery":
        video_path = dataset_path / "Gallery" / "face" / subpath
    elif category == "probe":
        video_path = dataset_path / "Probe_clips" / subpath
    else:
        raise HTTPException(status_code=400, detail="Invalid category. Use 'gallery' or 'probe'.")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample video not found: {video_path}")

    frame = face_engine.extract_frame_from_video(str(video_path))
    if frame is None:
        raise HTTPException(status_code=500, detail="Could not extract frame from sample video.")

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode JPEG frame.")

    return Response(content=buffer.tobytes(), media_type="image/jpeg")
