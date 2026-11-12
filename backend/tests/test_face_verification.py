import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.database import SessionLocal, engine, Base
from app.models.user import User, BiometricTemplate, AuthLog
from app.core.face_engine import face_engine
from app.security.encryption import decrypt_embedding

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # Clean test users
    db.query(AuthLog).filter(AuthLog.user_id.in_(["test_subject_01", "test_subject_02"])).delete(synchronize_session=False)
    db.query(BiometricTemplate).filter(BiometricTemplate.user_id.in_(["test_subject_01", "test_subject_02"])).delete(synchronize_session=False)
    db.query(User).filter(User.user_id.in_(["test_subject_01", "test_subject_02"])).delete(synchronize_session=False)
    db.commit()
    db.close()
    yield

def test_system_status():
    """Verify backend system status and model readiness."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["database_connected"] is True
    assert data["models_ready"] is True
    assert "ArcFace" in data["face_model"]

def test_user_registration():
    """Verify registration endpoint."""
    payload = {
        "user_id": "test_subject_01",
        "full_name": "MSU Subject 01",
        "email": "subj01@example.com"
    }
    response = client.post("/api/v1/users/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["user_id"] == "test_subject_01"
    assert data["is_face_enrolled"] is False

from unittest.mock import patch
import cv2

def _get_mock_face_embedding(seed=42):
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    return emb / np.linalg.norm(emb)

def _get_valid_test_jpeg():
    dummy = np.zeros((160, 160, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy)
    return encoded.tobytes()

def test_face_enrollment_with_real_gallery_sample():
    """Enroll Subject 01 using real MSU-AVIS Gallery face clip or synthetic fallback if dataset absent."""
    dataset_path = Path(settings.DATASET_PATH)
    gallery_file = dataset_path / "Gallery" / "face" / "G_F 1.mp4"

    if gallery_file.exists():
        with open(gallery_file, "rb") as f:
            response = client.post(
                "/api/v1/face/enroll",
                data={"user_id": "test_subject_01"},
                files={"file": ("G_F 1.mp4", f, "video/mp4")}
            )
    else:
        # Graceful fallback when external MSU_AVIS dataset is not downloaded
        synthetic_emb = _get_mock_face_embedding(seed=101)
        valid_jpeg = _get_valid_test_jpeg()
        with patch.object(face_engine, "extract_face_embedding", return_value=(synthetic_emb, 0.92, {"bbox": [0,0,100,100], "det_score": 0.95})):
            response = client.post(
                "/api/v1/face/enroll",
                data={"user_id": "test_subject_01"},
                files={"file": ("synthetic_face.jpg", valid_jpeg, "image/jpeg")}
            )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ENROLLED"
    assert data["user_id"] == "test_subject_01"
    assert data["embedding_dim"] == 512
    assert data["quality_score"] > 0.0

    # Verify template is encrypted with AES-256-GCM in database
    db = SessionLocal()
    tmpl = db.query(BiometricTemplate).filter(BiometricTemplate.user_id == "test_subject_01", BiometricTemplate.modality == "face").first()
    assert tmpl is not None
    assert len(tmpl.encrypted_embedding) > 0
    assert len(tmpl.iv_nonce) == 24  # 12 bytes hex = 24 chars
    assert len(tmpl.auth_tag) == 32  # 16 bytes hex = 32 chars
    
    # Decrypt and verify valid 512-d unit vector
    decrypted = decrypt_embedding(
        tmpl.encrypted_embedding,
        tmpl.iv_nonce,
        tmpl.auth_tag,
        associated_data=b"test_subject_01"
    )
    assert len(decrypted) == 512
    assert abs(np.linalg.norm(decrypted) - 1.0) < 1e-4
    db.close()

def test_genuine_face_verification():
    """Verify Subject 01 with their own probe clip or genuine embedding. Expect match (ACCEPT)."""
    dataset_path = Path(settings.DATASET_PATH)
    probe_file = dataset_path / "Probe_clips" / "S01_V1" / "clip_01.mp4"

    if probe_file.exists():
        with open(probe_file, "rb") as f:
            response = client.post(
                "/api/v1/face/verify",
                data={"user_id": "test_subject_01"},
                files={"file": ("clip_01.mp4", f, "video/mp4")}
            )
    else:
        # Load enrolled template for subject 01 to craft genuine query probe (sim > threshold)
        db = SessionLocal()
        tmpl = db.query(BiometricTemplate).filter(BiometricTemplate.user_id == "test_subject_01", BiometricTemplate.modality == "face").first()
        enrolled = decrypt_embedding(tmpl.encrypted_embedding, tmpl.iv_nonce, tmpl.auth_tag, associated_data=b"test_subject_01")
        db.close()

        # Slightly jittered genuine probe (cosine similarity ~ 0.95 >= 0.45)
        genuine_probe = enrolled + 0.05 * np.random.RandomState(1).randn(512).astype(np.float32)
        genuine_probe /= np.linalg.norm(genuine_probe)
        valid_jpeg = _get_valid_test_jpeg()

        with patch.object(face_engine, "extract_face_embedding", return_value=(genuine_probe, 0.91, {"bbox": [0,0,100,100], "det_score": 0.94})):
            response = client.post(
                "/api/v1/face/verify",
                data={"user_id": "test_subject_01"},
                files={"file": ("probe_face.jpg", valid_jpeg, "image/jpeg")}
            )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["is_match"] is True
    assert data["decision"] == "ACCEPT"
    assert data["risk_level"] == "LOW"
    assert data["similarity_score"] >= settings.FACE_SIMILARITY_THRESHOLD
    print(f"\n[GENUINE VERIFICATION] Similarity: {data['similarity_score']:.4f} (Threshold: {data['threshold']}) -> {data['decision']}")

def test_impostor_face_verification():
    """Test impostor attempt: Present different subject probe against Subject 01 template. Expect REJECT."""
    dataset_path = Path(settings.DATASET_PATH)
    impostor_probe = dataset_path / "Probe_clips" / "S02_V1" / "clip_01.mp4"

    if impostor_probe.exists():
        with open(impostor_probe, "rb") as f:
            response = client.post(
                "/api/v1/face/verify",
                data={"user_id": "test_subject_01"},
                files={"file": ("clip_01.mp4", f, "video/mp4")}
            )
    else:
        # Impostor orthogonal embedding (cosine sim ~ 0.0 < 0.45)
        impostor_emb = _get_mock_face_embedding(seed=999)
        valid_jpeg = _get_valid_test_jpeg()
        with patch.object(face_engine, "extract_face_embedding", return_value=(impostor_emb, 0.88, {"bbox": [0,0,100,100], "det_score": 0.91})):
            response = client.post(
                "/api/v1/face/verify",
                data={"user_id": "test_subject_01"},
                files={"file": ("impostor_face.jpg", valid_jpeg, "image/jpeg")}
            )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["is_match"] is False
    assert data["decision"] in ["REJECT", "STEP_UP"]
    assert data["similarity_score"] < settings.FACE_SIMILARITY_THRESHOLD
    print(f"\n[IMPOSTOR VERIFICATION] Similarity: {data['similarity_score']:.4f} (Threshold: {data['threshold']}) -> {data['decision']}")

def test_auth_audit_logs():
    """Check that verification attempts were logged in database."""
    response = client.get("/api/v1/auth/logs")
    assert response.status_code == 200
    logs = response.json()
    assert len(logs) >= 2
    # Verify at least one genuine and one rejected log
    decisions = [l["decision"] for l in logs if l["user_id"] == "test_subject_01"]
    assert "ACCEPT" in decisions
    assert "REJECT" in decisions or "STEP_UP" in decisions

def test_json_base64_face_enroll_and_verify():
    """Verify that JSON payloads with base64 data URLs enroll and verify correctly."""
    import base64
    synthetic_emb = _get_mock_face_embedding(seed=303)
    valid_jpeg = _get_valid_test_jpeg()
    b64_str = "data:image/jpeg;base64," + base64.b64encode(valid_jpeg).decode()

    with patch.object(face_engine, "extract_face_embedding", return_value=(synthetic_emb, 0.89, {"bbox": [0,0,100,100], "det_score": 0.93})):
        # JSON enrollment
        r_enroll = client.post("/api/v1/face/enroll", json={
            "user_id": "test_json_user",
            "image_base64": b64_str
        })
        assert r_enroll.status_code == 200, r_enroll.text
        assert r_enroll.json()["status"] == "ENROLLED"

        # JSON verification
        r_verify = client.post("/api/v1/face/verify", json={
            "user_id": "test_json_user",
            "image_base64": b64_str
        })
        assert r_verify.status_code == 200, r_verify.text
        assert r_verify.json()["is_match"] is True
        assert r_verify.json()["decision"] == "ACCEPT"

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
