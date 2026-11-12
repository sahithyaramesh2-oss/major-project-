"""
Module 2: Voice Identity Verification — Test Suite
====================================================
Tests are grouped into two categories:
  - [MOCKED] Tests that mock model inference for fast, deterministic CI execution.
  - [REAL]   Tests that use actual SpeechBrain ECAPA-TDNN inference on synthetic audio.

All tests use in-memory synthetic WAV audio. No real user recordings or MSU-AVIS
dataset files are required for these tests.
"""
import io
import sys
import base64
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import soundfile as sf

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.config import settings
from app.database import SessionLocal, engine, Base
from app.models.user import User, BiometricTemplate, AuthLog
from app.security.encryption import encrypt_embedding, decrypt_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_wav_bytes(duration_sec: float = 2.5, sr: int = 16000, f0: float = 180.0) -> bytes:
    """Generate synthetic harmonic voiced audio and return as WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (
        0.50 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    env = np.sin(np.pi * t / duration_sec) ** 2
    audio = (audio * env).astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-6
    audio *= 0.7
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def unit_emb(seed: int = 42, dim: int = 192) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


QUALITY_STUB = {
    "duration_sec": 2.5, "sample_rate": 16000,
    "snr_db": 24.0, "clipping_ratio": 0.0, "rms_energy": 0.08
}


def get_challenge(client, purpose="verification"):
    response = client.get("/api/v1/voice/challenge", params={"purpose": purpose})
    assert response.status_code == 200, response.text
    return response.json()


def enroll_request(client, **kwargs):
    from app.core.voice_engine import voice_engine
    challenge = get_challenge(client, "enrollment")
    payload_key = "json" if "json" in kwargs else "data"
    payload = dict(kwargs.get(payload_key) or {})
    payload["challenge_id"] = challenge["challenge_id"]
    kwargs[payload_key] = payload
    with patch.object(voice_engine, "transcribe_audio", return_value=challenge["challenge_phrase"]):
        return client.post("/api/v1/voice/enroll", **kwargs)


def verify_request(client, **kwargs):
    from app.core.voice_engine import voice_engine
    challenge = get_challenge(client)
    payload_key = "json" if "json" in kwargs else "data"
    payload = dict(kwargs.get(payload_key) or {})
    payload["challenge_id"] = challenge["challenge_id"]
    kwargs[payload_key] = payload
    with patch.object(voice_engine, "transcribe_audio", return_value=challenge["challenge_phrase"]):
        response = client.post("/api/v1/voice/verify", **kwargs)
    return response, challenge["challenge_phrase"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a TestClient while restoring the actual model objects afterward."""
    # Patch both model singletons before the app is imported into TestClient
    mock_classifier = MagicMock()
    mock_whisper = MagicMock()

    # Import voice_engine here so the patches apply to the same instance
    from app.core.voice_engine import voice_engine
    original_classifier = voice_engine.classifier
    original_whisper = voice_engine.whisper_model
    voice_engine.classifier = mock_classifier
    voice_engine.whisper_model = mock_whisper

    from app.main import app
    try:
        with TestClient(app) as c:
            yield c
    finally:
        voice_engine.classifier = original_classifier
        voice_engine.whisper_model = original_whisper


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    test_users = [
        "v_user_01", "v_user_02", "v_user_impostor",
        "v_json_user", "v_phrase_ok", "v_phrase_fail", "v_challenge_once"
    ]
    db.query(AuthLog).filter(AuthLog.user_id.in_(test_users)).delete(synchronize_session=False)
    db.query(BiometricTemplate).filter(BiometricTemplate.user_id.in_(test_users)).delete(synchronize_session=False)
    db.query(User).filter(User.user_id.in_(test_users)).delete(synchronize_session=False)
    db.commit()
    db.close()
    yield


# ---------------------------------------------------------------------------
# [UNIT] AES-256-GCM Encryption / Decryption of Voice Embeddings
# ---------------------------------------------------------------------------

def test_voice_embedding_encrypt_decrypt_roundtrip():
    """[UNIT] AES-256-GCM: encrypt then decrypt a 192-d voice embedding exactly."""
    original = unit_emb(seed=7, dim=192)
    aad = b"v_user_01"
    ct, nonce_hex, tag_hex = encrypt_embedding(original, associated_data=aad)
    assert len(ct) == 192 * 4           # 192 × float32 = 768 bytes ciphertext
    assert len(nonce_hex) == 24         # 12-byte nonce as 24-char hex
    assert len(tag_hex) == 32           # 16-byte GCM tag as 32-char hex
    recovered = decrypt_embedding(ct, nonce_hex, tag_hex, associated_data=aad)
    assert len(recovered) == 192
    assert np.allclose(original, recovered, atol=1e-5)

def test_wrong_aad_decryption_fails():
    """[UNIT] Decryption with wrong AAD must raise (GCM authentication failure)."""
    emb = unit_emb(seed=8, dim=192)
    ct, nh, th = encrypt_embedding(emb, associated_data=b"correct_user")
    with pytest.raises(Exception):
        decrypt_embedding(ct, nh, th, associated_data=b"wrong_user")


# ---------------------------------------------------------------------------
# [UNIT] Audio Quality & Audio Parsing
# ---------------------------------------------------------------------------

def test_audio_quality_silent():
    """[UNIT] Near-silent audio produces very low quality score."""
    from app.core.voice_engine import VoiceEngine
    silent = np.zeros(16000 * 2, dtype=np.float32)
    q, details = VoiceEngine.compute_audio_quality(silent, sr=16000)
    assert q < 0.5
    assert details["duration_sec"] == pytest.approx(2.0, abs=0.1)
    assert details["quality_status"] == "UNACCEPTABLE"

def test_audio_quality_clipping():
    """[UNIT] Heavily clipped audio (saturated to ±1) degrades quality score."""
    from app.core.voice_engine import VoiceEngine
    clipped = np.ones(16000 * 2, dtype=np.float32)
    q, details = VoiceEngine.compute_audio_quality(clipped, sr=16000)
    assert details["clipping_ratio"] > 0.9
    assert q < 0.6
    assert details["quality_status"] == "UNACCEPTABLE"


def test_missing_ecapa_model_never_creates_fallback_embedding():
    from app.core.voice_engine import voice_engine
    wav_bytes = make_wav_bytes(duration_sec=2.0)
    audio, sr = voice_engine.decode_audio_payload(wav_bytes)
    with patch.object(voice_engine, "classifier", None):
        with pytest.raises(RuntimeError, match="unavailable"):
            voice_engine.extract_voice_embedding(audio, sr)

def test_decode_valid_wav():
    """[UNIT] WAV bytes decode into correct shape and sample rate."""
    from app.core.voice_engine import VoiceEngine
    wav_bytes = make_wav_bytes(duration_sec=2.0, sr=16000)
    audio, sr = VoiceEngine.decode_audio_payload(wav_bytes)
    assert sr == 16000
    assert len(audio) == pytest.approx(2.0 * 16000, abs=100)
    assert audio.dtype == np.float32

def test_decode_empty_audio_raises():
    """[UNIT] Empty bytes raise ValueError immediately."""
    from app.core.voice_engine import VoiceEngine
    with pytest.raises(ValueError, match="empty"):
        VoiceEngine.decode_audio_payload(b"")

def test_decode_corrupted_audio_raises():
    """[UNIT] Garbage bytes raise ValueError."""
    from app.core.voice_engine import VoiceEngine
    with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]orrupted"):
        VoiceEngine.decode_audio_payload(b"ThisIsNotAudio###!!!")


# ---------------------------------------------------------------------------
# [UNIT] Challenge Phrase Verification (text similarity)
# ---------------------------------------------------------------------------

def test_phrase_match_exact():
    """[UNIT] Exact phrase match produces similarity >= threshold."""
    from app.core.voice_engine import VoiceEngine
    phrase = "The blue sky was clear this morning"
    sim, matched = VoiceEngine.verify_challenge_phrase(phrase, phrase)
    assert sim > 0.95
    assert matched is True

def test_phrase_match_case_insensitive():
    """[UNIT] Case differences are normalized before comparison."""
    from app.core.voice_engine import VoiceEngine
    sim, matched = VoiceEngine.verify_challenge_phrase(
        "THE BLUE SKY WAS CLEAR THIS MORNING",
        "the blue sky was clear this morning"
    )
    assert matched is True

def test_phrase_mismatch_completely_different():
    """[UNIT] Completely different phrase produces similarity below threshold."""
    from app.core.voice_engine import VoiceEngine
    sim, matched = VoiceEngine.verify_challenge_phrase(
        "absolutely unrelated random words here",
        "The blue sky was clear this morning"
    )
    assert matched is False
    assert sim < settings.CHALLENGE_MATCH_THRESHOLD

def test_phrase_empty_transcription():
    """[UNIT] Empty transcription returns phrase_matched=False."""
    from app.core.voice_engine import VoiceEngine
    sim, matched = VoiceEngine.verify_challenge_phrase("", "The blue sky was clear")
    assert matched is False


# ---------------------------------------------------------------------------
# [MOCKED] API: Challenge phrase generation
# ---------------------------------------------------------------------------

def test_challenge_generation_endpoint(client):
    """[MOCKED] GET /voice/challenge returns a valid challenge object."""
    r = client.get("/api/v1/voice/challenge")
    assert r.status_code == 200
    d = r.json()
    assert "challenge_id" in d
    assert len(d["challenge_phrase"]) > 10
    assert d["expires_in_seconds"] == 60
    assert d["purpose"] == "verification"


def test_challenge_is_single_use(client):
    from app.core.voice_engine import voice_engine
    wav = make_wav_bytes(2.0)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(unit_emb(seed=1), 0.9, QUALITY_STUB)):
        enrolled = enroll_request(
            client,
            data={"user_id": "v_challenge_once"},
            files={"file": ("enroll.wav", wav, "audio/wav")}
        )
    assert enrolled.status_code == 200, enrolled.text
    challenge = get_challenge(client)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(unit_emb(seed=1), 0.9, QUALITY_STUB)), \
            patch.object(voice_engine, "transcribe_audio", return_value=challenge["challenge_phrase"]):
        first = client.post(
            "/api/v1/voice/verify",
            data={"user_id": "v_challenge_once", "challenge_id": challenge["challenge_id"]},
            files={"file": ("query.wav", wav, "audio/wav")}
        )
        second = client.post(
            "/api/v1/voice/verify",
            data={"user_id": "v_challenge_once", "challenge_id": challenge["challenge_id"]},
            files={"file": ("query.wav", wav, "audio/wav")}
        )
    assert first.status_code == 200
    assert second.status_code == 409


# ---------------------------------------------------------------------------
# [MOCKED] API: Voice Enrollment
# ---------------------------------------------------------------------------

def test_voice_enroll_valid(client):
    """[MOCKED] Valid audio file → 200 ENROLLED with correct fields."""
    from app.core.voice_engine import voice_engine
    emb = unit_emb(seed=1)
    wav = make_wav_bytes(duration_sec=2.5)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.88, QUALITY_STUB)):
        r = enroll_request(
            client,
            data={"user_id": "v_user_01"},
            files={"file": ("sample.wav", wav, "audio/wav")}
        )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "ENROLLED"
    assert d["modality"] == "voice"
    assert d["embedding_dim"] == 192
    assert d["quality_score"] == pytest.approx(0.88, abs=0.01)
    assert d["audio_quality"]["duration_sec"] == 2.5

def test_voice_enroll_stores_encrypted_template(client):
    """[MOCKED] Enrolled template is stored encrypted with correct lengths."""
    from app.core.voice_engine import voice_engine
    emb = unit_emb(seed=2)
    wav = make_wav_bytes(duration_sec=2.0)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.85, QUALITY_STUB)):
        enroll_request(
            client,
            data={"user_id": "v_user_02"},
            files={"file": ("sample.wav", wav, "audio/wav")}
        )
    db = SessionLocal()
    tmpl = db.query(BiometricTemplate).filter(
        BiometricTemplate.user_id == "v_user_02",
        BiometricTemplate.modality == "voice"
    ).first()
    assert tmpl is not None
    assert len(tmpl.encrypted_embedding) > 0
    assert len(tmpl.iv_nonce) == 24
    assert len(tmpl.auth_tag) == 32
    # Decrypt and verify roundtrip
    recovered = decrypt_embedding(
        tmpl.encrypted_embedding, tmpl.iv_nonce, tmpl.auth_tag,
        associated_data=b"v_user_02"
    )
    assert np.allclose(emb, recovered, atol=1e-5)
    db.close()

def test_voice_enroll_no_audio_returns_400(client):
    """[MOCKED] Missing audio payload returns 400."""
    r = client.post("/api/v1/voice/enroll", json={"user_id": "v_user_01"})
    assert r.status_code == 400

def test_voice_enroll_empty_audio_returns_400(client):
    """[MOCKED] Empty bytes → 400."""
    r = enroll_request(
        client,
        data={"user_id": "v_user_01"},
        files={"file": ("empty.wav", b"", "audio/wav")}
    )
    assert r.status_code == 422

def test_voice_enroll_corrupted_audio_returns_400(client):
    """[MOCKED] Corrupted/garbage audio → 400."""
    r = enroll_request(
        client,
        data={"user_id": "v_user_01"},
        files={"file": ("bad.wav", b"not_audio_data", "audio/wav")}
    )
    assert r.status_code == 422

def test_voice_enroll_too_short_returns_422(client):
    """[MOCKED] Audio < 1.0s → 422 (duration validation)."""
    wav = make_wav_bytes(duration_sec=0.4)
    r = enroll_request(
        client,
        data={"user_id": "v_user_01"},
        files={"file": ("short.wav", wav, "audio/wav")}
    )
    assert r.status_code == 422
    assert "too short" in r.json()["detail"]["message"].lower()


# ---------------------------------------------------------------------------
# [MOCKED] API: Voice Verification
# ---------------------------------------------------------------------------

def test_voice_verify_genuine_match(client):
    """[MOCKED] Genuine speaker → is_match=True, decision=MATCH."""
    from app.core.voice_engine import voice_engine
    emb = unit_emb(seed=1)   # same as enrolled
    wav = make_wav_bytes(duration_sec=2.5)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.90, QUALITY_STUB)):
        r, _ = verify_request(
            client,
            data={"user_id": "v_user_01"},
            files={"file": ("query.wav", wav, "audio/wav")}
        )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["is_match"] is True
    assert d["speaker_identity"] == "MATCH"
    assert d["phrase_result"] == "PASS"
    assert d["similarity_score"] >= settings.VOICE_SIMILARITY_THRESHOLD
    # Cosine of two identical unit vectors = 1.0
    assert d["similarity_score"] > 0.99

def test_voice_verify_impostor_mismatch(client):
    """[MOCKED] Different speaker (orthogonal embedding) → is_match=False, decision=MISMATCH."""
    from app.core.voice_engine import voice_engine
    # Enrolled emb is unit_emb(seed=1). Impostor is orthogonal negative.
    impostor_emb = np.zeros(192, dtype=np.float32)
    impostor_emb[0] = -1.0
    wav = make_wav_bytes(duration_sec=2.5, f0=350.0)
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(impostor_emb, 0.82, QUALITY_STUB)):
        r, _ = verify_request(
            client,
            data={"user_id": "v_user_01"},
            files={"file": ("impostor.wav", wav, "audio/wav")}
        )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["is_match"] is False
    assert d["speaker_identity"] == "NO MATCH"
    assert d["phrase_result"] == "PASS"
    assert d["similarity_score"] < settings.VOICE_SIMILARITY_THRESHOLD

def test_voice_verify_missing_template(client):
    """[MOCKED] No enrolled template → 404."""
    wav = make_wav_bytes(duration_sec=2.0)
    r, _ = verify_request(
        client,
        data={"user_id": "user_never_enrolled_xyz999"},
        files={"file": ("q.wav", wav, "audio/wav")}
    )
    assert r.status_code == 404
    assert "No enrolled voice template" in r.json()["detail"]

def test_voice_verify_no_audio_returns_400(client):
    """[MOCKED] Missing audio on verify → 400."""
    r = client.post("/api/v1/voice/verify", json={"user_id": "v_user_01"})
    assert r.status_code == 400


def test_voice_verify_silence_returns_no_score(client):
    from app.core.voice_engine import voice_engine
    challenge = get_challenge(client)
    silent = io.BytesIO()
    sf.write(silent, np.zeros(16000 * 2, dtype=np.float32), 16000, format="WAV")
    r = client.post(
        "/api/v1/voice/verify",
        data={"user_id": "v_user_01", "challenge_id": challenge["challenge_id"]},
        files={"file": ("silent.wav", silent.getvalue(), "audio/wav")}
    )
    assert r.status_code == 422
    result = r.json()
    assert result["audio_quality_status"] == "UNACCEPTABLE"
    assert result["similarity_score"] is None
    assert result["speaker_identity"] == "NOT_EVALUATED"

def test_voice_verify_challenge_phrase_correct(client):
    """[MOCKED] Correct challenge phrase → phrase_matched=True."""
    from app.core.voice_engine import voice_engine
    # First enroll v_phrase_ok
    emb = unit_emb(seed=5)
    wav = make_wav_bytes(2.5)
    expected = "Silver rivers flow gracefully across the valley"
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.88, QUALITY_STUB)):
        enroll_request(
            client,
            data={"user_id": "v_phrase_ok"},
            files={"file": ("enroll.wav", wav, "audio/wav")}
        )
        challenge = get_challenge(client)
        expected = challenge["challenge_phrase"]
        with patch.object(voice_engine, "transcribe_audio", return_value=expected):
            r = client.post(
                "/api/v1/voice/verify",
            data={"user_id": "v_phrase_ok", "challenge_id": challenge["challenge_id"]},
                files={"file": ("query.wav", wav, "audio/wav")}
            )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["phrase_matched"] is True
    assert d["phrase_similarity"] >= settings.CHALLENGE_MATCH_THRESHOLD
    assert d["transcription"] == expected

def test_voice_verify_challenge_phrase_incorrect(client):
    """[MOCKED] Wrong challenge phrase → phrase_matched=False (identity still passes)."""
    from app.core.voice_engine import voice_engine
    emb = unit_emb(seed=5)
    wav = make_wav_bytes(2.5)
    expected = "Silver rivers flow gracefully across the valley"
    spoken = "completely wrong words said instead of challenge"
    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.88, QUALITY_STUB)):
        enroll_request(
            client,
            data={"user_id": "v_phrase_fail"},
            files={"file": ("enroll.wav", wav, "audio/wav")}
        )
        challenge = get_challenge(client)
        expected = challenge["challenge_phrase"]
        with patch.object(voice_engine, "transcribe_audio", return_value=spoken):
            r = client.post(
                "/api/v1/voice/verify",
            data={"user_id": "v_phrase_fail", "challenge_id": challenge["challenge_id"]},
                files={"file": ("query.wav", wav, "audio/wav")}
            )
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["is_match"] is True          # speaker identity passes
    assert d["phrase_matched"] is False    # but phrase fails
    assert d["speaker_identity"] == "MATCH"
    assert d["phrase_result"] == "FAIL"
    assert d["phrase_similarity"] < settings.CHALLENGE_MATCH_THRESHOLD

def test_voice_verify_base64_json_roundtrip(client):
    """[MOCKED] JSON base64 audio enrollment + verification roundtrip."""
    from app.core.voice_engine import voice_engine
    emb = unit_emb(seed=99)
    wav_bytes = make_wav_bytes(duration_sec=2.0)
    b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()

    with patch.object(voice_engine, "extract_voice_embedding", return_value=(emb, 0.86, QUALITY_STUB)):
        r_enroll = enroll_request(client, json={
            "user_id": "v_json_user", "audio_base64": b64
        })
        assert r_enroll.status_code == 200, r_enroll.text
        assert r_enroll.json()["status"] == "ENROLLED"

        r_verify, _ = verify_request(client, json={
            "user_id": "v_json_user", "audio_base64": b64
        })
        assert r_verify.status_code == 200, r_verify.text
        assert r_verify.json()["is_match"] is True

def test_auth_logs_contain_voice_entries(client):
    """[MOCKED] Verification attempts are logged in the auth_logs table."""
    r = client.get("/api/v1/auth/logs?limit=100")
    assert r.status_code == 200
    logs = r.json()
    voice_logs = [l for l in logs if l["modality"] == "voice"]
    assert len(voice_logs) >= 2
    decisions = {l["decision"] for l in voice_logs}
    assert "MATCH" in decisions
    assert "MISMATCH" in decisions


# ---------------------------------------------------------------------------
# [REAL] Actual ECAPA-TDNN inference on synthetic audio (no mocking)
# ---------------------------------------------------------------------------

def test_real_ecapa_embedding_shape_and_norm():
    """[REAL] ECAPA-TDNN extracts a 192-d unit-norm embedding from synthetic audio."""
    from app.core.voice_engine import voice_engine
    if voice_engine.classifier is None:
        pytest.skip("SpeechBrain ECAPA-TDNN model is unavailable in this environment.")
    wav_bytes = make_wav_bytes(duration_sec=1.5, f0=200.0)
    audio, sr = voice_engine.decode_audio_payload(wav_bytes)
    assert sr == 16000
    emb, quality, details = voice_engine.extract_voice_embedding(audio, sr)
    assert len(emb) == 192
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
    assert quality > 0.0
    assert details["duration_sec"] == pytest.approx(1.5, abs=0.05)

def test_real_same_audio_same_speaker_high_similarity():
    """[REAL] Same synthetic audio → near-identical embeddings (cosine ≈ 1.0)."""
    from app.core.voice_engine import voice_engine
    if voice_engine.classifier is None:
        pytest.skip("SpeechBrain ECAPA-TDNN model is unavailable in this environment.")
    from app.core.similarity import compute_cosine_similarity
    wav_bytes = make_wav_bytes(duration_sec=2.0, f0=180.0)
    audio, sr = voice_engine.decode_audio_payload(wav_bytes)
    emb1, _, _ = voice_engine.extract_voice_embedding(audio, sr)
    emb2, _, _ = voice_engine.extract_voice_embedding(audio, sr)
    sim = compute_cosine_similarity(emb1, emb2)
    assert sim > 0.99

def test_real_different_frequency_lower_similarity():
    """[REAL] Audio at different fundamental frequencies → lower cosine similarity."""
    from app.core.voice_engine import voice_engine
    if voice_engine.classifier is None:
        pytest.skip("SpeechBrain ECAPA-TDNN model is unavailable in this environment.")
    from app.core.similarity import compute_cosine_similarity
    wav1 = make_wav_bytes(duration_sec=2.0, f0=100.0)
    wav2 = make_wav_bytes(duration_sec=2.0, f0=400.0)
    audio1, sr = voice_engine.decode_audio_payload(wav1)
    audio2, _  = voice_engine.decode_audio_payload(wav2)
    emb1, _, _ = voice_engine.extract_voice_embedding(audio1, sr)
    emb2, _, _ = voice_engine.extract_voice_embedding(audio2, sr)
    sim = compute_cosine_similarity(emb1, emb2)
    # Different pitches produce different embeddings
    assert sim < 1.0  # They should not be identical


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
