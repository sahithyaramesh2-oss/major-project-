import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import update
import numpy as np

logger = logging.getLogger(__name__)

from app.database import get_db
from app.models.user import User, BiometricTemplate, AuthLog, VoiceChallenge
from app.schemas.voice import (
    AudioQualityInfo,
    VoiceEnrollRequest,
    VoiceVerifyRequest,
    VoiceEnrollmentResponse,
    VoiceVerificationResponse,
    ChallengePhraseResponse
)
from app.core.voice_engine import AudioQualityError, voice_engine
from app.core.similarity import evaluate_voice_verification
from app.security.encryption import encrypt_embedding, decrypt_embedding
from app.config import settings

router = APIRouter(prefix="/voice", tags=["Voice Biometrics"])


def _get_valid_challenge(db: Session, challenge_id: Optional[str], purpose: str) -> VoiceChallenge:
    if not challenge_id:
        raise HTTPException(status_code=400, detail="A fresh server-issued challenge is required.")
    challenge = db.query(VoiceChallenge).filter(VoiceChallenge.challenge_id == challenge_id).first()
    if not challenge or challenge.purpose != purpose:
        raise HTTPException(status_code=400, detail="Challenge was not found or has the wrong purpose.")
    if challenge.consumed_at is not None or challenge.expires_at <= datetime.utcnow():
        raise HTTPException(status_code=409, detail="Challenge expired or has already been used. Request a new phrase.")
    return challenge


def _consume_challenge(db: Session, challenge_id: str, purpose: str) -> bool:
    now = datetime.utcnow()
    result = db.execute(
        update(VoiceChallenge)
        .where(
            VoiceChallenge.challenge_id == challenge_id,
            VoiceChallenge.purpose == purpose,
            VoiceChallenge.consumed_at.is_(None),
            VoiceChallenge.expires_at > now,
        )
        .values(consumed_at=now)
    )
    return result.rowcount == 1

@router.get("/challenge", response_model=ChallengePhraseResponse)
def get_challenge_phrase(purpose: str = "verification", db: Session = Depends(get_db)):
    """
    Generate a fresh, unpredictable spoken challenge phrase for ATMCG challenge-response.
    Single-use challenge nonce with short validity window.
    """
    if purpose not in {"verification", "enrollment"}:
        raise HTTPException(status_code=400, detail="Challenge purpose must be verification or enrollment.")
    db.query(VoiceChallenge).filter(VoiceChallenge.expires_at <= datetime.utcnow()).delete(synchronize_session=False)
    challenge = voice_engine.generate_random_challenge()
    expires_at = datetime.utcnow() + timedelta(seconds=60)
    db.add(VoiceChallenge(
        challenge_id=challenge["challenge_id"],
        phrase=challenge["challenge_phrase"],
        purpose=purpose,
        expires_at=expires_at
    ))
    db.commit()
    return ChallengePhraseResponse(**challenge, purpose=purpose)

@router.post("/enroll", response_model=VoiceEnrollmentResponse)
async def enroll_voice(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Enroll a user's voice template.
    Accepts either an uploaded audio/video file (multipart/form-data) or a base64 microphone audio (JSON).
    Preprocesses audio into 16 kHz mono, extracts a 192-d SpeechBrain ECAPA-TDNN speaker embedding,
    and encrypts it using AES-256-GCM prior to storage.
    """
    content_type = request.headers.get("content-type", "")
    target_user_id = None
    file = None
    audio_base64 = None
    challenge_id = None

    if "application/json" in content_type:
        body = await request.json()
        target_user_id = body.get("user_id")
        audio_base64 = body.get("audio_base64")
        challenge_id = body.get("challenge_id")
    else:
        form = await request.form()
        target_user_id = form.get("user_id")
        file = form.get("file")
        challenge_id = form.get("challenge_id")
        if not file and "audio_base64" in form:
            audio_base64 = form.get("audio_base64")

    if not target_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required for voice enrollment."
        )

    challenge = _get_valid_challenge(db, challenge_id, "enrollment")

    user = db.query(User).filter(User.user_id == target_user_id).first()
    if not user:
        user = User(user_id=target_user_id, full_name=target_user_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    source_filename = "microphone_recording.wav"
    try:
        if file and hasattr(file, "read"):
            source_filename = getattr(file, "filename", "uploaded_voice.wav")
            content = await file.read()
            audio_data, sr = voice_engine.decode_audio_payload(content, source_filename)
        elif audio_base64:
            audio_data, sr = voice_engine.decode_base64_audio(audio_base64)
            source_filename = "recorded_audio.webm"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No voice sample provided. Please upload an audio file or record via microphone."
            )
    except ValueError as e:
        raise HTTPException(status_code=422, detail={
            "processing_status": "FAILURE",
            "audio_quality_status": "UNACCEPTABLE",
            "model_processing_status": "NOT_RUN",
            "message": str(e),
        })

    # Mark templates created by this genuine ECAPA path. Older rows may contain
    # development fallback vectors and must be re-enrolled rather than compared.
    source_filename = f"ecapa-tdnn-v1:{source_filename}"[:255]

    try:
        embedding, quality_score, quality_details = voice_engine.extract_voice_embedding(audio_data, sr)
    except AudioQualityError as e:
        raise HTTPException(status_code=422, detail={
            "processing_status": "FAILURE",
            "audio_quality_status": "UNACCEPTABLE",
            "audio_quality": e.quality_details,
            "message": str(e),
        })
    except ValueError as e:
        quality_score, quality_details = voice_engine.compute_audio_quality(audio_data, sr)
        quality_details["quality_status"] = "UNACCEPTABLE"
        quality_details["quality_message"] = str(e)
        raise HTTPException(status_code=422, detail={
            "processing_status": "FAILURE",
            "audio_quality_status": "UNACCEPTABLE",
            "audio_quality": quality_details,
            "message": str(e),
        })
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "processing_status": "FAILURE",
            "model_processing_status": "FAILURE",
            "message": f"Voice feature extraction failed: {e}",
        })

    try:
        enrollment_transcription = voice_engine.transcribe_audio(audio_data, sr)
        enrollment_phrase_score, enrollment_phrase_match = voice_engine.verify_challenge_phrase(
            enrollment_transcription, challenge.phrase
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail={
            "processing_status": "FAILURE",
            "model_processing_status": "FAILURE",
            "audio_quality_status": "ACCEPTABLE",
            "message": f"Whisper enrollment phrase processing failed: {e}",
        })

    if not enrollment_phrase_match:
        raise HTTPException(status_code=422, detail={
            "processing_status": "FAILURE",
            "model_processing_status": "SUCCESS",
            "audio_quality_status": "ACCEPTABLE",
            "phrase_result": "FAIL",
            "transcription": enrollment_transcription,
            "phrase_similarity": enrollment_phrase_score,
            "message": "The recording did not match the displayed enrollment phrase. Please record it again.",
        })

    # Encrypt 192-d embedding with AES-256-GCM using target_user_id as authenticated associated data
    ciphertext, nonce_hex, tag_hex = encrypt_embedding(
        embedding,
        associated_data=target_user_id.encode("utf-8")
    )

    existing_template = db.query(BiometricTemplate).filter(
        BiometricTemplate.user_id == target_user_id,
        BiometricTemplate.modality == "voice"
    ).first()

    now = datetime.utcnow()
    if existing_template:
        existing_template.encrypted_embedding = ciphertext
        existing_template.iv_nonce = nonce_hex
        existing_template.auth_tag = tag_hex
        existing_template.embedding_dim = len(embedding)
        existing_template.quality_score = quality_score
        existing_template.source_filename = source_filename
        existing_template.updated_at = now
    else:
        new_template = BiometricTemplate(
            user_id=target_user_id,
            modality="voice",
            encrypted_embedding=ciphertext,
            iv_nonce=nonce_hex,
            auth_tag=tag_hex,
            embedding_dim=len(embedding),
            quality_score=quality_score,
            source_filename=source_filename,
            created_at=now,
            updated_at=now
        )
        db.add(new_template)

    if not _consume_challenge(db, challenge.challenge_id, "enrollment"):
        db.rollback()
        raise HTTPException(status_code=409, detail="Enrollment challenge expired or has already been used. Request a new phrase.")
    db.commit()

    return VoiceEnrollmentResponse(
        user_id=target_user_id,
        modality="voice",
        status="ENROLLED",
        message="Voice enrolled successfully with AES-256-GCM privacy protection.",
        quality_score=quality_score,
        embedding_dim=len(embedding),
        audio_quality=AudioQualityInfo(**quality_details),
        audio_quality_status="ACCEPTABLE",
        processing_status="SUCCESS",
        model_processing_status="SUCCESS",
        phrase_result="PASS",
        transcription=enrollment_transcription,
        created_at=now
    )

@router.post("/verify", response_model=VoiceVerificationResponse)
async def verify_voice(
    request: Request,
    db: Session = Depends(get_db)
):
    """Return separate speaker identity, phrase compliance, quality, and processing evidence."""
    content_type = request.headers.get("content-type", "")
    target_user_id = None
    file = None
    audio_base64 = None
    challenge_id = None

    if "application/json" in content_type:
        body = await request.json()
        target_user_id = body.get("user_id")
        audio_base64 = body.get("audio_base64")
        challenge_id = body.get("challenge_id")
    else:
        form = await request.form()
        target_user_id = form.get("user_id")
        file = form.get("file")
        if not file and "audio_base64" in form:
            audio_base64 = form.get("audio_base64")
        challenge_id = form.get("challenge_id")

    if not target_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User ID is required for voice verification."
        )

    try:
        challenge = _get_valid_challenge(db, challenge_id, "verification")
    except HTTPException:
        raise
    challenge_phrase = challenge.phrase

    if not _consume_challenge(db, challenge.challenge_id, "verification"):
        db.rollback()
        raise HTTPException(status_code=409, detail="Challenge expired or has already been used. Request a new phrase.")
    db.commit()

    # 1. Retrieve registered voice template
    template = db.query(BiometricTemplate).filter(
        BiometricTemplate.user_id == target_user_id,
        BiometricTemplate.modality == "voice"
    ).first()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enrolled voice template found for user '{target_user_id}'."
        )

    if not (template.source_filename or "").startswith("ecapa-tdnn-v1:"):
        raise HTTPException(
            status_code=409,
            detail="This voice template predates verified ECAPA-TDNN enrollment. Re-enroll this user's voice before verification."
        )

    # Decrypt and authenticate the enrolled template.
    try:
        registered_emb = decrypt_embedding(
            template.encrypted_embedding,
            template.iv_nonce,
            template.auth_tag,
            associated_data=target_user_id.encode("utf-8")
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice template decryption / authentication failed: {e}"
        )
    if registered_emb.size != 192 or not np.all(np.isfinite(registered_emb)):
        raise HTTPException(status_code=500, detail="Stored voice template has an invalid embedding. Re-enroll this user.")

    # Decode and normalize query audio.
    try:
        if file and hasattr(file, "read"):
            content = await file.read()
            source_filename = getattr(file, "filename", "query_voice.wav")
            audio_data, sr = voice_engine.decode_audio_payload(content, source_filename)
        elif audio_base64:
            audio_data, sr = voice_engine.decode_base64_audio(audio_base64)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No query voice sample provided."
            )
    except ValueError as e:
        return _verification_failure(target_user_id, challenge_phrase, str(e), status_code=422)

    # Reject unusable audio before any speaker comparison. Never synthesize scores.
    try:
        query_emb, query_quality, quality_details = voice_engine.extract_voice_embedding(audio_data, sr)
    except AudioQualityError as e:
        return _verification_failure(
            target_user_id, challenge_phrase, str(e),
            quality_details=e.quality_details, status_code=422
        )
    except ValueError as e:
        quality_score, quality_details = voice_engine.compute_audio_quality(audio_data, sr)
        quality_details["quality_status"] = "UNACCEPTABLE"
        quality_details["quality_message"] = str(e)
        return _verification_failure(
            target_user_id, challenge_phrase, str(e),
            quality_details=quality_details, status_code=422
        )
    except Exception as e:
        return _verification_failure(
            target_user_id, challenge_phrase,
            f"ECAPA-TDNN processing failed: {e}", status_code=503,
            model_processing_status="FAILURE"
        )

    # Speaker identity is independent from challenge phrase compliance.
    eval_result = evaluate_voice_verification(
        query_emb=query_emb,
        template_emb=registered_emb,
        quality_score=query_quality,
        threshold=settings.VOICE_SIMILARITY_THRESHOLD
    )

    transcription = None
    phrase_similarity = 0.0
    phrase_matched = False
    processing_status = "SUCCESS"
    processing_message = "ECAPA-TDNN and Whisper processing completed."
    try:
        transcription = voice_engine.transcribe_audio(audio_data, sr)
        phrase_similarity, phrase_matched = voice_engine.verify_challenge_phrase(
            transcription=transcription,
            expected_phrase=challenge_phrase
        )
    except Exception as e:
        logger.exception("Whisper challenge transcription failed")
        processing_status = "FAILURE"
        processing_message = f"Speaker identity was evaluated, but Whisper processing failed: {e}"

    # 7. Create audit log
    attempt_id = str(uuid.uuid4())
    now = datetime.utcnow()
    details_payload = {
        "audio_quality": quality_details,
        "challenge_phrase": challenge_phrase,
        "transcription": transcription,
        "phrase_similarity": phrase_similarity,
        "phrase_matched": phrase_matched,
        "speaker_identity": eval_result["decision"],
        "audio_quality_status": "ACCEPTABLE",
        "processing_status": processing_status,
    }

    auth_log = AuthLog(
        attempt_id=attempt_id,
        user_id=target_user_id,
        modality="voice",
        similarity_score=eval_result["similarity_score"],
        threshold=eval_result["threshold"],
        decision=eval_result["decision"],
        risk_level="LOW" if eval_result["is_match"] else "HIGH",
        is_match=1 if eval_result["is_match"] else 0,
        quality_score=eval_result["quality_score"],
        details_json=json.dumps(details_payload),
        created_at=now
    )
    db.add(auth_log)
    db.commit()

    return VoiceVerificationResponse(
        attempt_id=attempt_id,
        user_id=target_user_id,
        modality="voice",
        similarity_score=eval_result["similarity_score"],
        threshold=eval_result["threshold"],
        is_match=eval_result["is_match"],
        speaker_identity="MATCH" if eval_result["is_match"] else "NO MATCH",
        quality_score=eval_result["quality_score"],
        audio_quality=AudioQualityInfo(**quality_details),
        audio_quality_status="ACCEPTABLE",
        phrase_result="PASS" if phrase_matched else "FAIL",
        processing_status=processing_status,
        processing_message=processing_message,
        model_processing_status="SUCCESS" if processing_status == "SUCCESS" else "FAILURE",
        challenge_phrase=challenge_phrase,
        transcription=transcription,
        phrase_similarity=phrase_similarity,
        phrase_matched=phrase_matched,
        decision=eval_result["decision"],
        message=eval_result["message"],
        timestamp=now
    )


def _verification_failure(
    user_id: str,
    challenge_phrase: str,
    message: str,
    *,
    quality_details: Optional[dict] = None,
    status_code: int = 422,
    model_processing_status: str = "NOT_RUN",
) -> JSONResponse:
    """Return structured failure evidence without inventing biometric scores."""
    timestamp = datetime.utcnow()
    details = quality_details
    return JSONResponse(
        status_code=status_code,
        content={
            "attempt_id": str(uuid.uuid4()),
            "user_id": user_id,
            "modality": "voice",
            "similarity_score": None,
            "threshold": settings.VOICE_SIMILARITY_THRESHOLD,
            "is_match": None,
            "speaker_identity": "NOT_EVALUATED",
            "quality_score": None,
            "audio_quality": details,
            "audio_quality_status": "UNACCEPTABLE",
            "phrase_result": "NOT_EVALUATED",
            "processing_status": "FAILURE",
            "processing_message": message,
            "model_processing_status": model_processing_status,
            "challenge_phrase": challenge_phrase,
            "transcription": None,
            "phrase_similarity": None,
            "phrase_matched": None,
            "decision": "NOT_EVALUATED",
            "message": message,
            "timestamp": timestamp.isoformat(),
        },
    )
