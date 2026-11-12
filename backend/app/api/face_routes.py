import os
import uuid
import json
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User, BiometricTemplate, AuthLog
from app.schemas.user import (
    UserCreate,
    UserResponse,
    EnrollmentResponse,
    VerificationResponse,
    AuthLogResponse,
    FaceEnrollRequest,
    FaceVerifyRequest
)
from app.core.face_engine import face_engine
from app.core.similarity import evaluate_face_verification
from app.security.encryption import encrypt_embedding, decrypt_embedding
from app.config import settings

router = APIRouter(prefix="", tags=["Face Biometrics"])

@router.post("/users/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user identity in the system."""
    existing = db.query(User).filter(User.user_id == user_in.user_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with ID '{user_in.user_id}' already exists."
        )
    
    if user_in.email:
        email_exists = db.query(User).filter(User.email == user_in.email).first()
        if email_exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email '{user_in.email}' is already registered."
            )

    user = User(
        user_id=user_in.user_id,
        full_name=user_in.full_name,
        email=user_in.email
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return UserResponse(
        id=user.id,
        user_id=user.user_id,
        full_name=user.full_name,
        email=user.email,
        created_at=user.created_at,
        is_face_enrolled=False,
        is_voice_enrolled=False
    )

@router.get("/users", response_model=List[UserResponse])
def list_users(db: Session = Depends(get_db)):
    """List all registered users and their enrollment status."""
    users = db.query(User).all()
    results = []
    for u in users:
        has_face = db.query(BiometricTemplate).filter(
            BiometricTemplate.user_id == u.user_id,
            BiometricTemplate.modality == "face"
        ).first() is not None
        
        has_voice = db.query(BiometricTemplate).filter(
            BiometricTemplate.user_id == u.user_id,
            BiometricTemplate.modality == "voice"
        ).first() is not None
        
        results.append(UserResponse(
            id=u.id,
            user_id=u.user_id,
            full_name=u.full_name,
            email=u.email,
            created_at=u.created_at,
            is_face_enrolled=has_face,
            is_voice_enrolled=has_voice
        ))
    return results

@router.post("/face/enroll", response_model=EnrollmentResponse)
async def enroll_face(
    user_id: str = Form(None),
    file: Optional[UploadFile] = File(None),
    json_data: Optional[FaceEnrollRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Enroll a user's face template.
    Accepts either an uploaded image/video file or a base64 webcam snapshot.
    The extracted 512-d ArcFace embedding is encrypted with AES-256-GCM before database storage.
    """
    # Resolve user_id and image source
    target_user_id = user_id or (json_data.user_id if json_data else None)
    if not target_user_id:
        raise HTTPException(status_code=400, detail="User ID is required for enrollment.")

    user = db.query(User).filter(User.user_id == target_user_id).first()
    if not user:
        # Auto-create user if registering directly through enrollment screen
        user = User(user_id=target_user_id, full_name=target_user_id)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Extract image
    img = None
    source_filename = None
    if file:
        source_filename = file.filename
        content = await file.read()
        if file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            # Save temporary video to extract frame
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            tmp_video_path = os.path.join(settings.UPLOAD_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
            with open(tmp_video_path, "wb") as f_out:
                f_out.write(content)
            img = face_engine.extract_frame_from_video(tmp_video_path)
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass
        else:
            img = face_engine.read_image_bytes(content)
    elif json_data and json_data.image_base64:
        img = face_engine.decode_base64_image(json_data.image_base64)
        source_filename = "webcam_capture.jpg"
    else:
        raise HTTPException(status_code=400, detail="No image file or webcam frame provided.")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not read valid image frame.")

    # Detect and extract ArcFace embedding
    try:
        embedding, quality_score, metadata = face_engine.extract_face_embedding(img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Encrypt embedding using AES-256-GCM
    ciphertext, nonce_hex, tag_hex = encrypt_embedding(
        embedding,
        associated_data=target_user_id.encode("utf-8")
    )

    # Store or update template in database
    existing_template = db.query(BiometricTemplate).filter(
        BiometricTemplate.user_id == target_user_id,
        BiometricTemplate.modality == "face"
    ).first()

    now = datetime.utcnow()
    if existing_template:
        existing_template.encrypted_embedding = ciphertext
        existing_template.iv_nonce = nonce_hex
        existing_template.auth_tag = tag_hex
        existing_template.quality_score = quality_score
        existing_template.source_filename = source_filename
        existing_template.updated_at = now
    else:
        new_template = BiometricTemplate(
            user_id=target_user_id,
            modality="face",
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

    db.commit()

    return EnrollmentResponse(
        user_id=target_user_id,
        modality="face",
        status="ENROLLED",
        message="Face enrolled successfully with AES-256-GCM privacy protection.",
        quality_score=quality_score,
        embedding_dim=len(embedding),
        created_at=now
    )

@router.post("/face/verify", response_model=VerificationResponse)
async def verify_face(
    user_id: str = Form(None),
    file: Optional[UploadFile] = File(None),
    json_data: Optional[FaceVerifyRequest] = None,
    db: Session = Depends(get_db)
):
    """
    Verify a face query against the user's enrolled template.
    1. Loads AES-256-GCM encrypted template from database.
    2. Decrypts and authenticates template.
    3. Extracts ArcFace embedding from the query image.
    4. Computes cosine similarity and assesses risk-adaptive decision.
    5. Saves full audit log in MySQL.
    """
    target_user_id = user_id or (json_data.user_id if json_data else None)
    if not target_user_id:
        raise HTTPException(status_code=400, detail="User ID is required for verification.")

    template = db.query(BiometricTemplate).filter(
        BiometricTemplate.user_id == target_user_id,
        BiometricTemplate.modality == "face"
    ).first()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No enrolled face template found for user '{target_user_id}'."
        )

    # Decrypt registered template embedding
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
            detail=f"Template decryption / authentication failed: {e}"
        )

    # Process query image
    img = None
    if file:
        content = await file.read()
        if file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            tmp_video_path = os.path.join(settings.UPLOAD_DIR, f"temp_{uuid.uuid4().hex}_{file.filename}")
            with open(tmp_video_path, "wb") as f_out:
                f_out.write(content)
            img = face_engine.extract_frame_from_video(tmp_video_path)
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass
        else:
            img = face_engine.read_image_bytes(content)
    elif json_data and json_data.image_base64:
        img = face_engine.decode_base64_image(json_data.image_base64)
    else:
        raise HTTPException(status_code=400, detail="No query image provided.")

    if img is None:
        raise HTTPException(status_code=400, detail="Could not read valid query image frame.")

    # Extract query ArcFace embedding
    try:
        query_emb, query_quality, metadata = face_engine.extract_face_embedding(img)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Evaluate cosine similarity and decision
    eval_result = evaluate_face_verification(
        query_emb=query_emb,
        template_emb=registered_emb,
        quality_score=query_quality,
        threshold=settings.FACE_SIMILARITY_THRESHOLD
    )

    # Create audit record
    attempt_id = str(uuid.uuid4())
    now = datetime.utcnow()
    auth_log = AuthLog(
        attempt_id=attempt_id,
        user_id=target_user_id,
        modality="face",
        similarity_score=eval_result["similarity_score"],
        threshold=eval_result["threshold"],
        decision=eval_result["decision"],
        risk_level=eval_result["risk_level"],
        is_match=1 if eval_result["is_match"] else 0,
        quality_score=eval_result["quality_score"],
        details_json=json.dumps(metadata),
        created_at=now
    )
    db.add(auth_log)
    db.commit()

    return VerificationResponse(
        attempt_id=attempt_id,
        user_id=target_user_id,
        modality="face",
        similarity_score=eval_result["similarity_score"],
        threshold=eval_result["threshold"],
        is_match=eval_result["is_match"],
        decision=eval_result["decision"],
        risk_level=eval_result["risk_level"],
        quality_score=eval_result["quality_score"],
        message=eval_result["message"],
        timestamp=now
    )

@router.get("/auth/logs", response_model=List[AuthLogResponse])
def get_auth_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Retrieve recent authentication attempt logs."""
    logs = db.query(AuthLog).order_by(AuthLog.created_at.desc()).limit(limit).all()
    return logs
