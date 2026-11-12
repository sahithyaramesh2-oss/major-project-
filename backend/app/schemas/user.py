from datetime import datetime
from typing import Optional, List, Any
from pydantic import BaseModel, Field

class UserCreate(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64, description="Unique username or subject ID")
    full_name: str = Field(..., min_length=2, max_length=128, description="User's full name")
    email: Optional[str] = Field(None, max_length=128, description="Optional email address")

class UserResponse(BaseModel):
    id: int
    user_id: str
    full_name: str
    email: Optional[str] = None
    created_at: datetime
    is_face_enrolled: bool = False
    is_voice_enrolled: bool = False

    class Config:
        from_attributes = True

class FaceEnrollRequest(BaseModel):
    user_id: str
    image_base64: Optional[str] = None

class FaceVerifyRequest(BaseModel):
    user_id: str
    image_base64: Optional[str] = None

class EnrollmentResponse(BaseModel):
    user_id: str
    modality: str
    status: str
    message: str
    quality_score: float
    embedding_dim: int = 512
    created_at: datetime

class VerificationResponse(BaseModel):
    attempt_id: str
    user_id: str
    modality: str
    similarity_score: float
    threshold: float
    is_match: bool
    decision: str
    risk_level: str
    quality_score: float
    message: str
    timestamp: datetime

class AuthLogResponse(BaseModel):
    id: int
    attempt_id: str
    user_id: str
    modality: str
    similarity_score: float
    threshold: float
    decision: str
    risk_level: str
    is_match: int
    quality_score: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True

class SystemStatusResponse(BaseModel):
    status: str
    project: str
    version: str
    database: str
    face_model: str
    models_ready: bool
