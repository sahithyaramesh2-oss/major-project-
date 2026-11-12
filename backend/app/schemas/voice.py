from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class AudioQualityInfo(BaseModel):
    duration_sec: float = Field(..., description="Duration of processed audio in seconds")
    sample_rate: int = Field(16000, description="Sampling rate in Hz (standard 16 kHz)")
    snr_db: float = Field(..., description="Estimated Signal-to-Noise Ratio in dB")
    clipping_ratio: float = Field(..., description="Ratio of clipped audio samples")
    rms_energy: float = Field(..., description="Root Mean Square energy of signal")
    active_frame_ratio: float = 0.0
    quality_status: str = "ACCEPTABLE"
    quality_message: str = "Audio quality is acceptable."

class VoiceEnrollRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64, description="Subject ID to enroll")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio or data URL")
    challenge_id: Optional[str] = None

class VoiceVerifyRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=64, description="Claimed Subject ID")
    audio_base64: Optional[str] = Field(None, description="Base64-encoded audio or data URL")
    challenge_id: str = Field(..., description="Fresh, server-issued verification challenge")

class VoiceEnrollmentResponse(BaseModel):
    user_id: str
    modality: str = "voice"
    status: str = "ENROLLED"
    message: str
    quality_score: float
    embedding_dim: int = 192
    audio_quality: AudioQualityInfo
    audio_quality_status: str = "ACCEPTABLE"
    processing_status: str = "SUCCESS"
    model_processing_status: str = "SUCCESS"
    phrase_result: str = "PASS"
    transcription: Optional[str] = None
    created_at: datetime

class VoiceVerificationResponse(BaseModel):
    attempt_id: str
    user_id: str
    modality: str = "voice"
    similarity_score: Optional[float] = None
    threshold: float
    is_match: Optional[bool] = None
    speaker_identity: str = "NOT_EVALUATED"
    quality_score: Optional[float] = None
    audio_quality: Optional[AudioQualityInfo] = None
    audio_quality_status: str = "UNACCEPTABLE"
    phrase_result: str = "NOT_EVALUATED"
    processing_status: str = "FAILURE"
    processing_message: Optional[str] = None
    model_processing_status: str = "NOT_RUN"
    challenge_phrase: Optional[str] = None
    transcription: Optional[str] = None
    phrase_similarity: Optional[float] = None
    phrase_matched: Optional[bool] = None
    decision: str = "MATCH"
    message: str
    timestamp: datetime

class ChallengePhraseResponse(BaseModel):
    challenge_id: str
    challenge_phrase: str
    expires_in_seconds: int = 60
    purpose: str = "verification"
