from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, LargeBinary, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(64), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=False)
    email = Column(String(128), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    templates = relationship("BiometricTemplate", back_populates="user", cascade="all, delete-orphan")
    auth_logs = relationship("AuthLog", back_populates="user", cascade="all, delete-orphan")


class BiometricTemplate(Base):
    __tablename__ = "biometric_templates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, index=True)
    modality = Column(String(32), nullable=False, default="face")  # "face", "voice"
    
    # AES-256-GCM Encrypted biometric embedding
    encrypted_embedding = Column(LargeBinary, nullable=False)
    iv_nonce = Column(String(64), nullable=False)  # Hex-encoded 12-byte IV / nonce
    auth_tag = Column(String(64), nullable=False)  # Hex-encoded 16-byte authentication tag
    
    embedding_dim = Column(Integer, default=512)
    quality_score = Column(Float, nullable=True, default=1.0)
    source_filename = Column(String(255), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="templates")


class AuthLog(Base):
    __tablename__ = "auth_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    attempt_id = Column(String(64), unique=True, index=True, nullable=False)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=False, index=True)
    modality = Column(String(32), nullable=False, default="face")
    
    similarity_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    decision = Column(String(32), nullable=False)  # "ACCEPT", "REJECT", "STEP_UP"
    risk_level = Column(String(32), nullable=False, default="LOW")  # "LOW", "MEDIUM", "HIGH"
    
    is_match = Column(Integer, nullable=False, default=0)  # 1 genuine match, 0 mismatch / impostor
    quality_score = Column(Float, nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="auth_logs")


class VoiceChallenge(Base):
    __tablename__ = "voice_challenges"

    challenge_id = Column(String(64), primary_key=True)
    phrase = Column(String(255), nullable=False)
    purpose = Column(String(32), nullable=False, default="verification")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)
