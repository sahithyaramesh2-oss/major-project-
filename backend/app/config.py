import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Risk-Adaptive Face-Voice Biometric Auth"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "1234")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "biometric_auth")

    # AES-256-GCM Encryption Key (32 bytes / 64 hex chars or generated)
    # Default key for development, can be overridden via environment variable
    AES_SECRET_KEY_HEX: str = os.getenv(
        "AES_SECRET_KEY_HEX",
        "4a72d3f9b8c1e5a27894d0e1f3a5b7c9e0d2f4a6b8c0e2d4f6a8b0c2e4f6a8b0"
    )
    
    # Biometric Thresholds
    FACE_SIMILARITY_THRESHOLD: float = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.45"))
    VOICE_SIMILARITY_THRESHOLD: float = float(os.getenv("VOICE_SIMILARITY_THRESHOLD", "0.55"))

    # Voice Processing Configurations
    VOICE_MIN_DURATION_SEC: float = float(os.getenv("VOICE_MIN_DURATION_SEC", "1.0"))
    VOICE_MAX_DURATION_SEC: float = float(os.getenv("VOICE_MAX_DURATION_SEC", "30.0"))
    VOICE_SAMPLE_RATE: int = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    CHALLENGE_MATCH_THRESHOLD: float = float(os.getenv("CHALLENGE_MATCH_THRESHOLD", "0.70"))
    VOICE_MODEL_DIR: str = os.getenv(
        "VOICE_MODEL_DIR",
        str(Path(__file__).resolve().parent.parent / "pretrained_models" / "speechbrain")
    )
    
    # Dataset path for gallery / probe validation
    DATASET_PATH: str = os.getenv("DATASET_PATH", r"C:\Users\sahit\Downloads\MSU_AVIS\MSU_AVIS")
    
    # Upload folder
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent / "uploads")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    @property
    def SQLITE_FALLBACK_URI(self) -> str:
        db_path = Path(__file__).resolve().parent.parent / "fallback.db"
        return f"sqlite:///{db_path}"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
