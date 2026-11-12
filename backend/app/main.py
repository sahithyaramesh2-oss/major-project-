import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.models import user as user_models
from app.api.face_routes import router as face_router
from app.api.voice_routes import router as voice_router
from app.api.system_routes import router as system_router
from app.core.face_engine import face_engine
from app.core.voice_engine import voice_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("biometric_auth")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    logger.info("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")
    
    # Pre-warm InsightFace model (non-blocking - graceful if model files missing)
    logger.info("Pre-warming InsightFace ArcFace models...")
    try:
        _ = face_engine.app
        logger.info("InsightFace models ready.")
    except Exception as e:
        logger.warning("InsightFace pre-warm failed (non-fatal): %s", e)

    # Voice models are loaded lazily on first request to avoid blocking startup
    logger.info("Voice models (ECAPA-TDNN + Whisper) will load on first request if not already cached.")
    logger.info("Voice model pre-loaded: %s", voice_engine.is_ready())
    
    yield
    logger.info("Shutting down biometric auth service...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Risk-Adaptive, Privacy-Preserving Face–Voice Biometric Authentication backend.",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v1 routes
app.include_router(face_router, prefix=settings.API_V1_STR)
app.include_router(voice_router, prefix=settings.API_V1_STR)
app.include_router(system_router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health": "OK"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
