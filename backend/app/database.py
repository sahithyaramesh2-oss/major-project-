import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

# Try connecting to MySQL; if not reachable, use SQLite fallback
try:
    engine = create_engine(
        settings.SQLALCHEMY_DATABASE_URI,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    # Test connection
    with engine.connect() as conn:
        logger.info("Successfully connected to MySQL database: %s", settings.MYSQL_DATABASE)
except Exception as e:
    logger.warning("Could not connect to MySQL (%s). Falling back to SQLite: %s", e, settings.SQLITE_FALLBACK_URI)
    engine = create_engine(
        settings.SQLITE_FALLBACK_URI,
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
