"""
FastAPI dependencies for database and other shared resources.
"""
from app.database import SessionLocal


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
