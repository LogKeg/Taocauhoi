"""
CRUD operations for UsageHistory model.
"""

from typing import List

from sqlalchemy.orm import Session

from app.database.models import UsageHistory


class HistoryCRUD:
    @staticmethod
    def create(db: Session, timestamp: str, history_type: str, filename: str = None,
               count: int = 0, difficulty: str = None, questions_json: str = None) -> UsageHistory:
        # Check if already exists
        existing = db.query(UsageHistory).filter(UsageHistory.timestamp == timestamp).first()
        if existing:
            return existing
        history = UsageHistory(
            timestamp=timestamp,
            history_type=history_type,
            filename=filename,
            count=count,
            difficulty=difficulty,
            questions_json=questions_json
        )
        db.add(history)
        db.commit()
        db.refresh(history)
        return history

    @staticmethod
    def get_all(db: Session, skip: int = 0, limit: int = 50) -> List[UsageHistory]:
        return db.query(UsageHistory).order_by(UsageHistory.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def delete_by_timestamp(db: Session, timestamp: str) -> bool:
        history = db.query(UsageHistory).filter(UsageHistory.timestamp == timestamp).first()
        if history:
            db.delete(history)
            db.commit()
            return True
        return False

    @staticmethod
    def delete_all(db: Session) -> int:
        count = db.query(UsageHistory).delete()
        db.commit()
        return count
