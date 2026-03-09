"""
CRUD operations for Question model.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.database.models import Question


class QuestionCRUD:
    @staticmethod
    def create(db: Session, **kwargs) -> Question:
        question = Question(**kwargs)
        db.add(question)
        db.commit()
        db.refresh(question)
        return question

    @staticmethod
    def get(db: Session, question_id: int) -> Optional[Question]:
        return db.query(Question).filter(Question.id == question_id).first()

    @staticmethod
    def get_all(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
        question_type: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Question]:
        query = db.query(Question)

        if subject:
            query = query.filter(Question.subject == subject)
        if topic:
            query = query.filter(Question.topic == topic)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)
        if question_type:
            query = query.filter(Question.question_type == question_type)
        if search:
            query = query.filter(Question.content.contains(search))

        return query.order_by(Question.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def count(
        db: Session,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> int:
        query = db.query(Question)

        if subject:
            query = query.filter(Question.subject == subject)
        if topic:
            query = query.filter(Question.topic == topic)
        if difficulty:
            query = query.filter(Question.difficulty == difficulty)

        return query.count()

    @staticmethod
    def update(db: Session, question_id: int, **kwargs) -> Optional[Question]:
        question = db.query(Question).filter(Question.id == question_id).first()
        if question:
            for key, value in kwargs.items():
                if hasattr(question, key):
                    setattr(question, key, value)
            db.commit()
            db.refresh(question)
        return question

    @staticmethod
    def delete(db: Session, question_id: int) -> bool:
        question = db.query(Question).filter(Question.id == question_id).first()
        if question:
            db.delete(question)
            db.commit()
            return True
        return False

    @staticmethod
    def exists_by_content(db: Session, content: str) -> bool:
        """Check if a question with this exact content already exists."""
        return db.query(Question).filter(Question.content == content).first() is not None

    @staticmethod
    def bulk_create(db: Session, questions: List[dict]) -> List[Question]:
        db_questions = [Question(**q) for q in questions]
        db.add_all(db_questions)
        db.commit()
        for q in db_questions:
            db.refresh(q)
        return db_questions

    @staticmethod
    def get_by_subject_stats(db: Session) -> dict:
        """Get statistics grouped by subject"""
        from sqlalchemy import func

        stats = db.query(
            Question.subject,
            Question.difficulty,
            func.count(Question.id).label("count")
        ).group_by(Question.subject, Question.difficulty).all()

        result = {}
        for subject, difficulty, count in stats:
            if subject not in result:
                result[subject] = {"total": 0, "easy": 0, "medium": 0, "hard": 0}
            result[subject][difficulty] = count
            result[subject]["total"] += count

        return result
