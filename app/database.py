"""
Database module for Question Bank system
Using SQLite with SQLAlchemy ORM
"""

import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./question_bank.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Enums
class DifficultyLevel(str):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(str):
    MCQ = "mcq"
    BLANK = "blank"
    ESSAY = "essay"
    MATCHING = "matching"
    ORDERING = "ordering"


# Models
class Question(Base):
    """Model for storing individual questions"""
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    options = Column(Text, nullable=True)  # JSON string for MCQ options
    answer = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)

    # Classification
    subject = Column(String(50), nullable=False, index=True)
    topic = Column(String(100), nullable=True, index=True)
    grade = Column(String(20), nullable=True)
    question_type = Column(String(20), default="mcq")

    # Difficulty
    difficulty = Column(String(20), default="medium")
    difficulty_score = Column(Float, nullable=True)  # AI-calculated score 0-1

    # Metadata
    tags = Column(Text, nullable=True)  # JSON array of tags
    source = Column(String(255), nullable=True)  # Where the question came from
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Quality metrics (from AI review)
    quality_score = Column(Float, nullable=True)  # 0-1 score
    quality_issues = Column(Text, nullable=True)  # JSON array of issues

    # Usage stats
    times_used = Column(Integer, default=0)

    # Relationships
    exam_questions = relationship("ExamQuestion", back_populates="question")


class Exam(Base):
    """Model for storing exam/test history"""
    __tablename__ = "exams"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Classification
    subject = Column(String(50), nullable=False)
    grade = Column(String(20), nullable=True)

    # Configuration
    total_questions = Column(Integer, default=0)
    duration_minutes = Column(Integer, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    exam_questions = relationship("ExamQuestion", back_populates="exam", cascade="all, delete-orphan")
    variants = relationship("ExamVariant", back_populates="exam", cascade="all, delete-orphan")


class ExamQuestion(Base):
    """Association table for Exam-Question with order"""
    __tablename__ = "exam_questions"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    order = Column(Integer, default=0)
    points = Column(Float, default=1.0)

    # Relationships
    exam = relationship("Exam", back_populates="exam_questions")
    question = relationship("Question", back_populates="exam_questions")


class ExamVariant(Base):
    """Model for storing different versions of an exam"""
    __tablename__ = "exam_variants"

    id = Column(Integer, primary_key=True, index=True)
    exam_id = Column(Integer, ForeignKey("exams.id"), nullable=False)
    variant_code = Column(String(10), nullable=False)  # e.g., "A", "B", "C"

    # Question order (JSON array of question IDs in shuffled order)
    question_order = Column(Text, nullable=False)
    # Answer key mapping (JSON object mapping original -> shuffled)
    answer_mapping = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    exam = relationship("Exam", back_populates="variants")


class Tag(Base):
    """Model for question tags"""
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=True)  # e.g., "topic", "skill", "bloom"
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# CRUD Operations for Questions
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


# CRUD Operations for Exams
class ExamCRUD:
    @staticmethod
    def create(db: Session, **kwargs) -> Exam:
        exam = Exam(**kwargs)
        db.add(exam)
        db.commit()
        db.refresh(exam)
        return exam

    @staticmethod
    def get(db: Session, exam_id: int) -> Optional[Exam]:
        return db.query(Exam).filter(Exam.id == exam_id).first()

    @staticmethod
    def get_all(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        subject: Optional[str] = None,
    ) -> List[Exam]:
        query = db.query(Exam)

        if subject:
            query = query.filter(Exam.subject == subject)

        return query.order_by(Exam.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def update(db: Session, exam_id: int, **kwargs) -> Optional[Exam]:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if exam:
            for key, value in kwargs.items():
                if hasattr(exam, key):
                    setattr(exam, key, value)
            db.commit()
            db.refresh(exam)
        return exam

    @staticmethod
    def delete(db: Session, exam_id: int) -> bool:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if exam:
            db.delete(exam)
            db.commit()
            return True
        return False

    @staticmethod
    def add_questions(db: Session, exam_id: int, question_ids: List[int]) -> Exam:
        exam = db.query(Exam).filter(Exam.id == exam_id).first()
        if not exam:
            return None

        # Get current max order
        max_order = db.query(ExamQuestion).filter(
            ExamQuestion.exam_id == exam_id
        ).count()

        for i, qid in enumerate(question_ids):
            eq = ExamQuestion(
                exam_id=exam_id,
                question_id=qid,
                order=max_order + i + 1
            )
            db.add(eq)

        exam.total_questions = max_order + len(question_ids)
        db.commit()
        db.refresh(exam)
        return exam

    @staticmethod
    def remove_question(db: Session, exam_id: int, question_id: int) -> bool:
        eq = db.query(ExamQuestion).filter(
            ExamQuestion.exam_id == exam_id,
            ExamQuestion.question_id == question_id
        ).first()

        if eq:
            db.delete(eq)
            # Update exam total
            exam = db.query(Exam).filter(Exam.id == exam_id).first()
            if exam:
                exam.total_questions = db.query(ExamQuestion).filter(
                    ExamQuestion.exam_id == exam_id
                ).count() - 1
            db.commit()
            return True
        return False

    @staticmethod
    def create_variant(db: Session, exam_id: int, variant_code: str, question_order: str, answer_mapping: str = None) -> ExamVariant:
        variant = ExamVariant(
            exam_id=exam_id,
            variant_code=variant_code,
            question_order=question_order,
            answer_mapping=answer_mapping
        )
        db.add(variant)
        db.commit()
        db.refresh(variant)
        return variant


# Initialize database on import
init_db()
