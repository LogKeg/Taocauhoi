"""
Database models for Question Bank system.
Using SQLite with SQLAlchemy ORM.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

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

    # Image support
    image_url = Column(Text, nullable=True)  # Relative path: "images/{id}/filename.png"

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


class UsageHistory(Base):
    """Model for storing usage history (generated exams, topic questions)"""
    __tablename__ = "usage_history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String(50), unique=True, nullable=False, index=True)
    history_type = Column(String(20), nullable=False)  # "create", "topic"
    filename = Column(String(255), nullable=True)
    count = Column(Integer, default=0)
    difficulty = Column(String(20), nullable=True)
    questions_json = Column(Text, nullable=True)  # JSON array of questions
    created_at = Column(DateTime, default=datetime.utcnow)


class Curriculum(Base):
    """Model for storing curriculum framework from Bộ GD&ĐT"""
    __tablename__ = "curriculum"

    id = Column(Integer, primary_key=True, index=True)
    subject = Column(String(50), nullable=False, index=True)  # toan, van, anh, etc.
    grade = Column(Integer, nullable=False, index=True)  # 1-12

    # Content structure
    chapter = Column(String(255), nullable=True)  # Chương/Phần
    topic = Column(String(255), nullable=True)  # Chủ đề
    lesson = Column(String(255), nullable=True)  # Bài học

    # Learning objectives
    knowledge = Column(Text, nullable=True)  # Kiến thức cần đạt
    skills = Column(Text, nullable=True)  # Kỹ năng cần đạt
    competencies = Column(Text, nullable=True)  # Năng lực cần hình thành

    # Time allocation
    periods = Column(Integer, nullable=True)  # Số tiết

    # Metadata
    source_url = Column(String(500), nullable=True)  # URL nguồn
    source_document = Column(String(255), nullable=True)  # Tên văn bản
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


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
