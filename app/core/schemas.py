"""
Pydantic models for request/response validation.
"""
from typing import List, Optional
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Request model for generating exam variants."""
    samples: List[str]
    topic: str
    custom_keywords: List[str]
    paraphrase: bool
    change_numbers: bool
    change_context: bool
    variants_per_question: int
    use_ai: bool = False
    ai_engine: str = "openai"


class ParseSamplesRequest(BaseModel):
    """Request model for parsing sample URLs."""
    urls: List[str]


class QuestionCreate(BaseModel):
    """Model for creating a new question."""
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    subject: str
    topic: Optional[str] = None
    grade: Optional[str] = None
    question_type: str = "mcq"
    difficulty: str = "medium"
    tags: Optional[str] = None
    source: Optional[str] = None


class QuestionUpdate(BaseModel):
    """Model for updating an existing question."""
    content: Optional[str] = None
    options: Optional[str] = None
    answer: Optional[str] = None
    explanation: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    grade: Optional[str] = None
    question_type: Optional[str] = None
    difficulty: Optional[str] = None
    tags: Optional[str] = None
    source: Optional[str] = None


class ExamCreate(BaseModel):
    """Model for creating a new exam."""
    title: str
    description: Optional[str] = None
    subject: str
    grade: Optional[str] = None
    duration_minutes: Optional[int] = None


class ExamUpdate(BaseModel):
    """Model for updating an existing exam."""
    title: Optional[str] = None
    description: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[str] = None
    duration_minutes: Optional[int] = None


class BulkSaveRequest(BaseModel):
    """Request model for bulk saving questions."""
    questions: List[QuestionCreate]


class AIAnalyzeRequest(BaseModel):
    """Request model for AI difficulty analysis."""
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    subject: Optional[str] = None
    ai_engine: str = "openai"


class AISuggestRequest(BaseModel):
    """Request model for AI question suggestions."""
    content: str
    subject: Optional[str] = None
    count: int = 3
    ai_engine: str = "openai"


class AIReviewRequest(BaseModel):
    """Request model for AI quality review."""
    content: str
    options: Optional[str] = None
    answer: Optional[str] = None
    subject: Optional[str] = None
    ai_engine: str = "openai"
