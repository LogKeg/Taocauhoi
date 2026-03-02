"""
Core module containing constants and schemas.
"""
from .constants import (
    TOPICS,
    TEMPLATES,
    TOPIC_AI_GUIDE,
    SUBJECTS,
    SUBJECT_TOPICS,
    QUESTION_TYPES,
    SYNONYMS,
    NUMBER_RE,
    LEADING_NUM_RE,
    MCQ_OPTION_RE,
    ANSWER_TEMPLATES,
)

from .schemas import (
    GenerateRequest,
    ParseSamplesRequest,
    QuestionCreate,
    QuestionUpdate,
    ExamCreate,
    ExamUpdate,
    BulkSaveRequest,
    AIAnalyzeRequest,
    AISuggestRequest,
    AIReviewRequest,
)

__all__ = [
    # Constants
    "TOPICS",
    "TEMPLATES",
    "TOPIC_AI_GUIDE",
    "SUBJECTS",
    "SUBJECT_TOPICS",
    "QUESTION_TYPES",
    "SYNONYMS",
    "NUMBER_RE",
    "LEADING_NUM_RE",
    "MCQ_OPTION_RE",
    "ANSWER_TEMPLATES",
    # Schemas
    "GenerateRequest",
    "ParseSamplesRequest",
    "QuestionCreate",
    "QuestionUpdate",
    "ExamCreate",
    "ExamUpdate",
    "BulkSaveRequest",
    "AIAnalyzeRequest",
    "AISuggestRequest",
    "AIReviewRequest",
]
