"""
Database module for Question Bank system.
Re-exports all models and CRUD classes for backward compatibility.
"""

import importlib.util
import os

from app.database.models import (
    Base,
    Curriculum,
    DifficultyLevel,
    Exam,
    ExamQuestion,
    ExamVariant,
    Question,
    QuestionType,
    SessionLocal,
    Tag,
    UsageHistory,
    engine,
    get_db,
    init_db,
)

# Load CRUD modules using importlib (kebab-case filenames)
_current_dir = os.path.dirname(os.path.abspath(__file__))


def _load_module(filename: str):
    filepath = os.path.join(_current_dir, filename)
    module_name = filename.replace('-', '_').replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_question_crud = _load_module('question-crud-operations.py')
_exam_crud = _load_module('exam-crud-operations.py')
_history_crud = _load_module('history-crud-operations.py')
_curriculum_crud = _load_module('curriculum-crud-operations.py')

QuestionCRUD = _question_crud.QuestionCRUD
ExamCRUD = _exam_crud.ExamCRUD
HistoryCRUD = _history_crud.HistoryCRUD
CurriculumCRUD = _curriculum_crud.CurriculumCRUD

__all__ = [
    # Database setup
    'Base',
    'engine',
    'SessionLocal',
    'init_db',
    'get_db',
    # Enums
    'DifficultyLevel',
    'QuestionType',
    # Models
    'Question',
    'Exam',
    'ExamQuestion',
    'ExamVariant',
    'Tag',
    'UsageHistory',
    'Curriculum',
    # CRUD classes
    'QuestionCRUD',
    'ExamCRUD',
    'HistoryCRUD',
    'CurriculumCRUD',
]

# Initialize database on import
init_db()
