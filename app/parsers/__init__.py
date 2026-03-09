"""
Parsers for various file formats.
"""

import importlib.util
import os

from app.parsers.exam_math_parser import _parse_math_exam_questions
from app.parsers.docx import (
    _parse_english_exam_questions,
    is_matching_section,
    is_matching_table_line,
    is_dialogue_completion,
    is_blank_only_line,
    is_dialogue_blank_line,
    is_dialogue_prompt_line,
    is_reading_passage_start,
    is_question_with_single_word_options,
    extract_passage_questions,
    is_passage_with_blanks,
    extract_cloze_questions,
)

# Load kebab-case modules using importlib (Python cannot import hyphened filenames directly)
_current_dir = os.path.dirname(__file__)

# Load exam-envie-bilingual-question-parser.py
_envie_parser_path = os.path.join(_current_dir, "exam-envie-bilingual-question-parser.py")
_envie_spec = importlib.util.spec_from_file_location("exam_envie_bilingual_question_parser", _envie_parser_path)
_envie_module = importlib.util.module_from_spec(_envie_spec)
_envie_spec.loader.exec_module(_envie_module)
_parse_envie_questions = _envie_module._parse_envie_questions

# Load envie-bilingual-dedup-helper.py
_dedup_path = os.path.join(_current_dir, "envie-bilingual-dedup-helper.py")
_dedup_spec = importlib.util.spec_from_file_location("envie_bilingual_dedup_helper", _dedup_path)
_dedup_module = importlib.util.module_from_spec(_dedup_spec)
_dedup_spec.loader.exec_module(_dedup_module)
_dedup_bilingual_science = _dedup_module._dedup_bilingual_science

__all__ = [
    "_parse_math_exam_questions",
    "_parse_english_exam_questions",
    "_parse_envie_questions",
    "_dedup_bilingual_science",
    # English helpers
    "is_matching_section",
    "is_matching_table_line",
    "is_dialogue_completion",
    "is_blank_only_line",
    "is_dialogue_blank_line",
    "is_dialogue_prompt_line",
    "is_reading_passage_start",
    "is_question_with_single_word_options",
    "extract_passage_questions",
    "is_passage_with_blanks",
    "extract_cloze_questions",
]
