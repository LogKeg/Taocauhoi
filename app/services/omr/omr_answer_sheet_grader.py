"""OMR Grader: Re-exports grading functions from modularized files.

This file maintains backward compatibility by re-exporting functions from:
- omr-single-sheet-grader.py: _grade_single_sheet
- omr-mixed-format-sheet-grader.py: _grade_mixed_format_sheet
- omr-answer-key-parser-and-text-extractor.py: _extract_answers_from_text, _parse_answer_key_for_template
"""

import importlib.util
import os

# Import from kebab-case module files using importlib
_current_dir = os.path.dirname(__file__)

# Load omr-single-sheet-grader.py
_single_sheet_path = os.path.join(_current_dir, "omr-single-sheet-grader.py")
_single_sheet_spec = importlib.util.spec_from_file_location("omr_single_sheet_grader", _single_sheet_path)
_single_sheet_module = importlib.util.module_from_spec(_single_sheet_spec)
_single_sheet_spec.loader.exec_module(_single_sheet_module)
_grade_single_sheet = _single_sheet_module._grade_single_sheet

# Load omr-mixed-format-sheet-grader.py
_mixed_format_path = os.path.join(_current_dir, "omr-mixed-format-sheet-grader.py")
_mixed_format_spec = importlib.util.spec_from_file_location("omr_mixed_format_grader", _mixed_format_path)
_mixed_format_module = importlib.util.module_from_spec(_mixed_format_spec)
_mixed_format_spec.loader.exec_module(_mixed_format_module)
_grade_mixed_format_sheet = _mixed_format_module._grade_mixed_format_sheet

# Load omr-answer-key-parser-and-text-extractor.py
_answer_key_path = os.path.join(_current_dir, "omr-answer-key-parser-and-text-extractor.py")
_answer_key_spec = importlib.util.spec_from_file_location("omr_answer_key_parser", _answer_key_path)
_answer_key_module = importlib.util.module_from_spec(_answer_key_spec)
_answer_key_spec.loader.exec_module(_answer_key_module)
_extract_answers_from_text = _answer_key_module._extract_answers_from_text
_parse_answer_key_for_template = _answer_key_module._parse_answer_key_for_template

__all__ = [
    "_grade_single_sheet",
    "_grade_mixed_format_sheet",
    "_extract_answers_from_text",
    "_parse_answer_key_for_template",
]
