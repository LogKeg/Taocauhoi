"""
DOCX parsing utilities for exam questions.
"""
from .extractor import (
    extract_paragraph_with_math,
    extract_cell_with_math,
    extract_docx_content,
    extract_docx_lines,
)
from .math_parser import (
    parse_cell_based_questions,
    has_highlight,
    get_highlighted_option_from_cell,
    get_cell_text_from_row,
)
from .bilingual_parser import (
    extract_docx_lines as extract_docx_lines_with_options,
    parse_bilingual_questions,
)
from .exam_english_parser import (
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

__all__ = [
    # Extractor functions
    "extract_paragraph_with_math",
    "extract_cell_with_math",
    "extract_docx_content",
    "extract_docx_lines",
    # Math parser functions
    "parse_cell_based_questions",
    "has_highlight",
    "get_highlighted_option_from_cell",
    "get_cell_text_from_row",
    # Bilingual parser functions
    "extract_docx_lines_with_options",
    "parse_bilingual_questions",
    # English exam parser functions
    "_parse_english_exam_questions",
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
