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
]
