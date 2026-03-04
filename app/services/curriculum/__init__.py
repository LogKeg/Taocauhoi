"""
Curriculum service for fetching and managing curriculum framework from Bộ GD&ĐT.
"""

from .parser import parse_curriculum_text, extract_curriculum_from_pdf
from .fetcher import fetch_curriculum_pdf, get_curriculum_urls
from .data import SAMPLE_CURRICULUM_DATA, get_sample_curriculum

__all__ = [
    "parse_curriculum_text",
    "extract_curriculum_from_pdf",
    "fetch_curriculum_pdf",
    "get_curriculum_urls",
    "SAMPLE_CURRICULUM_DATA",
    "get_sample_curriculum",
]
