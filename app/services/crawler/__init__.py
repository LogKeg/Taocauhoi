"""
Web crawler services for importing questions from external sources.
"""
from .fetcher import fetch_page, fetch_questions_from_url, crawl_multiple_urls
from .parsers import parse_vietjack, parse_hoc247, parse_loigiaihay, parse_generic

__all__ = [
    "fetch_page",
    "fetch_questions_from_url",
    "crawl_multiple_urls",
    "parse_vietjack",
    "parse_hoc247",
    "parse_loigiaihay",
    "parse_generic",
]
