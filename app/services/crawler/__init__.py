"""
Web crawler services for importing questions from external sources.
"""
from .fetcher import fetch_page, fetch_questions_from_url, crawl_multiple_urls
from .parsers import parse_vietjack, parse_hoc247, parse_loigiaihay, parse_generic
from .thuvienhoclieu import scrape_category, scrape_single_quiz, QUIZ_CATEGORIES

__all__ = [
    "fetch_page",
    "fetch_questions_from_url",
    "crawl_multiple_urls",
    "parse_vietjack",
    "parse_hoc247",
    "parse_loigiaihay",
    "parse_generic",
    "scrape_category",
    "scrape_single_quiz",
    "QUIZ_CATEGORIES",
]
