"""
Web fetcher for crawling question content from URLs.
"""
import re
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from .parsers import parse_vietjack, parse_hoc247, parse_loigiaihay, parse_generic


# Map domains to their specific parsers
DOMAIN_PARSERS = {
    "vietjack.com": parse_vietjack,
    "hoc247.net": parse_hoc247,
    "loigiaihay.com": parse_loigiaihay,
}


def fetch_page(url: str, timeout: int = 30) -> Tuple[str, Optional[str]]:
    """
    Fetch HTML content from a URL.

    Returns:
        Tuple of (html_content, error_message)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        }
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.text, None
    except httpx.TimeoutException:
        return "", f"Timeout khi tải trang: {url}"
    except httpx.HTTPStatusError as e:
        return "", f"Lỗi HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return "", f"Lỗi khi tải trang: {str(e)}"


def get_parser_for_url(url: str):
    """Get the appropriate parser for a URL based on its domain."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # Remove www. prefix if present
    if domain.startswith("www."):
        domain = domain[4:]

    # Check for known domains
    for known_domain, parser in DOMAIN_PARSERS.items():
        if known_domain in domain:
            return parser

    return parse_generic


def fetch_questions_from_url(
    url: str,
    subject: str = "",
    difficulty: str = "medium",
) -> Dict:
    """
    Fetch and parse questions from a URL.

    Returns:
        Dict with keys: questions, count, source, error
    """
    html, error = fetch_page(url)
    if error:
        return {"questions": [], "count": 0, "source": url, "error": error}

    # Get appropriate parser
    parser = get_parser_for_url(url)

    try:
        soup = BeautifulSoup(html, "html.parser")
        questions = parser(soup, subject=subject, difficulty=difficulty)

        return {
            "questions": questions,
            "count": len(questions),
            "source": url,
            "error": None,
        }
    except Exception as e:
        return {
            "questions": [],
            "count": 0,
            "source": url,
            "error": f"Lỗi khi phân tích trang: {str(e)}",
        }


def crawl_multiple_urls(
    urls: List[str],
    subject: str = "",
    difficulty: str = "medium",
    save_to_db: bool = True,
) -> Dict:
    """
    Crawl multiple URLs and optionally save questions to database.

    Returns:
        Dict with keys: total_questions, results, saved_count
    """
    from app.database import SessionLocal, QuestionCRUD

    results = []
    all_questions = []

    for url in urls:
        if not url or not url.strip():
            continue

        result = fetch_questions_from_url(url.strip(), subject, difficulty)
        results.append(result)

        if result["questions"]:
            all_questions.extend(result["questions"])

    saved_count = 0
    if save_to_db and all_questions:
        db = SessionLocal()
        try:
            for q in all_questions:
                QuestionCRUD.create(
                    db,
                    content=q["content"],
                    options=q.get("options"),
                    answer=q.get("answer", ""),
                    subject=q.get("subject", subject) or "general",
                    source=q.get("source", "crawled"),
                    difficulty=q.get("difficulty", difficulty),
                    question_type=q.get("question_type", "mcq"),
                )
                saved_count += 1
        finally:
            db.close()

    return {
        "total_questions": len(all_questions),
        "results": results,
        "saved_count": saved_count,
    }
