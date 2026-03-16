"""
Scraper for vietjack.com — Vietnamese education exam site.

Supports:
  - Exam/test pages with multiple choice questions
  - Category/listing pages for discovering exam URLs
  - Math formulas rendered with MathJax
"""
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

try:
    from . import image_filter as img_filter
except ImportError:
    import image_filter as img_filter

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://vietjack.com/",
}

REQUEST_DELAY = 1.0  # seconds between requests

# Subject mapping
SUBJECT_MAP = {
    "toan": "math",
    "ngu-van": "literature",
    "van": "literature",
    "vat-ly": "physics",
    "vat-li": "physics",
    "hoa": "chemistry",
    "hoa-hoc": "chemistry",
    "sinh": "biology",
    "sinh-hoc": "biology",
    "tieng-anh": "english",
    "lich-su": "history",
    "dia-li": "geography",
    "dia-ly": "geography",
    "gdcd": "civic_education",
    "kinh-te": "economics",
    "tin": "informatics",
    "tin-hoc": "informatics",
    "cong-nghe": "technology",
}

GRADE_RE = re.compile(r"(?:lop|lớp)[- ]?(\d{1,2})", re.IGNORECASE)
QUESTION_NUM_RE = re.compile(r"^Câu\s*(\d+)[.:]\s*", re.IGNORECASE)
OPTION_RE = re.compile(r"^([A-D])[.):]\s*(.+)$", re.IGNORECASE)


def _fetch_html(url: str, timeout: int = 30) -> Tuple[str, Optional[str]]:
    """Fetch HTML with browser-like headers."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=HEADERS)
            resp.raise_for_status()
            return resp.text, None
    except httpx.TimeoutException:
        return "", f"Timeout: {url}"
    except httpx.HTTPStatusError as e:
        return "", f"HTTP {e.response.status_code}: {url}"
    except Exception as e:
        return "", f"Lỗi: {str(e)}"


def _detect_subject(url: str, title: str = "") -> str:
    """Detect subject from URL or title."""
    text = (url + " " + title).lower()
    for slug, subj in SUBJECT_MAP.items():
        if slug in text:
            return subj
    return "general"


def _detect_grade(url: str, title: str = "") -> str:
    """Detect grade from URL or title."""
    text = url + " " + title
    m = GRADE_RE.search(text)
    return m.group(1) if m else ""


def _clean_text(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_math_text(el: Tag) -> str:
    """Extract text including MathJax formulas."""
    parts = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == 'script' and child.get('type') == 'math/tex':
                # MathJax inline formula
                parts.append(f"${child.get_text()}$")
            elif child.name == 'span' and 'MathJax' in child.get('class', []):
                # Skip MathJax rendered spans (we have the script)
                continue
            else:
                parts.append(_extract_math_text(child))
    return ''.join(parts)


def parse_exam_page(html: str, url: str = "") -> List[Dict]:
    """
    Parse VietJack exam page to extract multiple choice questions.

    VietJack structure:
    - Full text contains "Câu X." followed by question and options
    - Options can be A. B. C. D. on same line or separate lines
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Get page title
    title_el = soup.select_one("h1, .entry-title, .post-title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Detect metadata
    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Get full page text
    full_text = soup.get_text()

    # Split by question pattern: "Câu X." or "Câu X:"
    # Each match is one question block
    question_pattern = r'Câu\s*(\d+)[.:]\s*'
    parts = re.split(question_pattern, full_text)

    # parts[0] is intro, then alternating: question_num, question_content
    for i in range(1, len(parts) - 1, 2):
        q_num = parts[i]
        q_content = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if not q_content:
            continue

        # Extract question text and options
        # Options pattern: A. ... B. ... C. ... D. ...
        option_pattern = r'([A-D])[.)]\s*'

        # Find where options start
        option_match = re.search(r'\b[A-D][.)]\s*\S', q_content)
        if option_match:
            question_text = q_content[:option_match.start()].strip()
            options_text = q_content[option_match.start():]
        else:
            # No clear options, skip
            continue

        # Parse options
        option_parts = re.split(option_pattern, options_text)
        options = []

        for j in range(1, len(option_parts) - 1, 2):
            letter = option_parts[j]
            opt_text = option_parts[j + 1].strip() if j + 1 < len(option_parts) else ""

            # Clean option text - stop at next question or answer section
            opt_text = re.split(r'(?=Câu\s*\d+[.:]|Đáp án|Lời giải|Hướng dẫn)', opt_text)[0]
            opt_text = _clean_text(opt_text)

            if opt_text and len(opt_text) > 0:
                options.append(opt_text)

        # Only save if we have question and options
        if question_text and len(options) >= 2:
            questions.append({
                "question": _clean_text(question_text),
                "options": options[:4],  # Max 4 options A-D
                "answer": "",
                "subject": subject,
                "grade": grade,
                "source": url,
            })

    return questions


def discover_exam_urls(html: str, base_url: str) -> List[str]:
    """
    Discover exam page URLs from a category/listing page.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True).lower()

        # Look for exam-related links
        if any(kw in href.lower() or kw in text for kw in [
            'de-thi', 'de-kiem-tra', 'trac-nghiem', 'bai-tap',
            'đề thi', 'đề kiểm tra', 'trắc nghiệm', 'bài tập'
        ]):
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://vietjack.com/') and full_url not in urls:
                urls.append(full_url)

    return urls


def scrape_exam(url: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Scrape a single exam page.

    Returns:
        (questions, error_message)
    """
    html, error = _fetch_html(url)
    if error:
        return [], error

    questions = parse_exam_page(html, url)
    if not questions:
        return [], "Không tìm thấy câu hỏi trắc nghiệm"

    return questions, None


def scrape_category(url: str, max_pages: int = 5) -> Tuple[List[Dict], List[str]]:
    """
    Scrape multiple exams from a category page.

    Returns:
        (all_questions, errors)
    """
    all_questions = []
    errors = []

    html, error = _fetch_html(url)
    if error:
        return [], [error]

    exam_urls = discover_exam_urls(html, url)[:max_pages]

    for exam_url in exam_urls:
        time.sleep(REQUEST_DELAY)
        questions, err = scrape_exam(exam_url)
        if err:
            errors.append(f"{exam_url}: {err}")
        else:
            all_questions.extend(questions)

    return all_questions, errors
