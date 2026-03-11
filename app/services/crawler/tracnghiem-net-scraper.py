"""
Scraper for tracnghiem.net — Vietnamese education site with 2M+ questions.

Supports:
  - Quiz/test pages with multiple choice questions
  - Category/listing pages for discovering quiz URLs
  - Math formulas rendered with MathJax (LaTeX)
  - Grades 1-12, university, and national exams
"""
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://tracnghiem.net/",
}

REQUEST_DELAY = 1.0  # seconds between requests

# Subject mapping from URL slugs (longer slugs first for priority matching)
SUBJECT_MAP = {
    "tieng-anh": "english",
    "ngu-van": "literature",
    "vat-ly": "physics",
    "vat-li": "physics",
    "hoa-hoc": "chemistry",
    "sinh-hoc": "biology",
    "lich-su": "history",
    "dia-ly": "geography",
    "dia-li": "geography",
    "tin-hoc": "informatics",
    "cong-nghe": "technology",
    "gdcd": "civic_education",
    "giao-duc-cong-dan": "civic_education",
    "toan": "math",
    "van": "literature",
    "hoa": "chemistry",
    "sinh": "biology",
    "su": "history",
    "dia": "geography",
}

GRADE_RE = re.compile(r"(?:lop|lớp)[- ]?(\d{1,2})", re.IGNORECASE)


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
    """Detect subject from URL or title, prioritizing longer/more specific matches."""
    filename = url.split('/')[-1].lower() if '/' in url else url.lower()
    title_lower = title.lower()

    # Sort by slug length (longer = more specific)
    sorted_subjects = sorted(SUBJECT_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    # Check filename first
    for slug, subj in sorted_subjects:
        if slug in filename:
            return subj

    # Check title
    for slug, subj in sorted_subjects:
        if slug in title_lower:
            return subj

    # Check full URL
    url_lower = url.lower()
    for slug, subj in sorted_subjects:
        if slug in url_lower:
            return subj

    return "general"


def _detect_grade(url: str, title: str = "") -> str:
    """Detect grade from URL or title."""
    text = url + " " + title
    # Check for "-12-", "-11-", etc. in URL
    m = re.search(r"-(\d{1,2})-", url)
    if m:
        grade = m.group(1)
        if 1 <= int(grade) <= 12:
            return grade
    m = GRADE_RE.search(text)
    return m.group(1) if m else ""


def _clean_text(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_quiz_page(html: str, url: str = "") -> List[Dict]:
    """
    Parse TracNghiem.net quiz page to extract multiple choice questions.

    The site uses a React-based structure where questions are split by "Câu X:" pattern.
    Options are marked with A., B., C., D. prefixes.
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Get page title
    title_el = soup.select_one("h1, .title, .page-title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Detect metadata
    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Extract all text and split by "Câu X:" pattern
    all_text = soup.get_text(separator='\n')

    # Split into question blocks
    # Pattern: "Câu X:" followed by content until next "Câu" or end
    question_pattern = re.compile(r'Câu\s*\d+\s*[:.]\s*', re.IGNORECASE)
    parts = question_pattern.split(all_text)

    for part in parts[1:]:  # Skip first part (before Câu 1)
        if not part.strip():
            continue

        # Split options using A., B., C., D. pattern
        # Options pattern: newline or space followed by A. B. C. D.
        option_pattern = re.compile(r'\n\s*([A-D])[.)]\s*')
        option_splits = option_pattern.split(part)

        if len(option_splits) < 3:
            # Try alternative pattern (options on same line)
            option_pattern2 = re.compile(r'\s+([A-D])[.)]\s+')
            option_splits = option_pattern2.split(part)

        if len(option_splits) >= 3:
            # First part is the question
            question_text = option_splits[0].strip()

            # Clean question text - remove trailing content after options
            question_text = re.sub(r'\s*Lời giải.*$', '', question_text, flags=re.DOTALL)
            question_text = _clean_text(question_text)

            if not question_text or len(question_text) < 5:
                continue

            # Extract options (pairs of letter + content)
            options = []
            i = 1
            while i < len(option_splits) - 1:
                letter = option_splits[i]
                content = option_splits[i + 1].strip() if i + 1 < len(option_splits) else ""

                # Clean option content - remove next question markers
                content = re.sub(r'\s*Câu\s*\d+.*$', '', content, flags=re.DOTALL)
                content = re.sub(r'\s*Lời giải.*$', '', content, flags=re.DOTALL)
                content = _clean_text(content)

                if content:
                    options.append(content)
                i += 2

            # Only save if we have valid question and at least 2 options
            if question_text and len(options) >= 2:
                questions.append({
                    "question": question_text,
                    "options": options[:4],
                    "answer": "",  # TracNghiem.net requires login to see answers
                    "subject": subject,
                    "grade": grade,
                    "source": url,
                })

    return questions


def discover_quiz_urls(html: str, base_url: str) -> List[str]:
    """Discover quiz page URLs from a category/listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Look for quiz-related links (ends with .html and has numeric ID)
        if href.endswith('.html') and re.search(r'/\d+\.html$', href):
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://tracnghiem.net/') and full_url not in urls:
                urls.append(full_url)

    return urls


def scrape_quiz(url: str) -> Tuple[List[Dict], Optional[str]]:
    """Scrape a single quiz page."""
    html, error = _fetch_html(url)
    if error:
        return [], error

    questions = parse_quiz_page(html, url)
    if not questions:
        return [], "Không tìm thấy câu hỏi trắc nghiệm"

    return questions, None


def scrape_category(url: str, max_pages: int = 20) -> Tuple[List[Dict], List[str]]:
    """
    Scrape multiple quizzes from a category page.

    Returns:
        (all_questions, errors)
    """
    all_questions = []
    errors = []
    scraped_urls = set()

    html, error = _fetch_html(url)
    if error:
        return [], [error]

    # First try to parse the page itself
    questions = parse_quiz_page(html, url)
    if questions:
        all_questions.extend(questions)

    # Discover quiz URLs
    quiz_urls = discover_quiz_urls(html, url)

    # Scrape quiz pages
    for quiz_url in quiz_urls[:max_pages]:
        if quiz_url in scraped_urls:
            continue
        scraped_urls.add(quiz_url)

        time.sleep(REQUEST_DELAY)
        questions, err = scrape_quiz(quiz_url)
        if err:
            errors.append(f"{quiz_url}: {err}")
        else:
            all_questions.extend(questions)

    return all_questions, errors


# Categories for auto-scraping
TRACNGHIEM_NET_CATEGORIES = {
    # THPT National Exam
    "tnthpt_toan": {"url": "https://tracnghiem.net/tnthpt/", "name": "Tốt nghiệp THPT - Toán"},

    # High school by grade
    "thpt_12_toan": {"url": "https://tracnghiem.net/de-thi-thpt/toan-hoc-lop-12/", "name": "Toán 12"},
    "thpt_12_ly": {"url": "https://tracnghiem.net/de-thi-thpt/vat-ly-lop-12/", "name": "Vật lý 12"},
    "thpt_12_hoa": {"url": "https://tracnghiem.net/de-thi-thpt/hoa-hoc-lop-12/", "name": "Hóa học 12"},
    "thpt_12_sinh": {"url": "https://tracnghiem.net/de-thi-thpt/sinh-hoc-lop-12/", "name": "Sinh học 12"},
    "thpt_12_anh": {"url": "https://tracnghiem.net/de-thi-thpt/tieng-anh-lop-12/", "name": "Tiếng Anh 12"},
    "thpt_12_su": {"url": "https://tracnghiem.net/de-thi-thpt/lich-su-lop-12/", "name": "Lịch sử 12"},
    "thpt_12_dia": {"url": "https://tracnghiem.net/de-thi-thpt/dia-ly-lop-12/", "name": "Địa lý 12"},

    "thpt_11_toan": {"url": "https://tracnghiem.net/de-thi-thpt/toan-hoc-lop-11/", "name": "Toán 11"},
    "thpt_11_ly": {"url": "https://tracnghiem.net/de-thi-thpt/vat-ly-lop-11/", "name": "Vật lý 11"},
    "thpt_11_hoa": {"url": "https://tracnghiem.net/de-thi-thpt/hoa-hoc-lop-11/", "name": "Hóa học 11"},

    "thpt_10_toan": {"url": "https://tracnghiem.net/de-thi-thpt/toan-hoc-lop-10/", "name": "Toán 10"},
    "thpt_10_ly": {"url": "https://tracnghiem.net/de-thi-thpt/vat-ly-lop-10/", "name": "Vật lý 10"},
    "thpt_10_hoa": {"url": "https://tracnghiem.net/de-thi-thpt/hoa-hoc-lop-10/", "name": "Hóa học 10"},

    # THCS (Secondary school)
    "thcs_9_toan": {"url": "https://tracnghiem.net/de-thi-thcs/toan-hoc-lop-9/", "name": "Toán 9"},
    "thcs_9_anh": {"url": "https://tracnghiem.net/de-thi-thcs/tieng-anh-lop-9/", "name": "Tiếng Anh 9"},
    "thcs_8_toan": {"url": "https://tracnghiem.net/de-thi-thcs/toan-hoc-lop-8/", "name": "Toán 8"},
    "thcs_7_toan": {"url": "https://tracnghiem.net/de-thi-thcs/toan-hoc-lop-7/", "name": "Toán 7"},
    "thcs_6_toan": {"url": "https://tracnghiem.net/de-thi-thcs/toan-hoc-lop-6/", "name": "Toán 6"},

    # Elementary
    "tieuhoc_5_toan": {"url": "https://tracnghiem.net/de-thi-tieu-hoc/toan-hoc-lop-5/", "name": "Toán 5"},
    "tieuhoc_4_toan": {"url": "https://tracnghiem.net/de-thi-tieu-hoc/toan-hoc-lop-4/", "name": "Toán 4"},
    "tieuhoc_3_toan": {"url": "https://tracnghiem.net/de-thi-tieu-hoc/toan-hoc-lop-3/", "name": "Toán 3"},
}
