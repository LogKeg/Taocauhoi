"""
Scraper for hoc247.net — Vietnamese education site.

Supports:
  - Quiz/test pages with multiple choice questions
  - Category/listing pages for discovering quiz URLs
  - Math formulas rendered with MathJax (LaTeX)
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
    "Referer": "https://hoc247.net/",
}

REQUEST_DELAY = 1.0  # seconds between requests

# Subject mapping from URL slugs
SUBJECT_MAP = {
    "toan": "math",
    "ngu-van": "literature",
    "van": "literature",
    "vat-ly": "physics",
    "vat-li": "physics",
    "hoa-hoc": "chemistry",
    "hoa": "chemistry",
    "sinh-hoc": "biology",
    "sinh": "biology",
    "tieng-anh": "english",
    "lich-su": "history",
    "dia-li": "geography",
    "dia": "geography",
    "gdcd": "civic_education",
    "cong-nghe": "technology",
    "tin-hoc": "informatics",
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
    """Detect subject from URL or title."""
    text = (url + " " + title).lower()
    for slug, subj in SUBJECT_MAP.items():
        if slug in text:
            return subj
    return "general"


def _detect_grade(url: str, title: str = "") -> str:
    """Detect grade from URL or title."""
    text = url + " " + title
    # Also check for "-12-", "-11-", etc. in URL
    m = re.search(r"-(\d{1,2})-", url)
    if m:
        return m.group(1)
    m = GRADE_RE.search(text)
    return m.group(1) if m else ""


def _clean_text(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_latex(el) -> str:
    """Extract text including LaTeX formulas from element."""
    if el is None:
        return ""

    # Get text content
    text = el.get_text(separator=' ', strip=True)

    # Find LaTeX patterns like \(formula\) or \[formula\]
    # These are already in the text, just clean up
    text = _clean_text(text)

    return text


def parse_quiz_page(html: str, url: str = "") -> List[Dict]:
    """
    Parse Hoc247 quiz page to extract multiple choice questions.

    Structure:
    - Questions have "Câu X:" headers in h3.b-title
    - Question text in <a> tag with <p> content
    - Options in <ul class="dstl"> with <li> items
    - Option labels in <span class="cautl">
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Get page title
    title_el = soup.select_one("h1, .title, .page-title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Detect metadata
    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Find all dstl (danh sach tra loi = answer list) elements
    option_lists = soup.find_all('ul', class_='dstl')

    for ul in option_lists:
        # Find the question - look for h3 with b-title class before this ul
        h3 = ul.find_previous('h3', class_='b-title')
        if not h3:
            # Try finding any h3
            h3 = ul.find_previous('h3')
        if not h3:
            continue

        # Find the question text in <a> tag inside h3
        question_link = h3.find('a')
        if question_link:
            # Get text from <p> inside <a> or the <a> itself
            p_tag = question_link.find('p')
            if p_tag:
                question_text = p_tag.get_text(separator=' ', strip=True)
            else:
                question_text = question_link.get_text(separator=' ', strip=True)
        else:
            question_text = h3.get_text(separator=' ', strip=True)

        # Remove "Câu X:" prefix if present
        question_text = re.sub(r'^Câu\s*\d+[.:]\s*', '', question_text, flags=re.IGNORECASE)
        question_text = _clean_text(question_text)

        if not question_text or len(question_text) < 5:
            continue

        # Extract options from <li> items
        options = []
        for li in ul.find_all('li', recursive=False):
            # Get option label (A, B, C, D)
            label_el = li.find('span', class_='cautl')
            if not label_el:
                continue

            # Get option text - find span siblings after cautl
            option_spans = li.find_all('span')
            option_text = ""
            for span in option_spans:
                if 'cautl' not in span.get('class', []):
                    option_text += span.get_text(separator=' ', strip=True) + " "

            option_text = _clean_text(option_text)
            if option_text:
                options.append(option_text)

        # Only save if we have valid question and options
        if question_text and len(options) >= 2:
            questions.append({
                "question": question_text,
                "options": options[:4],  # Max 4 options A-D
                "answer": "",  # Hoc247 doesn't show answers directly
                "subject": subject,
                "grade": grade,
                "source": url,
            })

    return questions


def discover_quiz_urls(html: str, base_url: str) -> List[str]:
    """
    Discover quiz page URLs from a category/listing page.
    """
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Look for quiz-related links
        if any(pattern in href.lower() for pattern in [
            'trac-nghiem-', 'cau-hoi-', 'bai-tap-', 'de-thi-'
        ]) and href.endswith('.html'):
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://hoc247.net/') and full_url not in urls:
                # Skip index pages, we want actual quiz pages
                if '-index.html' not in full_url:
                    urls.append(full_url)

    return urls


def scrape_quiz(url: str) -> Tuple[List[Dict], Optional[str]]:
    """
    Scrape a single quiz page.

    Returns:
        (questions, error_message)
    """
    html, error = _fetch_html(url)
    if error:
        return [], error

    questions = parse_quiz_page(html, url)
    if not questions:
        return [], "Không tìm thấy câu hỏi trắc nghiệm"

    return questions, None


def scrape_category(url: str, max_pages: int = 10) -> Tuple[List[Dict], List[str]]:
    """
    Scrape multiple quizzes from a category page.

    Returns:
        (all_questions, errors)
    """
    all_questions = []
    errors = []

    html, error = _fetch_html(url)
    if error:
        return [], [error]

    # First try to parse the page itself (might be a quiz page)
    questions = parse_quiz_page(html, url)
    if questions:
        all_questions.extend(questions)

    # Then discover and scrape linked quiz pages
    quiz_urls = discover_quiz_urls(html, url)[:max_pages]

    for quiz_url in quiz_urls:
        time.sleep(REQUEST_DELAY)
        questions, err = scrape_quiz(quiz_url)
        if err:
            errors.append(f"{quiz_url}: {err}")
        else:
            all_questions.extend(questions)

    return all_questions, errors
