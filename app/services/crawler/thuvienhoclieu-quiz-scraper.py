"""
Scraper for thuvienhoclieu.com — Vietnamese education site.

Supports:
  - Quiz pages using wpProQuiz plugin
  - Math formulas in LaTeX format ($...$)
  - Multiple choice questions with 4 options
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
}

REQUEST_DELAY = 1.0

# Subject mapping from URL patterns
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
    "su": "history",
    "dia-ly": "geography",
    "dia-li": "geography",
    "dia": "geography",
    "gdcd": "civic_education",
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
    url_lower = url.lower()
    title_lower = title.lower()

    # Sort by slug length for priority matching
    sorted_subjects = sorted(SUBJECT_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    for slug, subj in sorted_subjects:
        if slug in url_lower or slug in title_lower:
            return subj

    # Check Vietnamese names
    if "toán" in title_lower or "toan" in url_lower:
        return "math"
    if "lịch sử" in title_lower:
        return "history"
    if "địa" in title_lower:
        return "geography"

    return "general"


def _detect_grade(url: str, title: str = "") -> Optional[str]:
    """Extract grade from URL or title."""
    combined = f"{url} {title}"
    match = GRADE_RE.search(combined)
    if match:
        return match.group(1)

    # Check for "tốt nghiệp" or "THPT" -> grade 12
    if "tot-nghiep" in url.lower() or "thpt" in url.lower():
        return "12"

    return None


def _clean_text(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_image_url(element, base_url: str = "") -> Optional[str]:
    """Extract first meaningful image URL from element."""
    if not element:
        return None

    img = element.find('img')
    if img and img.get('src'):
        src = img['src']
        # Skip icons, logos, avatars
        if any(skip in src.lower() for skip in ['icon', 'avatar', 'logo', 'placeholder', 'loading']):
            return None
        # Skip very small images
        width = img.get('width', '')
        height = img.get('height', '')
        if width and height:
            try:
                if int(width) < 50 or int(height) < 50:
                    return None
            except ValueError:
                pass
        # Make absolute URL
        if src.startswith('//'):
            return 'https:' + src
        elif src.startswith('/'):
            return urljoin(base_url or 'https://thuvienhoclieu.com', src)
        elif src.startswith('http'):
            return src
        else:
            return urljoin(base_url or 'https://thuvienhoclieu.com', src)
    return None


def parse_quiz_page(html: str, url: str = "") -> List[Dict]:
    """
    Parse ThuVienHocLieu quiz page using wpProQuiz plugin structure.

    Structure:
    - Questions in .wpProQuiz_question
    - Question text in .wpProQuiz_question_text
    - Options in .wpProQuiz_questionListItem
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Get page title
    title_el = soup.select_one("h1, .entry-title, .page-title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Detect metadata
    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Find all question containers
    question_containers = soup.select('.wpProQuiz_question')

    for container in question_containers:
        # Get question text
        text_el = container.select_one('.wpProQuiz_question_text')
        if not text_el:
            continue

        question_text = _clean_text(text_el.get_text(separator=' ', strip=True))

        if not question_text or len(question_text) < 5:
            continue

        # Extract image from question
        image_url = _extract_image_url(text_el, url)
        if not image_url:
            image_url = _extract_image_url(container, url)

        # Get options
        options = []
        option_els = container.select('.wpProQuiz_questionListItem')
        for opt_el in option_els:
            opt_text = _clean_text(opt_el.get_text(separator=' ', strip=True))
            if opt_text:
                options.append(opt_text)

        # Only save if valid
        if question_text and len(options) >= 2:
            question_data = {
                "question": question_text,
                "options": options[:4],
                "answer": "",  # wpProQuiz hides answers
                "subject": subject,
                "grade": grade,
                "source": url,
            }
            if image_url:
                question_data["image_source_url"] = image_url
            questions.append(question_data)

    return questions


def discover_quiz_urls(html: str, base_url: str) -> List[str]:
    """Discover quiz page URLs from a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Look for quiz-related links
        if any(pattern in href.lower() for pattern in [
            'de-thi', 'trac-nghiem', 'online', 'quiz'
        ]):
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://thuvienhoclieu.com/') and full_url not in urls:
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


def scrape_multiple(urls: List[str], delay: float = REQUEST_DELAY) -> Tuple[List[Dict], List[str]]:
    """Scrape multiple quiz pages with delay between requests."""
    all_questions = []
    errors = []

    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(delay)

        questions, error = scrape_quiz(url)
        if error:
            errors.append(f"{url}: {error}")
        else:
            all_questions.extend(questions)

    return all_questions, errors
