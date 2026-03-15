"""
Scraper for vietjack.com — Vietnamese education site.

Supports:
  - Exam/quiz pages with multiple choice questions
  - Math formulas (MathJax/LaTeX)
  - Text-based question parsing (Câu X: pattern)
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
    """Detect subject from URL or title."""
    filename = url.split('/')[-1].lower() if '/' in url else url.lower()
    title_lower = title.lower()

    # Sort by slug length for priority matching
    sorted_subjects = sorted(SUBJECT_MAP.items(), key=lambda x: len(x[0]), reverse=True)

    for slug, subj in sorted_subjects:
        if slug in filename:
            return subj

    for slug, subj in sorted_subjects:
        if slug in title_lower:
            return subj

    return "general"


def _detect_grade(url: str, title: str = "") -> Optional[str]:
    """Extract grade from URL or title."""
    combined = f"{url} {title}"
    match = GRADE_RE.search(combined)
    if match:
        return match.group(1)
    return None


def _clean_text(text: str) -> str:
    """Clean up whitespace and normalize text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _extract_question_images(soup, base_url: str = "") -> Dict[int, str]:
    """
    Extract images associated with questions.
    Maps question index to image URL.
    """
    images = {}

    # VietJack often has images between question text
    # Look for img tags and try to associate with nearby questions
    all_imgs = soup.find_all('img')

    for img in all_imgs:
        src = img.get('src', '')
        if not src:
            continue

        # Skip icons, logos, ads
        if any(skip in src.lower() for skip in [
            'icon', 'logo', 'avatar', 'ads', 'banner', 'button', 'pixel'
        ]):
            continue

        # Make absolute URL
        if src.startswith('//'):
            src = 'https:' + src
        elif src.startswith('/'):
            src = urljoin(base_url or 'https://vietjack.com', src)
        elif not src.startswith('http'):
            src = urljoin(base_url or 'https://vietjack.com', src)

        # Try to find associated question number from nearby text
        parent = img.parent
        if parent:
            nearby_text = parent.get_text()[:100]
            match = re.search(r'Câu\s*(\d+)', nearby_text)
            if match:
                q_idx = int(match.group(1)) - 1  # 0-indexed
                if q_idx not in images:
                    images[q_idx] = src

    return images


def parse_quiz_page(html: str, url: str = "") -> List[Dict]:
    """
    Parse VietJack quiz page using text-based parsing.

    Structure:
    - Questions split by "Câu X:" pattern
    - Options marked with A. B. C. D.
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Get page title
    title_el = soup.select_one("h1, .entry-title, title")
    title = title_el.get_text(strip=True) if title_el else ""

    # Detect metadata
    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Pre-extract images
    question_images = _extract_question_images(soup, url)

    # Get all text and split by question pattern
    all_text = soup.get_text(separator='\n')

    # Split by "Câu X:" pattern
    parts = re.split(r'\n\s*Câu\s*(\d+)[.:]\s*', all_text)

    # Parse each question
    for i in range(1, len(parts) - 1, 2):
        q_num = parts[i]
        q_content = parts[i + 1] if i + 1 < len(parts) else ""

        if not q_content.strip():
            continue

        # Split by options A. B. C. D.
        opt_splits = re.split(r'\n\s*([A-D])[.)]\s*', q_content)

        if len(opt_splits) >= 5:
            # First part is question text
            question_text = opt_splits[0].strip()

            # Clean up - remove extra content after options
            question_text = re.sub(r'\s*Lời giải.*$', '', question_text, flags=re.DOTALL)
            question_text = _clean_text(question_text)

            if len(question_text) < 5:
                continue

            # Extract options
            options = []
            j = 1
            while j < len(opt_splits) - 1:
                opt_content = opt_splits[j + 1].strip() if j + 1 < len(opt_splits) else ""
                # Clean option - remove next question markers
                opt_content = re.sub(r'\s*Câu\s*\d+.*$', '', opt_content, flags=re.DOTALL)
                opt_content = re.sub(r'\s*Lời giải.*$', '', opt_content, flags=re.DOTALL)
                opt_content = _clean_text(opt_content)
                if opt_content:
                    options.append(opt_content)
                j += 2

            # Skip if not enough content
            if len(question_text) < 10 or len(options) < 2:
                continue

            # Build question data
            question_data = {
                "question": question_text,
                "options": options[:4],
                "answer": "",
                "subject": subject,
                "grade": grade,
                "source": url,
            }

            # Add image if found
            q_idx = int(q_num) - 1
            if q_idx in question_images:
                question_data["image_source_url"] = question_images[q_idx]

            questions.append(question_data)

    return questions


def discover_quiz_urls(html: str, base_url: str) -> List[str]:
    """Discover quiz page URLs from a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Look for exam/quiz links
        if href.endswith('.jsp') and any(pattern in href.lower() for pattern in [
            'de-thi', 'de-kiem-tra', 'trac-nghiem', 'bai-tap'
        ]):
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://vietjack.com/') and full_url not in urls:
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
