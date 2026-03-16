"""
Scraper for thuvienhoclieu.com — Vietnamese education quiz site.

Supports:
  - Interactive WP Pro Quiz pages (questions + correct answers in JS)
  - Category/listing pages (discover quiz URLs)
  - Document download pages (.docx files)
"""
import json
import re
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Referer": "https://thuvienhoclieu.com/",
}

# Delay between requests to be respectful
REQUEST_DELAY = 1.0  # seconds

# Subject mapping from Vietnamese URL slugs (sorted by length for priority matching)
SUBJECT_MAP = {
    "lich-su": "history",
    "tieng-anh": "english",
    "sinh-hoc": "biology",
    "hoa-hoc": "chemistry",
    "tin-hoc": "informatics",
    "ngu-van": "literature",
    "vat-ly": "physics",
    "vat-li": "physics",
    "dia-li": "geography",
    "dia-ly": "geography",
    "mon-su": "history",  # mon-su in URLs like de-thi-thu-...-online-mon-su
    "mon-dia": "geography",
    "mon-sinh": "biology",
    "mon-hoa": "chemistry",
    "toan": "math",
    "hoa": "chemistry",
    "sinh": "biology",
    "gdcd": "civic_education",
    "tin": "informatics",
    "dia": "geography",
    "su": "history",
}

# Grade detection
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
    """Detect subject from URL or page title."""
    text = (url + " " + title).lower()
    for slug, subj in SUBJECT_MAP.items():
        if slug in text:
            return subj
    return ""


def _detect_grade(url: str, title: str = "") -> str:
    """Detect grade level from URL or page title."""
    text = url + " " + title
    m = GRADE_RE.search(text)
    return m.group(1) if m else ""


def _extract_image_url(element, base_url: str = "") -> Optional[str]:
    """Extract the first meaningful image URL from an element."""
    if not element:
        return None

    img = element.find('img')
    if img and img.get('src'):
        src = img['src']
        # Skip icons, logos, avatars, placeholders
        if any(skip in src.lower() for skip in ['icon', 'avatar', 'logo', 'placeholder', 'loading', 'pixel']):
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
    Parse a WP Pro Quiz page to extract questions with correct answers.

    Questions are in HTML: <div class="wpProQuiz_question_text">
    Options: <ul class="wpProQuiz_questionList"> → <li> items
    Correct answers: embedded in JS `wpProQuizInitList` JSON
    """
    soup = BeautifulSoup(html, "html.parser")
    questions = []

    # Extract page title
    title_el = soup.select_one("h1.entry-title, h1.tdb-title-text, h1")
    title = title_el.get_text(strip=True) if title_el else ""

    subject = _detect_subject(url, title)
    grade = _detect_grade(url, title)

    # Find correct answer data from JavaScript
    correct_answers = {}
    page_text = str(soup)
    # Look for wpProQuizInitList.push({...}) blocks
    for push_match in re.finditer(
        r'wpProQuizInitList\.push\(\s*\{(.*?)\}\s*\)',
        page_text,
        re.DOTALL,
    ):
        block = push_match.group(1)
        # Extract json: {...} — find the balanced braces
        json_start = re.search(r'json\s*:\s*\{', block)
        if not json_start:
            continue
        start_pos = json_start.end() - 1  # position of opening {
        depth = 0
        end_pos = start_pos
        for i in range(start_pos, len(block)):
            if block[i] == '{':
                depth += 1
            elif block[i] == '}':
                depth -= 1
                if depth == 0:
                    end_pos = i + 1
                    break
        try:
            json_data = json.loads(block[start_pos:end_pos])
            for qid, qinfo in json_data.items():
                correct_arr = qinfo.get("correct", [])
                for idx, val in enumerate(correct_arr):
                    if val == 1:
                        correct_answers[str(qid)] = chr(65 + idx)
                        break
        except (json.JSONDecodeError, AttributeError):
            pass

    # Parse question items
    for item in soup.select("li.wpProQuiz_listItem"):
        q_text_el = item.select_one(".wpProQuiz_question_text")
        if not q_text_el:
            continue

        content = q_text_el.get_text(strip=True)
        # Clean up the question text
        content = re.sub(r"^Câu\s*\d+[.:]\s*", "", content)
        content = content.strip()

        if not content or len(content) < 5:
            continue

        # Extract options
        options = []
        option_list = item.select_one("ul.wpProQuiz_questionList")
        question_id = ""
        if option_list:
            question_id = option_list.get("data-question_id", "")
            for li in option_list.select("li.wpProQuiz_questionListItem"):
                label = li.select_one("label")
                if label:
                    # Get text without the radio input
                    opt_text = label.get_text(strip=True)
                    if opt_text:
                        options.append(opt_text)

        # Get correct answer
        answer = correct_answers.get(str(question_id), "")

        # Extract image from question text area or the question item
        image_url = _extract_image_url(q_text_el, url)
        if not image_url:
            image_url = _extract_image_url(item, url)

        question_data = {
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": answer,
            "subject": subject,
            "grade": grade,
            "difficulty": "medium",
            "question_type": "mcq" if options else "essay",
            "source": url or "thuvienhoclieu.com",
        }
        if image_url:
            question_data["image_source_url"] = image_url
        questions.append(question_data)

    return questions


def discover_quiz_links(html: str, base_url: str = "") -> List[str]:
    """
    Extract quiz page links from a category/listing page.

    Looks for links to individual quiz pages (not sub-category pages).
    """
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Patterns that identify category/listing pages (NOT individual quizzes)
    category_re = re.compile(
        r"/trac-nghiem-online/"  # sub-paths under trac-nghiem-online/ are categories
        r"|/trac-nghiem-online$"
    )

    # Patterns that identify individual quiz pages
    quiz_re = re.compile(
        r"de-trac-nghiem-online-[^/]{10,}"
        r"|trac-nghiem-online-[^/]+-de-\d+"
        r"|trac-nghiem-online-bai-[^/]+"
        r"|trac-nghiem-online-de-kiem-tra-[^/]+"
        r"|kiem-tra-\d+-phut-online-[^/]+"
        r"|trac-nghiem-truc-tuyen-[^/]+"
        r"|de-thi-thu-[^/]+-online[^/]*"
        r"|de-\d+-phut-[^/]+-online[^/]*"
        r"|de-thi-thu-[^/]*-online-[^/]+"  # New: de-thi-thu-...-online-mon-...
        r"|de-thi-thu-tn-[^/]*-online-[^/]+"  # de-thi-thu-tn-...-online-...
        r"|de-thi-thu-tot-nghiep-[^/]*-online[^/]*"  # de-thi-thu-tot-nghiep-...-online
    )

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href) if base_url else href

        parsed = urlparse(full_url)
        if "thuvienhoclieu.com" not in parsed.netloc:
            continue

        path = parsed.path.rstrip("/")

        # Skip fragments like #respond
        if parsed.fragment:
            continue

        # Skip category/listing pages and series pages
        if category_re.search(path):
            continue
        if "/series/" in path:
            continue

        # Match quiz URL patterns
        if quiz_re.search(path):
            links.add(full_url.split("#")[0].rstrip("/"))

    return sorted(links)


def discover_category_links(html: str, base_url: str = "") -> List[str]:
    """Extract sub-category links from a listing page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href) if base_url else href
        parsed = urlparse(full_url)

        if "thuvienhoclieu.com" not in parsed.netloc:
            continue

        path = parsed.path.rstrip("/")
        # Category pages like /trac-nghiem-online/trac-nghiem-toan/
        if "/trac-nghiem-online/" in path and path != "/trac-nghiem-online":
            links.add(full_url.rstrip("/"))

    return sorted(links)


def discover_docx_links(html: str, base_url: str = "") -> List[str]:
    """Extract .docx download links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.lower().endswith(".docx"):
            full_url = urljoin(base_url, href) if base_url else href
            links.add(full_url)

    # Also check embedded iframes for Office Online viewer
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        if "view.officeapps.live.com" in src:
            # Extract the original doc URL
            m = re.search(r"src=([^&]+)", src)
            if m:
                from urllib.parse import unquote
                doc_url = unquote(m.group(1))
                if doc_url.lower().endswith(".docx"):
                    links.add(doc_url)

    return sorted(links)


def scrape_category(
    category_url: str,
    max_pages: int = 50,
    max_quizzes: int = 200,
    on_progress=None,
) -> Dict:
    """
    Scrape all quizzes from a category page.

    1. Fetch category page
    2. Discover quiz links (+ pagination)
    3. Scrape each quiz page

    Args:
        category_url: URL of the category page
        max_pages: Max listing pages to scan for quiz links
        max_quizzes: Max quiz pages to scrape
        on_progress: Optional callback(current, total, message)

    Returns:
        Dict with questions, quiz_count, error_count, errors
    """
    all_questions = []
    all_quiz_urls = set()
    errors = []
    pages_scanned = 0

    # Phase 1: Discover quiz URLs from category pages
    urls_to_scan = [category_url]
    scanned = set()

    while urls_to_scan and pages_scanned < max_pages:
        url = urls_to_scan.pop(0)
        url_clean = url.split("#")[0].rstrip("/")
        if url_clean in scanned:
            continue
        scanned.add(url_clean)

        if on_progress:
            on_progress(pages_scanned, max_pages, f"Đang quét danh mục: {url}")

        html, error = _fetch_html(url)
        if error:
            errors.append({"url": url, "error": error})
            continue

        pages_scanned += 1

        # Find quiz links
        quiz_links = discover_quiz_links(html, url)
        all_quiz_urls.update(quiz_links)

        # Find sub-category and series links to explore
        soup = BeautifulSoup(html, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].split("#")[0].rstrip("/")
            if "thuvienhoclieu.com" not in href or href in scanned:
                continue
            parsed = urlparse(href)
            path = parsed.path.rstrip("/")
            # Sub-categories and series pages to explore for more quiz links
            is_subcategory = "/trac-nghiem-online/" in path and path.count("/") <= 4
            is_series = "/series/" in path and "trac-nghiem" in path
            is_pagination = bool(re.search(r"/page/\d+", path))
            if is_subcategory or is_series or is_pagination:
                urls_to_scan.append(href)

        # Pagination: numbered page links
        for a_tag in soup.select(".page-nav a, .pages a, .pagination a, .nav-links a"):
            href = a_tag.get("href", "").split("#")[0].rstrip("/")
            if href and "thuvienhoclieu.com" in href and href not in scanned:
                urls_to_scan.append(href)

        time.sleep(REQUEST_DELAY * 0.5)

    # Phase 2: Scrape each quiz page
    quiz_urls = sorted(all_quiz_urls)[:max_quizzes]
    total_quizzes = len(quiz_urls)

    for i, quiz_url in enumerate(quiz_urls):
        if on_progress:
            on_progress(i + 1, total_quizzes, f"Đang lấy câu hỏi ({i + 1}/{total_quizzes}): {quiz_url}")

        html, error = _fetch_html(quiz_url)
        if error:
            errors.append({"url": quiz_url, "error": error})
            continue

        questions = parse_quiz_page(html, quiz_url)
        if questions:
            all_questions.extend(questions)
        else:
            errors.append({"url": quiz_url, "error": "Không tìm thấy câu hỏi"})

        time.sleep(REQUEST_DELAY)

    return {
        "questions": all_questions,
        "question_count": len(all_questions),
        "quiz_count": total_quizzes,
        "scanned_pages": pages_scanned,
        "error_count": len(errors),
        "errors": errors[:20],  # Limit error list
    }


def scrape_single_quiz(url: str) -> Dict:
    """Scrape a single quiz page."""
    html, error = _fetch_html(url)
    if error:
        return {"questions": [], "count": 0, "error": error}

    questions = parse_quiz_page(html, url)
    return {
        "questions": questions,
        "count": len(questions),
        "error": None,
    }


# Available categories for the UI
QUIZ_CATEGORIES = [
    {
        "name": "Toán lớp 10",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-toan/trac-nghiem-online-toan-10/",
        "subject": "math",
        "grade": "10",
    },
    {
        "name": "Toán lớp 11",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-toan/trac-nghiem-online-toan-11/",
        "subject": "math",
        "grade": "11",
    },
    {
        "name": "Toán lớp 12",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-toan/trac-nghiem-online-toan-12/",
        "subject": "math",
        "grade": "12",
    },
    {
        "name": "Toán thi TN THPT",
        "url": "https://thuvienhoclieu.com/series/trac-nghiem-online-de-thi-thu-thpt-quoc-gia-mon-toan/",
        "subject": "math",
        "grade": "12",
    },
    {
        "name": "Vật Lý",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-vat-ly/",
        "subject": "physics",
        "grade": "",
    },
    {
        "name": "Hóa Học",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-hoa/",
        "subject": "chemistry",
        "grade": "",
    },
    {
        "name": "Sinh Học",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-mon-sinh/",
        "subject": "biology",
        "grade": "",
    },
    {
        "name": "Tiếng Anh",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-tieng-anh/",
        "subject": "english",
        "grade": "",
    },
    {
        "name": "Tiếng Anh thi TN THPT",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-tieng-anh/trac-nghiem-online-tieng-anh-on-thi-tn-thpt/",
        "subject": "english",
        "grade": "12",
    },
    {
        "name": "Lịch Sử",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-lich-su/",
        "subject": "history",
        "grade": "",
    },
    {
        "name": "GDCD",
        "url": "https://thuvienhoclieu.com/trac-nghiem-online/trac-nghiem-online-gdcd/",
        "subject": "civic_education",
        "grade": "",
    },
]
