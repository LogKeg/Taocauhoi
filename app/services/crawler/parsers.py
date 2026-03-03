"""
HTML parsers for different education websites.
"""
import json
import re
from typing import List, Dict, Optional

from bs4 import BeautifulSoup, Tag


def _clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def _extract_options(element: Tag) -> List[str]:
    """Extract MCQ options from an element."""
    options = []

    # Try common patterns for options
    # Pattern 1: List items
    for li in element.find_all("li"):
        text = _clean_text(li.get_text())
        if text:
            # Remove option label if present (A., B., etc.)
            text = re.sub(r"^[A-Da-d][.):\s]+", "", text)
            options.append(text)

    if options:
        return options[:4]  # Max 4 options

    # Pattern 2: Paragraphs with A), B), C), D)
    text = element.get_text()
    matches = re.findall(r"[A-Da-d][.)]\s*([^A-Da-d\n]+?)(?=[A-Da-d][.)]|$)", text, re.DOTALL)
    if matches:
        return [_clean_text(m) for m in matches[:4]]

    # Pattern 3: Separate divs/spans with option labels
    for label in ["A", "B", "C", "D"]:
        for tag in element.find_all(string=re.compile(f"^{label}[.):]")):
            parent = tag.parent
            if parent:
                text = _clean_text(parent.get_text())
                text = re.sub(r"^[A-Da-d][.):\s]+", "", text)
                if text:
                    options.append(text)

    return options[:4]


def _extract_answer(element: Tag) -> str:
    """Extract the correct answer from an element."""
    text = element.get_text()

    # Pattern: "Đáp án: A" or "Answer: B"
    match = re.search(r"(?:Đáp án|Answer|Đ/A)[:\s]*([A-Da-d])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Pattern: Bold/highlighted option
    for label in ["A", "B", "C", "D"]:
        bold = element.find("b", string=re.compile(f"^{label}"))
        strong = element.find("strong", string=re.compile(f"^{label}"))
        if bold or strong:
            return label

    return ""


def parse_vietjack(soup: BeautifulSoup, subject: str = "", difficulty: str = "medium") -> List[Dict]:
    """
    Parse questions from vietjack.com

    VietJack typically structures questions in:
    - <div class="question"> or similar containers
    - Question text followed by options A, B, C, D
    - Answer often in a separate section or highlighted
    """
    questions = []

    # Try different selectors for question containers
    containers = (
        soup.select(".box-question") or
        soup.select(".question-item") or
        soup.select(".cau-hoi") or
        soup.select("div[class*='question']")
    )

    for container in containers:
        content = ""
        options = []
        answer = ""

        # Find question text
        q_elem = (
            container.select_one(".question-content") or
            container.select_one(".noi-dung") or
            container.select_one("p:first-child") or
            container.select_one("div:first-child")
        )

        if q_elem:
            content = _clean_text(q_elem.get_text())
            # Remove "Câu X:" prefix
            content = re.sub(r"^Câu\s*\d+[.:]\s*", "", content)

        if not content:
            continue

        # Find options
        opt_elem = (
            container.select_one(".answer-list") or
            container.select_one(".dap-an") or
            container.select_one("ul") or
            container
        )
        if opt_elem:
            options = _extract_options(opt_elem)

        # Find answer
        ans_elem = (
            container.select_one(".correct-answer") or
            container.select_one(".dap-an-dung") or
            container.select_one(".answer")
        )
        if ans_elem:
            answer = _extract_answer(ans_elem)
        else:
            answer = _extract_answer(container)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": answer,
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "vietjack.com",
        })

    # Fallback: try to parse inline questions
    if not questions:
        questions = _parse_inline_questions(soup, subject, difficulty, "vietjack.com")

    return questions


def parse_hoc247(soup: BeautifulSoup, subject: str = "", difficulty: str = "medium") -> List[Dict]:
    """
    Parse questions from hoc247.net

    Hoc247 typically structures questions in:
    - <div class="content-question"> containers
    - Options in ordered lists or labeled paragraphs
    """
    questions = []

    containers = (
        soup.select(".content-question") or
        soup.select(".question-box") or
        soup.select(".quiz-item") or
        soup.select("div[class*='question']")
    )

    for container in containers:
        content = ""
        options = []
        answer = ""

        # Find question text
        q_elem = (
            container.select_one(".question-text") or
            container.select_one("p.question") or
            container.select_one(".noidung")
        )

        if q_elem:
            content = _clean_text(q_elem.get_text())
            content = re.sub(r"^Câu\s*\d+[.:]\s*", "", content)

        if not content:
            # Try getting first paragraph
            first_p = container.find("p")
            if first_p:
                content = _clean_text(first_p.get_text())
                content = re.sub(r"^Câu\s*\d+[.:]\s*", "", content)

        if not content:
            continue

        # Find options
        options = _extract_options(container)

        # Find answer
        answer = _extract_answer(container)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": answer,
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "hoc247.net",
        })

    if not questions:
        questions = _parse_inline_questions(soup, subject, difficulty, "hoc247.net")

    return questions


def parse_loigiaihay(soup: BeautifulSoup, subject: str = "", difficulty: str = "medium") -> List[Dict]:
    """
    Parse questions from loigiaihay.com

    Loigiaihay typically has:
    - Question in <div class="box-content">
    - Structured question-answer format
    """
    questions = []

    containers = (
        soup.select(".box-content") or
        soup.select(".question-content") or
        soup.select(".bai-tap")
    )

    for container in containers:
        content = ""
        options = []
        answer = ""

        # Get main text
        text_parts = []
        for p in container.find_all("p"):
            text = _clean_text(p.get_text())
            if text:
                text_parts.append(text)

        if text_parts:
            # First part is usually the question
            content = text_parts[0]
            content = re.sub(r"^(Câu|Bài)\s*\d+[.:]\s*", "", content)

        if not content:
            continue

        # Extract options from remaining text
        full_text = " ".join(text_parts[1:]) if len(text_parts) > 1 else ""
        if full_text:
            matches = re.findall(r"[A-Da-d][.)]\s*([^A-Da-d]+?)(?=[A-Da-d][.)]|$)", full_text)
            options = [_clean_text(m) for m in matches[:4]]

        # Find answer
        answer = _extract_answer(container)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": answer,
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "loigiaihay.com",
        })

    if not questions:
        questions = _parse_inline_questions(soup, subject, difficulty, "loigiaihay.com")

    return questions


def parse_generic(soup: BeautifulSoup, subject: str = "", difficulty: str = "medium") -> List[Dict]:
    """
    Generic parser for unknown websites.
    Tries multiple heuristics to find questions.
    """
    questions = []

    # Remove script, style, nav elements
    for tag in soup.select("script, style, nav, header, footer, aside"):
        tag.decompose()

    # Try to find question containers
    containers = (
        soup.select("[class*='question']") or
        soup.select("[class*='cau-hoi']") or
        soup.select("[class*='bai-tap']") or
        soup.select("article") or
        soup.select(".content")
    )

    for container in containers:
        # Try to extract question
        content = ""
        options = []

        # Get text content
        paragraphs = container.find_all("p")
        if paragraphs:
            content = _clean_text(paragraphs[0].get_text())
            content = re.sub(r"^(Câu|Question|Bài)\s*\d+[.:]\s*", "", content)

        if not content or len(content) < 10:
            continue

        # Try to extract options
        options = _extract_options(container)

        # Extract answer
        answer = _extract_answer(container)

        questions.append({
            "content": content,
            "options": json.dumps(options, ensure_ascii=False) if options else None,
            "answer": answer,
            "subject": subject,
            "difficulty": difficulty,
            "question_type": "mcq" if options else "essay",
            "source": "web",
        })

    # Fallback to inline parsing
    if not questions:
        questions = _parse_inline_questions(soup, subject, difficulty, "web")

    return questions


def _parse_inline_questions(
    soup: BeautifulSoup,
    subject: str,
    difficulty: str,
    source: str
) -> List[Dict]:
    """
    Fallback parser that looks for inline question patterns in text.
    Useful when questions aren't in structured containers.
    """
    questions = []

    # Get main content area
    main = soup.select_one("main, article, .content, .post-content, #content")
    if not main:
        main = soup.body if soup.body else soup

    if not main:
        return questions

    text = main.get_text()

    # Pattern: "Câu X: <question text>" followed by options
    pattern = r"Câu\s*(\d+)[.:]\s*(.+?)(?=Câu\s*\d+[.:]|$)"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

    for num, q_text in matches:
        q_text = _clean_text(q_text)
        if len(q_text) < 10:
            continue

        # Try to split question from options
        content = q_text
        options = []

        # Check for A) B) C) D) pattern
        opt_match = re.search(r"[A-D][.)]\s", q_text)
        if opt_match:
            content = q_text[:opt_match.start()].strip()
            opt_text = q_text[opt_match.start():]
            opt_matches = re.findall(r"[A-Da-d][.)]\s*([^A-Da-d]+?)(?=[A-Da-d][.)]|$)", opt_text)
            options = [_clean_text(m) for m in opt_matches[:4]]

        # Extract answer if present
        answer = ""
        ans_match = re.search(r"(?:Đáp án|Answer)[:\s]*([A-Da-d])", q_text, re.IGNORECASE)
        if ans_match:
            answer = ans_match.group(1).upper()
            # Remove answer text from content
            content = re.sub(r"(?:Đáp án|Answer)[:\s]*[A-Da-d].*", "", content).strip()

        if content:
            questions.append({
                "content": content,
                "options": json.dumps(options, ensure_ascii=False) if options else None,
                "answer": answer,
                "subject": subject,
                "difficulty": difficulty,
                "question_type": "mcq" if options else "essay",
                "source": source,
            })

    return questions
