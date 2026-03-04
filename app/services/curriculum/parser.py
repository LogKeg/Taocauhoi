"""
Parser for curriculum PDF documents from Bộ GD&ĐT.
"""

import re
from typing import List, Dict, Optional


def parse_curriculum_text(text: str, subject: str, grade: int) -> List[Dict]:
    """
    Parse curriculum text content into structured data.

    Args:
        text: Raw text content from PDF
        subject: Subject key (toan, ly, hoa, etc.)
        grade: Grade level (10, 11, 12)

    Returns:
        List of curriculum items
    """
    items = []
    current_chapter = ""
    current_topic = ""

    # Split by lines
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect chapter (Chương, Phần)
        chapter_match = re.match(r'^(Chương|Phần)\s*(\d+|[IVX]+)[.:]\s*(.+)', line, re.IGNORECASE)
        if chapter_match:
            current_chapter = line
            continue

        # Detect topic (Chủ đề, Bài)
        topic_match = re.match(r'^(Chủ đề|Bài|Mục)\s*(\d+)?[.:]\s*(.+)', line, re.IGNORECASE)
        if topic_match:
            current_topic = topic_match.group(3).strip()
            items.append({
                "subject": subject,
                "grade": grade,
                "chapter": current_chapter,
                "topic": current_topic,
                "lesson": "",
                "knowledge": "",
                "skills": "",
            })
            continue

        # Detect learning objectives
        if "kiến thức" in line.lower() or "yêu cầu cần đạt" in line.lower():
            if items:
                items[-1]["knowledge"] = line
            continue

        if "kỹ năng" in line.lower() or "năng lực" in line.lower():
            if items:
                items[-1]["skills"] = line
            continue

    return items


def extract_curriculum_from_pdf(pdf_path: str, subject: str, grade: int) -> List[Dict]:
    """
    Extract curriculum data from a PDF file.

    Args:
        pdf_path: Path to PDF file
        subject: Subject key
        grade: Grade level

    Returns:
        List of curriculum items
    """
    try:
        import PyPDF2

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

        return parse_curriculum_text(text, subject, grade)

    except ImportError:
        print("PyPDF2 not installed. Please install: pip install PyPDF2")
        return []
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return []


def parse_education_standard_text(text: str) -> Dict:
    """
    Parse text following Vietnamese education standard format.

    Expected format:
    - Chương/Phần headers
    - Chủ đề/Nội dung sections
    - Yêu cầu cần đạt (learning objectives)
    - Số tiết (periods)

    Returns:
        Structured curriculum data
    """
    result = {
        "chapters": [],
        "total_periods": 0
    }

    current_chapter = None
    current_content = None

    # Patterns
    chapter_pattern = re.compile(r'^(CHƯƠNG|PHẦN)\s*(\d+|[IVX]+)[.:]?\s*(.+)', re.IGNORECASE | re.MULTILINE)
    content_pattern = re.compile(r'^(\d+\.|\d+\)|\-)\s*(.+)', re.MULTILINE)
    period_pattern = re.compile(r'(\d+)\s*(tiết|tiêt)', re.IGNORECASE)

    # Find chapters
    chapter_matches = list(chapter_pattern.finditer(text))

    for i, match in enumerate(chapter_matches):
        chapter_start = match.end()
        chapter_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)

        chapter_text = text[chapter_start:chapter_end]
        chapter_name = match.group(3).strip()

        chapter_data = {
            "name": f"{match.group(1)} {match.group(2)}: {chapter_name}",
            "contents": [],
            "periods": 0
        }

        # Find periods in chapter
        period_match = period_pattern.search(chapter_text)
        if period_match:
            chapter_data["periods"] = int(period_match.group(1))
            result["total_periods"] += chapter_data["periods"]

        # Find content items
        for content_match in content_pattern.finditer(chapter_text):
            content_text = content_match.group(2).strip()
            if len(content_text) > 5:  # Skip short items
                chapter_data["contents"].append(content_text)

        result["chapters"].append(chapter_data)

    return result
