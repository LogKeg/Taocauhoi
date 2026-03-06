import io
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from fastapi import FastAPI, Form, Request, UploadFile, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# Import from core module
from app.core import (
    TOPICS,
    TEMPLATES,
    TOPIC_AI_GUIDE,
    SUBJECTS,
    SUBJECT_TOPICS,
    QUESTION_TYPES,
    SYNONYMS,
    NUMBER_RE,
    LEADING_NUM_RE,
    MCQ_OPTION_RE,
    ANSWER_TEMPLATES,
    GenerateRequest,
    ParseSamplesRequest,
    QuestionCreate,
    QuestionUpdate,
    ExamCreate,
    ExamUpdate,
    BulkSaveRequest,
    AIAnalyzeRequest,
    AISuggestRequest,
    AIReviewRequest,
)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API routers (will be added after all imports)
# app.include_router(settings_router)  # Settings routes moved to router

# Import AI services
from app.services.ai import (
    call_ai as _call_ai,
    call_openai as _call_openai,
    call_gemini as _call_gemini,
    call_ollama as _call_ollama,
    extract_text_from_response as _extract_text_from_response,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_API_BASE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE,
    OLLAMA_MODEL,
    load_saved_settings as _load_saved_settings,
    save_settings_to_file as _save_settings_to_file,
    SETTINGS_FILE,
    BASE_DIR,
)

# Import text services
from app.services.text import (
    normalize_name as _normalize_name,
    normalize_question as _normalize_question,
    strip_leading_numbering as _strip_leading_numbering,
    apply_synonyms as _apply_synonyms,
    apply_context as _apply_context,
    replace_numbers as _replace_numbers,
)

# Import math services
from app.services.math import (
    omml_to_latex,
    omml_children_to_latex,
)

# Import DOCX parsers
from app.parsers.docx import (
    extract_paragraph_with_math,
    extract_cell_with_math,
    extract_docx_content,
    extract_docx_lines,
    parse_cell_based_questions,
    extract_docx_lines_with_options,
    parse_bilingual_questions,
)

# Import API routers
from app.api.routers import (
    settings_router,
    questions_router,
    exams_router,
    import_export_router,
    ai_features_router,
    generation_router,
    grading_router,
    parsing_router,
    crawler_router,
    storage_router,
    curriculum_router,
)


def _resolve_sample_dir() -> Optional[Path]:
    env_dir = os.getenv("SAMPLE_DIR")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
        if path.exists() and path.is_dir():
            return path

    target = _normalize_name("đề mẫu")
    for entry in BASE_DIR.iterdir():
        if entry.is_dir() and _normalize_name(entry.name) == target:
            return entry
    return None


SAMPLE_DIR = _resolve_sample_dir()


def _is_mcq_block(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return len(lines) > 1 and any(MCQ_OPTION_RE.match(ln) for ln in lines[1:])


def _force_variation(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    if _is_mcq_block(stripped):
        return _rewrite_mcq_block(stripped)
    # Heuristic: if mostly ASCII, use English prefix; otherwise Vietnamese.
    ascii_ratio = sum(1 for ch in stripped if ord(ch) < 128) / max(1, len(stripped))
    if ascii_ratio > 0.9:
        return f"Choose the correct option: {stripped}"
    return f"Hãy cho biết: {stripped}"


def _rewrite_english_question(question: str) -> str:
    q = question.strip()
    if not q:
        return q
    pattern = re.compile(
        r"^The\s+\.\.\.\s+you\s+try,\s+the\s+more\s+likely\s+you\s+are\s+to\s+be\s+successful\.?$",
        re.IGNORECASE,
    )
    if pattern.match(q):
        return "The more ... you attempt, the higher your chance of success."

    pattern2 = re.compile(
        r"^All the players did their best apart from Johnson\. Johnson was \.\.\. his best\.?$",
        re.IGNORECASE,
    )
    if pattern2.match(q):
        variants = [
            "Only Johnson failed to give maximum effort. Complete the sentence: Johnson did not ... his best.",
            "Everyone except Johnson gave their best; Johnson did not ... his best.",
            "Johnson was the only player who didn't perform to his best. Choose the correct completion.",
            "All the other players performed at their peak. Johnson, however, did not ... his best.",
            "Unlike the rest of the team, Johnson didn't ... his best. Select the best completion.",
            "The entire squad tried their hardest, but Johnson did not ... his best. Choose the correct option.",
        ]
        return random.choice(variants)

    candidates = [
        (r"\bapart from\b", "except for"),
        (r"\bdid their best\b", "performed to the best of their ability"),
        (r"\bwas \.\.\. his best\b", "did not ... his best"),
        (r"\bmore likely\b", "more probable"),
        (r"\byou are to be successful\b", "you will succeed"),
        (r"\bthe more\b", "the greater"),
        (r"\btry\b", "attempt"),
    ]
    rewritten = q
    for src, dst in candidates:
        rewritten = re.sub(src, dst, rewritten, flags=re.IGNORECASE)
    if _normalize_question(rewritten) != _normalize_question(q):
        return rewritten
    return q


def _rewrite_mcq_block(block: str) -> str:
    lines = block.splitlines()
    if len(lines) < 2:
        return block
    if not MCQ_OPTION_RE.match(lines[1]):
        return block
    question = lines[0]
    prefix = ""
    if ":" in question:
        head, tail = question.split(":", 1)
        if len(head.split()) <= 5:
            prefix = head.strip()
            question = tail.strip()
    ascii_ratio = sum(1 for ch in question if ord(ch) < 128) / max(1, len(question))
    if ascii_ratio > 0.9:
        rewritten = _rewrite_english_question(question)
    else:
        rewritten = question
    if _normalize_question(rewritten) == _normalize_question(question):
        rewritten = f"Complete the sentence: {question}"
    # Context swap for English MCQ to force stronger change.
    if ascii_ratio > 0.9 and re.search(r"players|team|squad", rewritten, re.IGNORECASE):
        rewritten = re.sub(r"players", "athletes", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"team|squad", "group", rewritten, flags=re.IGNORECASE)
    new_prefix = "Select the best completion"
    if re.match(r"^(Choose|Select|Complete)\\b", rewritten, re.IGNORECASE):
        qline = rewritten
    else:
        qline = f"{new_prefix}: {rewritten}"
    return "\n".join([qline] + lines[1:])


# Include API routers
app.include_router(settings_router)
app.include_router(questions_router)
app.include_router(exams_router)
app.include_router(import_export_router)
app.include_router(ai_features_router)
app.include_router(generation_router)
app.include_router(grading_router)
app.include_router(parsing_router)
app.include_router(crawler_router)
app.include_router(storage_router)
app.include_router(curriculum_router)


@app.get("/")
def index() -> HTMLResponse:
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/upload-sample")
def upload_sample(subject: str = Form(...), file: UploadFile = Form(...)) -> dict:
    if not subject.strip():
        return {"ok": False, "message": "Thiếu tên môn"}
    if SAMPLE_DIR is None:
        return {"ok": False, "message": "Không tìm thấy thư mục đề mẫu"}
    if not SAMPLE_DIR.exists():
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    subject_dir = (SAMPLE_DIR / subject.strip()).resolve()
    if SAMPLE_DIR not in subject_dir.parents:
        return {"ok": False, "message": "Đường dẫn không hợp lệ"}
    subject_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(file.filename or "").name
    if not filename:
        return {"ok": False, "message": "Tên file không hợp lệ"}
    if Path(filename).suffix.lower() not in {".txt", ".docx", ".md"}:
        return {"ok": False, "message": "Chỉ hỗ trợ .txt, .docx, .md"}
    target = subject_dir / filename
    with target.open("wb") as f:
        f.write(file.file.read())
    return {"ok": True, "message": "Đã tải lên", "filename": filename, "subject": subject.strip()}



# Question bank, exams, history, import/export APIs moved to:
# - app.api.routers.questions
# - app.api.routers.exams
# - app.api.routers.import_export


# ============================================================================
# MATH EXAM PARSING
# ============================================================================

def _parse_math_exam_questions(lines: List[str]) -> List[dict]:
    """
    Parse Math exam questions with "Question X." format.
    Handles bilingual (English + Vietnamese) content and MCQ options A/B/C/D.
    Supports both multi-line format and single-line format (all content in one cell).
    """
    questions = []

    # Pattern to detect "Question X." header (standalone)
    question_header = re.compile(r'^Question\s+(\d+)\s*[.\)]?\s*$', re.IGNORECASE)
    # Pattern to detect options: A. xxx  B. xxx  C. xxx  D. xxx (with tabs or multiple spaces)
    option_pattern = re.compile(r'([A-D])\s*[.\)]\s*([^\t]+?)(?=\s{2,}[B-D]\s*[.\)]|\t[B-D]\s*[.\)]|$)', re.IGNORECASE)

    def extract_options_from_text(text: str) -> tuple:
        """
        Extract options from text that may contain A. xxx B. xxx C. xxx D. xxx
        Returns (content_without_options, list_of_options)
        Handles both spaced format (A. opt1  B. opt2) and compact format (A. opt1B. opt2)
        """
        # Try to find where options start - look for A. or A) pattern
        # Options are typically at the end of the text, separated by tabs or multiple spaces
        option_start_pattern = re.compile(r'(?:\s+|^)A\s*[.\)]\s*\S', re.IGNORECASE)
        match = option_start_pattern.search(text)

        if not match:
            return text, []

        # Split text into content and options part
        content_part = text[:match.start()].strip()
        options_part = text[match.start():].strip()

        # Extract individual options using different strategies
        options = []

        # Check if compact format (no spaces between options): "A. 2B. 4C. 5D. 3"
        # \S[B-D] means any non-whitespace character immediately followed by B/C/D
        is_compact = bool(re.search(r'\S[B-D]\s*[.\)]', options_part))

        if is_compact:
            # Strategy 1: Compact format - split by B. C. D. markers
            parts = re.split(r'([B-D])\s*[.\)]', options_part, flags=re.IGNORECASE)
            # First part after A.
            a_match = re.match(r'A\s*[.\)]\s*(.+?)$', parts[0].strip(), re.IGNORECASE)
            if a_match:
                options.append(a_match.group(1).strip())
            # Rest of options (B, C, D values)
            i = 1
            while i < len(parts) - 1:
                value = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if value:
                    options.append(value)
                i += 2
        else:
            # Strategy 2: Normal spaced format
            opt_matches = list(option_pattern.finditer(options_part))

            if opt_matches and len(opt_matches) >= 2:
                for m in opt_matches:
                    opt_text = m.group(2).strip()
                    # Clean up trailing whitespace and tabs
                    opt_text = re.sub(r'\s+$', '', opt_text)
                    if opt_text:
                        options.append(opt_text)

        return content_part, options

    def process_content_line(line: str, content_lines: list, options: list):
        """Process a line that may contain both content and options."""
        # Check if line has embedded options (A. B. C. D. pattern)
        if re.search(r'A\s*[.\)]\s*.+B\s*[.\)]', line, re.IGNORECASE):
            content_part, extracted_opts = extract_options_from_text(line)
            if content_part:
                content_lines.append(content_part)
            if extracted_opts:
                options.extend(extracted_opts)
        else:
            content_lines.append(line)

    # First, check if we have single-line format (each line contains full question with options)
    single_line_format = False
    for line in lines[:5]:
        if re.match(r'^Question\s+\d+\s*[.\)]?\s*.+A\s*[.\)]', line, re.IGNORECASE):
            single_line_format = True
            break

    if single_line_format:
        # Parse single-line format: "Question X. content... A. opt1 B. opt2 C. opt3 D. opt4"
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match question header with content
            combined = re.match(r'^Question\s+(\d+)\s*[.\)]?\s*(.+)$', line, re.IGNORECASE)
            if combined:
                q_num = int(combined.group(1))
                remaining = combined.group(2).strip()

                # Extract content and options
                content_part, options = extract_options_from_text(remaining)

                questions.append({
                    "question": content_part.strip(),  # No "Question X." prefix
                    "options": options[:4],  # Max 4 options
                    "answer": "",
                    "number": q_num
                })
    else:
        # Parse multi-line format
        current_question_num = None
        current_content_lines = []
        current_options = []

        def save_current_question():
            nonlocal current_question_num, current_content_lines, current_options

            if current_question_num is not None and (current_content_lines or current_options):
                question_text = "\n".join(current_content_lines)

                questions.append({
                    "question": question_text.strip(),
                    "options": current_options[:4],
                    "answer": "",
                    "number": current_question_num
                })

            current_question_num = None
            current_content_lines = []
            current_options = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for standalone "Question X." header
            header_match = question_header.match(line)
            if header_match:
                save_current_question()
                current_question_num = int(header_match.group(1))
                continue

            # Check for question header combined with content
            combined_header = re.match(r'^Question\s+(\d+)\s*[.\)]?\s*(.+)$', line, re.IGNORECASE)
            if combined_header:
                save_current_question()
                current_question_num = int(combined_header.group(1))
                remaining = combined_header.group(2).strip()
                if remaining:
                    process_content_line(remaining, current_content_lines, current_options)
                continue

            # If we're in a question, process content
            if current_question_num is not None:
                # Check for options line - first try compact format (A. 2B. 4C. 5D. 3)
                is_compact_options = bool(re.match(r'^A\s*[.\)]\s*\S', line, re.IGNORECASE) and
                                          re.search(r'\S[B-D]\s*[.\)]', line))
                if is_compact_options:
                    # Parse compact format options
                    parts = re.split(r'([B-D])\s*[.\)]', line, flags=re.IGNORECASE)
                    a_match = re.match(r'A\s*[.\)]\s*(.+?)$', parts[0].strip(), re.IGNORECASE)
                    if a_match:
                        current_options.append(a_match.group(1).strip())
                    i = 1
                    while i < len(parts) - 1:
                        value = parts[i + 1].strip() if i + 1 < len(parts) else ""
                        if value:
                            current_options.append(value)
                        i += 2
                    continue

                # Check for spaced options line
                opt_matches = list(option_pattern.finditer(line))
                if opt_matches and len(opt_matches) >= 2:
                    for m in opt_matches:
                        opt_text = m.group(2).strip()
                        opt_text = re.sub(r'\s+$', '', opt_text)
                        if opt_text:
                            current_options.append(opt_text)
                    continue

                # Check for single option line (A. xxx)
                single_match = re.match(r'^([A-D])\s*[.\)]\s*(.+)$', line, re.IGNORECASE)
                if single_match:
                    current_options.append(single_match.group(2).strip())
                    continue

                # Otherwise process as content (may contain embedded options)
                process_content_line(line, current_content_lines, current_options)

        save_current_question()

    return questions


# ============================================================================
# ENGLISH EXAM PARSING
# ============================================================================

def _parse_english_exam_questions(doc: Document) -> List[dict]:
    """
    Parse English exam questions from Word document.
    Handles multiple formats specific to English Level exams:
    1. Nested 2x2 table options
    2. Paragraphs as options (reading comprehension, dialogue)
    3. Cloze passages with options in following table rows
    4. Orphan options (options without visible question text - may have image)
    5. Textbox content (dialogue context, additional question text)
    """
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    questions = []
    seen_texts = set()
    orphan_options = []  # Options without questions (text may be in image)

    # Track cloze passages and their options
    cloze_passage = None
    cloze_blanks = []
    cloze_options = []

    def extract_textbox_content(cell) -> str:
        """Extract text from textboxes in a cell (used for dialogue context)."""
        xml = cell._element.xml
        if 'w:txbxContent' not in xml:
            return ''

        matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
        texts = []
        seen = set()
        for m in matches:
            t_texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', m)
            content = ' '.join(t_texts).strip()
            # Deduplicate (textboxes often duplicated in Word)
            if content and content not in seen:
                seen.add(content)
                texts.append(content)
        return ' '.join(texts)

    def flush_cloze_questions():
        """Add cloze questions to the list when we have collected all options."""
        nonlocal cloze_passage, cloze_blanks, cloze_options
        if cloze_passage and cloze_options:
            for idx, blank_num in enumerate(cloze_blanks):
                if idx < len(cloze_options):
                    # Include full passage text for cloze questions
                    questions.append({
                        'question': f'Cloze ({blank_num}): {cloze_passage}',
                        'options': cloze_options[idx][:4]
                    })
            cloze_passage = None
            cloze_blanks = []
            cloze_options = []

    for ti, table in enumerate(doc.tables):
        for ri, row in enumerate(table.rows):
            for ci, cell in enumerate(row.cells):
                paras = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
                cell_text = cell.text.strip()
                textbox_content = extract_textbox_content(cell)

                nested_tables = cell._element.findall('.//w:tbl', ns)

                # Extract options from nested tables first
                options = []
                if nested_tables:
                    for nt in nested_tables:
                        rows_elem = nt.findall('.//w:tr', ns)
                        for nrow in rows_elem:
                            cells_elem = nrow.findall('.//w:tc', ns)
                            for nc in cells_elem:
                                t_elems = nc.findall('.//w:t', ns)
                                text = ''.join([t.text or '' for t in t_elems]).strip()
                                if text:
                                    options.append(text)

                # Skip completely empty cells
                if not cell_text and not options and not textbox_content:
                    continue

                # Better duplicate key: include textbox content
                para_opts_key = str(paras[1:5]) if len(paras) >= 5 else ''
                cell_key = (cell_text[:100] if cell_text else '') + str(options[:2]) + para_opts_key + textbox_content[:50]
                if cell_key in seen_texts:
                    continue
                seen_texts.add(cell_key)

                # Check for cloze passage (has numbered blanks like (31))
                numbered_blanks = re.findall(r'\((\d+)\)', cell_text)
                if len(numbered_blanks) >= 2 and not nested_tables:
                    # Flush any previous cloze questions first
                    flush_cloze_questions()

                    cloze_passage = cell_text
                    cloze_blanks = numbered_blanks
                    cloze_options = []
                    continue

                # Process nested table options
                if options:
                    # Normal question with nested table options (has question text)
                    if len(options) >= 4 and paras:
                        # First check if we have pending cloze - if so, this is NOT cloze options
                        # because it has question text
                        flush_cloze_questions()

                        q_text = ' '.join(paras)
                        # Include textbox content for dialogue/reading context
                        if textbox_content:
                            q_text = q_text + ' ' + textbox_content
                        # Remove option text that may have leaked into question text
                        for opt in options:
                            q_text = q_text.replace(opt, '').strip()

                        if q_text:
                            questions.append({
                                'question': q_text,
                                'options': options[:4]
                            })
                    # Standalone options table (for cloze or orphan)
                    elif len(options) >= 4 and not paras:
                        if cloze_passage:
                            cloze_options.append(options)
                            # Check if we have collected all cloze options
                            if len(cloze_options) >= len(cloze_blanks):
                                flush_cloze_questions()
                        else:
                            orphan_options.append(options)
                    continue

                # Paragraphs as options (reading comprehension, dialogue, antonyms)
                if len(paras) >= 5:
                    # Flush any pending cloze questions first
                    flush_cloze_questions()

                    q_text = paras[0]
                    opts = paras[1:5]
                    # Include textbox content for context (dialogue, etc.)
                    if textbox_content:
                        q_text = q_text + ' ' + textbox_content
                    if all(len(o) >= 2 for o in opts):
                        questions.append({
                            'question': q_text,
                            'options': opts
                        })
                    continue

    # Don't forget last cloze passage if not yet flushed
    flush_cloze_questions()

    # Handle orphan options - options without visible question text
    # (question text may be in an image)
    for opts in orphan_options:
        questions.append({
            'question': 'Question with options only (text may be in image)',
            'options': opts[:4]
        })

    return questions


def _parse_envie_questions(doc: Document) -> List[dict]:
    """
    Parse EN-VIE bilingual English exam questions.
    These files have different formats:
    1. Fill-blank questions followed by tab-separated options: "text\tB) text\tC) text"
    2. Questions ending with ? followed by paragraph options (no A/B/C markers)
    3. Matching questions with A) B) C) D) E) options
    4. Questions in paragraphs with options in separate tables (1x4 or 2x2 grid)
    """
    from docx.oxml.ns import qn as docx_qn

    questions = []

    # Extract document elements in order (paragraphs and tables interleaved)
    # This is important for formats where questions are in paragraphs and options in tables
    def get_document_elements():
        """Get paragraphs and tables in document order."""
        elements = []
        body = doc._element.body

        # Track numbering counters per numId and ilvl
        # Format: {(numId, ilvl): current_count}
        numbering_counters = {}

        for child in body:
            tag = child.tag.split('}')[-1]

            if tag == 'p':  # Paragraph
                # Build text with line breaks preserved (w:br → \n)
                text_parts = []
                for run_elem in child.findall('.//' + docx_qn('w:r')):
                    for sub in run_elem:
                        sub_tag = sub.tag.split('}')[-1]
                        if sub_tag == 't':
                            text_parts.append(sub.text or '')
                        elif sub_tag == 'br':
                            text_parts.append('\n')
                text = ''.join(text_parts).strip()
                if not text:
                    # Fallback to w:t only
                    t_elements = child.findall('.//' + docx_qn('w:t'))
                    text = ''.join([t.text or '' for t in t_elements]).strip()
                if text:
                    # Check for highlighted text in paragraph (for inline options)
                    # Find which option (by position) has yellow highlight
                    highlighted_opt_idx = 0
                    if re.search(r'[A-E]\)', text):  # Has inline options
                        # Parse runs to find highlighted option
                        runs = child.findall('.//' + docx_qn('w:r'))
                        current_opt_letter = None
                        for run in runs:
                            # Get run text
                            run_texts = run.findall('.//' + docx_qn('w:t'))
                            run_text = ''.join([t.text or '' for t in run_texts])

                            # Check for option marker
                            opt_match = re.search(r'([A-E])\)', run_text)
                            if opt_match:
                                current_opt_letter = opt_match.group(1)

                            # Check for highlight in this run
                            shd_elements = run.findall('.//' + docx_qn('w:shd'))
                            hl_elements = run.findall('.//' + docx_qn('w:highlight'))
                            has_yellow = False
                            for shd in shd_elements:
                                fill = shd.get(docx_qn('w:fill'))
                                if fill and fill.upper() == 'FFFF00':
                                    has_yellow = True
                            for hl in hl_elements:
                                val = hl.get(docx_qn('w:val'))
                                if val and val.lower() == 'yellow':
                                    has_yellow = True

                            # If highlighted and we know which option, record it
                            if has_yellow and current_opt_letter:
                                highlighted_opt_idx = ord(current_opt_letter) - ord('A') + 1

                    # Check for Word numbering (List Paragraph style)
                    # This handles files where question numbers are in Word's numbering system
                    num_level = None  # 0 = question level, 1 = option level
                    num_value = None  # The actual number (1, 2, 3, ...)

                    pPr = child.find(docx_qn('w:pPr'))
                    if pPr is not None:
                        numPr = pPr.find(docx_qn('w:numPr'))
                        if numPr is not None:
                            numId_elem = numPr.find(docx_qn('w:numId'))
                            ilvl_elem = numPr.find(docx_qn('w:ilvl'))

                            if numId_elem is not None and ilvl_elem is not None:
                                numId = numId_elem.get(docx_qn('w:val'))
                                ilvl = int(ilvl_elem.get(docx_qn('w:val')) or '0')
                                num_level = ilvl

                                # Track and increment counter for this numbering
                                key = (numId, ilvl)
                                if key not in numbering_counters:
                                    numbering_counters[key] = 0
                                numbering_counters[key] += 1
                                num_value = numbering_counters[key]

                                # Reset sub-level counters when parent level increments
                                # e.g., when question (ilvl=0) increments, reset option counter (ilvl=1)
                                if ilvl == 0:
                                    sub_key = (numId, 1)
                                    numbering_counters[sub_key] = 0

                    # Split paragraphs containing \n with option patterns
                    if '\n' in text and re.search(r'\n[A-E]\.?\s*\S', text):
                        for sub_line in text.split('\n'):
                            sub_line = sub_line.strip()
                            if sub_line:
                                elements.append({
                                    'type': 'paragraph',
                                    'text': sub_line,
                                    'highlighted_option': highlighted_opt_idx,
                                    'num_level': num_level,
                                    'num_value': num_value
                                })
                    else:
                        # Remove any remaining \n from text
                        text = text.replace('\n', ' ').strip()
                        elements.append({
                            'type': 'paragraph',
                            'text': text,
                            'highlighted_option': highlighted_opt_idx,
                            'num_level': num_level,  # None = no numbering, 0 = question, 1 = option
                            'num_value': num_value   # The number (1, 2, 3, ...)
                        })

            elif tag == 'tbl':  # Table
                # Extract options from table cells
                rows = child.findall('.//' + docx_qn('w:tr'))
                options = []
                highlighted_idx = 0  # Track which option is highlighted (for single-question tables)
                highlighted_map = {}  # Track highlights per Cloze question number
                opt_counter = 0
                current_cloze_num = None
                opt_idx_in_cloze = 0

                def split_cell_options(cell_text: str) -> List[str]:
                    """Split cell text that may contain multiple tab-separated options.
                    E.g., 'Selfish.\tB) Emancipatory.' -> ['Selfish.', 'Emancipatory.']
                    """
                    # Check for tab-separated options with markers
                    if '\t' in cell_text and re.search(r'\t[A-E]\s*[.)]', cell_text):
                        parts = []
                        # Split by tab and process each part
                        raw_parts = cell_text.split('\t')
                        for part in raw_parts:
                            part = part.strip()
                            if not part:
                                continue
                            # Remove option marker (A), B., etc.)
                            cleaned = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', part)
                            if cleaned:
                                parts.append(cleaned)
                        return parts if parts else [cell_text]
                    return [cell_text]

                # Collect rows, then check if table contains embedded questions
                # (e.g., options for Q38 in row 0, then Q39 + options in rows 1-2)
                table_rows_data = []  # List of lists: each row -> list of (cell_text, is_highlighted)
                for tr in rows:
                    row_cells = []
                    for tc in tr.findall('.//' + docx_qn('w:tc')):
                        t_elements = tc.findall('.//' + docx_qn('w:t'))
                        cell_text = ''.join([t.text or '' for t in t_elements]).strip()
                        cell_is_highlighted = False
                        if cell_text:
                            shd_elements = tc.findall('.//' + docx_qn('w:shd'))
                            for shd in shd_elements:
                                fill = shd.get(docx_qn('w:fill'))
                                if fill and fill.upper() == 'FFFF00':
                                    cell_is_highlighted = True
                                    break
                        row_cells.append((cell_text, cell_is_highlighted))
                    table_rows_data.append(row_cells)

                # Detect embedded questions: a cell contains "Question text" + "Option A text"
                # concatenated (e.g., "Who discovered gravity?Albert Einstein.")
                # Heuristic: first cell of a row has text ending with '?' or '.' followed by more text
                # and other cells in that row + next rows look like B), C), D), E) options
                embedded_questions = []  # List of (question_text, options, row_start_idx)
                embedded_row_indices = set()

                for ri, row_cells in enumerate(table_rows_data):
                    if ri in embedded_row_indices:
                        continue
                    if not row_cells or not row_cells[0][0]:
                        continue
                    first_cell = row_cells[0][0]
                    # Check if first cell contains question+optionA pattern
                    # e.g., "Who discovered gravity?Albert Einstein."
                    q_opt_match = re.match(r'^(.+\?)\s*([A-Z].+)$', first_cell, re.DOTALL)
                    if not q_opt_match:
                        continue
                    q_text = q_opt_match.group(1).strip()
                    opt_a_text = q_opt_match.group(2).strip()
                    # Verify other cells have B), C), etc.
                    other_cells_text = [c[0] for c in row_cells[1:] if c[0]]
                    has_bc_markers = any(re.match(r'^[B-E]\)', ct) for ct in other_cells_text)
                    if not has_bc_markers and len(other_cells_text) < 1:
                        continue
                    # This row has an embedded question - collect its options
                    emb_opts = [opt_a_text]
                    emb_highlighted = 0
                    if row_cells[0][1]:  # first cell highlighted
                        emb_highlighted = 1
                    opt_num = 1
                    for ci in range(1, len(row_cells)):
                        ct, ch = row_cells[ci]
                        if ct:
                            cell_options = split_cell_options(ct)
                            for co in cell_options:
                                clean_co = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', co).strip()
                                if clean_co:
                                    opt_num += 1
                                    emb_opts.append(clean_co)
                                    if ch:
                                        emb_highlighted = opt_num
                    embedded_row_indices.add(ri)
                    # Check next rows for continuation (D), E))
                    for ri2 in range(ri + 1, len(table_rows_data)):
                        next_row_cells = table_rows_data[ri2]
                        next_texts = [c[0] for c in next_row_cells if c[0]]
                        if next_texts and all(re.match(r'^[A-E]\)', t) or not t for t in next_texts):
                            for ct, ch in next_row_cells:
                                if ct:
                                    cell_options = split_cell_options(ct)
                                    for co in cell_options:
                                        clean_co = re.sub(r'^[A-Ea-e]\s*[.)]\s*', '', co).strip()
                                        if clean_co:
                                            opt_num += 1
                                            emb_opts.append(clean_co)
                                            if ch:
                                                emb_highlighted = opt_num
                            embedded_row_indices.add(ri2)
                        else:
                            break
                    if len(emb_opts) >= 2:
                        embedded_questions.append((q_text, emb_opts, emb_highlighted, ri))

                # Process non-embedded rows as regular options
                # Track embedded questions found in cells (e.g., "A) Table tennis.34. Pick out the odd one.")
                row_groups = []  # List of (options_list, highlighted_idx, highlighted_map, embedded_q_text)
                current_group_options = []
                current_group_highlighted = 0
                current_group_hl_map = {}
                current_group_opt_counter = 0

                for ri, row_cells in enumerate(table_rows_data):
                    if ri in embedded_row_indices:
                        continue
                    embedded_q_in_row = None
                    for cell_text, cell_is_highlighted in row_cells:
                        if cell_text:
                            # Check for embedded question number in cell
                            # Pattern: "A) Table tennis.34. Pick out the odd one."
                            # Split into option part and embedded question part
                            emb_q_match = re.search(r'(?<=\.)\s*(\d+\.\s+.+)$', cell_text)
                            if emb_q_match and re.match(r'^[A-E]\)', cell_text):
                                # Found embedded question - split the cell
                                option_part = cell_text[:emb_q_match.start()].strip()
                                embedded_q_in_row = emb_q_match.group(1).strip()
                                cell_text = option_part

                            cell_options = split_cell_options(cell_text)
                            for cell_opt in cell_options:
                                current_group_opt_counter += 1
                                current_group_options.append(cell_opt)

                                cloze_match = re.match(r'^(\d+)\.\s*[A-E]\)', cell_opt)
                                if cloze_match:
                                    current_cloze_num = int(cloze_match.group(1))
                                    opt_idx_in_cloze = 1
                                elif re.match(r'^[B-E]\)', cell_opt) and current_cloze_num:
                                    opt_idx_in_cloze += 1

                                if cell_is_highlighted:
                                    if current_cloze_num:
                                        current_group_hl_map[current_cloze_num] = opt_idx_in_cloze
                                    else:
                                        current_group_highlighted = current_group_opt_counter

                    # If this row had an embedded question, save current group and start new one
                    if embedded_q_in_row:
                        row_groups.append((
                            current_group_options[:],
                            current_group_highlighted,
                            current_group_hl_map.copy(),
                            embedded_q_in_row
                        ))
                        current_group_options = []
                        current_group_highlighted = 0
                        current_group_hl_map = {}
                        current_group_opt_counter = 0

                # Save last group
                if current_group_options:
                    row_groups.append((current_group_options, current_group_highlighted, current_group_hl_map, None))

                # Count total cells across all rows (including empty ones)
                total_cell_count = sum(len(row) for row in table_rows_data)

                # Emit table elements with embedded question paragraphs between them
                if not row_groups:
                    # No row groups — emit table with whatever options we have
                    # Even if options is empty, emit if there are cells (image-based options)
                    if options or total_cell_count >= 3:
                        elements.append({
                            'type': 'table',
                            'options': options,
                            'highlighted': highlighted_idx,
                            'cloze_highlights': highlighted_map,
                            'cell_count': total_cell_count
                        })
                else:
                    for grp_opts, grp_hl, grp_hl_map, grp_emb_q in row_groups:
                        if grp_opts:
                            elements.append({
                                'type': 'table',
                                'options': grp_opts,
                                'highlighted': grp_hl,
                                'cloze_highlights': grp_hl_map,
                                'cell_count': total_cell_count
                            })
                        if grp_emb_q:
                            # Remove leading question number for clean question text
                            q_text_clean = re.sub(r'^\d+\.\s*', '', grp_emb_q)
                            elements.append({
                                'type': 'paragraph',
                                'text': q_text_clean,
                                'highlighted_option': 0,
                                'num_level': None,
                                'num_value': None
                            })

                # Append embedded questions as separate paragraph+table pairs
                for q_text, emb_opts, emb_hl, _ in embedded_questions:
                    elements.append({
                        'type': 'paragraph',
                        'text': q_text,
                        'highlighted_option': 0,
                        'num_level': None,
                        'num_value': None
                    })
                    elements.append({
                        'type': 'table',
                        'options': emb_opts,
                        'highlighted': emb_hl,
                        'cloze_highlights': {}
                    })

        return elements

    doc_elements = get_document_elements()

    # Merge bilingual EN+VN paragraph pairs for science exams (IKSC format)
    # Pattern: EN paragraph (numbered, no VN chars) followed by VN paragraph (has VN chars)
    # followed by option lines (A./B./C.) → merge EN+VN into one bilingual paragraph
    def _has_vietnamese_chars(text):
        return any(ord(c) > 127 for c in text[:200])

    # Check if this looks like a bilingual science document
    # Detect by bilingual options (e.g., "A. Refraction / Khúc xạ") or EN+VN question pairs
    bilingual_option_count = sum(
        1 for elem in doc_elements[:60]
        if elem['type'] == 'paragraph' and re.match(r'^[A-E][.)]\s*.+\s*/\s*.+', elem['text'])
    )
    bilingual_pair_count = 0
    for idx_e in range(len(doc_elements) - 1):
        e1 = doc_elements[idx_e]
        e2 = doc_elements[idx_e + 1]
        if (e1['type'] == 'paragraph' and e2['type'] == 'paragraph'
                and re.match(r'^\d+\.\s*', e1['text'])
                and not _has_vietnamese_chars(e1['text'])
                and _has_vietnamese_chars(e2['text'])
                and not re.match(r'^[A-E][.)]\s*', e2['text'])):
            bilingual_pair_count += 1
            if bilingual_pair_count >= 3:
                break
    is_bilingual_doc = bilingual_option_count >= 3 or bilingual_pair_count >= 3
    if is_bilingual_doc:
        merged_elements = []
        i = 0
        while i < len(doc_elements):
            elem = doc_elements[i]
            if (elem['type'] == 'paragraph'
                    and not _has_vietnamese_chars(elem['text'])
                    and re.match(r'^\d+\.\s*', elem['text'])
                    and i + 1 < len(doc_elements)
                    and doc_elements[i + 1]['type'] == 'paragraph'
                    and _has_vietnamese_chars(doc_elements[i + 1]['text'])):
                # Merge EN + VN into one bilingual paragraph
                en_text = re.sub(r'^\d+\.\s*', '', elem['text']).strip()
                vn_text = doc_elements[i + 1]['text']
                merged_elem = dict(doc_elements[i + 1])
                merged_elem['text'] = en_text + '\n' + vn_text
                merged_elements.append(merged_elem)
                i += 2
            else:
                merged_elements.append(elem)
                i += 1
        doc_elements = merged_elements

    # Build a map of option paragraph text -> highlighted option index
    # This helps when processing inline options in paragraphs
    para_highlight_map = {}
    # Build list of (doc_elem_idx, normalized_text) for fuzzy matching
    doc_elem_texts = []
    for idx, elem in enumerate(doc_elements):
        if elem['type'] == 'paragraph':
            text = elem['text']
            if elem.get('highlighted_option', 0) > 0:
                para_highlight_map[text] = elem['highlighted_option']
            # Store normalized text for matching
            normalized_text = text.replace('\n', '').replace('\r', '')
            doc_elem_texts.append((idx, normalized_text))

    # Also keep traditional paragraph extraction for backward compatibility
    # Split paragraphs containing \n with option patterns (e.g., "A.Moss\nB. Wheat\nC. Corn")
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if '\n' in text and re.search(r'\n[A-E]\.?\s*\S', text):
            for sub_line in text.split('\n'):
                sub_line = sub_line.strip()
                if sub_line:
                    paragraphs.append(sub_line)
        else:
            paragraphs.append(text)

    # Merge bilingual EN+VN paragraph pairs in paragraphs list too
    if is_bilingual_doc:
        merged_paras = []
        i = 0
        while i < len(paragraphs):
            text = paragraphs[i]
            if (not _has_vietnamese_chars(text)
                    and re.match(r'^\d+\.\s*', text)
                    and i + 1 < len(paragraphs)
                    and _has_vietnamese_chars(paragraphs[i + 1])
                    and not re.match(r'^[A-E][.)]\s*', paragraphs[i + 1])):
                en_text = re.sub(r'^\d+\.\s*', '', text).strip()
                vn_text = paragraphs[i + 1]
                merged_paras.append(en_text + '\n' + vn_text)
                i += 2
            else:
                merged_paras.append(text)
                i += 1
        paragraphs = merged_paras

    # Create a mapping from paragraphs index to doc_elements index
    # This ensures consistent ordering when sorting by _doc_pos
    # Use fuzzy matching: find doc_element that contains the paragraph text
    para_idx_to_doc_idx = {}
    for pi, p_text in enumerate(paragraphs):
        if not p_text:
            para_idx_to_doc_idx[pi] = 10000 + pi
            continue
        # Normalize paragraph text for matching
        normalized_p = p_text.replace('\n', '').replace('\r', '')
        # First try exact match
        found = False
        for doc_idx, doc_text in doc_elem_texts:
            if normalized_p == doc_text:
                para_idx_to_doc_idx[pi] = doc_idx
                found = True
                break
        if not found:
            # Try fuzzy match: check if doc_text contains or ends with normalized_p
            for doc_idx, doc_text in doc_elem_texts:
                if doc_text.endswith(normalized_p) or normalized_p in doc_text:
                    para_idx_to_doc_idx[pi] = doc_idx
                    found = True
                    break
        if not found:
            # Fallback: use paragraph index (paragraphs are roughly in order)
            para_idx_to_doc_idx[pi] = pi

    def extract_options_envie(line: str) -> List[str]:
        """Extract options from EN-VIE format line."""
        opts = []
        # Format: "optA\tB) optB\tC) optC\tD) optD" or "optA  B) optB C) optC"
        # Also handle format without tabs: "Beijing, China.	B) Athens, Greece.   C) Rome"
        if re.search(r'B\s*\)', line, re.IGNORECASE):
            # Find B) marker position
            b_match = re.search(r'B\s*\)', line, re.IGNORECASE)
            if b_match:
                # Option A is text before B)
                opt_a = line[:b_match.start()].strip().rstrip('\t ')
                if opt_a:
                    # Clean option A - remove A) or A. marker if present
                    opt_a_clean = re.sub(r'^[Aa]\s*[.)]\s*', '', opt_a)
                    # Also handle newline - take only first part
                    if '\n' in opt_a_clean:
                        opt_a_clean = opt_a_clean.split('\n')[0].strip()
                    opts.append(opt_a_clean if opt_a_clean else opt_a)
                else:
                    # Image option - add placeholder
                    opts.append('[Image A]')
                # Find all B-E markers
                markers = list(re.finditer(r'([B-E])\s*\)', line, re.IGNORECASE))
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(line)
                    opt_text = line[start:end].strip().rstrip('\t ')
                    if opt_text:
                        # Handle newline - take only first part
                        if '\n' in opt_text:
                            opt_text = opt_text.split('\n')[0].strip()
                        opts.append(opt_text)
                    else:
                        # Image option - add placeholder
                        opts.append(f'[Image {m.group(1).upper()}]')
        # Format: "A) optA B) optB C) optC" (with A marker)
        elif re.match(r'^A\s*\)', line, re.IGNORECASE):
            markers = list(re.finditer(r'([A-E])\s*\)', line, re.IGNORECASE))
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    # Handle newline - take only first part
                    if '\n' in opt_text:
                        opt_text = opt_text.split('\n')[0].strip()
                    opts.append(opt_text)
        # Format: "C) optC D) optD" or "D) optD E) optE" (continuation line)
        elif re.match(r'^[C-E]\s*\)', line, re.IGNORECASE):
            markers = list(re.finditer(r'([C-E])\s*\)', line, re.IGNORECASE))
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    # Handle newline - take only first part
                    if '\n' in opt_text:
                        opt_text = opt_text.split('\n')[0].strip()
                    opts.append(opt_text)
        # Format: "A. optA\tB. optB\tC. optC..." (dot-separated, tab/space delimited)
        elif re.match(r'^A\.\s*\S', line):
            markers = list(re.finditer(r'([A-E])\.\s*', line))
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    if '\n' in opt_text:
                        opt_text = opt_text.split('\n')[0].strip()
                    opts.append(opt_text)
        return opts

    def get_answer_from_option_line(opt_line: str) -> str:
        """Get answer from paragraph highlight map for an option line."""
        if opt_line in para_highlight_map:
            return str(para_highlight_map[opt_line])
        return ''

    def clean_option_text(opt: str) -> str:
        """Clean option text by removing markers and extracting just the content.

        Handles formats like:
        - 'A) Option text' -> 'Option text'
        - 'A. Option text' -> 'Option text'
        - 'a) Option text' -> 'Option text'
        - 'A) Option text\n34. Next question' -> 'Option text' (split at newline)
        """
        if not opt:
            return opt

        # First, split at newline and take only first part (in case question is embedded)
        if '\n' in opt:
            opt = opt.split('\n')[0].strip()

        # Remove option markers: A), A., a), a., etc.
        opt_match = re.match(r'^([A-Ea-e])[.)]\s*(.*)$', opt)
        if opt_match:
            return opt_match.group(2).strip()

        return opt.strip()

    def clean_options_list(options: List[str]) -> List[str]:
        """Clean a list of options, removing markers and embedded questions."""
        cleaned = []
        for opt in options:
            clean_opt = clean_option_text(opt)
            if clean_opt:
                cleaned.append(clean_opt)
        return cleaned

    def is_option_line_envie(line: str) -> bool:
        """Check if line contains options."""
        # Options with B) marker (tab or space separated)
        if re.search(r'B\s*\)', line, re.IGNORECASE):
            return True
        # Options starting with A)
        if re.match(r'^A\s*\)', line, re.IGNORECASE):
            return True
        # Image-only options: "B)\tC)\tD)"
        if re.match(r'^B\s*\)\s*\t', line, re.IGNORECASE):
            return True
        # Continuation line: "C) opt D) opt" or "D) opt E) opt"
        if re.match(r'^[C-E]\s*\)', line, re.IGNORECASE):
            return True
        # Options with A. B. format (e.g., "A. Stork" or "A.It occurs")
        if re.match(r'^A\.\s*\S', line):
            return True
        return False

    def has_fill_blank(text: str) -> bool:
        return '…' in text or '___' in text or '...' in text

    def is_sentence_with_blank(text: str) -> bool:
        """Check if text is a sentence containing a fill-blank (transform sentence)."""
        # Pattern: sentence with … marker, ending with period
        if has_fill_blank(text) and text.endswith('.'):
            return True
        return False

    def is_instruction_line(text: str) -> bool:
        """Check if line is an instruction (not a question)."""
        # If it has fill-blank marker, it's a question, not instruction
        if has_fill_blank(text):
            return False
        lower = text.lower()
        # Filter out artifact/watermark names (level names from competition exams)
        artifact_names = {'benjamin', 'ecolier', 'preecolier', 'student', 'junior', 'cadet', 'wallaby',
                          'pre-ecolier', 'pre ecolier', 'kadett', 'koala'}
        stripped = text.strip().lower()
        if stripped in artifact_names:
            return True
        # Also match repeated artifact like "StudentStudent" or "JuniorJunior"
        for name in artifact_names:
            if stripped == name * 2 or stripped == name + name:
                return True
        patterns = [
            'for each question',
            'read the following',
            'read and look',
            'read, look',
            'choose the correct',
            'choose the best',
            'answer the questions',
            'questions 1-',
            'questions (1-',
            'for questions',
            'for each sentence',
            'for each group',
            'choose the right answer to define',
            'read the text',
        ]
        return any(p in lower for p in patterns)

    # Track question numbers we've seen to detect standalone number patterns
    seen_q_numbers = set()

    # Track which paragraphs were already processed
    processed_paragraphs = set()

    # Track dialogue sections (e.g., Q11-15 with 3 response options each)
    in_dialogue_section = False
    dialogue_max_opts = 3

    # ============ WORD NUMBERING PASS: Handle Word List Paragraph format ============
    # For files like "Kangaroo Start R2 demo.docx" where:
    # - Questions have num_level=0 (numbered list items)
    # - Options have num_level=1 (sub-list items) with tab-separated B) C) D)
    # - IMPORTANT: Each question has exactly ONE option paragraph (1:1 ratio)
    # Check if document uses Word numbering format
    num_level_0_count = sum(1 for elem in doc_elements if elem['type'] == 'paragraph' and elem.get('num_level') == 0)
    num_level_1_count = sum(1 for elem in doc_elements if elem['type'] == 'paragraph' and elem.get('num_level') == 1)

    has_word_numbering = num_level_0_count > 0

    # Check if this is a clean 1:1 format (each question has exactly one option paragraph)
    # Story file has 30 questions but 54 options (options split across multiple paragraphs)
    # Start R2 file has 25 questions and 25 options (1:1)
    is_1to1_format = (num_level_0_count > 0 and num_level_0_count == num_level_1_count)

    # Check if options have duplicate pattern (old format files)
    # Old format: "book.book.sign.sign.map.map." - words repeated due to Word formatting
    # New R2 format: "park\tB) beach\tC) school" - clean options
    has_duplicate_options = False
    if has_word_numbering and is_1to1_format:
        for elem in doc_elements:
            if elem['type'] == 'paragraph' and elem.get('num_level') == 1:
                opt_text = elem['text']
                # Check for duplicate pattern like "word.word." or "wordword"
                # Split by B) and check the first option (A)
                if 'B)' in opt_text:
                    before_b = opt_text.split('B)')[0].strip()
                    # Remove StartStart prefix for checking
                    before_b = re.sub(r'^(Start)+', '', before_b)
                    # Check for duplicate words: "book.book." or "a cameraa camera"
                    if re.search(r'(\w{3,}\.)\1', before_b) or re.search(r'(\w{4,})\1', before_b):
                        has_duplicate_options = True
                        break

    # Only use Word numbering pass for:
    # 1. Clean 1:1 format (each question has exactly one option paragraph)
    # 2. No duplicate patterns in options
    if has_word_numbering and is_1to1_format and not has_duplicate_options:
        word_num_questions = []
        elem_idx = 0

        while elem_idx < len(doc_elements):
            elem = doc_elements[elem_idx]

            # Skip non-paragraph elements
            if elem['type'] != 'paragraph':
                elem_idx += 1
                continue

            # Check if this is a question (num_level=0) or instruction
            if elem.get('num_level') == 0:
                text = elem['text']
                q_num = elem.get('num_value')

                # Skip instruction lines
                if is_instruction_line(text):
                    elem_idx += 1
                    continue

                # Check if next element is options (num_level=1)
                if elem_idx + 1 < len(doc_elements):
                    next_elem = doc_elements[elem_idx + 1]

                    if next_elem['type'] == 'paragraph' and next_elem.get('num_level') == 1:
                        opt_text = next_elem['text']
                        # Clean watermark/header artifacts (e.g., "StartStart" prefix)
                        # These appear when document has repeated header text runs
                        opt_text = re.sub(r'^(Start)+', '', opt_text)
                        # Parse options from format: "optA\tB) optB\tC) optC\tD) optD"
                        opts = extract_options_envie(opt_text)

                        if len(opts) >= 2:
                            # Get answer from highlighted option if available
                            answer = ''
                            hl = next_elem.get('highlighted_option', 0)
                            if hl > 0:
                                answer = str(hl)

                            q_dict = {
                                'question': text,
                                'options': opts,
                                '_doc_pos': elem_idx,
                                '_q_num': q_num
                            }
                            if answer:
                                q_dict['answer'] = answer

                            word_num_questions.append(q_dict)
                            processed_paragraphs.add(text)
                            processed_paragraphs.add(opt_text)
                            elem_idx += 2
                            continue

            elem_idx += 1

        # If we found questions using Word numbering, use them and skip other parsing
        if word_num_questions:
            # Sort by question number
            word_num_questions.sort(key=lambda q: q.get('_q_num', 0))
            questions.extend(word_num_questions)
            # Return early - no need for other parsing passes
            return questions
    # ============ END WORD NUMBERING PASS ============

    # ============ PRE-PROCESS: Parse paragraph + table pairs ============
    # For formats where question is in paragraph and options are in following table
    # This handles Red Kangaroo style: "Question text ending with ? or :" followed by table with options
    para_table_questions = []

    elem_idx = 0
    while elem_idx < len(doc_elements):
        elem = doc_elements[elem_idx]

        if elem['type'] == 'paragraph':
            text = elem['text']
            # Check if this looks like a question
            # Patterns: ends with ? or :, has fill-blank (___), ends with quote
            has_fill_blank_marker = '___' in text or '______' in text
            # Also check if followed by table (synthetic paragraph from table splitting)
            next_is_table = (
                elem_idx + 1 < len(doc_elements) and
                doc_elements[elem_idx + 1]['type'] == 'table'
            )
            ends_with_ellipsis = text.endswith('...') or text.endswith('…')
            # Lower min length for texts ending with ... or followed by table
            min_len = 10 if (next_is_table or ends_with_ellipsis) else 15
            # Check for ? inside text followed by parenthetical, e.g., "...bao nhiêu? (g = 10 m/s²)"
            has_question_mark_inside = bool(re.search(r'\?\s*\([^)]+\)\s*$', text))
            is_question_text = (
                len(text) > min_len and
                not is_instruction_line(text) and
                (
                    text.endswith(':') or
                    text.endswith('?') or
                    text.endswith('"') or  # Dialogue/quote questions
                    has_fill_blank_marker or  # Fill-in-blank questions
                    ends_with_ellipsis or  # Questions ending with "..."
                    has_question_mark_inside or  # "...? (params)" format
                    (text.endswith('.') and next_is_table)  # Table-split questions like "Pick out the odd one."
                )
            )

            if is_question_text:
                # Look ahead for a table with options OR paragraph options (A., B., C., D.)
                if elem_idx + 1 < len(doc_elements):
                    next_elem = doc_elements[elem_idx + 1]

                    # Case 1: Options in table
                    table_cell_count = next_elem.get('cell_count', 0) if next_elem['type'] == 'table' else 0
                    if next_elem['type'] == 'table' and (len(next_elem['options']) >= 2 or table_cell_count >= 3):
                        options = next_elem['options']

                        # Check for Cloze passage tables (options start with number like "15.A)", "16.A)")
                        first_opt = options[0] if options else ''
                        is_cloze_table = re.match(r'^\d+\.\s*[A-E]\)', first_opt)
                        if is_cloze_table:
                            # Parse Cloze table - extract questions with passage
                            # Find passage before this table (look back for instruction line)
                            passage_lines = []
                            for back_idx in range(elem_idx - 1, max(0, elem_idx - 15), -1):
                                back_elem = doc_elements[back_idx]
                                if back_elem['type'] == 'paragraph':
                                    back_text = back_elem['text']
                                    # Stop at instruction line or previous question
                                    if 'Read the following' in back_text or 'choose the best' in back_text.lower():
                                        passage_lines.insert(0, back_text)
                                        break
                                    # Skip header/watermark lines
                                    if back_text.startswith('Red Kangaroo'):
                                        continue
                                    passage_lines.insert(0, back_text)

                            cloze_passage = '\n'.join(passage_lines) if passage_lines else ''

                            # Parse Cloze options - group by question number
                            cloze_opts_by_num = {}
                            cloze_highlights = {}  # Track highlighted option per question
                            current_q_num = None
                            current_opts = []
                            opt_idx_in_q = 0

                            for opt in options:
                                # Check if starts with question number (15.A), 16.A), etc.)
                                q_match = re.match(r'^(\d+)\.\s*[A-E]\)\s*(.*)$', opt)
                                if q_match:
                                    # Save previous question
                                    if current_q_num and current_opts:
                                        cloze_opts_by_num[current_q_num] = current_opts
                                    current_q_num = int(q_match.group(1))
                                    current_opts = [q_match.group(2)]
                                    opt_idx_in_q = 1
                                else:
                                    # Continuation option (B), C), D))
                                    opt_match = re.match(r'^[B-E]\)\s*(.*)$', opt)
                                    if opt_match:
                                        current_opts.append(opt_match.group(1))
                                        opt_idx_in_q += 1

                            # Save last question
                            if current_q_num and current_opts:
                                cloze_opts_by_num[current_q_num] = current_opts

                            # Get highlighted answers from cloze_highlights map
                            cloze_highlights = next_elem.get('cloze_highlights', {})

                            # Create Cloze questions
                            for q_num in sorted(cloze_opts_by_num.keys()):
                                opts = cloze_opts_by_num[q_num]
                                q_dict = {
                                    'question': f'Cloze question {q_num}\n\n{cloze_passage}' if cloze_passage else f'Cloze question {q_num}',
                                    'options': opts,
                                    '_doc_pos': elem_idx  # Track document position
                                }
                                # Add answer from highlighted option
                                if q_num in cloze_highlights:
                                    q_dict['answer'] = str(cloze_highlights[q_num])
                                para_table_questions.append(q_dict)

                            # Mark passage paragraphs as processed
                            for pl in passage_lines:
                                processed_paragraphs.add(pl)

                            elem_idx += 2  # Skip paragraph and table
                            continue

                        # Found question + table options pattern
                        # Clean option prefixes (A., B., C., D., E. or A), B), etc.)
                        cleaned_options = clean_options_list(options)

                        # If table has cells but few text options, fill with placeholders
                        if len(cleaned_options) < 3 and table_cell_count >= 3:
                            # Determine expected option count from cell layout
                            expected_opts = min(table_cell_count, 5)
                            while len(cleaned_options) < expected_opts:
                                letter = chr(ord('A') + len(cleaned_options))
                                cleaned_options.append(f'[{letter}]')

                        q_dict = {
                            'question': text,
                            'options': cleaned_options,
                            '_doc_pos': elem_idx  # Track document position
                        }
                        # Add correct answer from highlighted option
                        if next_elem.get('highlighted', 0) > 0:
                            q_dict['answer'] = str(next_elem['highlighted'])

                        para_table_questions.append(q_dict)
                        processed_paragraphs.add(text)
                        elem_idx += 2  # Skip both paragraph and table
                        continue

                    # Case 2: Options in paragraphs (A., B., C., D. as separate paragraphs)
                    elif next_elem['type'] == 'paragraph' and re.match(r'^A\.\s*\S', next_elem['text']):
                        # Collect paragraph options
                        para_options = []
                        opt_idx = elem_idx + 1
                        while opt_idx < len(doc_elements):
                            opt_elem = doc_elements[opt_idx]
                            if opt_elem['type'] != 'paragraph':
                                break
                            opt_match = re.match(r'^([A-E])\.\s*(.+)$', opt_elem['text'])
                            if opt_match:
                                para_options.append(opt_match.group(2))
                                processed_paragraphs.add(opt_elem['text'])
                                opt_idx += 1
                            else:
                                break

                        if len(para_options) >= 2:
                            q_dict = {
                                'question': text,
                                'options': para_options,
                                '_doc_pos': elem_idx  # Track document position
                            }
                            para_table_questions.append(q_dict)
                            processed_paragraphs.add(text)
                            elem_idx = opt_idx
                            continue

        elem_idx += 1

    # ============ SECOND PASS: Handle standalone Cloze tables ============
    # Cloze tables not preceded by a question paragraph (e.g., Red Kangaroo format)
    # These tables have options starting with "11.A)", "12.A)", etc.
    processed_tables = set()  # Track which tables were processed
    for pt_q in para_table_questions:
        # Mark tables used in para_table_questions
        pass

    elem_idx = 0
    while elem_idx < len(doc_elements):
        elem = doc_elements[elem_idx]

        if elem['type'] == 'table':
            options = elem.get('options', [])
            if options:
                first_opt = options[0]
                # Check for Cloze table format (starts with number like "11.A)", "12.A)")
                is_cloze_table = re.match(r'^(\d+)\.\s*[A-E]\)', first_opt)

                if is_cloze_table:
                    first_q_num = int(is_cloze_table.group(1))

                    # Check if this Cloze table was already processed
                    already_processed = any(
                        f'Cloze question {first_q_num}' in q.get('question', '')
                        for q in para_table_questions
                    )

                    if not already_processed:
                        # Find passage before this table (look back for instruction + content)
                        # Strategy: First find the instruction line, then collect passage after it
                        instruction_idx = -1
                        instruction_text = ''
                        for back_idx in range(elem_idx - 1, max(0, elem_idx - 25), -1):
                            back_elem = doc_elements[back_idx]
                            if back_elem['type'] == 'paragraph':
                                back_text = back_elem['text']
                                # Check for instruction line with question range (e.g., "(11-20)")
                                # Must check BEFORE skipping headers, as instruction may start with header text
                                if f'({first_q_num}-' in back_text or f'({first_q_num} -' in back_text:
                                    # Extract the instruction part (after header if present)
                                    if 'Read the' in back_text:
                                        read_pos = back_text.find('Read the')
                                        instruction_text = back_text[read_pos:]
                                    else:
                                        instruction_text = back_text
                                    instruction_idx = back_idx
                                    break
                                # Also check for generic instruction patterns
                                if ('Read the text' in back_text or 'Read the following' in back_text) and 'choose the best' in back_text.lower():
                                    instruction_idx = back_idx
                                    instruction_text = back_text
                                    break
                                # Skip header/watermark lines (only if no instruction pattern found)
                                if back_text.startswith('Red Kangaroo') or back_text.startswith('Grey Kangaroo'):
                                    continue

                        # Now collect passage lines AFTER instruction (between instruction and Cloze table)
                        passage_lines = []
                        if instruction_idx >= 0:
                            passage_lines.append(instruction_text)
                            for fwd_idx in range(instruction_idx + 1, elem_idx):
                                fwd_elem = doc_elements[fwd_idx]
                                if fwd_elem['type'] == 'paragraph':
                                    fwd_text = fwd_elem['text']
                                    # Skip header/watermark lines
                                    if fwd_text.startswith('Red Kangaroo') or fwd_text.startswith('Grey Kangaroo'):
                                        continue
                                    passage_lines.append(fwd_text)
                        else:
                            # Fallback: collect a few paragraphs before table if no instruction found
                            for back_idx in range(elem_idx - 1, max(0, elem_idx - 5), -1):
                                back_elem = doc_elements[back_idx]
                                if back_elem['type'] == 'paragraph':
                                    back_text = back_elem['text']
                                    if back_text.startswith('Red Kangaroo') or back_text.startswith('Grey Kangaroo'):
                                        continue
                                    passage_lines.insert(0, back_text)

                        cloze_passage = '\n'.join(passage_lines) if passage_lines else ''

                        # Parse Cloze options - group by question number
                        cloze_opts_by_num = {}
                        current_q_num = None
                        current_opts = []

                        for opt in options:
                            # Check if starts with question number (11.A), 12.A), etc.)
                            q_match = re.match(r'^(\d+)\.\s*[A-E]\)\s*(.*)$', opt)
                            if q_match:
                                # Save previous question
                                if current_q_num and current_opts:
                                    cloze_opts_by_num[current_q_num] = current_opts
                                current_q_num = int(q_match.group(1))
                                current_opts = [q_match.group(2)]
                            else:
                                # Continuation option (B), C), D))
                                opt_match = re.match(r'^[B-E]\)\s*(.*)$', opt)
                                if opt_match:
                                    current_opts.append(opt_match.group(1))

                        # Save last question
                        if current_q_num and current_opts:
                            cloze_opts_by_num[current_q_num] = current_opts

                        # Get highlighted answers from cloze_highlights map
                        cloze_highlights = elem.get('cloze_highlights', {})

                        # Create Cloze questions
                        for q_num in sorted(cloze_opts_by_num.keys()):
                            opts = cloze_opts_by_num[q_num]
                            q_dict = {
                                'question': f'Cloze question {q_num}\n\n{cloze_passage}' if cloze_passage else f'Cloze question {q_num}',
                                'options': opts,
                                '_doc_pos': elem_idx  # Track document position
                            }
                            # Add answer from highlighted option
                            if q_num in cloze_highlights:
                                q_dict['answer'] = str(cloze_highlights[q_num])
                            para_table_questions.append(q_dict)

                        # Mark passage paragraphs as processed
                        for pl in passage_lines:
                            processed_paragraphs.add(pl)

        elem_idx += 1
    # ============ END SECOND PASS ============

    # Add para_table_questions with document position for later sorting
    # Each question needs _doc_pos to maintain document order
    for q in para_table_questions:
        if '_doc_pos' not in q:
            # Find position based on question text in paragraphs
            q_text = q.get('question', '')[:50]
            for pi, para in enumerate(paragraphs):
                if q_text and q_text in para:
                    q['_doc_pos'] = pi
                    break
            else:
                q['_doc_pos'] = 9999  # Default to end if not found

    questions.extend(para_table_questions)
    # ============ END PRE-PROCESS ============

    i = 0
    while i < len(paragraphs):
        line = paragraphs[i].strip()
        if not line:
            i += 1
            continue

        # Skip paragraphs already processed in para+table phase
        if line in processed_paragraphs:
            i += 1
            continue

        # Skip pure option lines (they belong to previous question)
        # Cloze option lines are handled separately in the textbox section
        if is_option_line_envie(line) and not has_fill_blank(line):
            # Check if this is a numbered cloze question "22. A) opt B) opt..."
            # Skip these as they're processed in the textbox section later
            is_numbered_cloze = re.match(r'^(\d+)\.\s*A\s*\)', line)
            if is_numbered_cloze:
                i += 1
                continue
            # Check if this is a standalone cloze option (A) at start with tab-separated B) C) D))
            # Also skip these - handled in textbox section
            is_standalone_cloze = re.match(r'^A\s*\)', line) and '\t' in line and re.search(r'B\s*\).*C\s*\).*D\s*\)', line)
            if is_standalone_cloze:
                i += 1
                continue
            # Regular option line - skip
            i += 1
            continue

        # Case 0a: Wallaby Q1-5 special format (must be BEFORE instruction skip)
        # Structure: "For each question (1-5)..." instruction
        # Then: passage paragraphs for Q1 (no number marker, can be multiple paragraphs)
        # Then: "2.", "3.", "4.", "5." markers (all consecutive)
        # Then: 15 option paragraphs (3 for each Q1-5)
        # Question content is in textboxes (Q2-5) or paragraphs before "2." (Q1)
        if 'question (1-5)' in line.lower() or 'questions (1-5)' in line.lower():
            # Extract textbox content for Q2-5 passages
            from lxml import etree
            try:
                body = doc.element.body
                xml_str = etree.tostring(body, encoding='unicode')
                textbox_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml_str, re.DOTALL)

                # Get unique textbox contents (they're often duplicated)
                textbox_contents = []
                seen_tb = set()
                for tb in textbox_matches:
                    texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', tb)
                    content = ''.join(texts).strip()
                    # Skip short content, headers, and cloze options
                    if content and len(content) > 30 and content not in seen_tb:
                        if not re.match(r'^\d+\.?\s*A\s*\)', content):  # Not cloze options
                            if 'Wallaby' not in content and 'Answer a maximum' not in content:
                                seen_tb.add(content)
                                textbox_contents.append(content)
            except:
                textbox_contents = []

            # Find all consecutive standalone numbers and passages before them
            j = i + 1
            passages_before_nums = []
            standalone_nums = []

            # First pass: collect everything before we see standalone numbers
            while j < len(paragraphs):
                next_para = paragraphs[j].strip()
                if not next_para:
                    j += 1
                    continue
                # Check for standalone number
                num_match = re.match(r'^(\d+)\.$', next_para)
                if num_match:
                    standalone_nums.append(int(num_match.group(1)))
                    j += 1
                    # Continue collecting more numbers
                    while j < len(paragraphs):
                        np = paragraphs[j].strip()
                        if not np:
                            j += 1
                            continue
                        nm = re.match(r'^(\d+)\.$', np)
                        if nm:
                            standalone_nums.append(int(nm.group(1)))
                            j += 1
                        else:
                            break
                    break
                else:
                    passages_before_nums.append(next_para)
                    j += 1

            # If we found numbers 2-5 pattern
            if standalone_nums and 2 in standalone_nums:
                # Count questions (1 + number of standalone markers)
                num_questions = 1 + len(standalone_nums)
                # Collect 3 options per question
                opts_list = []
                while j < len(paragraphs) and len(opts_list) < num_questions * 3:
                    opt_para = paragraphs[j].strip()
                    if not opt_para:
                        j += 1
                        continue
                    if is_instruction_line(opt_para) or is_option_line_envie(opt_para):
                        break
                    opts_list.append(opt_para)
                    j += 1

                # Build question contents
                # Q1: from paragraphs before "2." marker
                q1_content = ' '.join(passages_before_nums) if passages_before_nums else 'Question 1'

                # Q2-5: from textboxes (first 4 unique textbox contents)
                q_contents = [q1_content]
                for tb_idx in range(min(4, len(textbox_contents))):
                    # Truncate long passages for display
                    tb_text = textbox_contents[tb_idx]
                    if len(tb_text) > 200:
                        tb_text = tb_text[:200] + '...'
                    q_contents.append(tb_text)

                # Pad with placeholders if not enough textboxes
                while len(q_contents) < num_questions:
                    q_contents.append(f'Question {len(q_contents) + 1}')

                # Create questions from collected options
                if len(opts_list) >= num_questions * 3:
                    for q_idx in range(num_questions):
                        q_opts = opts_list[q_idx*3:(q_idx+1)*3]
                        if len(q_opts) == 3:
                            questions.append({
                                'question': q_contents[q_idx],
                                'options': q_opts,
                                '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                            })
                            seen_q_numbers.add(str(q_idx + 1))
                    i = j
                    continue
            i += 1
            continue

        # Skip general instruction lines (after handling Q1-5 special case)
        # BUT: If instruction line is followed by option line, treat it as a question
        # EXCEPT: If instruction mentions question range like "(21-30)", always skip
        if is_instruction_line(line):
            # Detect dialogue section: "For each question (11-15), read and choose"
            # Dialogue questions have exactly 3 response options
            lower_line = line.lower()
            if re.search(r'\(\d+-\d+\)', line):
                # Reset dialogue section flag by default for any new section
                in_dialogue_section = False
                if 'read and choose' in lower_line or 'choose the best answer' in lower_line:
                    range_match = re.search(r'\((\d+)-(\d+)\)', line)
                    if range_match:
                        r_start, r_end = int(range_match.group(1)), int(range_match.group(2))
                        if r_end - r_start == 4:  # 5 questions like (11-15)
                            in_dialogue_section = True
                            dialogue_max_opts = 3
            # Always skip if it's a section instruction with question range
            if re.search(r'\(\d+-\d+\)', line):
                i += 1
                continue
            # Check if next non-empty line is an option line
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if np:
                    break
                j += 1
            if j < len(paragraphs) and is_option_line_envie(paragraphs[j].strip()):
                # Check if next line is a numbered cloze option (e.g., "22.\tA) neither")
                # If so, skip this instruction - it's for Cloze section
                next_line_text = paragraphs[j].strip()
                if re.match(r'^\d+\.\s*\t?A\s*\)', next_line_text):
                    i += 1
                    continue
                # This instruction acts as a question - process it below
                pass
            else:
                i += 1
                continue

        # Case 0b: Standalone question number "1." or "2." or just "1", "2" etc.
        # Ecolier format: number → question text → merged options (A+B on one line, C+D on next)
        # EN-VIE format: number → 3 plain option paragraphs
        q_num_match = re.match(r'^(\d+)\.?$', line)
        if q_num_match:
            q_num = q_num_match.group(1)
            if q_num not in seen_q_numbers:
                seen_q_numbers.add(q_num)
                # Look ahead: next non-empty line could be question text (Ecolier) or option (EN-VIE)
                j = i + 1
                # Find next non-empty paragraph
                while j < len(paragraphs) and not paragraphs[j].strip():
                    j += 1

                if j < len(paragraphs):
                    next_text = paragraphs[j].strip()

                    # Ecolier pattern: question text on next line, then merged options
                    # Check if the line AFTER the question text has merged options
                    if next_text and not is_option_line_envie(next_text) and not re.match(r'^\d+\.?$', next_text):
                        # This looks like a question text, check what follows
                        k = j + 1
                        while k < len(paragraphs) and not paragraphs[k].strip():
                            k += 1
                        if k < len(paragraphs) and is_option_line_envie(paragraphs[k].strip()):
                            # Ecolier format: number → question → merged options
                            question_text = next_text
                            opts = extract_options_envie(paragraphs[k].strip())
                            m = k + 1
                            # Check for continuation lines (C) D) on next line)
                            while m < len(paragraphs) and len(opts) < 5:
                                cont_line = paragraphs[m].strip()
                                if not cont_line:
                                    m += 1
                                    continue
                                if re.match(r'^[C-E]\s*\)', cont_line, re.IGNORECASE):
                                    more_opts = extract_options_envie(cont_line)
                                    if more_opts:
                                        opts.extend(more_opts)
                                        m += 1
                                        continue
                                break
                            if opts:
                                questions.append({
                                    'question': question_text,
                                    'options': opts[:5],
                                    'number': int(q_num),
                                    '_doc_pos': para_idx_to_doc_idx.get(i, i)
                                })
                                i = m
                                continue

                # Fallback: EN-VIE format - collect next 3 non-empty paragraphs as options
                opts = []
                j = i + 1
                while j < len(paragraphs) and len(opts) < 3:
                    opt_line = paragraphs[j].strip()
                    if not opt_line:
                        j += 1
                        continue
                    # Stop if we hit another question number or option line
                    if re.match(r'^\d+\.?$', opt_line) or is_option_line_envie(opt_line):
                        break
                    if is_instruction_line(opt_line):
                        j += 1
                        continue
                    opts.append(opt_line)
                    j += 1

                if len(opts) == 3:
                    questions.append({
                        'question': f'Question {q_num} (reading comprehension)',
                        'options': opts,
                        'number': int(q_num),
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)
                    })
                    i = j
                    continue
            i += 1
            continue

        # Find next non-empty line
        next_line = None
        next_idx = i + 1
        while next_idx < len(paragraphs):
            if paragraphs[next_idx].strip():
                next_line = paragraphs[next_idx].strip()
                break
            next_idx += 1

        # Case 1: Fill-blank question followed by option line(s)
        if has_fill_blank(line):
            # Case 1a: Options in tab-separated format
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for multi-line options (C) D) on next line)
                while j < len(paragraphs) and len(opts) < 4:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check if line starts with C) or D) (continuation)
                    if re.match(r'^[C-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if opts:
                    # Word Groups format: Look back to collect additional fill-blank paragraphs
                    question_parts = [line]
                    for back_idx in range(i - 1, max(0, i - 10), -1):
                        back_text = paragraphs[back_idx].strip()
                        if not back_text:
                            continue
                        # Stop at instruction line
                        if is_instruction_line(back_text):
                            break
                        # Stop at options line (previous question's options)
                        if is_option_line_envie(back_text) and not has_fill_blank(back_text):
                            break
                        # Collect consecutive fill-blank paragraphs
                        if has_fill_blank(back_text):
                            question_parts.insert(0, back_text)
                        else:
                            break

                    combined_question = ' '.join(question_parts)
                    questions.append({
                        'question': combined_question,
                        'options': opts,
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    })
                    i = j
                    continue

            # Case 1b: Options as separate paragraphs (no A/B/C markers)
            # Common in Story format, Red Kangaroo has 5 options (A-E)
            if next_line and not is_option_line_envie(next_line):
                opts = []
                j = i + 1
                while j < len(paragraphs) and len(opts) < 5:  # Max 5 options (A-E)
                    opt_line = paragraphs[j].strip()
                    if not opt_line:
                        j += 1
                        continue
                    # Stop if we hit another question or option line
                    if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                        break
                    # Skip instruction lines
                    if is_instruction_line(opt_line):
                        j += 1
                        continue
                    opts.append(opt_line)
                    j += 1

                if len(opts) >= 3:  # Need at least 3 options
                    questions.append({
                        'question': line,
                        'options': opts[:5],  # Max 5 options (A-E)
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    })
                    i = j
                    continue

        # Case 2: Question ending with ? followed by option line (text or image)
        if line.endswith('?') and len(line) > 10:
            # First check if next line is an option line (image options)
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for continuation lines (C) D) on next line)
                while j < len(paragraphs) and len(opts) < 5:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check if line starts with C), D), or E) (continuation)
                    if re.match(r'^[C-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if opts:
                    q_dict = {
                        'question': line,
                        'options': opts[:5],  # Max 5 options (A-E)
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    }
                    # Check for highlighted answer in option line
                    ans = get_answer_from_option_line(next_line)
                    if ans:
                        q_dict['answer'] = ans
                    questions.append(q_dict)
                    i = j
                    continue

            # Otherwise collect next 3-5 paragraphs as options (no markers)
            # Common in Q1-5 sections with reading comprehension
            # Benjamin has 5 options (A-E), other levels may have 3-4
            # In dialogue sections, limit to 3 options to avoid eating next question
            max_opts = dialogue_max_opts if in_dialogue_section else 5
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < max_opts:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                # Stop if we hit another question or option line
                if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                    break
                # Stop if line ends with comma (likely start of next question, not an option)
                if opt_line.endswith(','):
                    break
                # Skip instruction lines
                if is_instruction_line(opt_line):
                    j += 1
                    continue
                # Stop if line is too long (likely a passage, not an option)
                if len(opt_line) > 200:
                    break
                # This looks like an option
                opts.append(opt_line)
                j += 1

            if len(opts) >= 3:  # Reading comprehension has 3-5 options
                questions.append({
                    'question': line,
                    'options': opts[:max_opts],
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i = j
                continue

        # Case 2b: Dialogue question NOT ending with ? (Grey Kangaroo Q12, Q14, Q15)
        # Format: "I think we should come up with a plan B." followed by 3 response options
        # Distinguishing features:
        # - Short sentence ending with period (dialogue prompt)
        # - Following 3 lines are responses (not option markers like A) B))
        # - Must be in dialogue section (after "For each question (11-15)")
        if line.endswith('.') and len(line) > 15 and len(line) < 80 and not has_fill_blank(line):
            # First check for special format: next line has "optA\tB) optB" and following line has "C) optC"
            # This is Grey Kangaroo Q14 format
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                break

            if j < len(paragraphs):
                first_opt_line = paragraphs[j].strip()
                # Check for "optA\tB) optB" pattern - use extract_options_envie for proper parsing
                if re.search(r'\tB\s*\)', first_opt_line):
                    first_opts = extract_options_envie(first_opt_line)
                    if len(first_opts) >= 2:
                        # Look for continuation options (C), D), E)) on next lines
                        j += 1
                        while j < len(paragraphs) and len(first_opts) < 5:
                            np = paragraphs[j].strip()
                            if not np:
                                j += 1
                                continue
                            # Check if line starts with C), D), or E)
                            if re.match(r'^[C-E]\s*\)', np, re.IGNORECASE):
                                cont_opts = extract_options_envie(np)
                                if cont_opts:
                                    first_opts.extend(cont_opts)
                                    j += 1
                                    continue
                            break
                        if len(first_opts) >= 3:
                            all_opts = first_opts
                        questions.append({
                            'question': line,
                            'options': all_opts[:5],  # Max 5 options (A-E)
                            '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                        })
                        i = j
                        continue

            # Otherwise check for dialogue responses (no markers)
            dialogue_opts = []
            j = i + 1
            while j < len(paragraphs) and len(dialogue_opts) < 4:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                # Stop if we hit a question, option line with markers, or instruction
                if opt_line.endswith('?') or has_fill_blank(opt_line) or is_instruction_line(opt_line):
                    break
                # Stop if line has B) C) markers (this is an option line for different question)
                if re.search(r'B\s*\)', opt_line):
                    break
                # Dialogue responses are short sentences (< 60 chars) ending with period
                if len(opt_line) < 70 and opt_line.endswith('.'):
                    dialogue_opts.append(opt_line)
                    j += 1
                else:
                    break

            # Dialogue questions have exactly 3-4 response options
            if len(dialogue_opts) == 3 or len(dialogue_opts) == 4:
                questions.append({
                    'question': line,
                    'options': dialogue_opts[:3],
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i = j
                continue

        # Case 3: Matching question - description followed by option line with 3+ choices
        # e.g., "Maria loves comfort food..." followed by "The Safari. B) Origo. C)..."
        # Also handles instruction line as question: "Fill in the missing letter" + "A\tB) I\tC) U"
        # Skip numbered cloze format "22. A) ..."
        if len(line) > 30 and not line.endswith('?') and not has_fill_blank(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            if next_line and is_option_line_envie(next_line):
                # Check if options are on separate paragraphs (A., B., C., D., E.)
                if re.match(r'^[A]\.\s*\S', next_line) and not re.search(r'B\s*[.)]\s*\S', next_line):
                    # Collect A., B., C., D., E. from separate paragraphs
                    para_opts = []
                    j = next_idx
                    while j < len(paragraphs) and len(para_opts) < 5:
                        opt_line = paragraphs[j].strip()
                        if not opt_line:
                            j += 1
                            continue
                        opt_m = re.match(r'^([A-E])\.\s*(.+)$', opt_line)
                        if opt_m:
                            para_opts.append(opt_m.group(2).strip())
                            j += 1
                        else:
                            break
                    if len(para_opts) >= 3:
                        questions.append({
                            'question': line,
                            'options': para_opts[:5],
                            '_doc_pos': para_idx_to_doc_idx.get(i, i)
                        })
                        i = j
                        continue
                # Extract options from option line (inline format)
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for continuation lines (D), E) on next line)
                while j < len(paragraphs) and len(opts) < 5:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check if line starts with D) or E) (continuation)
                    if re.match(r'^[D-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if len(opts) >= 3:  # At least 3 options (A, B, C)
                    q_dict = {
                        'question': line,
                        'options': opts[:5],
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    }
                    ans = get_answer_from_option_line(next_line)
                    if ans:
                        q_dict['answer'] = ans
                    questions.append(q_dict)
                    i = j
                    continue

        # Case 4: Question ending with , followed by paragraph options (incomplete sentence)
        # e.g., "According to the notice," followed by 3 options
        # Or: Question split across paragraphs ending with , then . followed by option line
        if line.endswith(',') and len(line) > 15:
            # First check if next paragraph is a continuation (ends with .) followed by option line
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if np:
                    break
                j += 1

            if j < len(paragraphs):
                continuation = paragraphs[j].strip()
                # Check if this is a continuation ending with . (not a question)
                if continuation.endswith('.') and not continuation.endswith('?') and len(continuation) < 80:
                    # Check if next line after continuation is an option line
                    k = j + 1
                    while k < len(paragraphs):
                        np = paragraphs[k].strip()
                        if np:
                            break
                        k += 1
                    if k < len(paragraphs) and is_option_line_envie(paragraphs[k].strip()):
                        # Merge question parts and get options
                        combined_question = line + ' ' + continuation
                        opt_line = paragraphs[k].strip()
                        opts = extract_options_envie(opt_line)
                        if opts:
                            questions.append({
                                'question': combined_question,
                                'options': opts[:5],
                                '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                            })
                            i = k + 1
                            continue

            # Otherwise, collect paragraph options
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < 3:
                opt_line = paragraphs[j].strip()
                if not opt_line:
                    j += 1
                    continue
                if opt_line.endswith('?') or is_option_line_envie(opt_line) or has_fill_blank(opt_line):
                    break
                if is_instruction_line(opt_line):
                    j += 1
                    continue
                opts.append(opt_line)
                j += 1

            if len(opts) == 3:
                questions.append({
                    'question': line,
                    'options': opts,
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i = j
                continue

        # Case 5a: Reading comprehension with C) marker on option 2 line
        # Format: "Abbreviations in the likes of BRB and LOL" + opt1 + "opt2\tC) opt3"
        # Skip numbered cloze format "22. A) ..."
        # Skip if line ends with '.' AND has commas (looks like option list, not question)
        looks_like_option = line.endswith('.') and ',' in line and len(line) < 60
        if len(line) > 20 and len(line) < 150 and not line.endswith('?') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line) and not looks_like_option:
            j = i + 1
            opt1 = None
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                opt1 = np
                break

            if opt1 and j + 1 < len(paragraphs):
                opt2_line = paragraphs[j + 1].strip() if j + 1 < len(paragraphs) else ""
                # Skip if opt2_line has B) marker - this is a full option line, not Case 5a format
                if not re.search(r'B\s*\)', opt2_line):
                    c_marker = re.search(r'\tC\s*\)|(\s{2,})C\s*\)', opt2_line)
                    if c_marker:
                        opt2 = opt2_line[:c_marker.start()].strip()
                        opt3_match = re.search(r'C\s*\)\s*(.+)$', opt2_line, re.IGNORECASE)
                        opt3 = opt3_match.group(1).strip() if opt3_match else ""

                        if opt2 and opt3:
                            questions.append({
                                'question': line,
                                'options': [opt1, opt2, opt3],
                                '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                            })
                            i = j + 2
                            continue

        # Case 5b-new: Grey Kangaroo Q1-5 reading comprehension (MUST come before 5b)
        # Format: stem line (no ? ending), then "opt1\tB) opt2" then "C) opt3"
        # Example: "Traditionally, child narrators are regarded as"
        #          "voices that employ a gloomy tone.\tB) innocent and genuine."
        #          "C) devoid of the depth needed to explore serious themes."
        if len(line) > 15 and len(line) < 80 and not line.endswith('?') and not line.endswith('.') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            # Check if next line has format "opt1\tB) opt2" (option A + B in one line)
            j = i + 1
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                break

            if j < len(paragraphs):
                first_opt_line = paragraphs[j].strip()
                # Check for "opt1\tB) opt2" or "opt1  B) opt2" pattern
                b_match = re.search(r'(.+?)(?:\t|\s{2,})B\s*\)\s*(.+)$', first_opt_line)
                if b_match:
                    opt_a = b_match.group(1).strip()
                    opt_b = b_match.group(2).strip()

                    # Look for C) option on next line
                    j += 1
                    while j < len(paragraphs):
                        np = paragraphs[j].strip()
                        if not np:
                            j += 1
                            continue
                        break

                    opt_c = None
                    if j < len(paragraphs):
                        c_line = paragraphs[j].strip()
                        c_match = re.match(r'^C\s*\)\s*(.+)$', c_line, re.IGNORECASE)
                        if c_match:
                            opt_c = c_match.group(1).strip()
                            j += 1

                    if opt_a and opt_b and opt_c:
                        questions.append({
                            'question': line,
                            'options': [opt_a, opt_b, opt_c],
                            '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                        })
                        i = j
                        continue

        # Case 5b: Reading comprehension with 3 plain paragraph options (Q12-15 format)
        # Stem is short (< 70 chars), followed by 3 options without markers
        # Each option starts with lowercase (continuation of stem sentence)
        # Skip numbered cloze format "22. A) ..."
        if len(line) > 8 and len(line) < 70 and not line.endswith('?') and not line.endswith('.') and not has_fill_blank(line) and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            # Collect next 3 non-empty paragraphs
            opts = []
            j = i + 1
            while j < len(paragraphs) and len(opts) < 3:
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Stop if we hit instruction or question-like line
                if is_instruction_line(np) or np.endswith('?') or is_option_line_envie(np):
                    break
                # Options should start with lowercase (continuation) or be short sentences
                if len(np) < 100:
                    opts.append(np)
                    j += 1
                else:
                    break

            if len(opts) == 3:
                questions.append({
                    'question': line,
                    'options': opts,
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i = j
                continue

        # Case 5c: Matching question - "Match each prefix..." or "Match the questions..." followed by table rows then options
        if 'match' in line.lower() and ('column' in line.lower() or 'prefix' in line.lower() or 'questions' in line.lower() or 'left' in line.lower()):
            # Skip table rows until we find options A) 1.../2.../
            j = i + 1
            table_rows = []
            while j < len(paragraphs):
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Check for option line "A) 1e/2b/..."
                if re.match(r'^A\s*\)\s*\d', np):
                    break
                # Collect table rows
                if '\t' in np or re.match(r'^\w+\s+[a-e]\.', np):
                    table_rows.append(np)
                j += 1

            # Extract options from option lines
            opts = []
            while j < len(paragraphs) and len(opts) < 5:
                np = paragraphs[j].strip()
                if not np:
                    j += 1
                    continue
                # Extract A-E options
                if re.match(r'^[A-E]\s*\)', np):
                    markers = list(re.finditer(r'([A-E])\s*\)', np, re.IGNORECASE))
                    for idx, m in enumerate(markers):
                        start = m.end()
                        if idx + 1 < len(markers):
                            end = markers[idx + 1].start()
                        else:
                            end = len(np)
                        opt_text = np[start:end].strip()
                        if opt_text:
                            opts.append(opt_text)
                    j += 1
                    continue
                break

            if len(opts) >= 3:
                questions.append({
                    'question': line,
                    'options': opts[:5],
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i = j
                continue

        # Case 5d: Question with options in image (no text options)
        # Format: "Pick the sentence with the correct punctuation." followed by another question
        # These questions have their options shown as images
        if line.endswith('.') and len(line) > 20 and len(line) < 80:
            # Check if this looks like a question (not a regular statement)
            lower_line = line.lower()
            is_question_like = any(w in lower_line for w in ['pick', 'choose', 'select', 'which', 'what'])
            # Check if next line is NOT an option line (options are in images)
            if is_question_like and next_line and not is_option_line_envie(next_line):
                # Check that next line is another question (not table data)
                next_is_question = len(next_line) > 20 and not re.match(r'^\w+\s+[a-e]\.', next_line)
                if next_is_question:
                    questions.append({
                        'question': line,
                        'options': ['[Option A in image]', '[Option B in image]', '[Option C in image]', '[Option D in image]', '[Option E in image]'],
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    })
                    i += 1
                    continue

        # Case 6: Question + next line options with 5 choices (Wallaby Q31-50)
        # Format: "The Forbidden City is located in …" on one line
        # Then: "Beijing, China.\tB) Athens, Greece.   C) Rome..." on next line
        # Also handles multi-line options (D) E) on subsequent line)
        # Skip numbered cloze format "22. A) ..." which is handled by Case 7
        if len(line) > 15 and not is_instruction_line(line) and not re.match(r'^(\d+)\.\s*A\s*\)', line):
            if next_line and is_option_line_envie(next_line):
                opts = extract_options_envie(next_line)
                j = next_idx + 1
                # Check for continuation line (D) E) options)
                while j < len(paragraphs) and len(opts) < 5:
                    cont_line = paragraphs[j].strip()
                    if not cont_line:
                        j += 1
                        continue
                    # Check for D) E) continuation
                    if re.match(r'^[D-E]\s*\)', cont_line, re.IGNORECASE):
                        more_opts = extract_options_envie(cont_line)
                        if more_opts:
                            opts.extend(more_opts)
                            j += 1
                            continue
                    break
                if len(opts) >= 3:
                    # Check if this is a Word Groups format (multiple fill-blank sentences)
                    # Look back to find additional fill-blank paragraphs that belong to this question
                    # Strategy: Collect consecutive fill-blank paragraphs, stop at options line
                    question_parts = [line]
                    if has_fill_blank(line):
                        found_fill_blank_sequence = False
                        for back_idx in range(i - 1, max(0, i - 10), -1):
                            back_text = paragraphs[back_idx].strip()
                            if not back_text:
                                continue
                            # Stop if we hit instruction line
                            if is_instruction_line(back_text):
                                break
                            # Options line marks the boundary between questions
                            # Stop when we hit an options line (previous question's options)
                            if is_option_line_envie(back_text) and not has_fill_blank(back_text):
                                break
                            if has_fill_blank(back_text):
                                question_parts.insert(0, back_text)
                                found_fill_blank_sequence = True
                            else:
                                # Non-fill-blank, non-option line - stop
                                break

                    combined_question = ' '.join(question_parts)
                    questions.append({
                        'question': combined_question,
                        'options': opts[:5],
                        '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                    })
                    i = j
                    continue

        # Case 7: Cloze question in paragraph format
        # Format: "22.\tA) neither\tB) both\tC) either\tD) whether"
        cloze_match = re.match(r'^(\d+)\.\s*A\s*\)', line)
        if cloze_match:
            q_num = cloze_match.group(1)
            opts = extract_options_envie(line[cloze_match.start():])
            # If A) not extracted properly, re-extract starting from A)
            a_pos = re.search(r'A\s*\)', line)
            if a_pos:
                opts_text = line[a_pos.start():]
                markers = list(re.finditer(r'([A-E])\s*\)', opts_text, re.IGNORECASE))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(opts_text)
                    opt_text = opts_text[start:end].strip().rstrip('\t ')
                    if opt_text:
                        opts.append(opt_text)
            if len(opts) >= 3:
                questions.append({
                    'question': f'Cloze question {q_num}',
                    'options': opts[:4],
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i += 1
                continue

        # Case 8: Cloze options without number (Q25-30 format)
        # Format: "A) yet\tB) whereas\tC) also\tD) while" - only options, no question number
        # These appear after numbered cloze questions in the same section
        # Each line is OPTIONS for ONE cloze question (not question + options from next line)
        if re.match(r'^A\s*\)', line) and '\t' in line:
            opts_text = line
            markers = list(re.finditer(r'([A-E])\s*\)', opts_text, re.IGNORECASE))
            opts = []
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(opts_text)
                opt_text = opts_text[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
            if len(opts) >= 3:
                # This line IS the options for a cloze question (no question text, just options)
                # Question text is in the passage as a numbered blank like (26)
                questions.append({
                    'question': 'Cloze question (unnumbered)',
                    'options': opts[:4],
                    '_doc_pos': para_idx_to_doc_idx.get(i, i)  # Track document position
                })
                i += 1
                continue

        i += 1

    # Parse cloze questions from tables
    # Format: Table rows with "16.\tA) option" in first cell, B)/C)/D) in other cells
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if not cells or not cells[0]:
                continue

            # Check if first cell has numbered question format "16.\tA) option"
            first_cell = cells[0]
            match = re.match(r'^(\d+)\.\s*A\s*\)\s*(.+)$', first_cell.replace('\t', ' '))
            if match:
                q_num = match.group(1)
                opt_a = match.group(2).strip()
                opts = [opt_a]

                # Extract B, C, D options from other cells
                for cell_text in cells[1:]:
                    opt_match = re.match(r'^([B-E])\s*\)\s*(.+)$', cell_text.strip())
                    if opt_match:
                        opts.append(opt_match.group(2).strip())

                if len(opts) >= 3:
                    questions.append({
                        'question': f'Cloze question {q_num}',
                        'options': opts[:4],
                        '_doc_pos': 9000  # Tables come later in document
                    })
                continue

            # Check for table with mixed question/options (Wallaby format)
            # Cell might have "Question?\nOption A." format
            for cell_text in cells:
                # Look for question ending with ? followed by option on new line
                if '?' in cell_text and '\n' in cell_text:
                    lines = cell_text.split('\n')
                    for li, line in enumerate(lines):
                        if line.strip().endswith('?'):
                            q_text = line.strip()
                            # Next line is option A
                            opts = []
                            if li + 1 < len(lines):
                                opt_a = lines[li + 1].strip().rstrip('.')
                                if opt_a:
                                    opts.append(opt_a)
                            # Get B, C options from other cells in same row
                            for other_cell in cells[1:]:
                                opt_match = re.match(r'^([B-E])\s*\)\s*(.+)$', other_cell.strip())
                                if opt_match:
                                    opts.append(opt_match.group(2).strip().rstrip('.'))

                            if q_text and len(opts) >= 3:
                                questions.append({
                                    'question': q_text,
                                    'options': opts[:5],
                                    '_doc_pos': 9000  # Tables come later in document
                                })
                            break

            # Note: Joey format (A) option cells with embedded question numbers)
            # is now handled by get_document_elements() table splitting

    # Parse cloze questions from textboxes and paragraphs
    # Textboxes may contain: "21. A) sparked B) spotted C) split D) squandered"
    from lxml import etree
    try:
        body = doc.element.body
        xml_str = etree.tostring(body, encoding='unicode')
        textbox_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml_str, re.DOTALL)

        seen_cloze_nums = set()
        cloze_opts_from_textbox = {}  # Store options by question number

        for match in textbox_matches:
            texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', match)
            content = ' '.join(texts).strip()

            # Look for cloze option format: "21. A) option B) option C) option D) option"
            cloze_match = re.match(r'^(\d+)\.\s*A\s*\)\s*', content)
            if cloze_match:
                q_num = int(cloze_match.group(1))
                if q_num in seen_cloze_nums:
                    continue
                seen_cloze_nums.add(q_num)

                # Extract options
                markers = list(re.finditer(r'([A-D])\s*\)\s*', content))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(content)
                    opt_text = content[start:end].strip()
                    if opt_text:
                        opts.append(opt_text)

                if len(opts) >= 3:
                    cloze_opts_from_textbox[q_num] = opts[:4]

        # Also check paragraphs for numbered cloze options (e.g., "22. A) neither B) both...")
        for para in doc.paragraphs:
            text = para.text.strip()
            opt_match = re.match(r'^(\d+)\.\s*A\s*\)', text)
            if opt_match:
                q_num = int(opt_match.group(1))
                if q_num not in cloze_opts_from_textbox:
                    # Extract options
                    markers = list(re.finditer(r'([A-D])\s*\)\s*', text))
                    opts = []
                    for idx, m in enumerate(markers):
                        start = m.end()
                        if idx + 1 < len(markers):
                            end = markers[idx + 1].start()
                        else:
                            end = len(text)
                        opt_text = text[start:end].strip()
                        if opt_text:
                            opts.append(opt_text)
                    if len(opts) >= 3:
                        cloze_opts_from_textbox[q_num] = opts[:4]

        # Collect unnumbered cloze option lines from paragraphs
        # These are lines like "A) yet B) whereas C) also D) while" in cloze section
        unnumbered_cloze_opts = []
        in_cloze_section = False
        for para in doc.paragraphs:
            text = para.text.strip()
            # Detect start of cloze section
            if 'space (21-30)' in text.lower() or '(21-30)' in text:
                in_cloze_section = True
                continue
            # Detect end of cloze section
            if in_cloze_section and ('questions 31-' in text.lower() or 'For questions 31' in text):
                in_cloze_section = False
                continue
            # Collect unnumbered option lines in cloze section
            if in_cloze_section and re.match(r'^A\s*\)', text):
                # This is an unnumbered cloze option line
                markers = list(re.finditer(r'([A-D])\s*\)\s*', text))
                opts = []
                for idx, m in enumerate(markers):
                    start = m.end()
                    if idx + 1 < len(markers):
                        end = markers[idx + 1].start()
                    else:
                        end = len(text)
                    opt_text = text[start:end].strip()
                    if opt_text:
                        opts.append(opt_text)
                if len(opts) >= 3:
                    unnumbered_cloze_opts.append(opts[:4])

        # Find cloze numbers in passages (from textboxes)
        cloze_in_passage = set()
        for match in textbox_matches:
            texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', match)
            content = ''.join(texts).strip()
            # Passage blanks: "(26)" or "(27)"
            blanks = re.findall(r'\((\d+)\)', content)
            for b in blanks:
                cloze_in_passage.add(int(b))

        # Determine missing cloze numbers (numbers in passage but no options)
        cloze_with_options = set(cloze_opts_from_textbox.keys())
        missing_cloze = sorted(cloze_in_passage - cloze_with_options)

        # Assign unnumbered options to missing cloze numbers
        if missing_cloze and unnumbered_cloze_opts:
            for i, cloze_num in enumerate(missing_cloze):
                if i < len(unnumbered_cloze_opts):
                    cloze_opts_from_textbox[cloze_num] = unnumbered_cloze_opts[i]

        # Also handle cloze 30 which may be the last unnumbered option
        if 30 not in cloze_opts_from_textbox and unnumbered_cloze_opts:
            # Use the last available unnumbered option for cloze 30
            remaining_unnumbered = len(unnumbered_cloze_opts) - len(missing_cloze)
            if remaining_unnumbered > 0:
                cloze_opts_from_textbox[30] = unnumbered_cloze_opts[-1]

        # Add all cloze questions
        for q_num in sorted(cloze_opts_from_textbox.keys()):
            opts = cloze_opts_from_textbox[q_num]
            questions.append({
                'question': f'Cloze question {q_num}',
                'options': opts,
                '_doc_pos': 8000 + q_num  # Textbox cloze, position by question number
            })

        # Add placeholder for any remaining missing cloze numbers (image-based)
        all_cloze_nums = set(cloze_opts_from_textbox.keys())
        still_missing = cloze_in_passage - all_cloze_nums
        if all_cloze_nums:
            max_cloze = max(all_cloze_nums)
            for cloze_num in sorted(still_missing):
                if 20 <= cloze_num <= max_cloze:
                    questions.append({
                        'question': f'Cloze question {cloze_num} (image-based)',
                        'options': ['[Option A in image]', '[Option B in image]', '[Option C in image]', '[Option D in image]'],
                        '_doc_pos': 8000 + cloze_num  # Position by question number
                    })

    except Exception:
        pass  # lxml may not be available

    # Remove incorrectly parsed "A) ..." lines as questions
    questions = [q for q in questions if not q.get('question', '').startswith('A)')]

    # Remove duplicate Cloze questions (keep ones with passage/answer)
    # Group by Cloze number and keep the best version
    cloze_by_num = {}
    non_cloze = []
    for q in questions:
        text = q.get('question', '')
        m = re.search(r'Cloze question (\d+)', text)
        if m:
            q_num = int(m.group(1))
            existing = cloze_by_num.get(q_num)
            # Prefer version with passage (longer text) or with answer
            if existing is None:
                cloze_by_num[q_num] = q
            else:
                # Keep the one with more content or answer
                existing_len = len(existing.get('question', ''))
                new_len = len(text)
                existing_has_ans = bool(existing.get('answer'))
                new_has_ans = bool(q.get('answer'))
                if (new_has_ans and not existing_has_ans) or (new_len > existing_len and not existing_has_ans):
                    cloze_by_num[q_num] = q
        else:
            non_cloze.append(q)

    # Rebuild questions list
    questions = non_cloze + list(cloze_by_num.values())

    # Sort cloze questions by their question number and insert at correct position
    def get_cloze_num(q):
        text = q.get('question', '')
        m = re.search(r'Cloze question (\d+)', text)
        if m:
            return int(m.group(1))
        return 999

    # Separate cloze questions from others
    cloze_questions = [q for q in questions if 'Cloze question' in q.get('question', '')]
    other_questions = [q for q in questions if 'Cloze question' not in q.get('question', '')]

    # Sort other_questions by document position to maintain correct order
    other_questions.sort(key=lambda q: q.get('_doc_pos', 9999))

    # Sort cloze questions by their number
    cloze_questions.sort(key=get_cloze_num)

    # Insert cloze questions at correct position based on cloze number
    # E.g., "Cloze question 21" should be inserted at position 20 (0-indexed)
    if cloze_questions:
        # Get the first cloze question number to determine insertion position
        first_cloze_num = get_cloze_num(cloze_questions[0])
        # Insert at position = first_cloze_num - 1 (0-indexed)
        # E.g., Cloze Q11 -> insert at index 10, Cloze Q21 -> insert at index 20
        insert_idx = first_cloze_num - 1
        # Make sure we don't go beyond the available questions
        insert_idx = min(insert_idx, len(other_questions))
        questions = other_questions[:insert_idx] + cloze_questions + other_questions[insert_idx:]
    else:
        questions = other_questions

    # Deduplicate questions (same question text)
    seen_texts = set()
    unique_questions = []
    for q in questions:
        qt = q.get('question', '').strip()
        if qt and qt not in seen_texts:
            seen_texts.add(qt)
            unique_questions.append(q)
        elif not qt:
            unique_questions.append(q)
    questions = unique_questions

    # Clean up _doc_pos from final output
    for q in questions:
        q.pop('_doc_pos', None)

    return questions


def _dedup_bilingual_science(questions: List[dict]) -> List[dict]:
    """Post-process bilingual science exam questions (IKSC format).

    These documents have each question in both English and Vietnamese.
    This function removes English duplicates and cleans bilingual options.
    """
    def _has_vietnamese(text):
        return any(ord(c) > 127 for c in text)

    def _is_section_header(text):
        t = text.strip()
        return bool(re.match(r'^\d+\s*[–\-]\s*Point\s+Questions?$', t, re.IGNORECASE))

    # Step 1: Remove section headers and merge EN+VN pairs into bilingual questions
    cleaned = []
    i = 0
    while i < len(questions):
        q = questions[i]
        qtext = q.get('question', '').strip()

        # Skip section headers
        if _is_section_header(qtext):
            i += 1
            continue

        # If pure English question, check if next question is Vietnamese version
        if not _has_vietnamese(qtext):
            if i + 1 < len(questions):
                next_q = questions[i + 1]
                next_text = next_q.get('question', '').strip()
                if _has_vietnamese(next_text):
                    # Merge EN + VN into bilingual question
                    merged_q = dict(next_q)
                    merged_q['question'] = qtext + '\n' + next_text
                    # Use whichever has more options; merge bilingual opts if both have same count
                    en_opts = q.get('options', [])
                    vn_opts = next_q.get('options', [])
                    if len(en_opts) >= len(vn_opts) and len(en_opts) >= 3:
                        # EN has options — merge with VN if same count
                        if len(en_opts) == len(vn_opts):
                            merged_opts = []
                            for en_o, vn_o in zip(en_opts, vn_opts):
                                if _has_vietnamese(vn_o) and not _has_vietnamese(en_o):
                                    merged_opts.append(en_o + ' / ' + vn_o)
                                else:
                                    merged_opts.append(en_o)
                            merged_q['options'] = merged_opts
                        else:
                            merged_q['options'] = en_opts
                    # else: keep VN options from next_q
                    cleaned.append(merged_q)
                    i += 2
                    continue
            # Standalone English with few options — skip
            if len(q.get('options', [])) < 5:
                i += 1
                continue

        cleaned.append(q)
        i += 1

    # Step 2: Remove near-duplicate questions (same content from different sources)
    # When texts match after cleaning, merge: keep clean text + best options
    def _has_garbled_prefix(text):
        """Check if text starts with garbled diagram labels like K1K2K3."""
        m = re.match(r'^[A-Z0-9]{4,}', text)
        return bool(m)

    deduped = []
    skip_indices = set()
    for idx, q in enumerate(cleaned):
        if idx in skip_indices:
            continue
        qtext = q.get('question', '').strip()
        opts = q.get('options', [])
        opts_count = len(opts)
        # Clean garbled diagram labels like K1K2K3 for comparison
        def _clean_garbled(text):
            return re.sub(r'(?:[A-Z]\d){2,}', '', text)
        qtext_clean = _clean_garbled(qtext)
        merged = False
        for idx2 in range(idx + 1, min(idx + 5, len(cleaned))):
            if idx2 in skip_indices:
                continue
            q2 = cleaned[idx2]
            q2text = q2.get('question', '').strip()
            q2_opts = q2.get('options', [])
            q2text_clean = _clean_garbled(q2text)
            # Check if one text contains the other (near-duplicate)
            if len(qtext_clean) > 10 and len(q2text_clean) > 10:
                if (qtext_clean in q2text_clean or q2text_clean in qtext_clean
                        or qtext in q2text or q2text in qtext):
                    # Merge: prefer clean text, take more options
                    has_garbled1 = qtext_clean != qtext
                    has_garbled2 = q2text_clean != q2text
                    best_text = q2text if has_garbled1 and not has_garbled2 else qtext if not has_garbled1 else qtext_clean
                    best_opts = opts if opts_count >= len(q2_opts) else q2_opts
                    merged_q = dict(q)
                    merged_q['question'] = best_text
                    merged_q['options'] = best_opts
                    deduped.append(merged_q)
                    skip_indices.add(idx2)
                    merged = True
                    break
        if not merged:
            deduped.append(q)
    cleaned = deduped

    # Step 3: Keep bilingual options as-is (EN / VN format)
    # No stripping — options like "Refraction / Khúc xạ" stay bilingual

    return cleaned



# ============================================================================
# ADDITIONAL WORD PARSING UTILITIES
# ============================================================================

def is_matching_section(text: str) -> bool:
    """Check if we're in a matching section (column A matches column B)."""
    lower = text.lower()
    return 'match' in lower or 'matching' in lower or 'column a' in lower


def is_matching_table_line(text: str) -> bool:
    """Check if line is part of a matching table (numbered items with ellipsis)."""
    # Pattern: "1. something …" or "a. something ..."
    if re.match(r'^[1-9a-z][.\)]\s*.+[…\.]{2,}', text, re.IGNORECASE):
        return True
    return False


def is_dialogue_completion(text: str) -> bool:
    """Check if text is a dialogue completion question (Complete the dialogue...)."""
    lower = text.lower()
    return 'complete the dialogue' in lower or 'suitable response' in lower


def is_blank_only_line(text: str) -> bool:
    """Check if line is only underscores/blanks (dialogue placeholder)."""
    stripped = text.strip().replace('_', '').replace('.', '').replace(' ', '')
    return len(stripped) == 0 and len(text.strip()) >= 3


def is_dialogue_blank_line(text: str) -> bool:
    """Check if line is a dialogue blank (speaker: ___ format)."""
    # Pattern: "Speaker:" or "A:" or "Name:" followed by blank
    if re.match(r'^[A-Z][a-zA-Z]*\s*:\s*[_\.…]+\s*$', text):
        return True
    return False


def is_dialogue_prompt_line(text: str) -> bool:
    """Check if line is dialogue speaker line."""
    # Pattern: "A:" or "Speaker:" followed by text
    if re.match(r'^[A-Z][a-zA-Z]*\s*:\s*.+', text):
        return True
    return False


def is_reading_passage_start(text: str) -> bool:
    """Check if text starts a reading passage section."""
    lower = text.lower()
    patterns = [
        'read the following',
        'read the passage',
        'reading comprehension',
        'read the text',
    ]
    return any(p in lower for p in patterns)


def is_question_with_single_word_options(text: str) -> bool:
    """Check if question has single-word options (vocabulary questions)."""
    # Pattern: options like "A. word  B. word  C. word  D. word"
    options = re.findall(r'[A-D]\s*[.\)]\s*(\S+)', text, re.IGNORECASE)
    if len(options) >= 3:
        # Check if all options are single words
        return all(len(opt.split()) == 1 for opt in options)
    return False


def extract_passage_questions(lines: List[str], start_idx: int) -> Tuple[List[dict], int]:
    """
    Extract questions from a reading passage section.
    Returns list of questions and the end index.
    """
    questions = []
    passage_lines = []
    i = start_idx

    # Collect passage text until we hit questions
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Check if line starts a numbered question
        if re.match(r'^\d+\s*[.\)]', line):
            break

        passage_lines.append(line)
        i += 1

    passage_text = "\n".join(passage_lines)

    # Now collect questions
    question_pattern = re.compile(r'^\s*(\d+)\s*[.\)]\s*(.*)$')

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        q_match = question_pattern.match(line)
        if q_match:
            q_num = q_match.group(1)
            q_content = q_match.group(2)

            # Collect options
            options = []
            j = i + 1
            while j < len(lines):
                opt_line = lines[j].strip()
                if not opt_line:
                    j += 1
                    continue
                if re.match(r'^[A-D]\s*[.\)]', opt_line, re.IGNORECASE):
                    opt_text = re.sub(r'^[A-D]\s*[.\)]\s*', '', opt_line, flags=re.IGNORECASE)
                    options.append(opt_text)
                    j += 1
                    if len(options) >= 4:
                        break
                else:
                    break

            if options:
                questions.append({
                    "question": f"(Passage) {q_num}. {q_content}",
                    "options": options,
                    "passage": passage_text[:200] + "..." if len(passage_text) > 200 else passage_text,
                })
                i = j
            else:
                i += 1
        else:
            # Not a question, might be end of section
            break

    return questions, i


def is_passage_with_blanks(text: str) -> bool:
    """Check if text is a passage paragraph with numbered blanks like (16)."""
    # Long text (> 100 chars) with embedded numbers like (16), (21)
    if len(text) > 100 and re.search(r'\(\d+\)', text):
        return True
    return False


def extract_cloze_questions(start_idx: int, lines_list: List[str]) -> Tuple[List[dict], int]:
    """Extract cloze passage questions (passage with numbered blanks + batched options)."""
    result = []
    i = start_idx
    line = lines_list[i].strip()

    # Check if this line has numbered blanks like __________(31)
    blank_nums = re.findall(r'_+\s*\((\d+)\)', line)
    if not blank_nums:
        return [], start_idx

    # Collect all passage lines with blanks
    passage_lines = [line]
    all_blank_nums = list(blank_nums)
    j = i + 1

    while j < len(lines_list):
        next_line = lines_list[j].strip()
        if not next_line:
            j += 1
            continue
        # Stop if we hit A/B/C/D options
        if 'A)' in next_line and 'B)' in next_line:
            break
        # Check for more blanks
        more_blanks = re.findall(r'_+\s*\((\d+)\)', next_line)
        if more_blanks:
            passage_lines.append(next_line)
            all_blank_nums.extend(more_blanks)
            j += 1
            continue
        # If line is short and part of passage, include it
        if len(next_line) < 150 and not re.match(r'^[A-E]\s*\)', next_line):
            passage_lines.append(next_line)
            j += 1
            continue
        break

    # Now collect options (one A/B/C/D line per blank)
    options_lines = []
    while j < len(lines_list) and len(options_lines) < len(all_blank_nums):
        next_line = lines_list[j].strip()
        if not next_line:
            j += 1
            continue
        if 'A)' in next_line and 'B)' in next_line:
            options_lines.append(next_line)
            j += 1
        else:
            break

    # Helper to extract options from line
    def extract_options_from_line(line: str) -> List[str]:
        opts = []
        marker_pattern = re.compile(r'(?:^|(?<=\s)|(?<=\t))([A-E])\s*[.\)]', re.IGNORECASE)
        markers = list(marker_pattern.finditer(line))
        if markers:
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text:
                    opts.append(opt_text)
        return opts

    # Create one question per blank
    passage_text = " ".join(passage_lines)
    for idx, blank_num in enumerate(all_blank_nums):
        q_text = f"({blank_num}) {passage_text[:100]}..."
        opts = []
        if idx < len(options_lines):
            opts = extract_options_from_line(options_lines[idx])
        result.append({
            "question": q_text,
            "options": opts
        })

    return result, j



# ============================================================================
# OMR (Optical Mark Recognition) - Chấm bài trắc nghiệm
# ============================================================================
# OMR Grading Functions
# ============================================================================


def _detect_template_from_image(image_bytes: bytes, num_questions_detected: int = 0) -> dict:
    """Nhận diện loại đề và cấp độ từ phiếu bằng OCR

    Trả về dict với keys:
    - detected_template: template_type đầy đủ (ví dụ: "IKSC_BENJAMIN")
    - detected_contest: IKSC hoặc IKLC
    - detected_level: PRE_ECOLIER, ECOLIER, BENJAMIN, CADET, JUNIOR, STUDENT

    Sử dụng kết hợp:
    1. OCR để đọc text từ header
    2. Số câu hỏi được phát hiện để xác định level
    """
    import cv2
    import numpy as np

    result = {
        "detected_template": "",
        "detected_contest": "",
        "detected_level": ""
    }

    # Đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return result

    height, width = img.shape[:2]
    text = ""

    # Thử các OCR engines theo thứ tự ưu tiên
    ocr_success = False

    # 1. Thử EasyOCR
    try:
        # Fix SSL certificate issue
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context

        import easyocr
        # Lấy phần trên của ảnh (chứa thông tin loại đề) - khoảng 15% trên
        top_region = img[0:int(height * 0.15), :]
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        ocr_results = reader.readtext(top_region)
        text = ' '.join([r[1] for r in ocr_results])
        ocr_success = True
    except:
        pass

    # 2. Thử Pytesseract
    if not ocr_success:
        try:
            import pytesseract
            top_region = img[0:int(height * 0.15), :]
            # Chuyển sang grayscale và tăng contrast
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, lang='eng')
            ocr_success = True
        except:
            pass

    # 3. Nếu OCR không thành công, sử dụng số câu hỏi để đoán
    if not ocr_success and num_questions_detected > 0:
        # Dựa vào số câu hỏi để đoán template
        if num_questions_detected <= 24:
            # 24 câu -> Pre-Ecolier hoặc Ecolier (IKSC) hoặc Pre-Ecolier (IKLC)
            result["detected_level"] = "PRE_ECOLIER"  # Mặc định
        elif num_questions_detected <= 30:
            # 30 câu -> Benjamin/Cadet/Junior/Student (IKSC) hoặc Ecolier (IKLC)
            result["detected_level"] = "BENJAMIN"  # Mặc định cho IKSC
        elif num_questions_detected <= 50:
            # 50 câu -> Benjamin/Cadet/Junior/Student (IKLC)
            result["detected_contest"] = "IKLC"
            result["detected_level"] = "BENJAMIN"  # Mặc định

        return result

    text_lower = text.lower()

    # === NHẬN DIỆN LOẠI CUỘC THI (IKSC hoặc IKLC) ===
    if 'science' in text_lower or 'iksc' in text_lower:
        result["detected_contest"] = "IKSC"
    elif 'linguistic' in text_lower or 'iklc' in text_lower or 'english' in text_lower:
        result["detected_contest"] = "IKLC"

    # === NHẬN DIỆN CẤP ĐỘ (LEVEL) ===
    level_detected = ""

    # Tìm theo CLASS pattern (ví dụ: "CLASS 5 & 6", "CLASS 5&6", "5 & 6")
    class_match = re.search(r'class\s*(\d+)\s*[&]\s*(\d+)', text_lower)
    if not class_match:
        # Thử pattern không có "class"
        class_match = re.search(r'(\d+)\s*[&]\s*(\d+)', text_lower)

    if class_match:
        class1 = int(class_match.group(1))
        class2 = int(class_match.group(2))
        if class1 == 1 and class2 == 2:
            level_detected = "PRE_ECOLIER"
        elif class1 == 3 and class2 == 4:
            level_detected = "ECOLIER"
        elif class1 == 5 and class2 == 6:
            level_detected = "BENJAMIN"
        elif class1 == 7 and class2 == 8:
            level_detected = "CADET"
        elif class1 == 9 and class2 == 10:
            level_detected = "JUNIOR"
        elif class1 == 11 and class2 == 12:
            level_detected = "STUDENT"

    # Nếu không tìm thấy theo class, thử tìm theo tên level
    if not level_detected:
        if 'pre-ecolier' in text_lower or 'preecolier' in text_lower or 'pre_ecolier' in text_lower:
            level_detected = "PRE_ECOLIER"
        elif 'benjamin' in text_lower:
            level_detected = "BENJAMIN"
        elif 'cadet' in text_lower:
            level_detected = "CADET"
        elif 'junior' in text_lower:
            level_detected = "JUNIOR"
        elif 'student' in text_lower:
            level_detected = "STUDENT"
        elif 'ecolier' in text_lower:
            level_detected = "ECOLIER"
        # IKLC specific names
        elif 'start' in text_lower:
            level_detected = "PRE_ECOLIER"
        elif 'story' in text_lower:
            level_detected = "ECOLIER"
        elif 'joey' in text_lower:
            level_detected = "BENJAMIN"
        elif 'wallaby' in text_lower:
            level_detected = "CADET"
        elif 'grey' in text_lower:
            level_detected = "JUNIOR"
        elif 'red k' in text_lower:
            level_detected = "STUDENT"

    # Nếu vẫn không tìm được level nhưng có số câu hỏi
    if not level_detected and num_questions_detected > 0:
        if num_questions_detected <= 24:
            level_detected = "PRE_ECOLIER"
        elif num_questions_detected <= 30:
            level_detected = "BENJAMIN"
        elif num_questions_detected <= 50:
            level_detected = "BENJAMIN"

    result["detected_level"] = level_detected

    # Tạo template_type đầy đủ
    if result["detected_contest"] and level_detected:
        result["detected_template"] = f"{result['detected_contest']}_{level_detected}"
    elif level_detected:
        # Nếu chỉ có level, thử đoán contest từ số câu
        if num_questions_detected == 50:
            result["detected_contest"] = "IKLC"
        elif num_questions_detected == 30:
            result["detected_contest"] = "IKSC"
        elif num_questions_detected == 24:
            result["detected_contest"] = "IKSC"

        if result["detected_contest"]:
            result["detected_template"] = f"{result['detected_contest']}_{level_detected}"

    return result


def _extract_student_info_ocr(image_bytes: bytes) -> dict:
    """Trích xuất thông tin học sinh và loại đề từ phiếu bằng OCR (EasyOCR)"""
    import cv2
    import numpy as np

    # Bắt đầu với việc nhận diện template
    template_info = _detect_template_from_image(image_bytes)

    # Parse thông tin từ text
    info = {
        "full_name": "",
        "class": "",
        "dob": "",
        "id_no": "",
        "school_name": "",
        "detected_template": template_info.get("detected_template", ""),
        "detected_contest": template_info.get("detected_contest", ""),
        "detected_level": template_info.get("detected_level", "")
    }

    try:
        import easyocr
    except ImportError:
        return info

    # Đọc ảnh
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return info

    # Lấy phần trên của ảnh (chứa thông tin học sinh) - khoảng 25% trên
    height, width = img.shape[:2]
    top_region = img[0:int(height * 0.25), :]

    # Khởi tạo EasyOCR reader
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(top_region)
    except:
        return info

    # Ghép kết quả OCR thành text
    text = '\n'.join([result[1] for result in results])

    # === PARSE THÔNG TIN HỌC SINH ===
    lines = text.split('\n')
    for line in lines:
        line_lower = line.lower().strip()

        # Tìm Full Name
        if 'full name' in line_lower or 'họ tên' in line_lower or 'name:' in line_lower:
            # Lấy phần sau dấu :
            parts = line.split(':')
            if len(parts) > 1:
                info["full_name"] = parts[1].strip()
            else:
                # Tìm trên cùng dòng sau label
                match = re.search(r'(?:full name|họ tên|name)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["full_name"] = match.group(1).strip()

        # Tìm Class (thông tin lớp học của học sinh, không phải level)
        elif 'class:' in line_lower and 'school' not in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["class"] = parts[1].strip()

        # Tìm DOB
        elif 'dob' in line_lower or 'date of birth' in line_lower or 'ngày sinh' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["dob"] = parts[1].strip()
            else:
                match = re.search(r'(?:dob|date of birth|ngày sinh)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["dob"] = match.group(1).strip()

        # Tìm ID NO
        elif 'id no' in line_lower or 'id:' in line_lower or 'số báo danh' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["id_no"] = parts[1].strip()
            else:
                match = re.search(r'(?:id no|id|số báo danh)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["id_no"] = match.group(1).strip()

        # Tìm School Name
        elif 'school' in line_lower or 'trường' in line_lower:
            parts = line.split(':')
            if len(parts) > 1:
                info["school_name"] = parts[1].strip()
            else:
                match = re.search(r'(?:school name|school|trường)[:\s]*(.+)', line, re.IGNORECASE)
                if match:
                    info["school_name"] = match.group(1).strip()

    return info


def _order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    import numpy as np
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left có tổng nhỏ nhất
    rect[2] = pts[np.argmax(s)]  # bottom-right có tổng lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def _four_point_transform(image, pts):
    """Thực hiện perspective transform với 4 điểm"""
    import cv2
    import numpy as np

    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Tính chiều rộng mới
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Tính chiều cao mới
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def _deskew_image(image):
    """Tự động căn chỉnh ảnh bị nghiêng"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Phát hiện cạnh
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Tìm đường thẳng bằng Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return image, 0

    # Tính góc nghiêng trung bình
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Chỉ lấy các đường gần ngang (±15 độ)
            if abs(angle) < 15:
                angles.append(angle)

    if not angles:
        return image, 0

    # Lấy góc trung vị
    median_angle = np.median(angles)

    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated, median_angle


def _find_document_contour(image):
    """Tìm contour của tài liệu (phiếu trả lời)"""
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Làm dày cạnh
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def _preprocess_omr_image(image_bytes: bytes):
    """Tiền xử lý ảnh cho OMR với deskew và perspective correction"""
    import cv2
    import numpy as np

    # Đọc ảnh từ bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None

    original = img.copy()

    # Bước 1: Tìm và căn chỉnh tài liệu nếu bị méo
    # CHÚ Ý: Chỉ áp dụng perspective transform khi contour bao phủ gần như toàn bộ ảnh
    # để tránh cắt mất nội dung (ví dụ: hàng cuối của phiếu 50 câu)
    doc_contour = _find_document_contour(img)
    if doc_contour is not None:
        contour_area = cv2.contourArea(doc_contour)
        img_area = img.shape[0] * img.shape[1]
        # Chỉ transform nếu contour bao phủ > 80% diện tích ảnh
        if contour_area > 0.8 * img_area:
            img = _four_point_transform(img, doc_contour)

    # Bước 2: Deskew (căn chỉnh góc nghiêng)
    img, skew_angle = _deskew_image(img)

    # Bước 3: Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bước 4: Tăng contrast bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Bước 5: Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 6: Adaptive threshold (tốt hơn cho điều kiện ánh sáng khác nhau)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Bước 7: Morphological operations để làm sạch
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return img, gray, binary


def _find_answer_grid_region(gray_image, binary_image):
    """Tìm vùng chứa lưới đáp án trong ảnh dựa trên cấu trúc grid"""
    import cv2
    import numpy as np

    height, width = gray_image.shape[:2]

    # Tìm các đường ngang và dọc
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

    # Kết hợp các đường
    grid = cv2.add(horizontal_lines, vertical_lines)

    # Tìm contours của grid
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Tìm bounding box lớn nhất
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Mở rộng một chút
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)

        return (x, y, w, h)

    # Fallback: Giả sử vùng đáp án nằm ở phần dưới 2/3 của ảnh
    return (0, int(height * 0.25), width, int(height * 0.75))


def _detect_all_rectangles(binary_image, min_size=15, max_size=80):
    """Phát hiện tất cả hình chữ nhật (ô đáp án) trong ảnh"""
    import cv2
    import numpy as np

    # Tìm contours
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for i, contour in enumerate(contours):
        # Lấy bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Lọc theo kích thước
        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue

        # Kiểm tra tỉ lệ gần vuông
        aspect_ratio = w / float(h) if h > 0 else 0
        if not (0.6 <= aspect_ratio <= 1.4):
            continue

        # Kiểm tra diện tích contour so với bounding box (phải gần vuông/chữ nhật)
        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        if bbox_area > 0:
            extent = contour_area / bbox_area
            if extent < 0.5:  # Bỏ qua các hình không đầy đặn
                continue

        rectangles.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'cx': x + w // 2, 'cy': y + h // 2,
            'contour': contour
        })

    return rectangles


def _cluster_by_rows(rectangles, tolerance=15):
    """Nhóm các hình chữ nhật theo hàng dựa trên tọa độ y"""
    import numpy as np

    if not rectangles:
        return []

    # Sắp xếp theo y
    sorted_rects = sorted(rectangles, key=lambda r: r['cy'])

    rows = []
    current_row = [sorted_rects[0]]

    for rect in sorted_rects[1:]:
        # Nếu tọa độ y gần với hàng hiện tại
        if abs(rect['cy'] - current_row[0]['cy']) <= tolerance:
            current_row.append(rect)
        else:
            # Sắp xếp hàng theo x và thêm vào danh sách
            rows.append(sorted(current_row, key=lambda r: r['cx']))
            current_row = [rect]

    # Thêm hàng cuối cùng
    rows.append(sorted(current_row, key=lambda r: r['cx']))

    return rows


def _detect_bubbles_grid_based(gray_image, binary_image, template_type: str):
    """Phát hiện bubble dựa trên cấu trúc grid của phiếu"""
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]  # A-E = 5
    questions_per_row = template["questions_per_row"]  # 4

    height, width = gray_image.shape[:2]

    # Phát hiện tất cả hình chữ nhật
    rectangles = _detect_all_rectangles(binary_image)

    if len(rectangles) < num_questions * num_options * 0.3:
        # Thử với ngưỡng khác
        _, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        rectangles = _detect_all_rectangles(binary_otsu)

    if not rectangles:
        return [], []

    # Phân tích phân bố kích thước để tìm kích thước phổ biến nhất (ô đáp án)
    widths = [r['w'] for r in rectangles]
    heights = [r['h'] for r in rectangles]

    # Tìm mode (giá trị phổ biến nhất) cho width và height
    from collections import Counter
    width_counts = Counter([int(w/5)*5 for w in widths])  # Bin by 5 (rộng hơn)
    height_counts = Counter([int(h/5)*5 for h in heights])

    # Lấy top kích thước phổ biến nhất
    common_widths = [w for w, _ in width_counts.most_common(3)]
    common_heights = [h for h, _ in height_counts.most_common(3)]

    # Lọc các ô có kích thước nằm trong nhóm phổ biến
    target_width = common_widths[0] if common_widths else np.median(widths)
    target_height = common_heights[0] if common_heights else np.median(heights)

    # Lọc với tolerance ±40% (rộng hơn để bắt các ô hơi khác kích thước)
    filtered_rects = [
        r for r in rectangles
        if 0.6 * target_width <= r['w'] <= 1.4 * target_width
        and 0.6 * target_height <= r['h'] <= 1.4 * target_height
    ]

    if not filtered_rects:
        # Fallback: dùng median
        avg_width = np.median(widths)
        avg_height = np.median(heights)
        filtered_rects = [
            r for r in rectangles
            if 0.5 * avg_width <= r['w'] <= 1.5 * avg_width
            and 0.5 * avg_height <= r['h'] <= 1.5 * avg_height
        ]

    # Nhóm theo hàng
    avg_height = np.median([r['h'] for r in filtered_rects]) if filtered_rects else 30
    rows = _cluster_by_rows(filtered_rects, tolerance=int(avg_height * 0.6))

    # Phân loại các hàng
    expected_per_row = questions_per_row * num_options
    valid_rows = []
    partial_rows = []

    for row in rows:
        if len(row) >= expected_per_row * 0.8:
            # Hàng đầy đủ (4 câu/hàng)
            valid_rows.append(row)
        elif len(row) >= num_options:
            # Hàng không đầy đủ (1-3 câu) - có thể là hàng cuối
            partial_rows.append(row)

    # Nếu không đủ hàng valid, thử relax điều kiện
    if len(valid_rows) < num_questions / questions_per_row * 0.5:
        valid_rows = [row for row in rows if len(row) >= num_options]
        partial_rows = []

    # Kết hợp valid_rows và partial_rows, sắp xếp theo y
    all_rows = valid_rows + partial_rows
    all_rows.sort(key=lambda row: row[0]['cy'] if row else 0)

    return all_rows, filtered_rects


def _analyze_bubble_fill_improved(gray_image, rect, threshold=0.4):
    """Phân tích bubble fill với phương pháp cải tiến

    Trả về tuple (is_filled, score, mean_val) để hỗ trợ so sánh tương đối
    """
    import cv2
    import numpy as np

    x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']

    # Lấy vùng bubble với margin nhỏ bên trong
    margin = max(2, int(min(w, h) * 0.15))
    roi = gray_image[y+margin:y+h-margin, x+margin:x+w-margin]

    if roi.size == 0:
        return False, 0.0, 255.0

    # Tính các chỉ số
    mean_val = np.mean(roi)
    min_val = np.min(roi)

    # Phương pháp chính: Đếm pixel tối
    # Bubble được tô bằng bút chì 2B sẽ có nhiều pixel rất tối
    dark_pixels = np.sum(roi < 100) / roi.size  # Pixel tối (< 100)
    very_dark_pixels = np.sum(roi < 60) / roi.size  # Pixel rất tối (< 60)

    # Phương pháp phụ: Binary threshold với Otsu (cho ảnh scan tốt)
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = np.sum(binary_roi > 0) / binary_roi.size

    # Tính score dựa trên pixel tối
    # Ưu tiên pixel rất tối (có trọng số cao hơn)
    darkness_score = dark_pixels * 0.6 + very_dark_pixels * 1.5

    # Kiểm tra có được tô không
    # Tiêu chí: có nhiều pixel tối HOẶC mean thấp
    if very_dark_pixels > 0.05 or dark_pixels > 0.15:
        # Có vùng được tô rõ ràng
        is_filled = True
        score = darkness_score
    elif mean_val < 120 and dark_pixels > 0.05:
        # Mean thấp và có một ít pixel tối
        is_filled = True
        score = darkness_score + (120 - mean_val) / 200
    elif fill_ratio > threshold and mean_val < 150:
        # Fallback: Otsu + mean thấp
        is_filled = True
        score = fill_ratio * 0.5  # Giảm trọng số của Otsu
    else:
        is_filled = False
        score = darkness_score

    return is_filled, score, mean_val


def _group_bubbles_to_questions_improved(rows, template_type: str):
    """Nhóm bubble thành câu hỏi dựa trên vị trí trong grid"""
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]
    layout = template.get("layout", "row")  # "row" hoặc "column"

    questions = []

    # Lọc các hàng có số bubble hợp lý
    expected_per_row = questions_per_row * num_options
    valid_rows = []
    partial_rows = []  # Hàng có ít bubble hơn (có thể là hàng cuối)

    for row in rows:
        # Loại bỏ các bubble trùng lặp (gap < 5 pixels)
        filtered_row = [row[0]] if row else []
        for i in range(1, len(row)):
            gap = row[i]['cx'] - filtered_row[-1]['cx']
            if gap > 10:  # Chỉ thêm nếu cách bubble trước > 10 pixels
                filtered_row.append(row[i])

        # Phân loại hàng theo số bubble
        if expected_per_row * 0.8 <= len(filtered_row) <= expected_per_row * 1.3:
            # Hàng đầy đủ (4 câu/hàng)
            valid_rows.append(filtered_row)
        elif num_options <= len(filtered_row) < expected_per_row * 0.8:
            # Hàng không đầy đủ (có thể là hàng cuối với 1-3 câu)
            partial_rows.append(filtered_row)

    # Loại bỏ các hàng partial ở đầu (trước hàng valid đầu tiên)
    # Đây thường là header hoặc phần thông tin học sinh
    if valid_rows and partial_rows:
        first_valid_y = valid_rows[0][0]['cy'] if valid_rows[0] else float('inf')
        # Chỉ giữ lại partial_rows sau hàng valid cuối cùng (hàng cuối của grid)
        last_valid_y = valid_rows[-1][0]['cy'] if valid_rows[-1] else 0
        partial_rows = [row for row in partial_rows if row and row[0]['cy'] > last_valid_y]

    # Tách mỗi hàng thành các nhóm câu hỏi (mỗi nhóm = 5 bubbles cho 1 câu)
    all_question_groups = []  # List of lists: mỗi hàng chứa các câu hỏi

    for row in valid_rows:
        if len(row) < num_options:
            continue

        row_questions = []

        # Tính khoảng cách giữa các bubble liên tiếp
        gaps = []
        for i in range(1, len(row)):
            gap = row[i]['cx'] - row[i-1]['cx']
            gaps.append((i, gap))

        if not gaps:
            continue

        # Phân tích gaps để tìm điểm phân tách câu hỏi
        gap_values = [g[1] for g in gaps]
        median_gap = np.median(gap_values)
        max_gap = max(gap_values)

        # Nếu max_gap > 1.5 * median_gap, đó là điểm phân tách câu hỏi
        if max_gap > median_gap * 1.4:
            # Có điểm phân tách rõ ràng
            large_gap_threshold = median_gap * 1.3

            current_question_bubbles = [row[0]]
            for i in range(1, len(row)):
                gap = row[i]['cx'] - row[i-1]['cx']

                # Cho phép split khi có large gap và có ít nhất 4 bubbles (thiếu 1 do không phát hiện được)
                if gap > large_gap_threshold and len(current_question_bubbles) >= num_options - 1:
                    row_questions.append(current_question_bubbles[:num_options])
                    current_question_bubbles = [row[i]]
                else:
                    current_question_bubbles.append(row[i])

            # Thêm câu hỏi cuối cùng trong hàng
            # Cho phép thiếu 1 bubble (4/5) vì có thể bubble không được phát hiện
            if len(current_question_bubbles) >= num_options - 1:
                row_questions.append(current_question_bubbles[:num_options])
        else:
            # Không có điểm phân tách rõ ràng, chia đều theo số options
            for i in range(0, len(row), num_options):
                question_bubbles = row[i:i+num_options]
                if len(question_bubbles) == num_options:
                    row_questions.append(question_bubbles)

        if row_questions:
            all_question_groups.append(row_questions)

    # Xử lý các hàng partial (hàng cuối có ít câu hơn)
    for row in partial_rows:
        if len(row) < num_options:
            continue

        row_questions = []
        # Chia hàng thành các câu hỏi
        for i in range(0, len(row), num_options):
            question_bubbles = row[i:i+num_options]
            if len(question_bubbles) == num_options:
                row_questions.append(question_bubbles)

        if row_questions:
            all_question_groups.append(row_questions)

    # Đánh số câu hỏi dựa trên layout
    if layout == "column":
        # Layout theo cột: cột 1 có câu 1,5,9..., cột 2 có câu 2,6,10...
        # Mỗi hàng có 4 câu hỏi (4 cột)
        # Câu hỏi thứ i ở cột (i-1) % 4, hàng (i-1) // 4
        num_cols = questions_per_row
        num_rows = len(all_question_groups)

        for row_idx, row_questions in enumerate(all_question_groups):
            for col_idx, bubbles in enumerate(row_questions):
                if col_idx >= num_cols:
                    break
                # Tính số thứ tự câu hỏi: hàng * 4 + cột + 1
                # Ví dụ: hàng 0, cột 0 = câu 1; hàng 0, cột 1 = câu 2
                # hàng 1, cột 0 = câu 5; hàng 1, cột 1 = câu 6
                q_num = row_idx * num_cols + col_idx + 1
                if q_num <= num_questions:
                    questions.append({
                        "index": q_num,
                        "bubbles": bubbles
                    })
    else:
        # Layout theo hàng (mặc định): đọc từ trái sang phải, trên xuống dưới
        question_idx = 0
        for row_questions in all_question_groups:
            for bubbles in row_questions:
                if question_idx >= num_questions:
                    break
                questions.append({
                    "index": question_idx + 1,
                    "bubbles": bubbles
                })
                question_idx += 1

    # Sắp xếp theo index
    questions.sort(key=lambda x: x["index"])

    return questions


def _detect_bubbles(binary_image, template_type: str = "IKSC_BENJAMIN"):
    """Phát hiện các bubble trong ảnh (legacy function for compatibility)"""
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]

    # Tìm contours
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lọc các contour có dạng bubble (gần vuông/tròn)
    bubbles = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Bubble phải gần vuông và có kích thước hợp lý
        if 0.6 <= aspect_ratio <= 1.4 and 12 <= w <= 80 and 12 <= h <= 80:
            bubbles.append((x, y, w, h, contour))

    return bubbles


def _analyze_bubble_fill(binary_image, bubble_contour, threshold: float = 0.35):
    """Phân tích xem bubble có được tô hay không (legacy)"""
    import cv2
    import numpy as np

    # Tạo mask cho bubble
    mask = np.zeros(binary_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [bubble_contour], -1, 255, -1)

    # Đếm pixel trong mask
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return False, 0

    # Đếm pixel được tô (giao của mask và binary image)
    filled = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    filled_pixels = cv2.countNonZero(filled)

    fill_ratio = filled_pixels / total_pixels

    return fill_ratio > threshold, fill_ratio


def _group_bubbles_to_questions(bubbles, template_type: str = "IKSC_BENJAMIN"):
    """Nhóm các bubble thành câu hỏi (legacy)"""
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    num_options = template["options"]
    questions_per_row = template["questions_per_row"]

    if not bubbles:
        return []

    # Sắp xếp bubble theo y (hàng) rồi theo x (cột)
    sorted_bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))

    # Nhóm theo hàng dựa trên tọa độ y
    rows = []
    current_row = [sorted_bubbles[0]]
    y_threshold = 30  # Ngưỡng để phân biệt hàng

    for bubble in sorted_bubbles[1:]:
        if abs(bubble[1] - current_row[0][1]) < y_threshold:
            current_row.append(bubble)
        else:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [bubble]
    rows.append(sorted(current_row, key=lambda b: b[0]))

    # Mỗi hàng chứa questions_per_row câu hỏi × num_options lựa chọn
    expected_bubbles_per_row = questions_per_row * num_options

    questions = []
    question_idx = 0

    for row in rows:
        # Chia hàng thành các nhóm 5 bubble (A-E) cho mỗi câu hỏi
        for i in range(0, len(row), num_options):
            if question_idx >= num_questions:
                break
            question_bubbles = row[i:i+num_options]
            if len(question_bubbles) == num_options:
                questions.append({
                    "index": question_idx + 1,
                    "bubbles": question_bubbles
                })
                question_idx += 1

    return questions


def _grade_single_sheet(image_bytes: bytes, answer_key: List[str], template_type: str = "IKSC_BENJAMIN", extract_info: bool = True):
    """Chấm một phiếu trả lời với thuật toán OMR cải tiến"""
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    option_labels = ["A", "B", "C", "D", "E"]

    # Trích xuất thông tin học sinh bằng OCR
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh với deskew và perspective correction
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # Thử phương pháp mới trước: phát hiện dựa trên grid
    rows, all_rects = _detect_bubbles_grid_based(gray, binary, template_type)

    questions = []
    use_new_method = False

    if rows and len(rows) >= 3:
        # Sử dụng phương pháp mới nếu phát hiện đủ hàng
        questions = _group_bubbles_to_questions_improved(rows, template_type)
        use_new_method = True

    # Fallback: Sử dụng phương pháp cũ nếu phương pháp mới không hiệu quả
    if len(questions) < num_questions * 0.3:
        bubbles = _detect_bubbles(binary, template_type)

        if len(bubbles) >= num_questions * 5 * 0.3:
            questions = _group_bubbles_to_questions(bubbles, template_type)
            use_new_method = False

    if len(questions) < num_questions * 0.3:
        return {
            "error": f"Không phát hiện đủ câu hỏi. Tìm thấy: {len(questions)}, cần: {num_questions}. "
                     f"Vui lòng đảm bảo ảnh rõ nét và phiếu được căn chỉnh đúng."
        }

    # Phân tích từng câu hỏi
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    # Tạo dict để tra cứu câu hỏi theo index (thay vì dùng vị trí trong mảng)
    questions_by_index = {q["index"]: q for q in questions}

    for q_idx in range(num_questions):
        q_num = q_idx + 1  # Số thứ tự câu hỏi (1-based)
        if q_num not in questions_by_index:
            # Không tìm thấy câu hỏi này
            student_answers.append(None)
            blank_count += 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        mean_vals = []  # Lưu mean value của từng option
        is_filled_list = []  # Lưu trạng thái is_filled của từng option
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            if use_new_method:
                # Phương pháp mới: bubble là dict
                result = _analyze_bubble_fill_improved(gray, bubble)
                if len(result) == 3:
                    is_filled, fill_ratio, mean_val = result
                else:
                    is_filled, fill_ratio = result
                    mean_val = 128.0
                mean_vals.append(mean_val)
                is_filled_list.append(is_filled)
            else:
                # Phương pháp cũ: bubble là tuple với contour
                is_filled, fill_ratio = _analyze_bubble_fill(binary, bubble[4])
                mean_vals.append(128.0)
                is_filled_list.append(is_filled)

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        # Ưu tiên option có is_filled=True và mean thấp nhất (được tô đậm nhất)
        filled_options = [(i, mean_vals[i]) for i in range(len(is_filled_list)) if is_filled_list[i]]
        if filled_options and selected_option is None:
            # Có option được đánh dấu filled nhưng chưa được chọn
            # Chọn option có mean thấp nhất (tối nhất = được tô)
            darkest_filled = min(filled_options, key=lambda x: x[1])
            selected_option = option_labels[darkest_filled[0]]
        elif len(filled_options) == 1:
            # Chỉ có 1 option filled -> chọn option đó
            selected_option = option_labels[filled_options[0][0]]

        # Phát hiện vùng tối bất thường (bóng/rìa ảnh)
        # Nếu nhiều options có mean rất thấp (<50), đây có thể là vùng tối
        dark_region_count = sum(1 for m in mean_vals if m < 50)
        is_dark_region = dark_region_count >= 3

        if is_dark_region:
            # Vùng tối: chọn option có mean CAO nhất (sáng nhất = không bị bóng che)
            # vì các vùng tối là do bóng, không phải do được tô
            max_mean = max(mean_vals)
            min_mean = min(mean_vals)

            # Chỉ chọn nếu có 1 option sáng hơn hẳn (chênh lệch > 50)
            if max_mean - min_mean > 50:
                bright_option_idx = mean_vals.index(max_mean)
                # Kiểm tra option sáng này có được tô không
                if fill_ratios[bright_option_idx] > 0.1:
                    selected_option = option_labels[bright_option_idx]
                else:
                    # Option sáng nhưng không được tô -> có thể là blank hoặc tô option khác
                    # Trong vùng tối, tìm option có score cao nhất trong các option không quá tối
                    valid_options = [(i, fill_ratios[i]) for i in range(len(mean_vals))
                                    if mean_vals[i] > 100 or fill_ratios[i] > 0.3]
                    if valid_options:
                        best_idx = max(valid_options, key=lambda x: x[1])[0]
                        selected_option = option_labels[best_idx]
        else:
            # Vùng bình thường: sử dụng logic cũ với cải tiến

            # Ngưỡng động: nếu max_fill > 0.25 và vượt trội hơn các option khác
            if selected_option is None and max_fill > 0.25:
                # Kiểm tra xem có một option nào vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)
                if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Option đầu lớn hơn 30% so với option thứ 2
                    selected_option = option_labels[fill_ratios.index(max_fill)]

            # Kiểm tra nếu có nhiều đáp án được chọn
            # Sử dụng ngưỡng động dựa trên max_fill
            if max_fill > 0.5:
                # Nếu có option được tô đậm, các option khác cần đạt ít nhất 60% của max
                filled_threshold = max_fill * 0.6
            else:
                filled_threshold = 0.35

            filled_count = sum(1 for r in fill_ratios if r > filled_threshold)
            if filled_count > 1:
                # Kiểm tra xem có 1 option rõ ràng vượt trội không
                sorted_ratios = sorted(fill_ratios, reverse=True)

                # Tính chênh lệch mean giữa option cao nhất và thấp nhất
                max_score_idx = fill_ratios.index(sorted_ratios[0])
                max_score_mean = mean_vals[max_score_idx]

                # Nếu option có score cao nhất cũng có mean thấp nhất -> đây là bubble được tô
                if max_score_mean == min(mean_vals) or sorted_ratios[0] > sorted_ratios[1] * 1.3:
                    # Có 1 option vượt trội rõ ràng
                    selected_option = option_labels[max_score_idx]
                else:
                    # Kiểm tra thêm: nếu max > 0.5 và second < 0.4, vẫn chọn max
                    if sorted_ratios[0] > 0.5 and sorted_ratios[1] < 0.4:
                        selected_option = option_labels[fill_ratios.index(sorted_ratios[0])]
                    else:
                        # Phân tích thêm bằng mean value
                        # Option được tô sẽ có mean thấp hơn các option không tô
                        mean_diff = max(mean_vals) - min(mean_vals)
                        if mean_diff > 30:
                            # Có sự khác biệt rõ ràng về độ sáng
                            darkest_idx = mean_vals.index(min(mean_vals))
                            selected_option = option_labels[darkest_idx]
                        else:
                            selected_option = "MULTI"  # Đánh dấu chọn nhiều đáp án

        student_answers.append(selected_option)

        # So sánh với đáp án
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if selected_option is None:
            status = "blank"
            blank_count += 1
        elif selected_option == "MULTI":
            status = "invalid"
            wrong_count += 1
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            status = "correct"
            correct_count += 1
        else:
            status = "wrong"
            wrong_count += 1

        details.append({
            "q": q_idx + 1,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # Tính điểm
    if scoring.get("type") == "tiered":
        # Tính điểm theo phần (tiered scoring) - dùng cho IKSC
        score = scoring.get("base", 0)
        tiers = scoring.get("tiers", [])

        for detail in details:
            q_num = detail["q"]
            status = detail["status"]

            # Tìm tier phù hợp cho câu hỏi này
            tier_points = {"correct": 1, "wrong": 0}  # Default
            for tier in tiers:
                if tier["start"] <= q_num <= tier["end"]:
                    tier_points = tier
                    break

            if status == "correct":
                score += tier_points.get("correct", 1)
            elif status in ["wrong", "invalid"]:
                score += tier_points.get("wrong", 0)
            # blank: không cộng/trừ điểm

    elif scoring.get("type") == "best_of":
        # Tính điểm chỉ lấy N câu tốt nhất - dùng cho IKLC Benjamin-Student
        # Công thức: base + (correct * points) + (wrong * penalty)
        # Chỉ tính count_best câu có điểm cao nhất
        count_best = scoring.get("count_best", 40)
        correct_pts = scoring.get("correct", 1)
        wrong_pts = scoring.get("wrong", -0.5)

        # Tính điểm từng câu
        question_scores = []
        for detail in details:
            status = detail["status"]
            if status == "correct":
                question_scores.append(correct_pts)
            elif status in ["wrong", "invalid"]:
                question_scores.append(wrong_pts)
            else:  # blank
                question_scores.append(0)

        # Sắp xếp giảm dần và lấy N câu tốt nhất
        question_scores.sort(reverse=True)
        best_scores = question_scores[:count_best]

        score = scoring.get("base", 0) + sum(best_scores)

    else:
        # Tính điểm đơn giản (flat scoring) - dùng cho IKLC Pre-Ecolier, Ecolier và các kỳ thi khác
        score = (
            scoring.get("base", 0) +
            correct_count * scoring.get("correct", 1) +
            wrong_count * scoring.get("wrong", 0) +
            blank_count * scoring.get("blank", 0)
        )

    return {
        "answers": student_answers,
        "score": round(score, 2),
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details,
        "student_info": student_info,
        "detection_method": "grid_based" if use_new_method else "contour_based",
        "questions_detected": len(questions)
    }


def _detect_seamo_bubbles_fixed_grid(gray_image):
    """Phát hiện bubbles trong phiếu SEAMO với dynamic grid detection

    Sử dụng kết hợp:
    1. Phát hiện đường kẻ ngang để tìm vị trí các hàng
    2. Phát hiện đường kẻ dọc để tìm vị trí các cột
    3. Fallback về tọa độ cố định nếu không detect được
    """
    import cv2
    import numpy as np

    h, w = gray_image.shape[:2]

    # Detect loại ảnh: PDF vector vs scan image
    # - Scan 300 DPI A4: ~2480 x 3508 (width > 2000)
    # - PDF vector render: ~1191 x 1685 (width ~1200)
    is_high_res_scan = w > 2000

    if not is_high_res_scan:
        # PDF vector render - Thử dynamic detection trước
        grid_info = _detect_seamo_grid_dynamic(gray_image)

        if grid_info is not None:
            grid_start_x = grid_info['start_x']
            grid_start_y = grid_info['start_y']
            option_spacing = grid_info['col_spacing']
            row_spacing = grid_info['row_spacing']
            bubble_w = grid_info['bubble_w']
            bubble_h = grid_info['bubble_h']
        else:
            # Fallback cho PDF vector (từ 72 DPI gốc)
            expected_w, expected_h = 1191, 1685
            scale_x = w / expected_w
            scale_y = h / expected_h
            grid_start_x = int(68 * scale_x)
            grid_start_y = int(541 * scale_y)
            option_spacing = int(49 * scale_x)
            row_spacing = int(42 * scale_y)
            bubble_w = int(30 * scale_x)
            bubble_h = int(18 * scale_y)
    else:
        # Ảnh scan - sử dụng vị trí cột tuyệt đối đã calibrate cẩn thận
        # (SEAMO có spacing không đều giữa các cột A-E)
        # Expected size: 2480 x 3508 (A4 @ 300 DPI)
        expected_scan_w, expected_scan_h = 2480, 3508
        scan_scale_x = w / expected_scan_w
        scan_scale_y = h / expected_scan_h

        # Vị trí tuyệt đối cho mỗi cột (đã calibrate từ scan thực tế)
        col_lefts_base = [238, 318, 435, 551, 663]  # A, B, C, D, E
        col_lefts = [int(c * scan_scale_x) for c in col_lefts_base]

        grid_start_y = int(1164 * scan_scale_y)
        row_spacing = int(82 * scan_scale_y)
        bubble_w = int(50 * scan_scale_x)
        bubble_h = int(35 * scan_scale_y)

        # Build questions với vị trí cột tuyệt đối
        questions = []
        for q_idx in range(20):
            row_y = grid_start_y + q_idx * row_spacing
            bubbles = []
            for opt_idx in range(5):
                bubble_x = col_lefts[opt_idx]
                bubble_cx = bubble_x + bubble_w // 2
                bubble_cy = row_y + bubble_h // 2
                bubbles.append({
                    'x': bubble_x,
                    'y': row_y,
                    'w': bubble_w,
                    'h': bubble_h,
                    'cx': bubble_cx,
                    'cy': bubble_cy
                })
            questions.append({
                'index': q_idx + 1,
                'bubbles': bubbles
            })
        return questions

    questions = []

    for q_idx in range(20):
        row_y = grid_start_y + q_idx * row_spacing

        bubbles = []
        for opt_idx in range(5):
            bubble_x = grid_start_x + opt_idx * option_spacing
            bubble_cx = bubble_x + bubble_w // 2
            bubble_cy = row_y + bubble_h // 2

            bubbles.append({
                'x': bubble_x,
                'y': row_y,
                'w': bubble_w,
                'h': bubble_h,
                'cx': bubble_cx,
                'cy': bubble_cy
            })

        questions.append({
            'index': q_idx + 1,
            'bubbles': bubbles
        })

    return questions


def _detect_seamo_grid_dynamic(gray_image):
    """Phát hiện động vị trí grid SEAMO bằng Canny edge detection + Hough Lines

    Cải thiện: Sử dụng Canny + HoughLinesP để detect đường kẻ chính xác hơn,
    hoạt động tốt với cả PDF vector và ảnh scan.

    Returns:
        dict với các key: start_x, start_y, col_spacing, row_spacing, bubble_w, bubble_h
        hoặc None nếu không detect được
    """
    import cv2
    import numpy as np

    h, w = gray_image.shape[:2]

    # ===== BƯỚC 1: Edge detection với Canny =====
    edges = cv2.Canny(gray_image, 50, 150)

    # Crop vùng câu hỏi (25%-90% chiều cao, 2%-40% chiều rộng)
    crop_y1, crop_y2 = int(h * 0.25), int(h * 0.9)
    crop_x1, crop_x2 = int(w * 0.02), int(w * 0.4)
    edges_crop = edges[crop_y1:crop_y2, crop_x1:crop_x2]

    # ===== BƯỚC 2: Detect đường ngang với Hough Lines =====
    h_lines = cv2.HoughLinesP(edges_crop, 1, np.pi/180,
                              threshold=80, minLineLength=80, maxLineGap=10)

    if h_lines is None or len(h_lines) < 10:
        return None

    # Lọc và nhóm đường ngang
    horizontal_y = []
    for line in h_lines:
        x1, y1, x2, y2 = line[0]
        # Đường ngang: góc < 5 độ
        if abs(y2 - y1) < 5 and abs(x2 - x1) > 50:
            y_center = (y1 + y2) // 2 + crop_y1
            horizontal_y.append(y_center)

    if len(horizontal_y) < 10:
        return None

    # Nhóm các đường gần nhau
    horizontal_y = sorted(horizontal_y)
    row_lines = []
    current_group = [horizontal_y[0]]

    for y in horizontal_y[1:]:
        if y - current_group[-1] <= 5:
            current_group.append(y)
        else:
            row_lines.append(int(np.mean(current_group)))
            current_group = [y]

    if current_group:
        row_lines.append(int(np.mean(current_group)))

    if len(row_lines) < 10:
        return None

    # ===== BƯỚC 3: Tính row_spacing =====
    row_gaps = []
    for i in range(1, len(row_lines)):
        gap = row_lines[i] - row_lines[i-1]
        if 25 < gap < 55:  # Điều chỉnh range phù hợp hơn
            row_gaps.append(gap)

    if len(row_gaps) < 5:
        return None

    row_spacing = int(np.median(row_gaps))

    # ===== BƯỚC 4: Detect đường dọc =====
    v_lines = cv2.HoughLinesP(edges_crop, 1, np.pi/180,
                              threshold=50, minLineLength=50, maxLineGap=10)

    if v_lines is None:
        return None

    # Lọc đường dọc
    vertical_x = []
    for line in v_lines:
        x1, y1, x2, y2 = line[0]
        # Đường dọc: góc > 85 độ
        if abs(x2 - x1) < 5 and abs(y2 - y1) > 30:
            x_center = (x1 + x2) // 2 + crop_x1
            vertical_x.append(x_center)

    if len(vertical_x) < 5:
        return None

    # Nhóm các đường gần nhau
    vertical_x = sorted(vertical_x)
    col_lines = []
    current_group = [vertical_x[0]]

    for x in vertical_x[1:]:
        if x - current_group[-1] <= 8:
            current_group.append(x)
        else:
            col_lines.append(int(np.mean(current_group)))
            current_group = [x]

    if current_group:
        col_lines.append(int(np.mean(current_group)))

    if len(col_lines) < 5:
        return None

    # ===== BƯỚC 5: Tính col_spacing =====
    col_gaps = []
    for i in range(1, len(col_lines)):
        gap = col_lines[i] - col_lines[i-1]
        if 30 < gap < 60:
            col_gaps.append(gap)

    if len(col_gaps) < 3:
        return None

    col_spacing = int(np.median(col_gaps))

    # ===== BƯỚC 6: Xác định vị trí bắt đầu =====
    # Tìm header - thường là 2 đường liên tiếp gần nhau ở đầu (header đen + viền)
    # Sau đó các row có khoảng cách đều (row_spacing)

    # Tìm vị trí bắt đầu của grid thực sự (sau header)
    # Header thường có khoảng cách < row_spacing * 0.8
    content_start_idx = 0
    for i in range(1, len(row_lines)):
        gap = row_lines[i] - row_lines[i-1]
        if gap < row_spacing * 0.7:
            # Đây vẫn là header area
            content_start_idx = i
        elif gap >= row_spacing * 0.85:
            # Đây là row đầu tiên của content
            content_start_idx = i
            break

    # Row đầu tiên của content (câu 1)
    first_content_row = row_lines[content_start_idx] if content_start_idx < len(row_lines) else row_lines[-1]

    # Tìm cột bubble A (sau cột số thứ tự)
    first_bubble_x = None
    for i in range(1, len(col_lines)):
        gap = col_lines[i] - col_lines[i-1]
        if gap >= col_spacing * 0.8:
            first_bubble_x = col_lines[i-1] + 5  # Offset nhỏ sau viền
            break

    if first_bubble_x is None and len(col_lines) > 1:
        first_bubble_x = col_lines[1] + 5

    if first_bubble_x is None:
        return None

    # start_y: vùng tô của câu 1 (sau đường kẻ, bỏ qua label A,B,C,D,E)
    # Offset ~45% row_spacing để vào vùng tô
    start_y = first_content_row + int(row_spacing * 0.5)

    # Bubble size
    bubble_w = int(col_spacing * 0.6)
    bubble_h = int(row_spacing * 0.35)

    return {
        'start_x': first_bubble_x,
        'start_y': start_y,
        'col_spacing': col_spacing,
        'row_spacing': row_spacing,
        'bubble_w': bubble_w,
        'bubble_h': bubble_h,
        'row_lines': row_lines[:25],
        'col_lines': col_lines
    }


def _grade_mixed_format_sheet(
    image_bytes: bytes,
    answer_key: List[str],
    template_type: str,
    extract_info: bool = True
):
    """Chấm phiếu trả lời có format hỗn hợp (trắc nghiệm + điền đáp án)

    Dùng cho SEAMO Math: 20 câu trắc nghiệm + 5 câu điền đáp án
    """
    import cv2
    import numpy as np

    template = ANSWER_TEMPLATES.get(template_type, ANSWER_TEMPLATES["IKSC_BENJAMIN"])
    num_questions = template["questions"]
    scoring = template["scoring"]
    mixed_format = template.get("mixed_format", None)

    if not mixed_format:
        # Không có mixed format, dùng hàm chấm thông thường
        return _grade_single_sheet(image_bytes, answer_key, template_type, extract_info)

    mcq_count = mixed_format.get("mcq", 20)
    fill_in_count = mixed_format.get("fill_in", 5)

    # Trích xuất thông tin học sinh
    student_info = {}
    if extract_info:
        try:
            student_info = _extract_student_info_ocr(image_bytes)
        except Exception:
            pass

    # Tiền xử lý ảnh
    result = _preprocess_omr_image(image_bytes)
    if result[0] is None:
        return {"error": "Không thể đọc ảnh"}

    original, gray, binary = result

    # ========== PHẦN 1: Chấm 20 câu trắc nghiệm bằng OMR ==========
    # Kiểm tra nếu là SEAMO, sử dụng fixed grid detection
    is_seamo = "SEAMO" in template_type.upper()

    mcq_questions = []
    if is_seamo:
        # SEAMO có layout cố định, dùng fixed grid
        mcq_questions = _detect_seamo_bubbles_fixed_grid(gray)
    else:
        # Các template khác dùng dynamic detection
        mcq_template_type = template_type
        rows, all_rects = _detect_bubbles_grid_based(gray, binary, mcq_template_type)

        if rows and len(rows) >= 2:
            mcq_questions = _group_bubbles_to_questions_improved(rows, mcq_template_type)
            # Chỉ lấy các câu từ 1 đến mcq_count
            mcq_questions = [q for q in mcq_questions if q["index"] <= mcq_count]

    # Chấm phần trắc nghiệm
    option_labels = ["A", "B", "C", "D", "E"]
    student_answers = []
    details = []
    correct_count = 0
    wrong_count = 0
    blank_count = 0

    questions_by_index = {q["index"]: q for q in mcq_questions}

    for q_idx in range(mcq_count):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if q_num not in questions_by_index:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "mcq",
                "fill_ratios": []
            })
            continue

        question = questions_by_index[q_num]
        fill_ratios = []
        max_fill = 0
        selected_option = None

        for opt_idx, bubble in enumerate(question["bubbles"]):
            if opt_idx >= len(option_labels):
                break

            result_fill = _analyze_bubble_fill_improved(gray, bubble)
            if len(result_fill) == 3:
                is_filled, fill_ratio, mean_val = result_fill
            else:
                is_filled, fill_ratio = result_fill

            fill_ratios.append(fill_ratio)

            if fill_ratio > max_fill:
                max_fill = fill_ratio
                if is_filled:
                    selected_option = option_labels[opt_idx]

        if selected_option is None and max_fill > 0.25:
            sorted_ratios = sorted(fill_ratios, reverse=True)
            if len(sorted_ratios) >= 2 and sorted_ratios[0] > sorted_ratios[1] * 1.3:
                selected_option = option_labels[fill_ratios.index(max_fill)]

        if selected_option is None:
            student_answers.append(None)
            blank_count += 1
            status = "blank"
        elif correct_answer and selected_option.upper() == correct_answer.upper():
            student_answers.append(selected_option)
            correct_count += 1
            status = "correct"
        else:
            student_answers.append(selected_option)
            wrong_count += 1
            status = "wrong"

        details.append({
            "q": q_num,
            "student": selected_option,
            "correct": correct_answer,
            "status": status,
            "type": "mcq",
            "fill_ratios": [round(r, 3) for r in fill_ratios]
        })

    # ========== PHẦN 2: Chấm 5 câu điền đáp án bằng OCR ==========
    try:
        reader = _get_easyocr_reader()
    except Exception as e:
        # Nếu không load được OCR, đánh dấu các câu fill-in là not_found
        for q_idx in range(mcq_count, num_questions):
            q_num = q_idx + 1
            correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0,
                "error": f"OCR error: {str(e)}"
            })

        # Tính điểm và trả về
        score = (
            scoring.get("base", 0) +
            correct_count * scoring.get("correct", 1) +
            wrong_count * scoring.get("wrong", 0) +
            blank_count * scoring.get("blank", 0)
        )

        return {
            "answers": student_answers,
            "score": round(score, 2),
            "correct": correct_count,
            "wrong": wrong_count,
            "blank": blank_count,
            "total": num_questions,
            "details": details,
            "student_info": student_info,
            "format": "mixed",
            "mcq_count": mcq_count,
            "fill_in_count": fill_in_count
        }

    # Tìm vùng chứa đáp án điền (thường ở phía dưới phiếu)
    # Phát hiện các ô điền đáp án
    height, width = gray.shape[:2]

    # Giả sử phần điền đáp án nằm ở 1/3 dưới của ảnh
    fill_in_region = gray[int(height * 0.6):, :]

    # Sử dụng OCR để đọc toàn bộ vùng
    try:
        ocr_results = reader.readtext(fill_in_region, detail=1, paragraph=False)
    except Exception:
        ocr_results = []

    # Tìm các đáp án số/chữ
    recognized_fill_ins = []
    for (bbox, text, confidence) in ocr_results:
        text = text.strip()
        # Lọc các text có vẻ là đáp án (số hoặc chữ ngắn)
        if text and len(text) <= 10:
            cx = (bbox[0][0] + bbox[2][0]) / 2
            cy = (bbox[0][1] + bbox[2][1]) / 2
            recognized_fill_ins.append({
                'text': text,
                'confidence': confidence,
                'cx': cx,
                'cy': cy + int(height * 0.6)  # Offset lại vị trí
            })

    # Sắp xếp theo vị trí (trái sang phải, trên xuống dưới)
    recognized_fill_ins.sort(key=lambda r: (r['cy'], r['cx']))

    # Gán đáp án cho các câu fill-in
    for i, q_idx in enumerate(range(mcq_count, num_questions)):
        q_num = q_idx + 1
        correct_answer = answer_key[q_idx] if q_idx < len(answer_key) else None

        if i < len(recognized_fill_ins):
            recognized = recognized_fill_ins[i]['text']
            confidence = recognized_fill_ins[i]['confidence']

            # So sánh đáp án (có thể là số hoặc chữ)
            if correct_answer:
                # Chuẩn hóa để so sánh
                student_norm = recognized.upper().strip()
                correct_norm = str(correct_answer).upper().strip()

                if student_norm == correct_norm:
                    status = "correct"
                    correct_count += 1
                else:
                    status = "wrong"
                    wrong_count += 1
            else:
                status = "unknown"

            student_answers.append(recognized)
            details.append({
                "q": q_num,
                "student": recognized,
                "correct": correct_answer,
                "status": status,
                "type": "fill_in",
                "confidence": round(confidence, 3)
            })
        else:
            student_answers.append(None)
            blank_count += 1
            details.append({
                "q": q_num,
                "student": None,
                "correct": correct_answer,
                "status": "not_found",
                "type": "fill_in",
                "confidence": 0.0
            })

    # Tính điểm
    score = (
        scoring.get("base", 0) +
        correct_count * scoring.get("correct", 1) +
        wrong_count * scoring.get("wrong", 0) +
        blank_count * scoring.get("blank", 0)
    )

    return {
        "answers": student_answers,
        "score": round(score, 2),
        "correct": correct_count,
        "wrong": wrong_count,
        "blank": blank_count,
        "total": num_questions,
        "details": details,
        "student_info": student_info,
        "format": "mixed",
        "mcq_count": mcq_count,
        "fill_in_count": fill_in_count
    }


def _extract_answers_from_text(text: str, num_questions: int) -> dict:
    """Trích xuất đáp án từ text (PDF/Word)"""
    # Hỗ trợ các format: "1. A", "1) A", "1: A", "1 A", "Câu 1: A"
    answer_patterns = [
        r'(?:Câu\s*)?(\d+)\s*[.:)]\s*([A-Ea-e])',  # Câu 1: A, 1. A, 1) A
        r'(\d+)\s+([A-Ea-e])\b',  # 1 A
    ]

    found_answers = {}
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            q_num = int(match[0])
            answer = match[1].upper()
            if 1 <= q_num <= num_questions:
                found_answers[q_num] = answer

    return found_answers



def _parse_answer_key_for_template(answer_file_content: bytes, file_ext: str, template_type: str) -> List[str]:
    """Parse đáp án từ file cho một template cụ thể"""
    from collections import defaultdict

    template = ANSWER_TEMPLATES.get(template_type)
    if not template:
        return []

    num_questions = template["questions"]
    answers = []

    if file_ext in ["xlsx", "xls"]:
        # Đọc từ file Excel
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(answer_file_content))
        ws = wb.active

        for row in ws.iter_rows(min_row=2, max_col=2):
            if row[1].value:
                answers.append(str(row[1].value).strip().upper())

    elif file_ext == "pdf":
        # Đọc từ file PDF
        import fitz  # PyMuPDF

        pdf_doc = fitz.open(stream=answer_file_content, filetype="pdf")
        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text() + "\n"

        found_answers = {}

        # Kiểm tra nếu là file IKLC (Linguistic Kangaroo) với format đặc biệt
        is_iklc_format = "LINGUISTIC KANGAROO" in pdf_text.upper() or all(
            level in pdf_text for level in ["Joey", "Wallaby"]
        )

        if is_iklc_format and "IKLC" in template_type.upper():
            # Parse IKLC PDF với format nhiều cột theo vị trí x
            # Cột: Start (25 câu), Story (30 câu), Joey (50 câu), Wallaby (50 câu), Grey K. (50 câu), Red K. (50 câu)
            iklc_levels = [
                ("Start", 25),      # Pre-Ecolier (Lớp 1-2)
                ("Story", 30),      # Ecolier (Lớp 3-4)
                ("Joey", 50),       # Benjamin (Lớp 5-6)
                ("Wallaby", 50),    # Cadet (Lớp 7-8)
                ("Grey", 50),       # Junior (Lớp 9-10)
                ("Red", 50),        # Student (Lớp 11-12)
            ]

            level_map = {
                "IKLC_PRE_ECOLIER": 0,
                "IKLC_ECOLIER": 1,
                "IKLC_BENJAMIN": 2,
                "IKLC_CADET": 3,
                "IKLC_JUNIOR": 4,
                "IKLC_STUDENT": 5,
            }

            target_level_idx = level_map.get(template_type.upper(), -1)

            if target_level_idx >= 0:
                target_level_name, target_num_q = iklc_levels[target_level_idx]

                # Đọc tất cả text blocks với vị trí từ tất cả các trang
                all_blocks = []
                for page in pdf_doc:
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"].strip()
                                    x0 = span["bbox"][0]
                                    y0 = span["bbox"][1]
                                    if text:
                                        all_blocks.append({"text": text, "x": x0, "y": y0})

                # Tách số câu và đáp án
                numbers = []  # Số câu hỏi (1-50)
                answers_list = []  # Đáp án A-E

                for b in all_blocks:
                    text = b["text"]
                    if text.isdigit() and 1 <= int(text) <= 50:
                        numbers.append({"num": int(text), "x": b["x"], "y": b["y"]})
                    elif len(text) == 1 and text in "ABCDE":
                        answers_list.append({"ans": text, "x": b["x"], "y": b["y"]})
                    elif len(text) >= 1 and text[0] in "ABCDE" and "," in text:
                        # Trường hợp "B, C" -> lấy ký tự đầu
                        answers_list.append({"ans": text[0], "x": b["x"], "y": b["y"]})

                # Tìm vị trí x của số 1 cho mỗi cột (mỗi level bắt đầu từ câu 1)
                ones = [n for n in numbers if n["num"] == 1]
                ones.sort(key=lambda o: o["x"])

                # Có 6 cột (6 số 1), gán level theo thứ tự x
                # ones[0] = Start, ones[1] = Story, ones[2] = Joey, ...
                if len(ones) >= 6:
                    # Xác định x boundaries giữa các cột
                    x_boundaries = []
                    for i in range(len(ones) - 1):
                        mid_x = (ones[i]["x"] + ones[i + 1]["x"]) / 2
                        x_boundaries.append(mid_x)
                    x_boundaries.append(9999)  # Boundary cuối cùng

                    # Hàm xác định cột của một số dựa trên vị trí x
                    def get_column_idx(x):
                        for i, boundary in enumerate(x_boundaries):
                            if x < boundary:
                                return i
                        return len(x_boundaries) - 1

                    # Nhóm số câu theo cột
                    column_numbers = defaultdict(list)
                    for n in numbers:
                        col_idx = get_column_idx(n["x"])
                        column_numbers[col_idx].append(n)

                    # Lấy số câu của cột target
                    target_numbers = column_numbers.get(target_level_idx, [])

                    # Với mỗi số câu, tìm đáp án gần nhất bên phải
                    for num_block in target_numbers:
                        q_num = num_block["num"]
                        q_x = num_block["x"]
                        q_y = num_block["y"]

                        if q_num > target_num_q:
                            continue

                        # Tìm đáp án gần nhất: cùng y (tolerance 3px trước, sau đó 8px) và x lớn hơn số câu
                        best_answer = None
                        best_dist = 9999
                        best_y_diff = 9999

                        for ans_block in answers_list:
                            ans_x = ans_block["x"]
                            ans_y = ans_block["y"]

                            # Đáp án ở bên phải số câu và cùng hàng
                            y_diff = abs(ans_y - q_y)
                            if ans_x > q_x and y_diff < 8:
                                dist = ans_x - q_x
                                if dist < 50:  # Không quá xa
                                    # Ưu tiên đáp án cùng y hơn (y_diff nhỏ hơn)
                                    # Nếu y_diff gần bằng nhau (< 3px), chọn x gần nhất
                                    if y_diff < best_y_diff - 3 or (abs(y_diff - best_y_diff) <= 3 and dist < best_dist):
                                        best_dist = dist
                                        best_y_diff = y_diff
                                        best_answer = ans_block["ans"]

                        if best_answer and q_num not in found_answers:
                            found_answers[q_num] = best_answer

        pdf_doc.close()

        # Fallback: parse đơn giản
        if not found_answers:
            found_answers = _extract_answers_from_text(pdf_text, num_questions)

        for i in range(1, num_questions + 1):
            answers.append(found_answers.get(i, ""))

    elif file_ext in ["docx", "doc"]:
        # Đọc từ file Word
        doc = Document(io.BytesIO(answer_file_content))
        found_answers = {}

        level_keywords = {
            "pre_ecolier": ["preecolier", "pre-ecolier", "pre ecolier", "pre_ecolier"],
            "ecolier": ["ecolier"],
            "benjamin": ["benjamin"],
            "cadet": ["cadet"],
            "junior": ["junior"],
            "student": ["student"],
        }

        for table in doc.tables:
            if len(table.rows) > 1 and len(table.columns) >= 2:
                header = [cell.text.strip().lower() for cell in table.rows[0].cells]

                level_col = -1
                search_keywords = []
                is_ecolier_only = False

                for key, keywords in level_keywords.items():
                    if key in template_type.lower():
                        search_keywords = keywords
                        if key == "ecolier" and "pre" not in template_type.lower():
                            is_ecolier_only = True
                        break

                for col_idx, col_header in enumerate(header):
                    if is_ecolier_only:
                        if col_header == "ecolier" or (col_header.endswith("ecolier") and not col_header.startswith("pre")):
                            level_col = col_idx
                            break
                    else:
                        for keyword in search_keywords:
                            if keyword in col_header:
                                level_col = col_idx
                                break
                    if level_col >= 0:
                        break

                if level_col >= 0:
                    # Detect paired-column format: each level has 2 cols (number, answer)
                    # Check if header has duplicate names (merged cells)
                    is_paired = (level_col + 1 < len(header)
                                 and header[level_col] == header[level_col + 1])
                    if is_paired:
                        # Paired format: level_col = numbers, level_col+1 = answers
                        num_col = level_col
                        ans_col = level_col + 1
                    else:
                        # Standard format: col 0 = numbers, level_col = answers
                        num_col = 0
                        ans_col = level_col

                    for row in table.rows[1:]:
                        try:
                            q_num = int(row.cells[num_col].text.strip())
                            answer = row.cells[ans_col].text.strip().upper()
                            if answer and answer in "ABCDE":
                                found_answers[q_num] = answer
                        except (ValueError, IndexError):
                            continue

                if not found_answers and len(table.columns) == 2:
                    for row in table.rows[1:]:
                        try:
                            q_num = int(row.cells[0].text.strip())
                            answer = row.cells[1].text.strip().upper()
                            if answer and answer in "ABCDE":
                                found_answers[q_num] = answer
                        except (ValueError, IndexError):
                            continue

        if not found_answers:
            doc_text = ""
            for para in doc.paragraphs:
                doc_text += para.text + "\n"
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        doc_text += cell.text + " "
                    doc_text += "\n"
            found_answers = _extract_answers_from_text(doc_text, num_questions)

        for i in range(1, num_questions + 1):
            answers.append(found_answers.get(i, ""))

    return answers
