"""
Bilingual question parser for Word documents.
Handles EN-VIE format and various option formats.
"""
import re
from typing import List

from .extractor import extract_paragraph_with_math, extract_cell_with_math


def extract_docx_lines(doc, include_textboxes: bool = True, use_latex: bool = False) -> tuple:
    """
    Extract all text lines from a Word document as a list.
    For Math exams, preserves line breaks within cells to keep options separate.

    Args:
        doc: A python-docx Document object
        include_textboxes: If True, extract text from text boxes
        use_latex: If True, convert math formulas to LaTeX notation

    Returns:
        tuple: (lines, table_options) where table_options is a list of option lists from standalone tables
    """
    all_lines = []
    seen_text = set()
    table_options = []
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    def add_line(text: str):
        text = text.strip()
        if not text:
            return
        # Allow duplicate option lines
        is_option_line = ('A)' in text and 'B)' in text) or re.match(r'^[A-E]\s*[.\)]\s*\S', text)
        if is_option_line or text not in seen_text:
            if not is_option_line:
                seen_text.add(text)
            all_lines.append(text)

    def add_multiline_text(text: str):
        """Add text that may contain multiple lines (from table cells)."""
        text = text.strip()
        if not text:
            return
        for line in text.split('\n'):
            line = line.strip()
            if line:
                add_line(line)

    def extract_nested_table_options(cell) -> str:
        """Extract options from nested tables within a cell (2x2 grid format)."""
        nested_tables = cell._element.findall('.//w:tbl', ns)
        options = []
        for nt in nested_tables:
            rows = nt.findall('.//w:tr', ns)
            for row_elem in rows:
                cells = row_elem.findall('.//w:tc', ns)
                for cell_elem in cells:
                    t_elems = cell_elem.findall('.//w:t', ns)
                    cell_text = ''.join([t.text or '' for t in t_elems]).strip()
                    if cell_text:
                        options.append(cell_text)
        if len(options) >= 2:
            labels = ['A', 'B', 'C', 'D', 'E']
            formatted = '\t'.join([f"{labels[i]}) {opt}" for i, opt in enumerate(options[:5])])
            return formatted
        return ''

    # Read paragraphs
    for para in doc.paragraphs:
        text = extract_paragraph_with_math(para, use_latex=use_latex)
        if text:
            add_line(text)

        # Text boxes
        if include_textboxes:
            try:
                xml = para._element.xml
                if 'w:txbxContent' in xml:
                    txbx_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
                    for txbx in txbx_matches:
                        texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', txbx)
                        content = ''.join(texts).strip()
                        if content:
                            add_line(content)
            except Exception:
                pass

    # Read tables
    for table in doc.tables:
        # Check if this table is a standalone options table
        is_options_table = False
        if len(table.rows) == 1 and len(table.columns) >= 4:
            first_row_texts = [cell.text.strip() for cell in table.rows[0].cells]
            if (first_row_texts and
                first_row_texts[0].startswith('A.') and
                len(first_row_texts) >= 2 and first_row_texts[1].startswith('B.')):
                is_options_table = True
                opts = []
                for cell_text in first_row_texts:
                    opt_text = re.sub(r'^[A-E]\.\s*', '', cell_text).strip()
                    if opt_text:
                        opts.append(opt_text)
                if opts:
                    table_options.append(opts)

        if not is_options_table:
            for row in table.rows:
                for cell in row.cells:
                    cell_text = extract_cell_with_math(cell, use_latex=use_latex)
                    if cell_text:
                        add_multiline_text(cell_text)

                    nested_opts = extract_nested_table_options(cell)
                    if nested_opts:
                        add_line(nested_opts)

    return all_lines, table_options


def parse_bilingual_questions(lines: List[str], table_options: List[List[str]] = None) -> List[dict]:
    """
    Parse questions from Word document lines.
    Returns list of dicts with 'question' and 'options' keys.

    Args:
        lines: List of text lines from document
        table_options: Optional list of option lists from standalone tables
    """
    questions = []
    table_options = table_options or []
    table_option_idx = 0

    # Patterns
    question_num_pattern = re.compile(r'^\s*(\d+)\s*[.\)]\s*(.*)$')
    section_header_pattern = re.compile(r'^Section\s+[A-Z]\s*:', re.IGNORECASE)

    def extract_options_from_line(line: str) -> List[str]:
        """Extract options from a line with A) B) C) D) markers or EN-VIE format."""
        opts = []
        marker_pattern = re.compile(r'(?:^|(?<=\s)|(?<=\t))([A-E])\s*[.\)]\s*(?=\S)', re.IGNORECASE)
        markers = list(marker_pattern.finditer(line))
        if markers and len(markers) >= 2:
            for idx, m in enumerate(markers):
                start = m.end()
                if idx + 1 < len(markers):
                    end = markers[idx + 1].start()
                else:
                    end = len(line)
                opt_text = line[start:end].strip().rstrip('\t ')
                if opt_text and opt_text not in ['...', '..', '.', '…']:
                    opts.append(opt_text)
        elif '\t' in line and re.search(r'B\s*\)', line, re.IGNORECASE):
            b_markers = list(re.finditer(r'(?:^|(?<=\t))([B-E])\s*\)\s*', line, re.IGNORECASE))
            if b_markers:
                first_b = b_markers[0]
                opt_a = line[:first_b.start()].strip().rstrip('\t ')
                if opt_a and opt_a not in ['...', '..', '.', '…']:
                    opts.append(opt_a)
                for idx, m in enumerate(b_markers):
                    start = m.end()
                    if idx + 1 < len(b_markers):
                        end = b_markers[idx + 1].start()
                    else:
                        end = len(line)
                    opt_text = line[start:end].strip().rstrip('\t ')
                    if opt_text and opt_text not in ['...', '..', '.', '…']:
                        opts.append(opt_text)
        return opts

    def is_option_line(line: str) -> bool:
        """Check if a line looks like options."""
        if re.match(r'^A\s*[.\)]', line, re.IGNORECASE):
            return True
        if '\t' in line and re.search(r'A\s*[.\)].*B\s*[.\)]', line, re.IGNORECASE):
            return True
        if '\t' in line and re.search(r'B\s*\)', line, re.IGNORECASE):
            markers = re.findall(r'[B-E]\s*\)', line, re.IGNORECASE)
            if len(markers) >= 2:
                return True
        return False

    def has_fill_blank(text: str) -> bool:
        """Check if text contains fill-in-blank marker."""
        return '…' in text or '...' in text or '___' in text

    def is_reading_passage(text: str) -> bool:
        """Check if text is a long reading passage."""
        return len(text) > 150 and not is_option_line(text) and not has_fill_blank(text)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Skip section headers
        if section_header_pattern.match(line):
            i += 1
            continue

        # Skip standalone option lines
        if is_option_line(line):
            i += 1
            continue

        # Get next non-empty lines for look-ahead
        next_nonempty_lines = []
        for k in range(i + 1, min(i + 10, len(lines))):
            if lines[k].strip():
                next_nonempty_lines.append(lines[k].strip())
                if len(next_nonempty_lines) >= 5:
                    break

        # Skip single option markers
        single_option_only = re.match(r'^[A-E]\s*[.\)]\s*\S+$', line, re.IGNORECASE) and len(line) < 30
        if single_option_only:
            i += 1
            continue

        # Skip standalone numbers
        if re.match(r'^\d+$', line):
            i += 1
            continue

        # Check for question detection
        q_match = question_num_pattern.match(line)
        is_fill_blank = has_fill_blank(line) and not is_option_line(line) and q_match
        is_direct_question = line.endswith('?') and len(line) > 15

        is_question_stem = False
        if len(line) < 200 and line and line[0].isupper():
            if next_nonempty_lines and is_option_line(next_nonempty_lines[0]):
                if q_match or is_direct_question:
                    is_question_stem = True

        if q_match or is_fill_blank or is_question_stem or is_direct_question:
            question_text_parts = []
            options = []

            # Check for cloze passage
            numbered_blank_pattern = re.compile(r'\((\d+)\)')
            numbered_blanks = numbered_blank_pattern.findall(line)
            has_blank_markers = '___' in line or ('...' in line and not line.strip().endswith('...'))
            is_cloze_passage = len(numbered_blanks) >= 2 and has_blank_markers

            if q_match:
                q_content = q_match.group(2).strip()
                if q_content:
                    line_opts = extract_options_from_line(q_content)
                    if line_opts:
                        question_text_parts.append(f"{q_match.group(1)}. ...")
                        options.extend(line_opts)
                    else:
                        question_text_parts.append(q_content)
            else:
                question_text_parts.append(line)

            # Handle cloze passages
            if is_cloze_passage:
                j = i + 1
                all_options = []
                while j < len(lines) and is_option_line(lines[j].strip()):
                    line_opts = extract_options_from_line(lines[j].strip())
                    if line_opts:
                        all_options.append(line_opts)
                    j += 1

                passage_text = line
                for idx, blank_num in enumerate(numbered_blanks):
                    if idx < len(all_options):
                        questions.append({
                            "question": f"Cloze ({blank_num}): {passage_text[:100]}...",
                            "options": all_options[idx]
                        })
                i = j
                continue

            # Collect subsequent lines until next question
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue

                if section_header_pattern.match(next_line):
                    break

                if question_num_pattern.match(next_line):
                    break

                if is_option_line(next_line):
                    line_opts = extract_options_from_line(next_line)
                    if line_opts:
                        options.extend(line_opts)
                    j += 1
                    if len(options) >= 5:
                        break
                    continue

                single_opt_match = re.match(r'^([A-E])\s*[.\)]\s*(.+)$', next_line, re.IGNORECASE)
                if single_opt_match:
                    opt_text = single_opt_match.group(2).strip()
                    if opt_text:
                        options.append(opt_text)
                    j += 1
                    if len(options) >= 5:
                        break
                    continue

                if (has_fill_blank(next_line) or is_reading_passage(next_line)) and options:
                    break

                if len(next_line) > 100 and options:
                    break

                if not options:
                    question_text_parts.append(next_line)
                j += 1

            question_text = "\n".join(question_text_parts)

            # Use table options if no options found
            if not options and table_option_idx < len(table_options):
                options = table_options[table_option_idx]
                table_option_idx += 1

            if question_text.strip():
                questions.append({
                    "question": question_text.strip(),
                    "options": options,
                    "number": int(q_match.group(1)) if q_match else None
                })

            i = j
        else:
            i += 1

    return questions
