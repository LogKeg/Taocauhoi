"""
Math exam parser for cell-based question formats.
Used for ASMO Science exams where each table cell contains a complete question.
"""
import re
from typing import List
from docx import Document
from docx.oxml.ns import qn


def has_highlight(element) -> bool:
    """Check if an XML element has highlight or shading (background color)."""
    # Check for w:highlight (text highlight like yellow, green, etc.)
    highlights = element.findall('.//' + qn('w:highlight'))
    for hl in highlights:
        val = hl.get(qn('w:val'))
        if val and val != 'none':
            return True

    # Check for w:shd (cell/paragraph shading)
    shadings = element.findall('.//' + qn('w:shd'))
    for shd in shadings:
        fill = shd.get(qn('w:fill'))
        # Has background color if fill is not empty/auto/white
        if fill and fill not in ('', 'auto', 'FFFFFF', 'ffffff', 'none'):
            return True

    return False


def get_highlighted_option_from_cell(cell_element) -> int:
    """
    Find which option (1-5 for A-E) is highlighted in a cell.
    Returns 0 if no option is highlighted.
    """
    # Look for nested table with options (2x2 or similar grid)
    nested_tables = cell_element.findall('.//' + qn('w:tbl'))
    if nested_tables:
        option_idx = 0
        for nt in nested_tables:
            for tr in nt.findall('.//' + qn('w:tr')):
                for tc in tr.findall('.//' + qn('w:tc')):
                    option_idx += 1
                    # Check if this option cell has highlight/shading
                    if has_highlight(tc):
                        return option_idx
        return 0

    # No nested table - check paragraphs for A., B., C. style options
    paragraphs = cell_element.findall('.//' + qn('w:p'))
    option_idx = 0
    for p in paragraphs:
        # Get paragraph text
        t_elements = p.findall('.//' + qn('w:t'))
        p_text = ''.join([t.text or '' for t in t_elements]).strip()

        # Check if this is an option line (A., B., C., etc.)
        opt_match = re.match(r'^([A-E])\.\s*', p_text)
        if opt_match:
            option_letter = opt_match.group(1)
            option_idx = ord(option_letter) - ord('A') + 1

            # Check if this paragraph has highlight
            if has_highlight(p):
                return option_idx

    return 0


def get_cell_text_from_row(row) -> str:
    """Get text from first cell, handling edge cases where row.cells is empty.
    Also extracts options from nested tables (2x2 grid format)."""
    # First try normal cell access
    if row.cells:
        cell = row.cells[0]
        # Check for nested tables with options
        nested_tables = cell._element.findall('.//' + qn('w:tbl'))
        if nested_tables:
            # Get main paragraphs (before nested table)
            main_paras = []
            for p in cell.paragraphs:
                p_text = p.text.strip()
                if p_text:
                    main_paras.append(p_text)
            # Get options from nested table
            options = []
            for nt in nested_tables:
                for tr in nt.findall('.//' + qn('w:tr')):
                    for tc in tr.findall('.//' + qn('w:tc')):
                        t_elements = tc.findall('.//' + qn('w:t'))
                        cell_text = ''.join([t.text or '' for t in t_elements]).strip()
                        if cell_text:
                            options.append(cell_text)
            # Format: question lines + option lines
            if main_paras and options:
                return '\n'.join(main_paras + options)
        return cell.text.strip()

    # Fallback: extract directly from XML, preserving paragraph breaks
    tc_elements = row._tr.findall(qn('w:tc'))
    if tc_elements:
        tc = tc_elements[0]
        # Check for nested tables
        nested_tbls = tc.findall('.//' + qn('w:tbl'))
        if nested_tbls:
            # Get text from main paragraphs (not in nested table)
            main_paras = []
            for p in tc.findall(qn('w:p')):  # Direct children only
                t_elements = p.findall('.//' + qn('w:t'))
                p_text = ''.join([t.text or '' for t in t_elements]).strip()
                if p_text:
                    main_paras.append(p_text)
            # Get options from nested table
            options = []
            for nt in nested_tbls:
                for tr in nt.findall('.//' + qn('w:tr')):
                    for tc_inner in tr.findall('.//' + qn('w:tc')):
                        t_elements = tc_inner.findall('.//' + qn('w:t'))
                        cell_text = ''.join([t.text or '' for t in t_elements]).strip()
                        if cell_text:
                            options.append(cell_text)
            if main_paras and options:
                return '\n'.join(main_paras + options)

        # No nested table - get all paragraphs
        p_elements = tc.findall('.//' + qn('w:p'))
        paragraphs = []
        for p in p_elements:
            t_elements = p.findall('.//' + qn('w:t'))
            p_text = ''.join([t.text or '' for t in t_elements]).strip()
            if p_text:
                paragraphs.append(p_text)
        return '\n'.join(paragraphs)
    return ''


def parse_cell_based_questions(doc: Document) -> List[dict]:
    """
    Parse questions from documents where each table cell contains a complete question.
    This format is used in ASMO Science exams where:
    - Each row has 1 cell
    - Each cell contains: Question EN, Question VN, Options A-E (or fill-in-blank)

    Options may be:
    - Prefixed: A. option / B. option
    - Non-prefixed bilingual: lines with "/" separator (EN/VN)
    - Non-prefixed simple: short answer options like "20 m", "40 m"

    Correct answer detection:
    - Highlighted option (background color/shading) = correct answer
    - A=1, B=2, C=3, D=4, E=5

    Returns list of parsed questions, or empty list if format doesn't match.
    """
    questions = []

    if not doc.tables:
        return []

    # Check if this is the expected format: table with multiple rows
    # Accept 1-2 columns (some exams have 2 columns but we only use the first)
    table = doc.tables[0]
    if len(table.columns) > 2 or len(table.rows) < 5:
        return []

    def is_bilingual_separator(line: str) -> bool:
        """Check if line contains bilingual separator (/ with or without spaces)."""
        return '/' in line and len(line) > 5

    # Check first few cells to see if it matches expected format
    has_valid_format = False
    for row in table.rows[:3]:
        cell_text = get_cell_text_from_row(row)
        if not cell_text:
            continue
        lines = [l.strip() for l in cell_text.split('\n') if l.strip()]
        # Valid if: has A. style options
        if re.search(r'\n[A-E]\.\s+', cell_text):
            has_valid_format = True
            break
        # Check for bilingual lines (with / separator - may or may not have spaces)
        bilingual_lines = [l for l in lines if '/' in l and len(l) > 5]
        if len(bilingual_lines) >= 3:
            has_valid_format = True
            break
        # Check for question structure: 2+ lines ending with ?, then 3+ short answer lines
        question_ends = [i for i, l in enumerate(lines) if l.endswith('?')]
        if question_ends and len(lines) - question_ends[-1] - 1 >= 3:
            has_valid_format = True
            break
    if not has_valid_format:
        return []

    # Parse each cell as a complete question
    for row in table.rows:
        cell_text = get_cell_text_from_row(row)
        if not cell_text:
            continue

        lines = [l.strip() for l in cell_text.split('\n') if l.strip()]
        if len(lines) < 2:  # Need at least 2 lines (EN + VN for fill-blank)
            continue

        # Detect highlighted option (correct answer)
        highlighted_option = 0
        if row.cells:
            cell_element = row.cells[0]._element
            highlighted_option = get_highlighted_option_from_cell(cell_element)
        else:
            tc_elements = row._tr.findall(qn('w:tc'))
            if tc_elements:
                highlighted_option = get_highlighted_option_from_cell(tc_elements[0])

        question_lines = []
        options = []

        # First pass: try to find A., B., C. style options
        has_prefixed_options = any(re.match(r'^[A-E]\.\s+', line) for line in lines)

        if has_prefixed_options:
            for line in lines:
                opt_match = re.match(r'^([A-E])\.\s+(.+)$', line)
                if opt_match:
                    options.append(opt_match.group(2))
                else:
                    question_lines.append(line)
        else:
            # Non-prefixed format: find where question ends and options begin
            # Check if we have bilingual options (lines with "/" that are shorter)
            bilingual_option_lines = []
            for i, line in enumerate(lines):
                # Options typically: shorter than question, contain "/" for bilingual
                if '/' in line and len(line) < 120 and i >= 2:
                    bilingual_option_lines.append(i)

            if len(bilingual_option_lines) >= 3:
                # Found bilingual options - split at first option
                first_opt_idx = bilingual_option_lines[0]
                question_lines = lines[:first_opt_idx]
                options = lines[first_opt_idx:]
            else:
                # Find the last line that looks like a question
                last_question_idx = -1
                for i, line in enumerate(lines):
                    if line.endswith('?') or (len(line) > 100 and i < len(lines) - 3):
                        last_question_idx = i

                if last_question_idx >= 0 and last_question_idx < len(lines) - 2:
                    question_lines = lines[:last_question_idx + 1]
                    options = lines[last_question_idx + 1:]
                else:
                    # Fallback: check for bilingual options with "/"
                    for line in lines:
                        if options:
                            if is_bilingual_separator(line) or (len(line) < 50 and not line.endswith('?')):
                                options.append(line)
                            else:
                                question_lines.append(line)
                        elif is_bilingual_separator(line) and not line.endswith('?'):
                            options.append(line)
                        else:
                            question_lines.append(line)

        # Accept questions with options OR fill-in-blank questions
        all_text = '\n'.join(lines)
        is_fill_blank = '___' in all_text or '________' in all_text

        if question_lines and options:
            q_dict = {
                "question": '\n'.join(question_lines),
                "options": options
            }
            if highlighted_option > 0:
                q_dict["answer"] = str(highlighted_option)
            questions.append(q_dict)
        elif is_fill_blank and len(lines) >= 2:
            questions.append({
                "question": all_text,
                "options": []
            })

    return questions
