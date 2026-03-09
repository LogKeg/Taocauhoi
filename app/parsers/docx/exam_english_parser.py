"""
English exam question parser for Word (.docx) documents.
Handles English-language exam formats with tables, dialogues, cloze passages, etc.
"""
import re
from typing import List, Tuple

from docx import Document


# ============================================================================
# Helper functions for English exam question detection
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
# Main parser function
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
