"""
Parser for Math exam questions with "Question X." format.
Handles bilingual (English + Vietnamese) content and MCQ options A/B/C/D.
"""

import re
from typing import List


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
