"""
DOCX content extraction utilities.
Handles paragraphs, tables, math formulas (OMML), and text boxes.
"""
import re
from typing import List
from docx import Document

from app.services.math import omml_to_latex, omml_children_to_latex

# Namespaces
MATH_NS = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
OMML_NS = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}


def extract_paragraph_with_math(para, use_latex: bool = False) -> str:
    """
    Extract text from a paragraph, including OMML math formulas.

    python-docx's para.text doesn't include math formulas (OMML).
    This function extracts both regular text and math formula text.

    Args:
        para: A python-docx paragraph object
        use_latex: If True, convert math formulas to LaTeX notation
    """
    result_parts = []

    # Iterate through all child elements in order
    for child in para._element:
        tag = child.tag

        # Regular text run
        if tag == f'{WORD_NS}r':
            # Get text from <w:t> elements
            for t_elem in child.iter(f'{WORD_NS}t'):
                if t_elem.text:
                    result_parts.append(t_elem.text)

        # Math formula (OMML)
        elif tag == f'{MATH_NS}oMath' or tag == f'{MATH_NS}oMathPara':
            if use_latex:
                # Convert OMML to LaTeX
                latex = omml_children_to_latex(child, OMML_NS)
                if latex:
                    result_parts.append(f'${latex}$')
            else:
                # Extract plain text from math formula
                math_texts = []
                for m_t in child.iter(f'{MATH_NS}t'):
                    if m_t.text:
                        math_texts.append(m_t.text)
                if math_texts:
                    result_parts.append(''.join(math_texts))

    return ''.join(result_parts).strip()


def extract_cell_with_math(cell, use_latex: bool = False) -> str:
    """
    Extract text from a table cell, including OMML math formulas.

    Args:
        cell: A python-docx table cell object
        use_latex: If True, convert math formulas to LaTeX notation
    """
    cell_parts = []
    for para in cell.paragraphs:
        para_text = extract_paragraph_with_math(para, use_latex=use_latex)
        if para_text:
            cell_parts.append(para_text)
    return '\n'.join(cell_parts)


def extract_docx_content(doc: Document, include_textboxes: bool = True, use_latex: bool = False) -> str:
    """
    Extract all text content from a Word document.
    Handles paragraphs, tables, and optionally text boxes.

    This is the unified function for reading Word documents across all features.

    Args:
        doc: A python-docx Document object
        include_textboxes: If True, extract text from text boxes
        use_latex: If True, convert math formulas to LaTeX notation
    """
    all_text = []
    seen_text = set()  # To avoid duplicates

    def add_text(text: str):
        """Add text if not duplicate"""
        text = text.strip()
        if text and text not in seen_text:
            seen_text.add(text)
            all_text.append(text)

    # 1. Read paragraphs (including text boxes if enabled)
    for para in doc.paragraphs:
        # Get paragraph text including math formulas
        text = extract_paragraph_with_math(para, use_latex=use_latex)
        if text:
            add_text(text)

        # Extract text from text boxes (embedded in paragraphs)
        if include_textboxes:
            try:
                xml = para._element.xml
                if 'w:txbxContent' in xml:
                    txbx_matches = re.findall(r'<w:txbxContent[^>]*>(.*?)</w:txbxContent>', xml, re.DOTALL)
                    for txbx in txbx_matches:
                        texts = re.findall(r'<w:t[^>]*>([^<]+)</w:t>', txbx)
                        content = ''.join(texts).strip()
                        if content:
                            add_text(content)
            except Exception:
                pass  # Ignore textbox extraction errors

    # 2. Read tables (important for Math exams which store questions in tables)
    for table in doc.tables:
        for row in table.rows:
            row_texts = []
            for cell in row.cells:
                cell_text = extract_cell_with_math(cell, use_latex=use_latex)
                if cell_text:
                    row_texts.append(cell_text)

            if row_texts:
                # Join cells with newline to separate different content in the same row
                row_content = "\n".join(row_texts)
                add_text(row_content)

    # Join with double newline to separate questions/sections
    return "\n\n".join(all_text)


def extract_docx_lines(doc: Document) -> List[str]:
    """
    Extract all lines from a Word document.
    Returns list of non-empty lines.
    """
    content = extract_docx_content(doc, include_textboxes=True, use_latex=False)
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    return lines
