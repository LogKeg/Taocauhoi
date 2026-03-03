"""
Math formula rendering utilities for Word and PDF export.
Converts LaTeX formulas to appropriate formats.
"""
import re
import subprocess
import tempfile
import os
from typing import Tuple, List, Optional
from pathlib import Path
from lxml import etree

# LaTeX delimiter patterns
LATEX_INLINE_RE = re.compile(r'\$([^\$]+)\$')
LATEX_BLOCK_RE = re.compile(r'\$\$([^\$]+)\$\$')

# MathML to OMML XSLT (embedded)
# This is the official Microsoft MML2OMML.XSL transformation
MML2OMML_XSL = None


def _load_mml2omml_xsl():
    """Load MathML to OMML XSLT transformation."""
    global MML2OMML_XSL
    if MML2OMML_XSL is not None:
        return MML2OMML_XSL

    # Try to find the XSLT file from Office installation or use embedded version
    xsl_paths = [
        "/Applications/Microsoft Word.app/Contents/Resources/MML2OMML.XSL",
        "/usr/share/texmf/MML2OMML.XSL",
        str(Path(__file__).parent / "MML2OMML.XSL"),
    ]

    for path in xsl_paths:
        if os.path.exists(path):
            try:
                MML2OMML_XSL = etree.parse(path)
                return MML2OMML_XSL
            except Exception:
                continue

    return None


def has_latex(text: str) -> bool:
    """Check if text contains LaTeX formulas."""
    return bool(LATEX_INLINE_RE.search(text) or LATEX_BLOCK_RE.search(text))


def extract_latex_parts(text: str) -> List[Tuple[str, bool]]:
    """
    Split text into parts, identifying LaTeX formulas.

    Returns list of (text, is_latex) tuples.
    """
    parts = []
    last_end = 0

    # Find all LaTeX patterns
    patterns = []

    # Block math first ($$...$$)
    for m in LATEX_BLOCK_RE.finditer(text):
        patterns.append((m.start(), m.end(), m.group(1), True))

    # Inline math ($...$)
    for m in LATEX_INLINE_RE.finditer(text):
        # Check if not part of block math
        is_block = any(
            p[0] <= m.start() < p[1] or p[0] < m.end() <= p[1]
            for p in patterns
        )
        if not is_block:
            patterns.append((m.start(), m.end(), m.group(1), False))

    # Sort by position
    patterns.sort(key=lambda x: x[0])

    for start, end, latex, is_block in patterns:
        # Add text before this formula
        if start > last_end:
            parts.append((text[last_end:start], False))
        # Add the formula
        parts.append((latex, True))
        last_end = end

    # Add remaining text
    if last_end < len(text):
        parts.append((text[last_end:], False))

    return parts if parts else [(text, False)]


def latex_to_unicode(latex: str) -> str:
    """
    Convert simple LaTeX to Unicode approximation.
    For basic formulas when full rendering is not available.
    """
    # Superscripts
    superscript_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ',
    }

    # Subscripts
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
        'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'n': 'ₙ', 'x': 'ₓ',
    }

    # Common symbols
    symbol_map = {
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\Delta': 'Δ', r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ',
        r'\mu': 'μ', r'\pi': 'π', r'\sigma': 'σ', r'\phi': 'φ', r'\omega': 'ω',
        r'\infty': '∞', r'\pm': '±', r'\times': '×', r'\div': '÷',
        r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\ne': '≠', r'\approx': '≈',
        r'\sqrt': '√', r'\sum': '∑', r'\prod': '∏', r'\int': '∫',
        r'\partial': '∂', r'\nabla': '∇', r'\forall': '∀', r'\exists': '∃',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
        r'\cup': '∪', r'\cap': '∩', r'\emptyset': '∅',
        r'\rightarrow': '→', r'\leftarrow': '←', r'\Rightarrow': '⇒',
        r'\cdot': '·', r'\ldots': '…', r'\cdots': '⋯',
        r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
        r'\log': 'log', r'\ln': 'ln', r'\exp': 'exp',
    }

    result = latex

    # Replace symbols first
    for tex, uni in symbol_map.items():
        result = result.replace(tex, uni)

    # Handle fractions: \frac{a}{b} -> a/b
    frac_re = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
    result = frac_re.sub(r'(\1)/(\2)', result)

    # Handle sqrt: \sqrt{x} -> √x
    sqrt_re = re.compile(r'\\sqrt\{([^}]+)\}')
    result = sqrt_re.sub(r'√(\1)', result)

    # Handle superscripts: x^2 or x^{2n}
    def replace_superscript(m):
        base = m.group(1) if m.group(1) else ''
        exp = m.group(2)
        sup = ''.join(superscript_map.get(c, c) for c in exp)
        return base + sup

    # Simple superscript: x^2
    result = re.sub(r'(\w?)\^(\w)', replace_superscript, result)
    # Braced superscript: x^{2n}
    result = re.sub(r'(\w?)\^\{([^}]+)\}', replace_superscript, result)

    # Handle subscripts: x_1 or x_{12}
    def replace_subscript(m):
        base = m.group(1) if m.group(1) else ''
        sub = m.group(2)
        subs = ''.join(subscript_map.get(c, c) for c in sub)
        return base + subs

    # Simple subscript: x_1
    result = re.sub(r'(\w?)_(\w)', replace_subscript, result)
    # Braced subscript: x_{12}
    result = re.sub(r'(\w?)_\{([^}]+)\}', replace_subscript, result)

    # Clean up remaining braces and backslashes
    result = re.sub(r'\\[a-zA-Z]+', '', result)  # Remove unknown commands
    result = result.replace('{', '').replace('}', '')
    result = result.strip()

    return result


def render_text_with_math(text: str) -> str:
    """
    Render text with LaTeX formulas converted to Unicode.
    Used for plain text output.
    """
    if not has_latex(text):
        return text

    parts = extract_latex_parts(text)
    result = []
    for part, is_latex in parts:
        if is_latex:
            result.append(latex_to_unicode(part))
        else:
            result.append(part)

    return ''.join(result)


# ============================================================================
# Word OMML Support (Office Math Markup Language)
# ============================================================================

def latex_to_mathml(latex: str) -> Optional[str]:
    """Convert LaTeX to MathML string."""
    try:
        import latex2mathml.converter
        mathml = latex2mathml.converter.convert(latex)
        return mathml
    except Exception as e:
        print(f"LaTeX to MathML error: {e}")
        return None


def mathml_to_omml(mathml_str: str) -> Optional[etree._Element]:
    """
    Convert MathML to OMML (Office Math Markup Language).
    Returns an lxml Element that can be inserted into Word document.
    """
    try:
        xsl = _load_mml2omml_xsl()
        if xsl is None:
            return None

        # Parse MathML
        mathml_tree = etree.fromstring(mathml_str.encode('utf-8'))

        # Transform to OMML
        transform = etree.XSLT(xsl)
        omml_tree = transform(mathml_tree)

        return omml_tree.getroot()
    except Exception as e:
        print(f"MathML to OMML error: {e}")
        return None


def latex_to_omml(latex: str) -> Optional[etree._Element]:
    """
    Convert LaTeX directly to OMML element.
    Returns an lxml Element or None if conversion fails.
    """
    mathml = latex_to_mathml(latex)
    if mathml is None:
        return None

    return mathml_to_omml(mathml)


def create_omml_element(latex: str) -> Optional[etree._Element]:
    """
    Create an OMML element from LaTeX that can be added to Word paragraph.
    This creates a proper oMath element with the formula.
    """
    try:
        import latex2mathml.converter

        # Convert LaTeX to MathML
        mathml_str = latex2mathml.converter.convert(latex)

        # Parse MathML
        mathml_tree = etree.fromstring(mathml_str.encode('utf-8'))

        # Create OMML structure manually since we may not have the XSL file
        # This is a simplified conversion for common cases
        omml = _convert_mathml_to_omml_simple(mathml_tree)
        return omml

    except Exception as e:
        print(f"OMML creation error: {e}")
        return None


def _convert_mathml_to_omml_simple(mathml_elem) -> Optional[etree._Element]:
    """
    Simple MathML to OMML converter for common math expressions.
    This handles basic cases without needing the full Microsoft XSL.
    """
    # OMML namespace
    OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    nsmap = {
        'm': OMML_NS,
        'w': WORD_NS,
    }

    def make_elem(tag, text=None):
        elem = etree.Element(f"{{{OMML_NS}}}{tag}", nsmap=nsmap)
        if text:
            elem.text = text
        return elem

    def make_run(text):
        """Create an m:r element with text."""
        r = make_elem('r')
        t = make_elem('t')
        t.text = text
        r.append(t)
        return r

    def convert_children_flat(node):
        """Convert children and return as flat list of elements."""
        results = []
        for child in node:
            converted = convert_node(child)
            if converted is not None:
                # If result is an oMath wrapper, extract its children
                if converted.tag == f"{{{OMML_NS}}}oMath":
                    for sub in converted:
                        results.append(sub)
                else:
                    results.append(converted)
        return results

    def convert_single_flat(node):
        """Convert a single node and return flat list of elements."""
        converted = convert_node(node)
        if converted is None:
            return []
        # If result is an oMath wrapper, extract its children
        if converted.tag == f"{{{OMML_NS}}}oMath":
            return list(converted)
        return [converted]

    def convert_node(node):
        """Recursively convert MathML node to OMML."""
        tag = etree.QName(node.tag).localname if isinstance(node.tag, str) else str(node.tag)

        if tag == 'math':
            omath = make_elem('oMath')
            for elem in convert_children_flat(node):
                omath.append(elem)
            return omath

        elif tag == 'mfrac':
            # Fraction
            f = make_elem('f')
            fpr = make_elem('fPr')
            f.append(fpr)

            children = list(node)
            if len(children) >= 2:
                num = make_elem('num')
                den = make_elem('den')
                # Get flat elements for numerator and denominator
                for elem in convert_single_flat(children[0]):
                    num.append(elem)
                for elem in convert_single_flat(children[1]):
                    den.append(elem)
                f.append(num)
                f.append(den)
            return f

        elif tag == 'msqrt':
            # Square root
            rad = make_elem('rad')
            radPr = make_elem('radPr')
            degHide = make_elem('degHide')
            degHide.set(f"{{{OMML_NS}}}val", "1")
            radPr.append(degHide)
            rad.append(radPr)

            deg = make_elem('deg')
            rad.append(deg)

            e = make_elem('e')
            for elem in convert_children_flat(node):
                e.append(elem)
            rad.append(e)
            return rad

        elif tag == 'msup':
            # Superscript
            sSup = make_elem('sSup')
            sSupPr = make_elem('sSupPr')
            sSup.append(sSupPr)

            children = list(node)
            if len(children) >= 2:
                e = make_elem('e')
                sup = make_elem('sup')
                for elem in convert_single_flat(children[0]):
                    e.append(elem)
                for elem in convert_single_flat(children[1]):
                    sup.append(elem)
                sSup.append(e)
                sSup.append(sup)
            return sSup

        elif tag == 'msub':
            # Subscript
            sSub = make_elem('sSub')
            sSubPr = make_elem('sSubPr')
            sSub.append(sSubPr)

            children = list(node)
            if len(children) >= 2:
                e = make_elem('e')
                sub = make_elem('sub')
                for elem in convert_single_flat(children[0]):
                    e.append(elem)
                for elem in convert_single_flat(children[1]):
                    sub.append(elem)
                sSub.append(e)
                sSub.append(sub)
            return sSub

        elif tag == 'mrow':
            # Group - collect children and flatten them
            children_flat = convert_children_flat(node)
            # Return first child if only one
            if len(children_flat) == 1:
                return children_flat[0]
            # For multiple children, wrap in oMath (will be flattened by parent)
            if len(children_flat) > 1:
                group = make_elem('oMath')
                for c in children_flat:
                    group.append(c)
                return group
            return None

        elif tag in ('mi', 'mn', 'mo', 'mtext'):
            # Identifier, number, operator, text
            text = node.text or ''
            return make_run(text)

        else:
            # Unknown tag - try to convert children
            for child in node:
                return convert_node(child)
            if node.text:
                return make_run(node.text)
            return None

    try:
        return convert_node(mathml_elem)
    except Exception as e:
        print(f"OMML conversion error: {e}")
        return None


def _flatten_omath(elem, ns):
    """
    Flatten nested oMath elements.
    If an oMath contains only another oMath as child, extract the inner children.
    """
    omath_tag = f"{{{ns}}}oMath"

    # Check if this oMath has only one child which is also an oMath
    children = list(elem)
    if len(children) == 1 and children[0].tag == omath_tag:
        # Move grandchildren to this element
        inner = children[0]
        elem.remove(inner)
        for grandchild in list(inner):
            elem.append(grandchild)

    # Recursively flatten children
    for child in list(elem):
        if child.tag == omath_tag:
            _flatten_omath(child, ns)


def add_math_to_paragraph(paragraph, latex: str) -> bool:
    """
    Add a LaTeX formula to a Word paragraph as native OMML equation.
    Returns True if successful, False otherwise.
    """
    OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

    try:
        import latex2mathml.converter
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement

        # Convert LaTeX to MathML
        mathml_str = latex2mathml.converter.convert(latex)

        # Parse MathML
        mathml_tree = etree.fromstring(mathml_str.encode('utf-8'))

        # Convert to OMML
        omml = _convert_mathml_to_omml_simple(mathml_tree)

        if omml is not None:
            # Flatten nested oMath elements
            _flatten_omath(omml, OMML_NS)

            # Add to paragraph
            paragraph._p.append(omml)
            return True

        return False

    except Exception as e:
        print(f"Add math error: {e}")
        return False


# ============================================================================
# PDF Math Rendering
# ============================================================================

def latex_to_image(latex: str, dpi: int = 150, fontsize: int = 14) -> Optional[bytes]:
    """
    Render LaTeX to PNG image using matplotlib.
    Returns PNG bytes or None if rendering fails.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO

        # Configure matplotlib for better math rendering
        plt.rcParams['mathtext.fontset'] = 'cm'  # Computer Modern fonts
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis('off')

        # Render LaTeX
        text = ax.text(0.5, 0.5, f'${latex}$',
                      fontsize=fontsize,
                      ha='center', va='center',
                      transform=ax.transAxes)

        # Get bounding box
        fig.canvas.draw()
        bbox = text.get_window_extent()

        # Resize figure to fit text
        width = bbox.width / fig.dpi + 0.1
        height = bbox.height / fig.dpi + 0.1
        fig.set_size_inches(width, height)

        # Save to bytes
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi,
                   bbox_inches='tight', pad_inches=0.02,
                   transparent=True)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        print(f"LaTeX rendering error: {e}")
        return None


def get_latex_image_size(latex: str, fontsize: int = 14) -> Tuple[float, float]:
    """
    Get the rendered size of a LaTeX formula in points.
    Returns (width, height) tuple.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'

        fig, ax = plt.subplots(figsize=(1, 1))
        ax.axis('off')

        text = ax.text(0, 0, f'${latex}$', fontsize=fontsize)
        fig.canvas.draw()
        bbox = text.get_window_extent()
        plt.close(fig)

        # Convert to points (72 points per inch)
        width_pt = bbox.width * 72 / fig.dpi
        height_pt = bbox.height * 72 / fig.dpi
        return (width_pt, height_pt)

    except Exception:
        return (50, 14)  # Default fallback size


def render_latex_for_pdf(canvas_obj, latex: str, x: float, y: float,
                         fontsize: int = 12, max_width: float = 400) -> float:
    """
    Render LaTeX formula directly on a ReportLab canvas.
    Returns the width of the rendered formula.

    Uses drawImage for complex formulas, drawString for simple ones.
    """
    from io import BytesIO
    from reportlab.lib.utils import ImageReader

    # For very simple expressions (just numbers/letters), use text
    if re.match(r'^[a-zA-Z0-9\s\+\-\=\(\)]+$', latex):
        canvas_obj.drawString(x, y, latex)
        return len(latex) * fontsize * 0.6

    # Render as image
    img_bytes = latex_to_image(latex, dpi=150, fontsize=fontsize)
    if img_bytes:
        try:
            img = ImageReader(BytesIO(img_bytes))
            img_width, img_height = img.getSize()

            # Scale to fit
            scale = min(1.0, max_width / img_width)
            draw_width = img_width * scale * 0.5  # Adjust scale factor
            draw_height = img_height * scale * 0.5

            # Center vertically relative to text baseline
            y_offset = (draw_height - fontsize) / 2

            canvas_obj.drawImage(img, x, y - y_offset,
                               width=draw_width, height=draw_height,
                               mask='auto')
            return draw_width
        except Exception as e:
            print(f"PDF image draw error: {e}")

    # Fallback to Unicode
    unicode_text = latex_to_unicode(latex)
    canvas_obj.drawString(x, y, unicode_text)
    return len(unicode_text) * fontsize * 0.6
