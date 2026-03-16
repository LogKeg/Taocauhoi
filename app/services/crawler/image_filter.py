"""
Helper functions to filter out formula/LaTeX images and keep only real images
(diagrams, charts, photos, etc.)

Strategy: Only keep images when the question text indicates it needs a visual
(e.g., "hình vẽ", "đồ thị", "bảng biến thiên", "như hình").
"""
import re
from typing import Optional
from urllib.parse import urlparse
from bs4 import Tag


# Keywords that indicate the question genuinely needs an image
# (not just a formula render)
IMAGE_REQUIRED_KEYWORDS = [
    'hình vẽ',
    'hình bên',
    'như hình',
    'hình sau',
    'hình dưới',
    'đồ thị',
    'biểu đồ',
    'bảng biến thiên',
    'bảng xét dấu',
    'sơ đồ',
    'mặt phẳng',
    'hệ trục',
    'trục tọa độ',
    'hình chữ nhật',
    'hình vuông',
    'hình tròn',
    'tam giác',
    'tứ giác',
    'hình hộp',
    'hình lăng trụ',
    'hình chóp',
    'hình cầu',
    'hình trụ',
    'hình nón',
    'mặt cầu',
    'khối cầu',
    'khối hộp',
    'khối chóp',
    'khối lăng trụ',
    'phần gạch',
    'phần tô màu',
    'vùng gạch',
]

# Compiled regex for faster matching
IMAGE_REQUIRED_PATTERN = re.compile(
    '|'.join(re.escape(kw) for kw in IMAGE_REQUIRED_KEYWORDS),
    re.IGNORECASE
)


def question_needs_image(question_text: str) -> bool:
    """
    Check if a question genuinely needs an image based on its text content.

    Args:
        question_text: The question content text

    Returns:
        True if the question mentions visual elements that require an image
    """
    if not question_text:
        return False
    return bool(IMAGE_REQUIRED_PATTERN.search(question_text))


# URL patterns that indicate formula/math images (render services)
FORMULA_URL_PATTERNS = [
    'latex.codecogs.com',
    'latex2png',
    'mathjax',
    'katex',
    'tex.s2cms',
    'mimetex',
    'mathurl',
    'quicklatex',
    'latex4technics',
    'sciweavers.org/tex2img',
    'chart.googleapis.com/chart?cht=tx',  # Google Chart LaTeX
    'i.upmath.me',
    'render.githubusercontent.com',
]

# URL path keywords that indicate formula images
FORMULA_PATH_PATTERNS = [
    '/latex/',
    '/tex/',
    '/math/',
    '/formula/',
    '/equation/',
]

# Image class/attribute patterns indicating math content
FORMULA_CLASS_PATTERNS = [
    'mathjax',
    'latex',
    'katex',
    'math',
    'tex',
    'mjx',
    'equation',
]

# Alt text patterns indicating formula (LaTeX-like content)
FORMULA_ALT_PATTERNS = [
    r'\\frac',
    r'\\sqrt',
    r'\\sum',
    r'\\int',
    r'\\alpha',
    r'\\beta',
    r'\\gamma',
    r'\\delta',
    r'\\theta',
    r'\\pi',
    r'\\infty',
    r'\\lim',
    r'\\begin{',
    r'\\end{',
    r'\\left',
    r'\\right',
    r'\\cdot',
    r'\\times',
    r'\\div',
    r'\\pm',
    r'\\leq',
    r'\\geq',
    r'\\neq',
    r'\\approx',
    r'\\equiv',
    r'\$',  # Dollar signs often indicate LaTeX
]

# Skip patterns for non-content images
SKIP_URL_PATTERNS = [
    'icon',
    'avatar',
    'logo',
    'placeholder',
    'loading',
    'pixel',
    'emoji',
    'button',
    'banner',
    'ad-',
    'ads/',
    'advertisement',
    'tracking',
    'analytics',
]


def is_formula_image(img: Tag) -> bool:
    """
    Detect if an image is a formula/math render rather than a real image.

    Args:
        img: BeautifulSoup img tag

    Returns:
        True if this appears to be a formula image, False otherwise
    """
    if not img:
        return False

    src = img.get('src', '').lower()
    alt = img.get('alt', '').lower()
    title = img.get('title', '').lower()
    classes = ' '.join(img.get('class', [])).lower()
    data_latex = img.get('data-latex', '')
    data_formula = img.get('data-formula', '')

    # Check URL for formula render services
    for pattern in FORMULA_URL_PATTERNS:
        if pattern.lower() in src:
            return True

    # Check URL path patterns
    for pattern in FORMULA_PATH_PATTERNS:
        if pattern in src:
            return True

    # Check class names
    for pattern in FORMULA_CLASS_PATTERNS:
        if pattern in classes:
            return True

    # Check data attributes
    if data_latex or data_formula:
        return True

    # Check alt text for LaTeX patterns
    combined_text = f"{alt} {title}"
    for pattern in FORMULA_ALT_PATTERNS:
        if pattern.lower() in combined_text:
            return True

    # Check for common LaTeX delimiters in alt
    if any(delim in alt for delim in ['$$', '\\(', '\\)', '\\[', '\\]']):
        return True

    return False


def should_skip_image(src: str) -> bool:
    """
    Check if an image URL should be skipped (icons, ads, tracking, etc.)

    Args:
        src: Image source URL

    Returns:
        True if this image should be skipped
    """
    src_lower = src.lower()

    for pattern in SKIP_URL_PATTERNS:
        if pattern in src_lower:
            return True

    return False


def is_meaningful_image(img: Tag, min_width: int = 100, min_height: int = 50) -> bool:
    """
    Determine if an image is a meaningful content image (diagram, chart, photo).

    Args:
        img: BeautifulSoup img tag
        min_width: Minimum width to consider (default 100px for real images)
        min_height: Minimum height to consider (default 50px)

    Returns:
        True if this is a meaningful image worth keeping
    """
    if not img:
        return False

    src = img.get('src', '')
    if not src:
        return False

    # Skip non-content images
    if should_skip_image(src):
        return False

    # Skip formula images
    if is_formula_image(img):
        return False

    # Check size if available
    width = img.get('width', '')
    height = img.get('height', '')

    if width and height:
        try:
            w = int(str(width).replace('px', ''))
            h = int(str(height).replace('px', ''))
            if w < min_width or h < min_height:
                return False
        except (ValueError, TypeError):
            pass

    # Check for inline formula patterns (very narrow width, small height)
    if width:
        try:
            w = int(str(width).replace('px', ''))
            # Inline formulas are typically narrow
            if w < 30:
                return False
        except (ValueError, TypeError):
            pass

    return True


def extract_meaningful_image_url(
    element,
    base_url: str = "",
    question_text: str = ""
) -> Optional[str]:
    """
    Extract the first meaningful (non-formula) image URL from an element.

    IMPORTANT: Only returns an image URL if the question text indicates
    that a visual is genuinely needed (e.g., mentions "hình vẽ", "đồ thị").

    Args:
        element: BeautifulSoup element to search
        base_url: Base URL for resolving relative URLs
        question_text: The question content text to check for image keywords

    Returns:
        Absolute URL of meaningful image, or None
    """
    from urllib.parse import urljoin

    if not element:
        return None

    # CRITICAL: Only look for images if the question text indicates
    # it genuinely needs a visual (not just formula images)
    if question_text and not question_needs_image(question_text):
        return None

    # Find all images and check each
    for img in element.find_all('img'):
        if not is_meaningful_image(img):
            continue

        src = img.get('src', '')
        if not src:
            continue

        # Make absolute URL
        if src.startswith('//'):
            return 'https:' + src
        elif src.startswith('/'):
            if base_url:
                return urljoin(base_url, src)
            return src
        elif src.startswith('http'):
            return src
        elif base_url:
            return urljoin(base_url, src)

    return None
