"""
Lazy import utilities for heavy modules.
Defers loading of matplotlib, PIL, etc. until actually needed.
"""

from typing import Any, Optional

# Cached imports
_matplotlib = None
_plt = None
_pil_image = None
_numpy = None


def get_matplotlib():
    """Lazy load matplotlib."""
    global _matplotlib
    if _matplotlib is None:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        _matplotlib = matplotlib
    return _matplotlib


def get_pyplot():
    """Lazy load matplotlib.pyplot."""
    global _plt
    if _plt is None:
        get_matplotlib()  # Ensure matplotlib is configured first
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def get_pil_image():
    """Lazy load PIL Image."""
    global _pil_image
    if _pil_image is None:
        from PIL import Image
        _pil_image = Image
    return _pil_image


def get_numpy():
    """Lazy load numpy."""
    global _numpy
    if _numpy is None:
        import numpy as np
        _numpy = np
    return _numpy


def is_heavy_module_loaded(module_name: str) -> bool:
    """Check if a heavy module is already loaded."""
    import sys
    return module_name in sys.modules


def get_loaded_heavy_modules() -> list:
    """Get list of currently loaded heavy modules."""
    import sys
    heavy_modules = ['matplotlib', 'numpy', 'PIL', 'cv2', 'torch', 'tensorflow']
    return [m for m in heavy_modules if m in sys.modules]
