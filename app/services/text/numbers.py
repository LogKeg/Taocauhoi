"""
Number processing utilities.
"""
import random
import re

from app.core.constants import NUMBER_RE


def replace_numbers(text: str) -> str:
    """Replace numbers with similar values (±15%)."""
    def _swap(match: re.Match) -> str:
        value = int(match.group(0))
        if value == 0:
            return "0"
        delta = max(1, int(round(value * 0.15)))
        new_value = max(1, value + random.choice([-delta, delta]))
        return str(new_value)

    return NUMBER_RE.sub(_swap, text)
