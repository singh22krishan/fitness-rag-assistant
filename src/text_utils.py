"""Small text/metadata helpers kept independent for unit testing."""

import re


def clean_text(text: str) -> str:
    """Normalize common PDF extraction artifacts and whitespace."""
    text = text.replace("\x00", " ")
    text = text.replace("\uf0b7", " ")
    return re.sub(r"\s+", " ", text).strip()


def detect_section(page_index: int) -> str:
    """Assign coarse source sections based on zero-based PDF page index."""
    if page_index <= 2:
        return "overview"
    if 3 <= page_index <= 6:
        return "guidelines"
    return "recipes_and_examples"
