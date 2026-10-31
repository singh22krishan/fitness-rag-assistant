import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_utils import clean_text, detect_section


def test_clean_text_normalizes_whitespace_and_pdf_bullet():
    assert clean_text("  Pre\uf0b7  workout\n meal  ") == "Pre workout meal"


def test_detect_section_boundaries():
    assert detect_section(0) == "overview"
    assert detect_section(2) == "overview"
    assert detect_section(3) == "guidelines"
    assert detect_section(6) == "guidelines"
    assert detect_section(7) == "recipes_and_examples"
