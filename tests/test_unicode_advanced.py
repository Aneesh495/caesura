"""
Advanced tests for Unicode script classification and combining character detection.
"""

import pytest
from caesura.unicode.tables import (
    get_script,
    is_combining,
    is_cjk,
    is_punctuation,
    is_whitespace,
)


def test_unicode_script_classification():
    # Latin
    assert get_script(ord("A")) == "Latin"
    assert get_script(ord("z")) == "Latin"
    assert get_script(ord("é")) == "Latin"

    # Greek
    assert get_script(ord("Ω")) == "Greek"
    assert get_script(ord("α")) == "Greek"

    # Cyrillic
    assert get_script(ord("Д")) == "Cyrillic"
    assert get_script(ord("я")) == "Cyrillic"

    # Hebrew & Arabic
    assert get_script(ord("א")) == "Hebrew"
    assert get_script(ord("ش")) == "Arabic"

    # Devanagari
    assert get_script(ord("अ")) == "Devanagari"

    # CJK
    assert get_script(ord("中")) == "Han"
    assert get_script(ord("あ")) == "Hiragana"
    assert get_script(ord("ア")) == "Katakana"
    assert get_script(ord("한")) == "Hangul"

    # Common
    assert get_script(ord("1")) == "Common"
    assert get_script(ord(",")) == "Common"
    assert get_script(ord(" ")) == "Common"


def test_unicode_combining_marks():
    # Combining acute U+0301
    assert is_combining(0x0301)
    # Combining tilde U+0303
    assert is_combining(0x0303)
    # Standard letter 'a' is not combining
    assert not is_combining(ord("a"))
