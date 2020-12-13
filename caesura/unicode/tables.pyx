# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Unicode Property and Character Classification Tables
"""

from libc.stdint cimport uint32_t

cdef bint c_is_whitespace(uint32_t cp) noexcept nogil:
    if cp == 0x20 or cp == 0x09 or cp == 0x0A or cp == 0x0D:
        return 1
    if cp == 0x0B or cp == 0x0C or cp == 0x85 or cp == 0xA0:
        return 1
    if cp == 0x1680 or (cp >= 0x2000 and cp <= 0x200A):
        return 1
    if cp == 0x2028 or cp == 0x2029 or cp == 0x202F or cp == 0x205F or cp == 0x3000:
        return 1
    return 0

cdef bint c_is_control(uint32_t cp) noexcept nogil:
    if cp <= 0x1F and not (cp == 0x09 or cp == 0x0A or cp == 0x0D):
        return 1
    if cp >= 0x7F and cp <= 0x9F:
        return 1
    return 0

cdef bint c_is_punctuation(uint32_t cp) noexcept nogil:
    # Standard ASCII punctuation
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or \
       (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return 1
    # General Punctuation block U+2000..U+206F
    if cp >= 0x2000 and cp <= 0x206F:
        return 1
    # Supplemental Punctuation U+2E00..U+2E7F
    if cp >= 0x2E00 and cp <= 0x2E7F:
        return 1
    # CJK Symbols and Punctuation U+3000..U+303F
    if cp >= 0x3000 and cp <= 0x303F:
        return 1
    return 0

cdef bint c_is_combining(uint32_t cp) noexcept nogil:
    # Combining Diacritical Marks U+0300..U+036F
    if cp >= 0x0300 and cp <= 0x036F:
        return 1
    # Combining Diacritical Marks Extended U+1AB0..U+1AFF
    if cp >= 0x1AB0 and cp <= 0x1AFF:
        return 1
    # Combining Diacritical Marks Supplement U+1DC0..U+1DFF
    if cp >= 0x1DC0 and cp <= 0x1DFF:
        return 1
    # Combining Half Marks U+FE20..U+FE2F
    if cp >= 0xFE20 and cp <= 0xFE2F:
        return 1
    return 0

cdef bint c_is_cjk(uint32_t cp) noexcept nogil:
    # CJK Unified Ideographs U+4E00..U+9FFF
    if cp >= 0x4E00 and cp <= 0x9FFF:
        return 1
    # CJK Unified Ideographs Extension A U+3400..U+4DBF
    if cp >= 0x3400 and cp <= 0x4DBF:
        return 1
    # CJK Unified Ideographs Extension B..F U+20000..U+2CEAF
    if cp >= 0x20000 and cp <= 0x2CEAF:
        return 1
    # CJK Compatibility Ideographs U+F900..U+FAFF
    if cp >= 0xF900 and cp <= 0xFAFF:
        return 1
    # Hangul Syllables U+AC00..U+D7AF
    if cp >= 0xAC00 and cp <= 0xD7AF:
        return 1
    # Hiragana U+3040..U+309F and Katakana U+30A0..U+30FF
    if cp >= 0x3040 and cp <= 0x30FF:
        return 1
    return 0

cdef bint c_is_alphanumeric(uint32_t cp) noexcept nogil:
    if (cp >= 0x30 and cp <= 0x39) or \
       (cp >= 0x41 and cp <= 0x5A) or \
       (cp >= 0x61 and cp <= 0x7A):
        return 1
    if c_is_cjk(cp):
        return 1
    if cp >= 0xC0 and cp <= 0x24F:
        return 1
    return 0

def is_whitespace(int codepoint) -> bool:
    return c_is_whitespace(<uint32_t>codepoint)

def is_control(int codepoint) -> bool:
    return c_is_control(<uint32_t>codepoint)

def is_punctuation(int codepoint) -> bool:
    return c_is_punctuation(<uint32_t>codepoint)

def is_combining(int codepoint) -> bool:
    return c_is_combining(<uint32_t>codepoint)

def is_cjk(int codepoint) -> bool:
    return c_is_cjk(<uint32_t>codepoint)

def is_alphanumeric(int codepoint) -> bool:
    return c_is_alphanumeric(<uint32_t>codepoint)

def get_script(int codepoint) -> str:
    """Classify codepoint into major Unicode script groups."""
    cdef uint32_t cp = <uint32_t>codepoint
    if (cp >= 0x41 and cp <= 0x5A) or (cp >= 0x61 and cp <= 0x7A) or (cp >= 0xC0 and cp <= 0x24F):
        return "Latin"
    if cp >= 0x0370 and cp <= 0x03FF:
        return "Greek"
    if cp >= 0x0400 and cp <= 0x04FF:
        return "Cyrillic"
    if cp >= 0x0590 and cp <= 0x05FF:
        return "Hebrew"
    if cp >= 0x0600 and cp <= 0x06FF:
        return "Arabic"
    if cp >= 0x0900 and cp <= 0x097F:
        return "Devanagari"
    if cp >= 0x3040 and cp <= 0x309F:
        return "Hiragana"
    if cp >= 0x30A0 and cp <= 0x30FF:
        return "Katakana"
    if cp >= 0xAC00 and cp <= 0xD7AF:
        return "Hangul"
    if (cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF):
        return "Han"
    if c_is_punctuation(cp) or c_is_whitespace(cp) or (cp >= 0x30 and cp <= 0x39):
        return "Common"
    return "Unknown"
