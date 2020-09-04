"""
Caesura Unicode and Normalization Module
"""

from caesura.unicode.utf8 import (
    Utf8Scanner,
    ByteFallback,
    is_valid_utf8,
    scan_ascii_prefix,
)
from caesura.unicode.tables import (
    is_whitespace,
    is_control,
    is_punctuation,
    is_cjk,
    is_alphanumeric,
)
from caesura.unicode.normalizer import (
    Normalizer,
    LowercaseNormalizer,
    StripAccentsNormalizer,
    NfcNormalizer,
    NfkdNormalizer,
    MetaspaceNormalizer,
    SequenceNormalizer,
)

__all__ = [
    "Utf8Scanner",
    "ByteFallback",
    "is_valid_utf8",
    "scan_ascii_prefix",
    "is_whitespace",
    "is_control",
    "is_punctuation",
    "is_cjk",
    "is_alphanumeric",
    "Normalizer",
    "LowercaseNormalizer",
    "StripAccentsNormalizer",
    "NfcNormalizer",
    "NfkdNormalizer",
    "MetaspaceNormalizer",
    "SequenceNormalizer",
]
