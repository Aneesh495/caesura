"""
Tests for Caesura Unicode Scanning, ByteFallback, and Normalization
"""

import pytest
from caesura.unicode import (
    Utf8Scanner,
    ByteFallback,
    is_valid_utf8,
    scan_ascii_prefix,
    is_whitespace,
    is_control,
    is_punctuation,
    is_cjk,
    is_alphanumeric,
    LowercaseNormalizer,
    StripAccentsNormalizer,
    NfcNormalizer,
    NfkdNormalizer,
    MetaspaceNormalizer,
    SequenceNormalizer,
)


def test_utf8_validity():
    assert is_valid_utf8(b"Hello world") is True
    assert is_valid_utf8(b"Pr\xc3\xa9cis") is True  # Précis
    assert is_valid_utf8(b"\xf0\x9f\x9a\x80") is True  # Rocket emoji
    assert is_valid_utf8(b"\xff\xfe") is False  # Invalid lead bytes
    assert is_valid_utf8(b"\xc3") is False  # Incomplete sequence


def test_scan_ascii_prefix():
    assert scan_ascii_prefix(b"hello world") == 11
    assert scan_ascii_prefix(b"hello\x80world") == 5
    assert scan_ascii_prefix(b"\xffabc") == 0


def test_utf8_scanner():
    scanner = Utf8Scanner("Hello 🚀".encode("utf-8"))
    codepoints = []
    while scanner.has_more():
        cp = scanner.next_codepoint()
        if cp is not None:
            codepoints.append(cp)

    assert codepoints == [ord("H"), ord("e"), ord("l"), ord("l"), ord("o"), ord(" "), 0x1F680]


def test_byte_fallback_roundtrip():
    bf = ByteFallback()
    # Check all 256 bytes roundtrip losslessly
    all_bytes = bytes(range(256))
    encoded_str = bf.encode_bytes_to_string(all_bytes)
    assert len(encoded_str) == 256
    decoded_bytes = bf.decode_string_to_bytes(encoded_str)
    assert decoded_bytes == all_bytes


def test_character_tables():
    assert is_whitespace(ord(" ")) is True
    assert is_whitespace(ord("\t")) is True
    assert is_whitespace(ord("\n")) is True
    assert is_whitespace(ord("a")) is False

    assert is_punctuation(ord(".")) is True
    assert is_punctuation(ord(",")) is True
    assert is_punctuation(ord("!")) is True
    assert is_punctuation(ord("x")) is False

    assert is_cjk(0x4E00) is True  # 一
    assert is_cjk(ord("A")) is False

    assert is_alphanumeric(ord("9")) is True
    assert is_alphanumeric(ord("Z")) is True
    assert is_alphanumeric(ord("!")) is False


def test_normalizers():
    lower = LowercaseNormalizer()
    assert lower.normalize("CAESURA Tokenizer") == "caesura tokenizer"

    strip_acc = StripAccentsNormalizer()
    assert strip_acc.normalize("café résumé") == "cafe resume"

    nfc = NfcNormalizer()
    assert nfc.normalize("caf\u0065\u0301") == "café"

    meta = MetaspaceNormalizer(replacement="\u2581", prepend_space=True)
    assert meta.normalize("hello world") == "\u2581hello\u2581world"

    seq = SequenceNormalizer([lower, strip_acc])
    assert seq.normalize("PRÉCIS") == "precis"
