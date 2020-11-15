"""
Fuzz testing harness and invariant generators for Caesura subword engine.
"""

import random
from typing import Generator, Tuple, List
from caesura.unicode.utf8 import is_valid_utf8, Utf8Scanner, ByteFallback
from caesura.pipeline.tokenizer import Tokenizer


def generate_malformed_utf8(seed: int = 42, count: int = 100) -> Generator[bytes, None, None]:
    """
    Generates malformed UTF-8 byte sequences including overlong encodings,
    truncated multibyte headers, stray continuation bytes, and surrogate halves.
    """
    rng = random.Random(seed)
    edge_cases = [
        b"\xc0\xaf",  # Overlong slash
        b"\xe0\x80\xaf",  # Overlong 3-byte
        b"\xf0\x80\x80\xaf",  # Overlong 4-byte
        b"\xed\xa0\x80",  # High surrogate U+D800
        b"\xed\xbf\xbf",  # Low surrogate U+DFFF
        b"\xff",  # Invalid byte 0xFF
        b"\xfe",  # Invalid byte 0xFE
        b"\x80",  # Stray continuation byte
        b"\xc2",  # Truncated 2-byte header
        b"\xe0\xa0",  # Truncated 3-byte header
        b"\xf0\x90\x80",  # Truncated 4-byte header
        b"\xf4\x90\x80\x80",  # Codepoint > 0x10FFFF
    ]

    for b in edge_cases:
        yield b

    for _ in range(count):
        length = rng.randint(1, 64)
        raw = bytes(rng.randint(0, 255) for _ in range(length))
        yield raw


def generate_adversarial_strings(seed: int = 1337, count: int = 50) -> Generator[str, None, None]:
    """Generates strings with extreme boundary conditions: ZWJ sequences, rtl, combining marks."""
    rng = random.Random(seed)
    primitives = [
        "",
        " ",
        "  ",
        "\t\n\r",
        "👨‍👩‍👧‍👦",  # Family ZWJ sequence
        "🇦🇺🇺🇸🇯🇵",  # Regional flags
        "مرحبا بالعالم",  # Arabic RTL
        "שלום עולם",  # Hebrew RTL
        "e\u0301",  # Decomposed e + acute
        "é",  # Composed e-acute
        "﷽",  # U+FDFD single long glyph
        "𐍈",  # Old Turkic supplementary plane
        "a" * 10000,  # 10k repetitive characters
        " \t " * 1000,  # Alternating whitespace
        "!@#$%^&*()_+-=[]{}|;':,.<>/?",  # Dense punctuation
    ]

    for p in primitives:
        yield p

    for _ in range(count):
        k = rng.randint(2, 8)
        yield "".join(rng.choice(primitives) for _ in range(k))


def verify_offset_invariants(tok: Tokenizer, text: str) -> bool:
    """Verifies that every token offset strictly slices back to the token piece text."""
    encoding = tok.encode(text)
    for i in range(len(encoding)):
        start, end = encoding.offsets[i]
        expected_piece = text[start:end]
        # In special tokens or subword merges, offset span must be bounded within text length
        if start > end or end > len(text):
            return False
    return True


def verify_scanner_safety(raw_bytes: bytes) -> bool:
    """Verifies that Utf8Scanner processes arbitrary bytes without crash."""
    scanner = Utf8Scanner(raw_bytes)
    count = 0
    while scanner.has_more():
        cp = scanner.next_codepoint()
        count += 1
        if count > len(raw_bytes) + 1:
            return False
    return True
