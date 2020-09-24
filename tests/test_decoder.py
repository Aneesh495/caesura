"""
Tests for Decoders and Incremental Streaming Reconstruction
"""

import pytest
from caesura.decoder import (
    ByteLevelDecoder,
    WordPieceDecoder,
    MetaspaceDecoder,
    SequenceDecoder,
    IncrementalDecoder,
)
from caesura.unicode import ByteFallback


def test_byte_level_decoder():
    bf = ByteFallback()
    dec = ByteLevelDecoder()

    # Raw string with special characters
    orig = "Hello, 世界! 🚀"
    encoded_str = bf.encode_bytes_to_string(orig.encode("utf-8"))
    decoded = dec.decode(encoded_str)
    assert decoded == orig


def test_wordpiece_decoder():
    dec = WordPieceDecoder(prefix="##", cleanup_spaces=True)
    assert dec.decode("play ##er ##s") == "players"
    assert dec.decode("hello , world !") == "hello, world!"


def test_metaspace_decoder():
    dec = MetaspaceDecoder(replacement="\u2581", add_prefix_space=True)
    assert dec.decode("\u2581hello\u2581world") == "hello world"


def test_incremental_decoder_split_multibyte():
    stream_dec = IncrementalDecoder()

    # UTF-8 for '🚀' is 4 bytes: F0 9F 9A 80
    rocket_bytes = "🚀".encode("utf-8")
    assert len(rocket_bytes) == 4

    part1 = rocket_bytes[:2]  # Incomplete
    part2 = rocket_bytes[2:]  # Completion

    # Feeding incomplete bytes should buffer without emitting broken character
    out1 = stream_dec.decode_bytes_step(part1)
    assert out1 == ""
    assert stream_dec.buffer_len == 2

    # Feeding remainder should emit the reconstructed emoji
    out2 = stream_dec.decode_bytes_step(part2)
    assert out2 == "🚀"
    assert stream_dec.buffer_len == 0


def test_incremental_decoder_ascii():
    stream_dec = IncrementalDecoder()
    assert stream_dec.decode_bytes_step(b"hello ") == "hello "
    assert stream_dec.decode_bytes_step(b"world") == "world"
    assert stream_dec.flush() == ""
