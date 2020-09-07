"""
Tests for Caesura Vocabulary and Special Token Management
"""

import pytest
from caesura.vocab import Vocabulary, create_byte_vocab, SpecialTokenRegistry


def test_vocabulary_basic():
    v = Vocabulary()
    assert len(v) == 0

    id0 = v.add_token("hello")
    id1 = v.add_token("world")
    assert id0 == 0
    assert id1 == 1
    assert len(v) == 2

    assert v.get_id("hello") == 0
    assert v.get_token(1) == "world"
    assert "hello" in v
    assert "missing" not in v

    # Re-adding returns existing id
    assert v.add_token("hello") == 0


def test_vocabulary_serialization_roundtrip():
    d = {"[CLS]": 0, "[SEP]": 1, "hello": 2, "world": 3}
    v = Vocabulary.from_dict(d)
    assert len(v) == 4
    assert v["[CLS]"] == 0
    assert v[3] == "world"
    assert v.to_dict() == d


def test_byte_vocabulary():
    bv = create_byte_vocab()
    assert len(bv) == 256
    # Each byte 0..255 has an entry
    for b in range(256):
        assert bv.get_token(b) is not None


def test_special_token_registry():
    reg = SpecialTokenRegistry()
    reg.register("<|bos|>", 1, role="bos")
    reg.register("<|eos|>", 2, role="eos")
    reg.register("<|unk|>", 0, role="unk")
    reg.register("<|pad|>", 3, role="pad")

    assert reg.bos_token_id == 1
    assert reg.eos_token_id == 2
    assert reg.unk_token_id == 0
    assert reg.pad_token_id == 3

    assert reg.is_special_token("<|bos|>") is True
    assert reg.is_special_token("normal_word") is False
    assert reg.is_special_id(1) is True
    assert reg.is_special_id(999) is False

    # Test split
    text = "<|bos|>Generate text<|eos|>"
    splits = reg.split(text)
    assert len(splits) == 3
    assert splits[0] == ("special", (1, "<|bos|>"))
    assert splits[1] == ("text", "Generate text")
    assert splits[2] == ("special", (2, "<|eos|>"))
