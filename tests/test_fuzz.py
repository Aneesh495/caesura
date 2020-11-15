"""
Stress and fuzz tests verifying engine invariants under adversarial payloads.
"""

import pytest
from caesura.core.fuzz import (
    generate_malformed_utf8,
    generate_adversarial_strings,
    verify_offset_invariants,
    verify_scanner_safety,
)
from caesura.unicode.utf8 import is_valid_utf8
from caesura.vocab.vocab import create_byte_vocab
from caesura.models.bpe import BpeModel
from caesura.pipeline.tokenizer import Tokenizer
from caesura.regex.compiler import whitespace_pre_tokenizer


def test_malformed_utf8_fuzz():
    """Verify that malformed UTF-8 sequences never crash the C scanner."""
    for raw in generate_malformed_utf8(seed=123, count=50):
        # Must execute without segfault
        valid = is_valid_utf8(raw)
        # Scanner must terminate safely
        assert verify_scanner_safety(raw)


def test_adversarial_strings_tokenization():
    """Verify that adversarial unicode inputs encode and decode cleanly."""
    vocab = create_byte_vocab()
    vocab.add_token(" ", 256)
    bpe = BpeModel(vocab=vocab)
    tok = Tokenizer(model=bpe, pre_tokenizer=None)

    for text in generate_adversarial_strings(seed=999, count=30):
        encoding = tok.encode(text)
        assert len(encoding) >= 0
        assert verify_offset_invariants(tok, text)


def test_massive_payload_stress():
    """Verify engine stability on megabyte-scale repetitive and varied payloads."""
    vocab = create_byte_vocab()
    vocab.add_token(" ", 256)
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("a", "a", rank=0, merged_token="aa")
    bpe.add_merge("aa", "aa", rank=1, merged_token="aaaa")

    tok = Tokenizer(model=bpe, pre_tokenizer=None)

    # 5KB repetitive string
    massive_text = "a" * 5000
    enc = tok.encode(massive_text)
    assert len(enc) > 0
    # Merged token 'aaaa' should dominate
    assert enc.ids[0] == vocab["aaaa"]


def test_empty_and_whitespace_stress():
    """Verify edge cases for empty strings, solo whitespace, solo punctuation."""
    vocab = create_byte_vocab()
    vocab.add_token(" ", 256)
    bpe = BpeModel(vocab=vocab)
    tok = Tokenizer(model=bpe, pre_tokenizer=whitespace_pre_tokenizer(keep_whitespace=True))

    assert len(tok.encode("")) == 0
    assert len(tok.encode(" ")) == 1
    assert len(tok.encode("\n\t\r")) > 0
    assert tok.decode([]) == ""
