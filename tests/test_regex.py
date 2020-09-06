"""
Tests for Caesura DFA and Regex Pre-Tokenizers
"""

import pytest
from caesura.regex import (
    DfaTable,
    DfaScanner,
    get_word_dfa,
    gpt2_pre_tokenizer,
    gpt4_pre_tokenizer,
    whitespace_pre_tokenizer,
    punctuation_pre_tokenizer,
)


def test_dfa_table_basic():
    dfa = DfaTable(3, start_state=0)
    assert dfa.num_states == 3
    assert dfa.start_state == 0

    dfa.set_transition(0, ord("a"), 1)
    dfa.set_transition(1, ord("b"), 2)
    dfa.set_accept(2, 42)

    scanner = DfaScanner(dfa, b"ab")
    spans = scanner.find_all_spans()
    assert spans == [(0, 2)]


def test_word_dfa():
    dfa = get_word_dfa()
    scanner = DfaScanner(dfa, b"hello  world! 123")
    slices = scanner.find_all_slices(b"hello  world! 123")
    assert slices == [b"hello", b"  ", b"world", b"!", b" ", b"123"]


def test_regex_pre_tokenizers():
    tok = whitespace_pre_tokenizer()
    chunks = tok.split("The quick brown fox")
    assert chunks == ["The", "quick", "brown", "fox"]

    gpt2 = gpt2_pre_tokenizer()
    chunks = gpt2.split("Don't stop believing!")
    assert "Don" in chunks or "Don't" in chunks
    assert "stop" in chunks
    assert "believing" in chunks

    offsets = gpt2.split_with_offsets("hello world")
    assert len(offsets) >= 2
    # Check span boundaries match substring
    for chunk, start, end in offsets:
        assert "hello world"[start:end] == chunk
