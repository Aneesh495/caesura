"""
Advanced test suite for WordPiece maximum matching and offset validation.
"""

import pytest
from caesura.vocab.vocab import Vocabulary
from caesura.models.wordpiece import WordPieceModel


def test_wordpiece_offsets_and_string_input():
    vocab = Vocabulary()
    vocab.add_token("[UNK]", 0)
    vocab.add_token("un", 1)
    vocab.add_token("##believ", 2)
    vocab.add_token("##able", 3)

    wp = WordPieceModel(vocab=vocab, unk_token="[UNK]")

    # Single word with offsets
    offsets = wp.encode_word_with_offsets("unbelievable")
    assert len(offsets) == 3

    tok1, s1, e1 = offsets[0]
    assert tok1 == 1
    assert "unbelievable"[s1:e1] == "un"

    tok2, s2, e2 = offsets[1]
    assert tok2 == 2
    assert "unbelievable"[s2:e2] == "believ"

    tok3, s3, e3 = offsets[2]
    assert tok3 == 3
    assert "unbelievable"[s3:e3] == "able"

    # String input support
    res_str = wp.encode("unbelievable unbelievable")
    assert res_str == [1, 2, 3, 1, 2, 3]

    # Sentence with offsets
    res_sent_offsets = wp.encode_with_offsets("unbelievable unbelievable")
    assert len(res_sent_offsets) == 6
    assert res_sent_offsets[0][0] == 1
    assert res_sent_offsets[3][0] == 1


def test_wordpiece_max_char_limit():
    vocab = Vocabulary()
    vocab.add_token("[UNK]", 0)
    vocab.add_token("short", 1)

    wp = WordPieceModel(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=10)
    assert wp.encode_word("short") == [1]

    # Word exceeding limit
    long_word = "a" * 20
    assert wp.encode_word(long_word) == [0]


def test_wordpiece_empty_inputs():
    vocab = Vocabulary()
    vocab.add_token("[UNK]", 0)
    wp = WordPieceModel(vocab=vocab)

    assert wp.encode_word("") == []
    assert wp.encode_word_with_offsets("") == []
    assert wp.encode([]) == []
    assert wp.encode("") == []
