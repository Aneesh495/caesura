"""
Advanced tests for vocabulary inversion, disk serialization, and dict parity.
"""

import os
import tempfile
import pytest
from caesura.vocab.vocab import Vocabulary, create_byte_vocab


def test_vocabulary_inversion():
    v = Vocabulary()
    v.add_token("hello", 10)
    v.add_token("world", 20)

    inv = v.invert()
    assert inv[10] == "hello"
    assert inv[20] == "world"
    assert len(inv) == 2


def test_vocabulary_save_and_load_txt():
    v = Vocabulary()
    tokens = ["token_a", "token_b", "token_c", "token_d"]
    for t in tokens:
        v.add_token(t)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vocab.txt")
        v.save_txt(path)
        assert os.path.exists(path)

        loaded = Vocabulary.load_txt(path)
        assert len(loaded) == len(tokens)
        for i, t in enumerate(tokens):
            assert loaded[t] == i
            assert loaded[i] == t


def test_vocabulary_dict_parity():
    v = Vocabulary()
    v.add_token("x", 0)
    v.add_token("y", 1)

    assert set(v.keys()) == {"x", "y"}
    assert set(v.values()) == {0, 1}
    assert set(v.items()) == {("x", 0), ("y", 1)}
