"""
Advanced tests for Unigram language model, scoring, and Viterbi trellis offsets.
"""

import pytest
from caesura.vocab.vocab import Vocabulary
from caesura.models.unigram import UnigramModel


def test_unigram_offsets_and_scoring():
    vocab = Vocabulary()
    vocab.add_token("<unk>", 0)
    vocab.add_token("new", 1)
    vocab.add_token("york", 2)
    vocab.add_token("newyork", 3)

    # Make "new" + "york" score higher than single "newyork"
    scores = {
        0: -15.0,
        1: -1.0,
        2: -1.0,
        3: -3.5,
    }

    model = UnigramModel(vocab=vocab, scores=scores)
    offsets = model.encode_with_offsets("newyork")
    assert len(offsets) == 2

    tok1, s1, e1 = offsets[0]
    assert tok1 == 1
    assert "newyork"[s1:e1] == "new"

    tok2, s2, e2 = offsets[1]
    assert tok2 == 2
    assert "newyork"[s2:e2] == "york"

    # Verify score sequence
    score = model.score_sequence([1, 2])
    assert pytest.approx(score) == -2.0


def test_unigram_unk_fallback():
    vocab = Vocabulary()
    vocab.add_token("<unk>", 0)
    vocab.add_token("a", 1)
    vocab.add_token("b", 2)

    scores = {
        0: -10.0,
        1: -1.0,
        2: -1.0,
    }

    model = UnigramModel(vocab=vocab, scores=scores, unk_token="<unk>", unk_score=-10.0)

    # 'z' is not in vocabulary, should trigger fallback
    encoded = model.encode("azb")
    assert len(encoded) == 3
    assert encoded[0] == 1  # 'a'
    assert encoded[1] == 0  # '<unk>'
    assert encoded[2] == 2  # 'b'


def test_unigram_empty_and_single_char():
    vocab = Vocabulary()
    vocab.add_token("<unk>", 0)
    vocab.add_token("x", 1)

    model = UnigramModel(vocab=vocab)
    assert model.encode("") == []
    assert model.encode_with_offsets("") == []
    assert model.encode("x") == [1]
