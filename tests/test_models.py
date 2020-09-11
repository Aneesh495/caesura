"""
Tests for Subword Models: BPE, WordPiece, and Unigram
"""

import pytest
from caesura.vocab import Vocabulary, create_byte_vocab
from caesura.models import BpeModel, WordPieceModel, UnigramModel


def test_bpe_basic_merge():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)

    # In ASCII: 'h' is 104, 'e' is 101, 'l' is 108, 'o' is 111
    # Add merge for 'h' + 'e' -> 'he'
    bpe.add_merge("h", "e", rank=0, merged_token="he")
    # Add merge for 'l' + 'l' -> 'll'
    bpe.add_merge("l", "l", rank=1, merged_token="ll")
    # Add merge for 'he' + 'll' -> 'hell'
    bpe.add_merge("he", "ll", rank=2, merged_token="hell")
    # Add merge for 'hell' + 'o' -> 'hello'
    bpe.add_merge("hell", "o", rank=3, merged_token="hello")

    encoded = bpe.encode("hello")
    assert len(encoded) == 1
    assert encoded[0] == vocab["hello"]

    offsets = bpe.encode_with_offsets("hello")
    assert len(offsets) == 1
    tid, start, end = offsets[0]
    assert tid == vocab["hello"]
    assert start == 0
    assert end == 5


def test_bpe_fallback_unmerged():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("a", "b", rank=0, merged_token="ab")

    # 'abc' should merge 'a'+'b' -> 'ab', leaving 'c' separate
    encoded = bpe.encode("abc")
    assert len(encoded) == 2
    assert encoded[0] == vocab["ab"]
    assert encoded[1] == ord("c")


def test_wordpiece_basic():
    vocab = Vocabulary()
    vocab.add_token("[UNK]", 0)
    vocab.add_token("[CLS]", 1)
    vocab.add_token("[SEP]", 2)
    vocab.add_token("play", 3)
    vocab.add_token("##er", 4)
    vocab.add_token("##s", 5)

    wp = WordPieceModel(vocab=vocab, unk_token="[UNK]")
    tokens = wp.encode_word("players")
    assert tokens == [3, 4, 5]

    unk_tokens = wp.encode_word("unknownwordxyz")
    assert unk_tokens == [0]


def test_unigram_viterbi():
    vocab = Vocabulary()
    vocab.add_token("<unk>", 0)
    vocab.add_token("c", 1)
    vocab.add_token("a", 2)
    vocab.add_token("t", 3)
    vocab.add_token("ca", 4)
    vocab.add_token("at", 5)
    vocab.add_token("cat", 6)

    # cat has the highest score (least negative)
    scores = {
        0: -10.0,
        1: -3.0,
        2: -3.0,
        3: -3.0,
        4: -2.0,
        5: -2.0,
        6: -0.5,
    }

    unigram = UnigramModel(vocab=vocab, scores=scores)
    tokens = unigram.encode("cat")
    assert tokens == [6]  # "cat" chosen over ["ca", "t"] or ["c", "at"]
