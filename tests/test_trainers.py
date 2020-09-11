"""
Tests for Vocabulary Trainers: BPE, WordPiece, and Unigram
"""

import pytest
from caesura.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer


def test_bpe_training():
    corpus = [
        "low low low low low",
        "lower lower",
        "newest newest newest newest newest newest",
        "widest widest widest",
    ]

    trainer = BpeTrainer(vocab_size=300, min_frequency=2)
    model = trainer.train(corpus)

    assert model.num_merges > 0
    # "low" should be encoded into fewer tokens than 3
    encoded = model.encode("low")
    assert len(encoded) <= 2


def test_wordpiece_training():
    corpus = [
        "play playing player players",
        "work working worker workers",
    ]

    trainer = WordPieceTrainer(vocab_size=100, min_frequency=1)
    model = trainer.train(corpus)

    assert "play" in model.vocab or "##er" in model.vocab


def test_unigram_training():
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog slept under the tree",
    ]

    trainer = UnigramTrainer(vocab_size=60)
    model = trainer.train(corpus)

    encoded = model.encode("the dog")
    assert len(encoded) >= 2
