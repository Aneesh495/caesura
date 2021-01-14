"""
Validation tests for bundled reference tokenizer artifacts.
"""

import os
import pytest
from caesura.io.format import BinarySerializer
from caesura.io.huggingface_io import load_huggingface_json
from caesura.io.tiktoken_io import TiktokenLoader
from caesura.models.bpe import BpeModel
from caesura.models.wordpiece import WordPieceModel


def test_bundled_gpt2_reference():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "gpt2_reference.caesura")
    assert os.path.exists(model_path)

    serializer = BinarySerializer()
    tok = serializer.load_mmap(model_path)

    assert isinstance(tok.model, BpeModel)
    assert len(tok.model.vocab) > 256
    assert "<|endoftext|>" in tok.special_tokens

    enc = tok.encode("the quick brown fox")
    assert len(enc) > 0
    assert tok.decode(enc.ids, skip_special_tokens=True) == "the quick brown fox"


def test_bundled_bert_reference():
    json_path = os.path.join(os.path.dirname(__file__), "..", "models", "bert_reference.json")
    assert os.path.exists(json_path)

    tok = load_huggingface_json(json_path)
    assert isinstance(tok.model, WordPieceModel)
    assert len(tok.model.vocab) > 20
    assert "[CLS]" in tok.special_tokens
    assert "[SEP]" in tok.special_tokens

    # WordPiece greedy match on players -> play + ##er + ##s
    enc = tok.encode("players")
    assert len(enc) == 3


def test_bundled_cl100k_reference():
    tiktoken_path = os.path.join(os.path.dirname(__file__), "..", "models", "cl100k_sample.tiktoken")
    assert os.path.exists(tiktoken_path)

    loader = TiktokenLoader()
    tok = loader.load_tiktoken(tiktoken_path)
    assert isinstance(tok.model, BpeModel)
    assert len(tok.model.vocab) == 14
