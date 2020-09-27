"""
Tests for Binary Serialization, mmap, Tiktoken, and Hugging Face I/O
"""

import tempfile
import pytest
from caesura.vocab import Vocabulary, create_byte_vocab, SpecialTokenRegistry
from caesura.models import BpeModel, WordPieceModel
from caesura.pipeline import Tokenizer
from caesura.io import (
    BinarySerializer,
    TiktokenLoader,
    load_huggingface_json,
    save_huggingface_json,
)


def test_binary_serializer_roundtrip():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("a", "b", rank=0, merged_token="ab")

    spec = SpecialTokenRegistry()
    spec.register("<|bos|>", 1000, role="bos")
    spec.register("<|eos|>", 1001, role="eos")

    tok = Tokenizer(model=bpe, special_tokens=spec)
    serializer = BinarySerializer()

    with tempfile.NamedTemporaryFile(suffix=".caesura", delete=False) as tmp:
        path = tmp.name

    serializer.save(tok, path)

    # Test standard load
    loaded = serializer.load(path)
    assert len(loaded.model.vocab) == len(vocab)
    assert loaded.special_tokens.is_special_token("<|bos|>") is True
    assert loaded.special_tokens.is_special_token("<|eos|>") is True

    # Test zero-copy mmap load
    loaded_mmap = serializer.load_mmap(path)
    assert len(loaded_mmap.model.vocab) == len(vocab)
    assert loaded_mmap.special_tokens.is_special_token("<|bos|>") is True


def test_huggingface_json_roundtrip():
    vocab = Vocabulary()
    vocab.add_token("[PAD]", 0)
    vocab.add_token("[UNK]", 1)
    vocab.add_token("[CLS]", 2)
    vocab.add_token("hello", 3)
    vocab.add_token("world", 4)

    wp = WordPieceModel(vocab=vocab)
    tok = Tokenizer(model=wp)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    save_huggingface_json(tok, path)
    loaded = load_huggingface_json(path)

    assert len(loaded.model.vocab) == 5
    assert loaded.model.vocab["hello"] == 3
    assert loaded.model.vocab["world"] == 4


def test_tiktoken_roundtrip():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    tok = Tokenizer(model=bpe)

    loader = TiktokenLoader()
    with tempfile.NamedTemporaryFile(suffix=".tiktoken", delete=False) as tmp:
        path = tmp.name

    loader.save_tiktoken(tok, path)
    loaded = loader.load_tiktoken(path)
    assert len(loaded.model.vocab) >= 256
