"""
Corruption, truncation, and CRC32 verification tests for Caesura binary format.
"""

import os
import tempfile
import pytest
from caesura.vocab.vocab import create_byte_vocab
from caesura.models.bpe import BpeModel
from caesura.pipeline.tokenizer import Tokenizer
from caesura.io.format import BinarySerializer


def test_bpe_merges_binary_roundtrip():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("t", "h", rank=0, merged_token="th")
    bpe.add_merge("th", "e", rank=1, merged_token="the")

    tok = Tokenizer(model=bpe)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.caesura")
        serializer = BinarySerializer()
        serializer.save(tok, path)

        # Reload
        loaded = serializer.load(path)
        assert isinstance(loaded.model, BpeModel)
        assert loaded.model.num_merges == 2
        assert len(loaded.model.merges) == 2

        # Check tokenization parity
        assert loaded.encode("the").ids == tok.encode("the").ids


def test_truncated_header():
    serializer = BinarySerializer()
    with pytest.raises(ValueError, match="too small"):
        serializer._parse_binary(b"SHORT_DATA")


def test_bad_magic():
    serializer = BinarySerializer()
    # 64-byte bad magic header
    bad_header = b"BADMAGIC" + b"\x00" * 56
    with pytest.raises(ValueError, match="Invalid magic header"):
        serializer._parse_binary(bad_header)


def test_crc32_corruption():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    tok = Tokenizer(model=bpe)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "corrupt.caesura")
        serializer = BinarySerializer()
        serializer.save(tok, path)

        with open(path, "rb") as f:
            data = bytearray(f.read())

        # Flip byte in payload
        data[70] ^= 0xFF

        with pytest.raises(ValueError, match="CRC32 checksum mismatch"):
            serializer._parse_binary(bytes(data))
