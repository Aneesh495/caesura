"""
Tests for Pipeline, Encodings, Batching, and Master Tokenizer
"""

import pytest
from caesura.vocab import Vocabulary, create_byte_vocab, SpecialTokenRegistry
from caesura.models import BpeModel
from caesura.unicode import LowercaseNormalizer
from caesura.regex import whitespace_pre_tokenizer
from caesura.pipeline import Encoding, BatchEncoding, Tokenizer


def test_encoding_padding_and_truncation():
    enc = Encoding(ids=[10, 20, 30], tokens=["a", "b", "c"])
    assert len(enc) == 3

    # Test pad right to 5
    enc.pad(5, pad_id=0, pad_token="<pad>", direction="right")
    assert len(enc) == 5
    assert enc.ids == [10, 20, 30, 0, 0]
    assert enc.attention_mask == [1, 1, 1, 0, 0]

    # Test truncate right to 2
    enc.truncate(2, direction="right")
    assert len(enc) == 2
    assert enc.ids == [10, 20]
    assert enc.attention_mask == [1, 1]


def test_batch_encoding():
    enc1 = Encoding(ids=[1, 2, 3])
    enc2 = Encoding(ids=[4, 5])
    batch = BatchEncoding([enc1, enc2])

    assert len(batch) == 2
    # Dynamic pad to batch maximum (3)
    batch.pad(pad_id=0)
    data = batch.as_dict()

    assert data["input_ids"] == [
        [1, 2, 3],
        [4, 5, 0],
    ]
    assert data["attention_mask"] == [
        [1, 1, 1],
        [1, 1, 0],
    ]


def test_master_tokenizer_end_to_end():
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("h", "e", rank=0, merged_token="he")
    bpe.add_merge("l", "l", rank=1, merged_token="ll")
    bpe.add_merge("he", "ll", rank=2, merged_token="hell")
    bpe.add_merge("hell", "o", rank=3, merged_token="hello")

    spec = SpecialTokenRegistry()
    spec.register("<|start|>", 500, role="bos")
    spec.register("<|end|>", 501, role="eos")

    tok = Tokenizer(
        model=bpe,
        normalizer=LowercaseNormalizer(),
        pre_tokenizer=whitespace_pre_tokenizer(keep_whitespace=True),
        special_tokens=spec,
    )

    text = "<|start|>HELLO<|end|>"
    encoding = tok.encode(text)

    # First token should be <|start|> (500), then 'hello', then <|end|> (501)
    assert encoding.ids[0] == 500
    assert encoding.ids[-1] == 501
    assert encoding.special_tokens_mask[0] == 1
    assert encoding.special_tokens_mask[-1] == 1

    # Test decode
    decoded = tok.decode(encoding.ids, skip_special_tokens=True)
    assert decoded == "hello"

    decoded_with_spec = tok.decode(encoding.ids, skip_special_tokens=False)
    assert "<|start|>" in decoded_with_spec
    assert "<|end|>" in decoded_with_spec
