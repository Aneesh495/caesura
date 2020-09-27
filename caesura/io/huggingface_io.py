"""
Hugging Face tokenizer.json Compatibility Parser and Exporter
"""

import json
from caesura.vocab.vocab import Vocabulary
from caesura.vocab.special import SpecialTokenRegistry
from caesura.models.bpe import BpeModel
from caesura.models.wordpiece import WordPieceModel
from caesura.models.unigram import UnigramModel
from caesura.unicode.normalizer import (
    SequenceNormalizer,
    LowercaseNormalizer,
    StripAccentsNormalizer,
    NfcNormalizer,
    NfkdNormalizer,
    MetaspaceNormalizer,
)
from caesura.regex.compiler import RegexPreTokenizer, whitespace_pre_tokenizer
from caesura.decoder.decoder import (
    ByteLevelDecoder,
    WordPieceDecoder,
    MetaspaceDecoder,
)
from caesura.pipeline.tokenizer import Tokenizer


def load_huggingface_json(file_path: str) -> Tokenizer:
    """Load a Tokenizer directly from a Hugging Face tokenizer.json file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_data = data.get("model", {})
    model_type = model_data.get("type", "BPE")
    vocab_dict = model_data.get("vocab", {})
    vocab = Vocabulary.from_dict(vocab_dict)

    if model_type == "BPE":
        model = BpeModel(vocab=vocab)
        merges = model_data.get("merges", [])
        for rank, merge_line in enumerate(merges):
            parts = merge_line.split()
            if len(parts) == 2:
                p1, p2 = parts
                model.add_merge(p1, p2, rank=rank)
    elif model_type == "WordPiece":
        unk_token = model_data.get("unk_token", "[UNK]")
        continuing_prefix = model_data.get("continuing_subword_prefix", "##")
        model = WordPieceModel(vocab=vocab, unk_token=unk_token, continuing_prefix=continuing_prefix)
    elif model_type == "Unigram":
        model = UnigramModel(vocab=vocab)
    else:
        model = BpeModel(vocab=vocab)

    # Special tokens
    spec = SpecialTokenRegistry()
    added_tokens = data.get("added_tokens", [])
    for entry in added_tokens:
        tok_str = entry.get("content")
        tok_id = entry.get("id")
        if tok_str and tok_id is not None:
            spec.register(tok_str, tok_id)

    # Normalizer
    normalizer = None
    norm_data = data.get("normalizer")
    if norm_data:
        norm_type = norm_data.get("type")
        if norm_type == "Lowercase":
            normalizer = LowercaseNormalizer()
        elif norm_type == "NFC":
            normalizer = NfcNormalizer()
        elif norm_type == "NFKD":
            normalizer = NfkdNormalizer()

    # Pre-tokenizer
    pre_tok = whitespace_pre_tokenizer(keep_whitespace=True)

    return Tokenizer(
        model=model,
        normalizer=normalizer,
        pre_tokenizer=pre_tok,
        special_tokens=spec,
    )


def save_huggingface_json(tokenizer: Tokenizer, file_path: str):
    """Export a Tokenizer to Hugging Face tokenizer.json format."""
    vocab_dict = tokenizer.model.vocab.to_dict() if hasattr(tokenizer.model, "vocab") else {}

    model_type = "BPE"
    if isinstance(tokenizer.model, WordPieceModel):
        model_type = "WordPiece"
    elif isinstance(tokenizer.model, UnigramModel):
        model_type = "Unigram"

    data = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": sid, "content": st, "special": True}
            for st, sid in tokenizer.special_tokens.token_to_id.items()
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {
            "type": model_type,
            "vocab": vocab_dict,
            "merges": [],
        },
        "decoder": None,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
