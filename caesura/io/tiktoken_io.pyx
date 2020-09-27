# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
OpenAI Tiktoken (.tiktoken) File Format Parser and Exporter
"""

import base64
from caesura.vocab.vocab cimport Vocabulary
from caesura.vocab.special cimport SpecialTokenRegistry
from caesura.models.bpe cimport BpeModel
from caesura.regex.compiler cimport RegexPreTokenizer
from caesura.pipeline.tokenizer cimport Tokenizer
from caesura.vocab.vocab import create_byte_vocab
from caesura.regex.patterns import GPT4_PATTERN
from caesura.unicode.utf8 import ByteFallback

cdef class TiktokenLoader:
    """
    Parses OpenAI .tiktoken files containing base64 token bytes and merge ranks.
    """

    cpdef Tokenizer load_tiktoken(self, str file_path, dict special_tokens=None, str pattern=None):
        cdef Vocabulary vocab = Vocabulary()
        cdef BpeModel model = BpeModel(vocab=vocab)
        cdef SpecialTokenRegistry spec = SpecialTokenRegistry()
        cdef object bf = ByteFallback()

        # Default special tokens for cl100k / gpt4
        cdef dict specials = special_tokens if special_tokens is not None else {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        }

        for st, sid in specials.items():
            spec.register(st, sid)

        cdef str pat = pattern if pattern else GPT4_PATTERN
        cdef RegexPreTokenizer pre_tok = RegexPreTokenizer(pat)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    raw_b64, rank_str = parts
                    token_bytes = base64.b64decode(raw_b64)
                    rank = int(rank_str)
                    token_str = bf.encode_bytes_to_string(token_bytes)
                    vocab.add_token(token_str, rank)

        return Tokenizer(
            model=model,
            pre_tokenizer=pre_tok,
            special_tokens=spec,
        )

    cpdef void save_tiktoken(self, Tokenizer tokenizer, str file_path):
        """Export BPE vocabulary to .tiktoken format."""
        cdef object bf = ByteFallback()
        with open(file_path, "w", encoding="utf-8") as f:
            if hasattr(tokenizer.model, "vocab"):
                for tok, tid in tokenizer.model.vocab.to_dict().items():
                    raw_bytes = bf.decode_string_to_bytes(tok)
                    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
                    f.write(f"{b64_str} {tid}\n")
