# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Master Tokenizer Orchestrator and Pipeline Integration
"""

from libc.stdint cimport uint32_t
from caesura.unicode.normalizer cimport Normalizer
from caesura.regex.compiler cimport RegexPreTokenizer
from caesura.vocab.special cimport SpecialTokenRegistry
from caesura.pipeline.encoding cimport Encoding
from caesura.pipeline.batch cimport BatchEncoding
from caesura.models.bpe import BpeModel

cdef class Tokenizer:
    """
    Unified end-to-end Tokenizer orchestrator combining normalizer,
    regex pre-tokenizer, subword model, special token registry, and decoder.
    """

    def __init__(
        self,
        object model,
        Normalizer normalizer=None,
        RegexPreTokenizer pre_tokenizer=None,
        SpecialTokenRegistry special_tokens=None,
        object decoder=None
    ):
        self._model = model
        self._normalizer = normalizer
        self._pre_tokenizer = pre_tokenizer
        self._special_tokens = special_tokens if special_tokens is not None else SpecialTokenRegistry()
        self._decoder = decoder

    cpdef Encoding encode(self, str text, bint add_special_tokens=True):
        """Encode raw text into an Encoding container."""
        if not text:
            return Encoding([], raw_text="")

        cdef list all_ids = []
        cdef list all_tokens = []
        cdef list all_offsets = []
        cdef list all_special_mask = []

        # Step 1: Split on special tokens using Aho-Corasick
        cdef list segments = self._special_tokens.split(text)
        cdef str kind, chunk
        cdef object spec_info
        cdef uint32_t spec_id
        cdef str spec_str
        cdef size_t global_char_offset = 0

        cdef list pre_chunks
        cdef str pc
        cdef list piece_ids
        cdef uint32_t pid
        cdef object tok_str

        for kind, content in segments:
            if kind == "special":
                spec_info = content
                spec_id = <uint32_t>spec_info[0]
                spec_str = spec_info[1]
                all_ids.append(spec_id)
                all_tokens.append(spec_str)
                all_offsets.append((global_char_offset, global_char_offset + len(spec_str)))
                all_special_mask.append(1)
                global_char_offset += len(spec_str)
            else:
                chunk = content
                # Step 2: Normalization
                if self._normalizer is not None:
                    chunk = self._normalizer.normalize(chunk)

                # Step 3: Pre-tokenization
                if self._pre_tokenizer is not None:
                    pre_chunks = self._pre_tokenizer.split(chunk)
                else:
                    pre_chunks = [chunk]

                # Step 4: Subword model encoding
                for pc in pre_chunks:
                    if hasattr(self._model, "encode"):
                        piece_ids = self._model.encode(pc)
                    elif hasattr(self._model, "encode_word"):
                        piece_ids = self._model.encode_word(pc)
                    else:
                        piece_ids = []

                    for pid in piece_ids:
                        all_ids.append(pid)
                        tok_str = self._model.vocab.get_token(pid) if hasattr(self._model, "vocab") else str(pid)
                        all_tokens.append(tok_str if tok_str is not None else str(pid))
                        all_offsets.append((global_char_offset, global_char_offset + len(pc)))
                        all_special_mask.append(0)

                    global_char_offset += len(pc)

        return Encoding(
            ids=all_ids,
            tokens=all_tokens,
            offsets=all_offsets,
            special_tokens_mask=all_special_mask,
            raw_text=text
        )

    cpdef BatchEncoding encode_batch(self, list texts, bint add_special_tokens=True, object padding=None, object max_length=None):
        """Batch encode a list of text strings."""
        cdef list encodings = []
        cdef str t
        for t in texts:
            encodings.append(self.encode(t, add_special_tokens=add_special_tokens))

        cdef BatchEncoding batch = BatchEncoding(encodings)
        if max_length is not None:
            batch.truncate(<size_t>max_length)
        if padding:
            batch.pad()
        return batch

    cpdef str decode(self, list ids, bint skip_special_tokens=True):
        """Decode token IDs back into natural text."""
        if not ids:
            return ""

        cdef list pieces = []
        cdef uint32_t tid
        cdef str tok

        for tid in ids:
            if self._special_tokens.is_special_id(tid):
                if not skip_special_tokens:
                    tok = self._special_tokens.get_token(tid)
                    if tok is not None:
                        pieces.append(tok)
                continue

            if hasattr(self._model, "vocab"):
                tok = self._model.vocab.get_token(tid)
                if tok is not None:
                    pieces.append(tok)
                else:
                    pieces.append("")
            else:
                pieces.append("")

        cdef str raw_concat = "".join(pieces)
        # If decoder is configured, pass through decoder
        if self._decoder is not None and hasattr(self._decoder, "decode"):
            return self._decoder.decode(raw_concat)

        return raw_concat

    cpdef list decode_batch(self, list batch_ids, bint skip_special_tokens=True):
        """Batch decode multiple sequences of token IDs."""
        cdef list res = []
        cdef list ids
        for ids in batch_ids:
            res.append(self.decode(ids, skip_special_tokens=skip_special_tokens))
        return res

    @property
    def model(self):
        return self._model

    @property
    def normalizer(self):
        return self._normalizer

    @property
    def pre_tokenizer(self):
        return self._pre_tokenizer

    @property
    def special_tokens(self) -> SpecialTokenRegistry:
        return self._special_tokens
