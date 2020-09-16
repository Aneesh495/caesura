# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Encoding Container with Alignment Spans, Padding, and Truncation
"""

cdef class Encoding:
    """
    Encapsulates token IDs, string pieces, type IDs, attention masks,
    special token indicators, and character alignment spans.
    """

    def __init__(
        self,
        list ids,
        list tokens=None,
        list type_ids=None,
        list attention_mask=None,
        list special_tokens_mask=None,
        list offsets=None,
        str raw_text=""
    ):
        cdef size_t n = len(ids)
        self._ids = list(ids)
        self._tokens = list(tokens) if tokens is not None else [str(x) for x in ids]
        self._type_ids = list(type_ids) if type_ids is not None else [0] * n
        self._attention_mask = list(attention_mask) if attention_mask is not None else [1] * n
        self._special_tokens_mask = list(special_tokens_mask) if special_tokens_mask is not None else [0] * n
        self._offsets = list(offsets) if offsets is not None else [(0, 0)] * n
        self._raw_text = raw_text

    cpdef void truncate(self, size_t max_length, str direction="right"):
        """Truncate sequence to max_length along given direction ('left' or 'right')."""
        if len(self._ids) <= max_length:
            return

        if direction == "left":
            self._ids = self._ids[-max_length:]
            self._tokens = self._tokens[-max_length:]
            self._type_ids = self._type_ids[-max_length:]
            self._attention_mask = self._attention_mask[-max_length:]
            self._special_tokens_mask = self._special_tokens_mask[-max_length:]
            self._offsets = self._offsets[-max_length:]
        else:
            self._ids = self._ids[:max_length]
            self._tokens = self._tokens[:max_length]
            self._type_ids = self._type_ids[:max_length]
            self._attention_mask = self._attention_mask[:max_length]
            self._special_tokens_mask = self._special_tokens_mask[:max_length]
            self._offsets = self._offsets[:max_length]

    cpdef void pad(self, size_t length, uint32_t pad_id=0, str pad_token="<pad>", str direction="right"):
        """Pad sequence up to length with pad_id and zero attention mask."""
        cdef size_t curr_len = len(self._ids)
        if curr_len >= length:
            return

        cdef size_t pad_count = length - curr_len
        cdef list pad_ids = [pad_id] * pad_count
        cdef list pad_toks = [pad_token] * pad_count
        cdef list pad_types = [0] * pad_count
        cdef list pad_mask = [0] * pad_count
        cdef list pad_spec = [1] * pad_count
        cdef list pad_offsets = [(0, 0)] * pad_count

        if direction == "left":
            self._ids = pad_ids + self._ids
            self._tokens = pad_toks + self._tokens
            self._type_ids = pad_types + self._type_ids
            self._attention_mask = pad_mask + self._attention_mask
            self._special_tokens_mask = pad_spec + self._special_tokens_mask
            self._offsets = pad_offsets + self._offsets
        else:
            self._ids.extend(pad_ids)
            self._tokens.extend(pad_toks)
            self._type_ids.extend(pad_types)
            self._attention_mask.extend(pad_mask)
            self._special_tokens_mask.extend(pad_spec)
            self._offsets.extend(pad_offsets)

    cpdef object char_to_token(self, size_t char_pos):
        """Map a character position in the original string to its token index."""
        cdef size_t idx
        cdef size_t s, e
        for idx in range(len(self._offsets)):
            s, e = self._offsets[idx]
            if s <= char_pos < e:
                return idx
        return None

    cpdef object token_to_char(self, size_t token_idx):
        """Map a token index to its (start_char, end_char) span in the original string."""
        if token_idx < len(self._offsets):
            return self._offsets[token_idx]
        return None

    @property
    def ids(self) -> list:
        return self._ids

    @property
    def tokens(self) -> list:
        return self._tokens

    @property
    def type_ids(self) -> list:
        return self._type_ids

    @property
    def attention_mask(self) -> list:
        return self._attention_mask

    @property
    def special_tokens_mask(self) -> list:
        return self._special_tokens_mask

    @property
    def offsets(self) -> list:
        return self._offsets

    @property
    def raw_text(self) -> str:
        return self._raw_text

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:
        return f"<Encoding len={len(self._ids)} tokens={self._tokens[:5]}{'...' if len(self._tokens) > 5 else ''}>"
