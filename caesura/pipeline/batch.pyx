# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Batch Encoding and Tensor Buffer Container
"""

from libc.stdint cimport uint32_t
from caesura.pipeline.encoding cimport Encoding

cdef class BatchEncoding:
    """
    Manages a batch of Encoding instances with dynamic padding, truncation,
    and direct dictionary export for deep learning inference frameworks.
    """

    def __init__(self, list encodings=None):
        self._encodings = list(encodings) if encodings is not None else []

    cpdef void pad(self, object length=None, int pad_id=0, str pad_token="<pad>", str direction="right"):
        """Pad all sequences in batch to specified length or batch maximum."""
        if not self._encodings:
            return

        cdef size_t target_len
        cdef size_t max_l = 0
        cdef Encoding enc

        if length is not None:
            target_len = <size_t>length
        else:
            for enc in self._encodings:
                if len(enc) > max_l:
                    max_l = len(enc)
            target_len = max_l

        for enc in self._encodings:
            enc.pad(target_len, pad_id=<uint32_t>pad_id, pad_token=pad_token, direction=direction)

    cpdef void truncate(self, size_t max_length, str direction="right"):
        """Truncate all sequences in batch to max_length."""
        cdef Encoding enc
        for enc in self._encodings:
            enc.truncate(max_length, direction=direction)

    cpdef dict as_dict(self):
        """Export batch as dictionary matching standard deep learning frameworks."""
        cdef list input_ids = []
        cdef list attention_mask = []
        cdef list type_ids = []
        cdef Encoding enc

        for enc in self._encodings:
            input_ids.append(enc.ids)
            attention_mask.append(enc.attention_mask)
            type_ids.append(enc.type_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": type_ids,
        }

    @property
    def encodings(self) -> list:
        return self._encodings

    def __len__(self) -> int:
        return len(self._encodings)

    def __iter__(self):
        return iter(self._encodings)

    def __getitem__(self, object idx) -> Encoding:
        return self._encodings[idx]

    def __repr__(self) -> str:
        return f"<BatchEncoding size={len(self._encodings)}>"
