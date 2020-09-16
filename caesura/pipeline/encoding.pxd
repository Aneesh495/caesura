# cython: language_level=3
from libc.stdint cimport uint32_t
from libc.stddef cimport size_t

cdef class Encoding:
    cdef list _ids
    cdef list _tokens
    cdef list _type_ids
    cdef list _attention_mask
    cdef list _special_tokens_mask
    cdef list _offsets
    cdef str _raw_text

    cpdef void truncate(self, size_t max_length, str direction=*)
    cpdef void pad(self, size_t length, uint32_t pad_id=*, str pad_token=*, str direction=*)
    cpdef object char_to_token(self, size_t char_pos)
    cpdef object token_to_char(self, size_t token_idx)
