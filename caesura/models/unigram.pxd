# cython: language_level=3
from libc.stdint cimport uint32_t
from caesura.vocab.vocab cimport Vocabulary

cdef class UnigramModel:
    cdef Vocabulary _vocab
    cdef dict _scores  # token_id -> float log_prob
    cdef str _unk_token
    cdef uint32_t _unk_id
    cdef double _unk_score

    cpdef list encode(self, str text)
    cpdef list encode_with_offsets(self, str text)
    cpdef double score(self, str token)
    cpdef double score_sequence(self, list ids)
