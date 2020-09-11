# cython: language_level=3
from caesura.vocab.vocab cimport Vocabulary
from caesura.models.wordpiece cimport WordPieceModel

cdef class WordPieceTrainer:
    cdef size_t _vocab_size
    cdef size_t _min_frequency
    cdef list _special_tokens
    cdef str _continuing_prefix

    cpdef WordPieceModel train(self, list texts)
