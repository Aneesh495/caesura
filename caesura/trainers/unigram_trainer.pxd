# cython: language_level=3
from caesura.vocab.vocab cimport Vocabulary
from caesura.models.unigram cimport UnigramModel

cdef class UnigramTrainer:
    cdef size_t _vocab_size
    cdef list _special_tokens
    cdef str _unk_token

    cpdef UnigramModel train(self, list texts)
