# cython: language_level=3
from libc.stdint cimport uint32_t
from caesura.vocab.vocab cimport Vocabulary
from caesura.models.bpe cimport BpeModel

cdef class BpeTrainer:
    cdef size_t _vocab_size
    cdef size_t _min_frequency
    cdef list _special_tokens
    cdef bint _show_progress

    cpdef BpeModel train(self, list texts)
    cpdef BpeModel train_from_files(self, list file_paths)
