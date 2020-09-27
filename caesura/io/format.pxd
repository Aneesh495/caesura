# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, uint64_t
from caesura.pipeline.tokenizer cimport Tokenizer

cdef class BinarySerializer:
    cpdef void save(self, Tokenizer tokenizer, str file_path)
    cpdef Tokenizer load(self, str file_path)
    cpdef Tokenizer load_mmap(self, str file_path)
