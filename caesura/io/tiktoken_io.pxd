# cython: language_level=3
from caesura.pipeline.tokenizer cimport Tokenizer

cdef class TiktokenLoader:
    cpdef Tokenizer load_tiktoken(self, str file_path, dict special_tokens=*, str pattern=*)
    cpdef void save_tiktoken(self, Tokenizer tokenizer, str file_path)
