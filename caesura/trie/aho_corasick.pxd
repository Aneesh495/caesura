# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, int32_t
from libc.stddef cimport size_t

cdef struct AcNode:
    int32_t[256] transitions
    int32_t fail
    uint32_t token_id
    uint32_t pattern_len
    bint is_match

cdef class AhoCorasick:
    cdef AcNode* _nodes
    cdef size_t _capacity
    cdef size_t _num_nodes
    cdef bint _compiled

    cdef int32_t c_alloc_node(self) noexcept nogil
    cdef void c_add_pattern(self, const uint8_t* pattern, size_t pat_len, uint32_t token_id) noexcept nogil
    cdef void c_compile(self)
