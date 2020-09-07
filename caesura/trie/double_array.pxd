# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, int32_t
from libc.stddef cimport size_t

cdef class DoubleArrayTrie:
    cdef int32_t* _base
    cdef int32_t* _check
    cdef uint32_t* _values
    cdef bint* _is_leaf
    cdef size_t _capacity
    cdef size_t _size

    cdef void c_reserve(self, size_t new_cap) noexcept nogil
    cdef bint c_exact_match(self, const uint8_t* key, size_t len, uint32_t* out_val) noexcept nogil
    cdef size_t c_longest_prefix(self, const uint8_t* key, size_t len, uint32_t* out_val) noexcept nogil
