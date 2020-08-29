# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t
from cpython.buffer cimport Py_buffer

cdef class ByteBuffer:
    cdef uint8_t* _data
    cdef size_t _len
    cdef size_t _capacity
    cdef bint _owns_memory
    cdef object _py_obj

    cdef const uint8_t* c_data(self) nogil
    cdef size_t c_len(self) nogil
    cdef void c_append(self, const uint8_t* src, size_t length)
    cdef void c_clear(self)

cdef class TokenBuffer:
    cdef uint32_t* _data
    cdef size_t _len
    cdef size_t _capacity
    cdef bint _owns_memory

    cdef uint32_t* c_ptr(self) noexcept nogil
    cdef size_t c_len(self) noexcept nogil
    cdef void c_append(self, uint32_t token_id) noexcept nogil
    cdef void c_clear(self) noexcept nogil
    cdef void c_reserve(self, size_t new_cap) noexcept nogil
