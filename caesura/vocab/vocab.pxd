# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t

cdef class Vocabulary:
    cdef dict _token_to_id
    cdef list _id_to_token
    cdef size_t _size

    cpdef int add_token(self, str token, object token_id=*)
    cpdef object get_id(self, str token)
    cpdef str get_token(self, uint32_t token_id)
    cpdef bint contains(self, str token)
