# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t

cdef bint c_is_whitespace(uint32_t cp) noexcept nogil
cdef bint c_is_punctuation(uint32_t cp) noexcept nogil
cdef bint c_is_alphanumeric(uint32_t cp) noexcept nogil
cdef bint c_is_control(uint32_t cp) noexcept nogil
cdef bint c_is_cjk(uint32_t cp) noexcept nogil
