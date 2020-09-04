# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t
from caesura.core.types cimport (
    caesura_scan_ascii,
    caesura_validate_utf8,
    caesura_utf8_next_codepoint,
    caesura_utf8_encode_codepoint,
    caesura_utf8_lead_len,
)

cdef class Utf8Scanner:
    cdef const uint8_t* _data
    cdef size_t _len
    cdef size_t _pos

    cdef uint32_t c_next(self) noexcept nogil
    cdef bint c_has_more(self) noexcept nogil
    cdef size_t c_pos(self) noexcept nogil
    cdef void c_reset(self) noexcept nogil

cdef class ByteFallback:
    cdef uint32_t[256] _byte_to_unicode
    cdef uint8_t[65536] _unicode_to_byte
    cdef bint[65536] _has_byte_mapping

    cdef uint32_t c_byte_to_cp(self, uint8_t b) noexcept nogil
    cdef bint c_cp_to_byte(self, uint32_t cp, uint8_t* out_byte) noexcept nogil
