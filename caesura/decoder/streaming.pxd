# cython: language_level=3
from caesura.unicode.utf8 import ByteFallback

cdef class IncrementalDecoder:
    cdef bytearray _buffer
    cdef object _bf

    cpdef str decode_step(self, str token_piece)
    cpdef str decode_bytes_step(self, bytes raw_bytes)
    cpdef str flush(self)
