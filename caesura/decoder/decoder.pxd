# cython: language_level=3

cdef class Decoder:
    cpdef str decode(self, str text)

cdef class ByteLevelDecoder(Decoder):
    cdef object _bf

    cpdef str decode(self, str text)

cdef class WordPieceDecoder(Decoder):
    cdef str _prefix
    cdef bint _cleanup_spaces

    cpdef str decode(self, str text)

cdef class MetaspaceDecoder(Decoder):
    cdef str _replacement
    cdef bint _add_prefix_space

    cpdef str decode(self, str text)

cdef class SequenceDecoder(Decoder):
    cdef list _decoders

    cpdef str decode(self, str text)
