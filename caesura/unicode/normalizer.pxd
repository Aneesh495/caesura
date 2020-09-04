# cython: language_level=3

cdef class Normalizer:
    cpdef str normalize(self, str text)

cdef class LowercaseNormalizer(Normalizer):
    cpdef str normalize(self, str text)

cdef class StripAccentsNormalizer(Normalizer):
    cpdef str normalize(self, str text)

cdef class NfcNormalizer(Normalizer):
    cpdef str normalize(self, str text)

cdef class NfkdNormalizer(Normalizer):
    cpdef str normalize(self, str text)

cdef class MetaspaceNormalizer(Normalizer):
    cdef str _replacement
    cdef bint _prepend_space

    cpdef str normalize(self, str text)

cdef class SequenceNormalizer(Normalizer):
    cdef list _normalizers

    cpdef str normalize(self, str text)
