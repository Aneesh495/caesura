# cython: language_level=3
from caesura.pipeline.encoding cimport Encoding

cdef class BatchEncoding:
    cdef list _encodings

    cpdef void pad(self, object length=*, int pad_id=*, str pad_token=*, str direction=*)
    cpdef void truncate(self, size_t max_length, str direction=*)
    cpdef dict as_dict(self)
