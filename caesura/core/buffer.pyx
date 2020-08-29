# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Zero-Copy Buffer and Contiguous Token Array Primitives
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy, memmove, memset
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE

cdef class ByteBuffer:
    """
    Zero-copy contiguous byte sequence supporting the Python buffer protocol.
    Can borrow from existing Python bytes/memoryviews or manage its own heap buffer.
    """

    def __cinit__(self, object source=None, size_t initial_capacity=0):
        cdef Py_buffer pybuf
        self._data = NULL
        self._len = 0
        self._capacity = 0
        self._owns_memory = True
        self._py_obj = None

        if source is not None:
            if isinstance(source, (bytes, bytearray, memoryview)):
                self._py_obj = source
                self._owns_memory = False
                if PyObject_GetBuffer(source, &pybuf, PyBUF_SIMPLE) == 0:
                    self._data = <uint8_t*>pybuf.buf
                    self._len = <size_t>pybuf.len
                    self._capacity = self._len
                    PyBuffer_Release(&pybuf)
            elif isinstance(source, int):
                initial_capacity = <size_t>source

        if self._data == NULL and initial_capacity > 0:
            self._data = <uint8_t*>malloc(initial_capacity)
            if self._data == NULL:
                raise MemoryError("Failed to allocate ByteBuffer")
            self._capacity = initial_capacity
            self._len = 0
            self._owns_memory = True

    def __dealloc__(self):
        if self._owns_memory and self._data != NULL:
            free(self._data)
            self._data = NULL

    cdef const uint8_t* c_data(self) nogil:
        return self._data

    cdef size_t c_len(self) nogil:
        return self._len

    cdef void c_append(self, const uint8_t* src, size_t length):
        cdef uint8_t* new_buf
        cdef size_t required
        cdef size_t new_cap

        if length == 0 or src == NULL:
            return
        if not self._owns_memory:
            # Must detach and copy to become independent owner
            new_buf = <uint8_t*>malloc(self._len + length + 16)
            if new_buf == NULL:
                raise MemoryError("Failed to reallocate ByteBuffer")
            if self._len > 0:
                memcpy(new_buf, self._data, self._len)
            memcpy(new_buf + self._len, src, length)
            self._data = new_buf
            self._len += length
            self._capacity = self._len + 16
            self._owns_memory = True
            self._py_obj = None
            return

        required = self._len + length
        if required > self._capacity:
            new_cap = self._capacity * 2 if self._capacity > 0 else 64
            while new_cap < required:
                new_cap *= 2
            self._data = <uint8_t*>realloc(self._data, new_cap)
            if self._data == NULL:
                raise MemoryError("Failed to expand ByteBuffer")
            self._capacity = new_cap

        memcpy(self._data + self._len, src, length)
        self._len += length

    cdef void c_clear(self):
        self._len = 0

    def append(self, bytes data):
        """Append raw bytes to the buffer."""
        cdef const uint8_t* p = <const uint8_t*>data
        cdef size_t l = len(data)
        self.c_append(p, l)

    def to_bytes(self) -> bytes:
        """Convert the buffer to standard immutable Python bytes."""
        if self._data == NULL or self._len == 0:
            return b""
        return (<char*>self._data)[:self._len]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, size_t idx):
        if idx >= self._len:
            raise IndexError("Index out of bounds")
        return self._data[idx]

    def __repr__(self) -> str:
        return f"<ByteBuffer len={self._len} capacity={self._capacity} owns={self._owns_memory}>"


cdef class TokenBuffer:
    """
    High-speed dynamic array of 32-bit unsigned integer token IDs.
    """

    def __cinit__(self, size_t initial_capacity=64):
        self._data = NULL
        self._len = 0
        self._capacity = 0
        self._owns_memory = True

        if initial_capacity > 0:
            self._data = <uint32_t*>malloc(initial_capacity * sizeof(uint32_t))
            if self._data == NULL:
                raise MemoryError("Failed to allocate TokenBuffer")
            self._capacity = initial_capacity

    def __dealloc__(self):
        if self._owns_memory and self._data != NULL:
            free(self._data)
            self._data = NULL

    cdef uint32_t* c_ptr(self) noexcept nogil:
        return self._data

    cdef size_t c_len(self) noexcept nogil:
        return self._len

    cdef void c_reserve(self, size_t new_cap) noexcept nogil:
        if new_cap <= self._capacity:
            return
        cdef uint32_t* p = <uint32_t*>realloc(self._data, new_cap * sizeof(uint32_t))
        if p != NULL:
            self._data = p
            self._capacity = new_cap

    cdef void c_append(self, uint32_t token_id) noexcept nogil:
        if self._len >= self._capacity:
            self.c_reserve(self._capacity * 2 if self._capacity > 0 else 32)
        self._data[self._len] = token_id
        self._len += 1

    cdef void c_clear(self) noexcept nogil:
        self._len = 0

    def append(self, uint32_t token_id):
        self.c_append(token_id)

    def extend(self, list ids):
        cdef uint32_t tid
        for tid in ids:
            self.c_append(tid)

    def to_list(self) -> list:
        cdef list res = []
        cdef size_t i
        for i in range(self._len):
            res.append(self._data[i])
        return res

    def clear(self):
        self.c_clear()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, size_t idx):
        if idx >= self._len:
            raise IndexError("Index out of bounds")
        return self._data[idx]

    def __repr__(self) -> str:
        return f"<TokenBuffer len={self._len} capacity={self._capacity}>"
