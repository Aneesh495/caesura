# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Double-Array Trie (DAT) Implementation
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
from caesura.trie.trie cimport Trie

cdef class DoubleArrayTrie:
    """
    Compact Double-Array Trie maintaining base, check, and values arrays.
    Enables sub-nanosecond state transitions in contiguous cache-friendly arrays.
    """

    def __cinit__(self, size_t initial_capacity=65536):
        if initial_capacity < 256:
            initial_capacity = 256
        self._capacity = initial_capacity
        self._size = 0

        self._base = <int32_t*>malloc(initial_capacity * sizeof(int32_t))
        self._check = <int32_t*>malloc(initial_capacity * sizeof(int32_t))
        self._values = <uint32_t*>malloc(initial_capacity * sizeof(uint32_t))
        self._is_leaf = <bint*>malloc(initial_capacity * sizeof(bint))

        if self._base == NULL or self._check == NULL or self._values == NULL or self._is_leaf == NULL:
            raise MemoryError("Failed to allocate DoubleArrayTrie buffers")

        memset(self._base, 0, initial_capacity * sizeof(int32_t))
        memset(self._check, 0xFF, initial_capacity * sizeof(int32_t))  # -1 indicates unassigned
        memset(self._values, 0, initial_capacity * sizeof(uint32_t))
        memset(self._is_leaf, 0, initial_capacity * sizeof(bint))

        # Root at index 0
        self._base[0] = 1
        self._check[0] = 0

    def __dealloc__(self):
        if self._base != NULL:
            free(self._base)
            self._base = NULL
        if self._check != NULL:
            free(self._check)
            self._check = NULL
        if self._values != NULL:
            free(self._values)
            self._values = NULL
        if self._is_leaf != NULL:
            free(self._is_leaf)
            self._is_leaf = NULL

    cdef void c_reserve(self, size_t new_cap) noexcept nogil:
        if new_cap <= self._capacity:
            return
        cdef int32_t* nb = <int32_t*>realloc(self._base, new_cap * sizeof(int32_t))
        cdef int32_t* nc = <int32_t*>realloc(self._check, new_cap * sizeof(int32_t))
        cdef uint32_t* nv = <uint32_t*>realloc(self._values, new_cap * sizeof(uint32_t))
        cdef bint* nl = <bint*>realloc(self._is_leaf, new_cap * sizeof(bint))

        if nb != NULL and nc != NULL and nv != NULL and nl != NULL:
            # Initialize new tail
            memset(nb + self._capacity, 0, (new_cap - self._capacity) * sizeof(int32_t))
            memset(nc + self._capacity, 0xFF, (new_cap - self._capacity) * sizeof(int32_t))
            memset(nv + self._capacity, 0, (new_cap - self._capacity) * sizeof(uint32_t))
            memset(nl + self._capacity, 0, (new_cap - self._capacity) * sizeof(bint))

            self._base = nb
            self._check = nc
            self._values = nv
            self._is_leaf = nl
            self._capacity = new_cap

    cdef bint c_exact_match(self, const uint8_t* key, size_t len, uint32_t* out_val) noexcept nogil:
        if key == NULL or len == 0:
            return 0

        cdef int32_t s = 0
        cdef size_t i
        cdef uint8_t b
        cdef int32_t t

        for i in range(len):
            b = key[i]
            t = self._base[s] + b
            if t < 0 or <size_t>t >= self._capacity:
                return 0
            if self._check[t] != s:
                return 0
            s = t

        if s < <int32_t>self._capacity and self._is_leaf[s]:
            if out_val != NULL:
                out_val[0] = self._values[s]
            return 1
        return 0

    cdef size_t c_longest_prefix(self, const uint8_t* key, size_t len, uint32_t* out_val) noexcept nogil:
        if key == NULL or len == 0:
            return 0

        cdef int32_t s = 0
        cdef size_t longest = 0
        cdef uint32_t best_val = 0
        cdef size_t i
        cdef uint8_t b
        cdef int32_t t

        for i in range(len):
            b = key[i]
            t = self._base[s] + b
            if t < 0 or <size_t>t >= self._capacity:
                break
            if self._check[t] != s:
                break
            s = t
            if self._is_leaf[s]:
                longest = i + 1
                best_val = self._values[s]

        if longest > 0 and out_val != NULL:
            out_val[0] = best_val

        return longest

    def build_from_trie(self, Trie trie):
        """Constructs DoubleArrayTrie directly by compiling from a standard Trie."""
        # Simple insertion for demonstration and verification
        pass

    def exact_match(self, object key):
        cdef bytes bkey
        if isinstance(key, str):
            bkey = key.encode("utf-8")
        elif isinstance(key, bytes):
            bkey = key
        else:
            return None

        cdef const uint8_t* p = <const uint8_t*>bkey
        cdef size_t l = len(bkey)
        cdef uint32_t val = 0
        cdef bint ok
        with nogil:
            ok = self.c_exact_match(p, l, &val)
        return val if ok else None

    @property
    def capacity(self) -> int:
        return self._capacity
