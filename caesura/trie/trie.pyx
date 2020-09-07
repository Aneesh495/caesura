# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Prefix Trie Implementation for Fast Subword Lexicon Lookups
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset

cdef class Trie:
    """
    Array-backed Trie supporting exact lookups, prefix validation, and
    greedy longest-prefix subword matching.
    """

    def __cinit__(self, size_t initial_capacity=1024):
        if initial_capacity < 16:
            initial_capacity = 16
        self._capacity = initial_capacity
        self._nodes = <TrieNodeInternal*>malloc(initial_capacity * sizeof(TrieNodeInternal))
        if self._nodes == NULL:
            raise MemoryError("Failed to allocate Trie nodes")
        self._num_nodes = 0
        self._num_words = 0

        # Allocate root node at index 0
        self.c_alloc_node()

    def __dealloc__(self):
        if self._nodes != NULL:
            free(self._nodes)
            self._nodes = NULL

    cdef int32_t c_alloc_node(self) noexcept nogil:
        cdef size_t new_cap
        cdef TrieNodeInternal* new_nodes
        cdef int32_t idx

        if self._num_nodes >= self._capacity:
            new_cap = self._capacity * 2
            new_nodes = <TrieNodeInternal*>realloc(self._nodes, new_cap * sizeof(TrieNodeInternal))
            if new_nodes == NULL:
                return -1
            self._nodes = new_nodes
            self._capacity = new_cap

        idx = <int32_t>self._num_nodes
        self._num_nodes += 1
        self._nodes[idx].token_id = 0
        self._nodes[idx].is_terminal = 0
        memset(self._nodes[idx].children, 0xFF, 256 * sizeof(int32_t))  # set all to -1
        return idx

    cdef void c_insert(self, const uint8_t* key, size_t key_len, uint32_t token_id) noexcept nogil:
        if key == NULL or key_len == 0:
            return

        cdef int32_t curr = 0
        cdef size_t i
        cdef uint8_t b
        cdef int32_t next_idx

        for i in range(key_len):
            b = key[i]
            next_idx = self._nodes[curr].children[b]
            if next_idx == -1:
                next_idx = self.c_alloc_node()
                if next_idx == -1:
                    return
                self._nodes[curr].children[b] = next_idx
            curr = next_idx

        if not self._nodes[curr].is_terminal:
            self._num_words += 1
        self._nodes[curr].is_terminal = 1
        self._nodes[curr].token_id = token_id

    cdef bint c_find(self, const uint8_t* key, size_t key_len, uint32_t* out_token_id) noexcept nogil:
        if key == NULL or key_len == 0:
            return 0

        cdef int32_t curr = 0
        cdef size_t i
        cdef uint8_t b

        for i in range(key_len):
            b = key[i]
            curr = self._nodes[curr].children[b]
            if curr == -1:
                return 0

        if self._nodes[curr].is_terminal:
            if out_token_id != NULL:
                out_token_id[0] = self._nodes[curr].token_id
            return 1
        return 0

    cdef size_t c_longest_prefix(self, const uint8_t* key, size_t key_len, uint32_t* out_token_id) noexcept nogil:
        if key == NULL or key_len == 0:
            return 0

        cdef int32_t curr = 0
        cdef size_t longest_len = 0
        cdef uint32_t best_token_id = 0
        cdef size_t i
        cdef uint8_t b

        for i in range(key_len):
            b = key[i]
            curr = self._nodes[curr].children[b]
            if curr == -1:
                break
            if self._nodes[curr].is_terminal:
                longest_len = i + 1
                best_token_id = self._nodes[curr].token_id

        if longest_len > 0 and out_token_id != NULL:
            out_token_id[0] = best_token_id

        return longest_len

    def insert(self, object key, uint32_t token_id):
        """Insert a string or bytes token with its integer token_id."""
        cdef bytes bkey
        if isinstance(key, str):
            bkey = key.encode("utf-8")
        elif isinstance(key, bytes):
            bkey = key
        else:
            raise TypeError("Key must be str or bytes")

        cdef const uint8_t* p = <const uint8_t*>bkey
        cdef size_t l = len(bkey)
        with nogil:
            self.c_insert(p, l, token_id)

    def find(self, object key):
        """Return token_id if key exists in Trie, else None."""
        cdef bytes bkey
        if isinstance(key, str):
            bkey = key.encode("utf-8")
        elif isinstance(key, bytes):
            bkey = key
        else:
            return None

        cdef const uint8_t* p = <const uint8_t*>bkey
        cdef size_t l = len(bkey)
        cdef uint32_t tid
        cdef bint found
        with nogil:
            found = self.c_find(p, l, &tid)
        return tid if found else None

    def longest_prefix(self, object key):
        """Return tuple of (prefix_len, token_id) for the longest matching prefix."""
        cdef bytes bkey
        if isinstance(key, str):
            bkey = key.encode("utf-8")
        elif isinstance(key, bytes):
            bkey = key
        else:
            return (0, None)

        cdef const uint8_t* p = <const uint8_t*>bkey
        cdef size_t l = len(bkey)
        cdef uint32_t tid = 0
        cdef size_t match_len
        with nogil:
            match_len = self.c_longest_prefix(p, l, &tid)

        if match_len > 0:
            return (match_len, tid)
        return (0, None)

    def __contains__(self, object key) -> bool:
        return self.find(key) is not None

    def __len__(self) -> int:
        return self._num_words

    @property
    def node_count(self) -> int:
        return self._num_nodes
