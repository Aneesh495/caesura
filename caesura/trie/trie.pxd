# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, int32_t
from libc.stddef cimport size_t

cdef struct TrieNodeInternal:
    uint32_t token_id
    bint is_terminal
    int32_t[256] children

cdef class Trie:
    cdef TrieNodeInternal* _nodes
    cdef size_t _capacity
    cdef size_t _num_nodes
    cdef size_t _num_words

    cdef int32_t c_alloc_node(self) noexcept nogil
    cdef void c_insert(self, const uint8_t* key, size_t key_len, uint32_t token_id) noexcept nogil
    cdef bint c_find(self, const uint8_t* key, size_t key_len, uint32_t* out_token_id) noexcept nogil
    cdef size_t c_longest_prefix(self, const uint8_t* key, size_t key_len, uint32_t* out_token_id) noexcept nogil
    cdef void _fuzzy_dfs(self, int32_t node, bytes prefix, bytes target, list prev_row, list results, int max_edits)
