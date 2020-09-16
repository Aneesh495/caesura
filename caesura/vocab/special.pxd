# cython: language_level=3
from libc.stdint cimport uint32_t
from caesura.trie.aho_corasick cimport AhoCorasick

cdef class SpecialTokenRegistry:
    cdef dict _token_to_id
    cdef dict _id_to_token
    cdef dict _role_to_token
    cdef AhoCorasick _matcher
    cdef bint _dirty

    cpdef void register(self, str token, uint32_t token_id, str role=*)
    cpdef bint is_special_token(self, str token)
    cpdef bint is_special_id(self, uint32_t token_id)
    cpdef object get_id(self, str token)
    cpdef str get_token(self, uint32_t token_id)
    cpdef object get_role_id(self, str role)
