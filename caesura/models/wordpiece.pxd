# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t
from caesura.trie.trie cimport Trie
from caesura.vocab.vocab cimport Vocabulary

cdef class WordPieceModel:
    cdef Vocabulary _vocab
    cdef Trie _trie
    cdef str _unk_token
    cdef uint32_t _unk_id
    cdef str _continuing_prefix
    cdef size_t _max_input_chars_per_word

    cpdef list encode_word(self, str word)
    cpdef list encode_word_with_offsets(self, str word, size_t word_offset=*)
    cpdef list encode(self, object words)
    cpdef list encode_with_offsets(self, object words)
