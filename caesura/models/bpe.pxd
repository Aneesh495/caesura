# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t
from libc.stddef cimport size_t
from caesura.core.types cimport (
    CaesuraPairMap,
    CaesuraPairSlot,
    TokenNode,
    BpeMinHeapEntry,
    caesura_pair_map_create,
    caesura_pair_map_destroy,
    caesura_pair_map_insert,
    caesura_pair_map_get,
    caesura_hash_pair,
)
from caesura.core.memory cimport MemoryArena
from caesura.vocab.vocab cimport Vocabulary

cdef class BpeModel:
    cdef CaesuraPairMap* _pair_map
    cdef Vocabulary _vocab
    cdef MemoryArena _arena
    cdef uint32_t _unk_id
    cdef bint _byte_fallback

    cdef size_t c_encode_chunk(
        self,
        const uint8_t* chunk_bytes,
        size_t chunk_len,
        size_t chunk_offset,
        uint32_t* out_ids,
        size_t* out_offsets_start,
        size_t* out_offsets_end,
        size_t max_out
    ) noexcept nogil

    cpdef list encode(self, str text)
    cpdef list encode_with_offsets(self, str text)
