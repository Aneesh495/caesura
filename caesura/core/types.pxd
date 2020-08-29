# cython: language_level=3
"""
Caesura Core Types and Native Signatures
"""

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t
from libc.stddef cimport size_t
from libcpp cimport bool

cdef extern from "arena.h" nogil:
    ctypedef struct CaesuraArenaBlock:
        CaesuraArenaBlock* next
        size_t capacity
        size_t used
        uint8_t data[]

    ctypedef struct CaesuraArena:
        CaesuraArenaBlock* head
        CaesuraArenaBlock* current
        size_t default_block_size
        size_t total_allocated
        size_t peak_used

    ctypedef struct CaesuraArenaMarker:
        CaesuraArenaBlock* block
        size_t used

    CaesuraArena* caesura_arena_create(size_t default_block_size)
    void caesura_arena_destroy(CaesuraArena* arena)
    void caesura_arena_reset(CaesuraArena* arena)
    void* caesura_arena_alloc(CaesuraArena* arena, size_t size)
    void* caesura_arena_alloc_aligned(CaesuraArena* arena, size_t size, size_t alignment)
    void* caesura_arena_alloc_zeroed(CaesuraArena* arena, size_t size)
    char* caesura_arena_strdup(CaesuraArena* arena, const char* str)
    void* caesura_arena_memdup(CaesuraArena* arena, const void* src, size_t size)
    CaesuraArenaMarker caesura_arena_mark(CaesuraArena* arena)
    void caesura_arena_rewind(CaesuraArena* arena, CaesuraArenaMarker marker)
    size_t caesura_arena_total_bytes(const CaesuraArena* arena)
    size_t caesura_arena_used_bytes(const CaesuraArena* arena)

cdef extern from "murmur3.h" nogil:
    uint32_t caesura_murmur3_32(const void* key, size_t len, uint32_t seed)
    void caesura_murmur3_128(const void* key, size_t len, uint32_t seed, void* out)
    uint64_t caesura_hash_pair(uint32_t left, uint32_t right)

cdef extern from "robin_hood.h" nogil:
    ctypedef struct CaesuraPairSlot:
        uint64_t key
        uint32_t rank
        uint32_t token_id
        uint16_t psl
        uint8_t occupied
        uint8_t _pad

    ctypedef struct CaesuraPairMap:
        CaesuraPairSlot* slots
        size_t capacity
        size_t size
        size_t mask
        size_t max_psl

    CaesuraPairMap* caesura_pair_map_create(size_t initial_capacity)
    void caesura_pair_map_destroy(CaesuraPairMap* map)
    void caesura_pair_map_clear(CaesuraPairMap* map)
    bool caesura_pair_map_insert(CaesuraPairMap* map, uint64_t key, uint32_t rank, uint32_t token_id)
    bool caesura_pair_map_get(const CaesuraPairMap* map, uint64_t key, uint32_t* out_rank, uint32_t* out_token_id)
    bool caesura_pair_map_contains(const CaesuraPairMap* map, uint64_t key)
    bool caesura_pair_map_remove(CaesuraPairMap* map, uint64_t key)
    size_t caesura_pair_map_size(const CaesuraPairMap* map)
    size_t caesura_pair_map_capacity(const CaesuraPairMap* map)

cdef extern from "simd_utf8.h" nogil:
    size_t caesura_scan_ascii(const uint8_t* ptr, size_t len)
    bool caesura_validate_utf8(const uint8_t* ptr, size_t len)
    uint32_t caesura_utf8_next_codepoint(const uint8_t** ptr, size_t* len)
    size_t caesura_utf8_encode_codepoint(uint32_t cp, uint8_t* out_buf)
    size_t caesura_utf8_lead_len(uint8_t byte)

# Pure C structs used throughout the Cython engine
ctypedef struct ByteSpan:
    const uint8_t* ptr
    size_t len
    size_t offset

ctypedef struct TokenNode:
    uint32_t token_id
    uint32_t byte_len
    size_t orig_offset
    int32_t prev
    int32_t next
    uint8_t flags

ctypedef struct BpeMinHeapEntry:
    uint32_t rank
    int32_t node_idx

ctypedef struct OffsetSpan:
    size_t start
    size_t end

ctypedef struct RawEncoding:
    uint32_t* ids
    size_t count
    size_t capacity
    OffsetSpan* offsets
    uint32_t* attention_mask
    uint32_t* special_tokens_mask
    uint32_t* type_ids
