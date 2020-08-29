# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t
from libc.stddef cimport size_t
from caesura.core.types cimport (
    CaesuraArena,
    CaesuraArenaMarker,
    caesura_arena_create,
    caesura_arena_destroy,
    caesura_arena_reset,
    caesura_arena_alloc,
    caesura_arena_alloc_aligned,
    caesura_arena_mark,
    caesura_arena_rewind,
    caesura_arena_total_bytes,
    caesura_arena_used_bytes,
)

cdef class MemoryArena:
    cdef CaesuraArena* _arena
    cdef size_t _default_block_size

    cdef void* c_alloc(self, size_t size) noexcept nogil
    cdef void* c_alloc_aligned(self, size_t size, size_t alignment) noexcept nogil
    cdef CaesuraArenaMarker c_mark(self) noexcept nogil
    cdef void c_rewind(self, CaesuraArenaMarker marker) noexcept nogil
    cdef void c_reset(self) noexcept nogil
