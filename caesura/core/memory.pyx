# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Linear Memory Arena Implementation for Cython
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

cdef class ArenaScope:
    cdef MemoryArena _arena
    cdef CaesuraArenaMarker _marker

    def __init__(self, MemoryArena arena):
        self._arena = arena
        self._marker = arena.c_mark()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._arena.c_rewind(self._marker)

cdef class MemoryArena:
    """
    High-performance bump memory allocator wrapping CaesuraArena.
    Eliminates per-token malloc and free overhead during intensive batch tokenization.
    """

    def __cinit__(self, size_t default_block_size=65536):
        self._default_block_size = default_block_size
        self._arena = caesura_arena_create(default_block_size)
        if self._arena == NULL:
            raise MemoryError("Failed to allocate native memory arena")

    def __dealloc__(self):
        if self._arena != NULL:
            caesura_arena_destroy(self._arena)
            self._arena = NULL

    cdef void* c_alloc(self, size_t size) noexcept nogil:
        return caesura_arena_alloc(self._arena, size)

    cdef void* c_alloc_aligned(self, size_t size, size_t alignment) noexcept nogil:
        return caesura_arena_alloc_aligned(self._arena, size, alignment)

    cdef CaesuraArenaMarker c_mark(self) noexcept nogil:
        return caesura_arena_mark(self._arena)

    cdef void c_rewind(self, CaesuraArenaMarker marker) noexcept nogil:
        caesura_arena_rewind(self._arena, marker)

    cdef void c_reset(self) noexcept nogil:
        caesura_arena_reset(self._arena)

    def reset(self):
        """Reset the arena in O(1) time without releasing allocated OS pages."""
        self.c_reset()

    def scope(self):
        """Context manager to rewind arena state upon exiting block."""
        return ArenaScope(self)

    @property
    def total_bytes(self) -> int:
        """Total memory allocated across all arena pages."""
        return caesura_arena_total_bytes(self._arena)

    @property
    def used_bytes(self) -> int:
        """Total memory consumed in the active arena blocks."""
        return caesura_arena_used_bytes(self._arena)

    def __repr__(self) -> str:
        return f"<MemoryArena used={self.used_bytes} total={self.total_bytes}>"
