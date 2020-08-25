/*
 * Caesura Native Core: Linear Memory Arena Implementation
 */

#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static inline size_t align_up(size_t n, size_t alignment) {
    return (n + (alignment - 1)) & ~(alignment - 1);
}

static CaesuraArenaBlock* create_block(size_t capacity) {
    size_t total_size = sizeof(CaesuraArenaBlock) + capacity;
    CaesuraArenaBlock* block = (CaesuraArenaBlock*)malloc(total_size);
    if (!block) {
        return NULL;
    }
    block->next = NULL;
    block->capacity = capacity;
    block->used = 0;
    return block;
}

CaesuraArena* caesura_arena_create(size_t default_block_size) {
    if (default_block_size == 0) {
        default_block_size = CAESURA_DEFAULT_PAGE_SIZE;
    }
    CaesuraArena* arena = (CaesuraArena*)malloc(sizeof(CaesuraArena));
    if (!arena) {
        return NULL;
    }

    CaesuraArenaBlock* first = create_block(default_block_size);
    if (!first) {
        free(arena);
        return NULL;
    }

    arena->head = first;
    arena->current = first;
    arena->default_block_size = default_block_size;
    arena->total_allocated = sizeof(CaesuraArenaBlock) + default_block_size;
    arena->peak_used = 0;
    return arena;
}

void caesura_arena_destroy(CaesuraArena* arena) {
    if (!arena) {
        return;
    }
    CaesuraArenaBlock* cur = arena->head;
    while (cur) {
        CaesuraArenaBlock* next = cur->next;
        free(cur);
        cur = next;
    }
    free(arena);
}

void caesura_arena_reset(CaesuraArena* arena) {
    if (!arena) {
        return;
    }
    CaesuraArenaBlock* cur = arena->head;
    while (cur) {
        cur->used = 0;
        cur = cur->next;
    }
    arena->current = arena->head;
}

void* caesura_arena_alloc_aligned(CaesuraArena* arena, size_t size, size_t alignment) {
    if (!arena || size == 0) {
        return NULL;
    }
    if (alignment == 0) {
        alignment = CAESURA_ALIGNMENT;
    }

    CaesuraArenaBlock* block = arena->current;
    size_t current_addr = (size_t)(block->data + block->used);
    size_t aligned_addr = align_up(current_addr, alignment);
    size_t padding = aligned_addr - current_addr;
    size_t required = size + padding;

    if (block->used + required <= block->capacity) {
        void* ptr = (void*)aligned_addr;
        block->used += required;
        return ptr;
    }

    /* Current block exhausted: check if next pre-allocated block suffices or allocate new */
    size_t next_cap = arena->default_block_size;
    if (size > next_cap) {
        next_cap = align_up(size + alignment, CAESURA_ALIGNMENT);
    }

    CaesuraArenaBlock* new_block = NULL;
    if (block->next && block->next->capacity >= next_cap) {
        new_block = block->next;
        new_block->used = 0;
    } else {
        new_block = create_block(next_cap);
        if (!new_block) {
            return NULL;
        }
        new_block->next = block->next;
        block->next = new_block;
        arena->total_allocated += sizeof(CaesuraArenaBlock) + next_cap;
    }

    arena->current = new_block;
    size_t base = (size_t)new_block->data;
    size_t aligned_base = align_up(base, alignment);
    size_t pad = aligned_base - base;
    new_block->used = size + pad;
    return (void*)aligned_base;
}

void* caesura_arena_alloc(CaesuraArena* arena, size_t size) {
    return caesura_arena_alloc_aligned(arena, size, CAESURA_ALIGNMENT);
}

void* caesura_arena_alloc_zeroed(CaesuraArena* arena, size_t size) {
    void* ptr = caesura_arena_alloc(arena, size);
    if (ptr) {
        memset(ptr, 0, size);
    }
    return ptr;
}

char* caesura_arena_strdup(CaesuraArena* arena, const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char* copy = (char*)caesura_arena_alloc(arena, len + 1);
    if (copy) {
        memcpy(copy, str, len + 1);
    }
    return copy;
}

void* caesura_arena_memdup(CaesuraArena* arena, const void* src, size_t size) {
    if (!src || size == 0) return NULL;
    void* copy = caesura_arena_alloc(arena, size);
    if (copy) {
        memcpy(copy, src, size);
    }
    return copy;
}

CaesuraArenaMarker caesura_arena_mark(CaesuraArena* arena) {
    CaesuraArenaMarker marker;
    marker.block = arena ? arena->current : NULL;
    marker.used = arena && arena->current ? arena->current->used : 0;
    return marker;
}

void caesura_arena_rewind(CaesuraArena* arena, CaesuraArenaMarker marker) {
    if (!arena || !marker.block) return;
    arena->current = marker.block;
    arena->current->used = marker.used;
}

size_t caesura_arena_total_bytes(const CaesuraArena* arena) {
    return arena ? arena->total_allocated : 0;
}

size_t caesura_arena_used_bytes(const CaesuraArena* arena) {
    if (!arena) return 0;
    size_t total = 0;
    CaesuraArenaBlock* b = arena->head;
    while (b) {
        total += b->used;
        if (b == arena->current) break;
        b = b->next;
    }
    return total;
}
