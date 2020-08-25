/*
 * Caesura Native Core: Linear Memory Arena Allocator
 *
 * Provides cache friendly bump allocation with 64 byte alignment for
 * token nodes, temporary parse buffers, and intermediate tokenization
 * arrays. Memory can be reset in O(1) time to eliminate per token
 * malloc and free overhead during intensive batch processing.
 */

#ifndef CAESURA_ARENA_H
#define CAESURA_ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CAESURA_DEFAULT_PAGE_SIZE (64 * 1024) /* 64 KiB pages */
#define CAESURA_ALIGNMENT 16

typedef struct CaesuraArenaBlock {
    struct CaesuraArenaBlock* next;
    size_t capacity;
    size_t used;
    uint8_t data[];
} CaesuraArenaBlock;

typedef struct CaesuraArena {
    CaesuraArenaBlock* head;
    CaesuraArenaBlock* current;
    size_t default_block_size;
    size_t total_allocated;
    size_t peak_used;
} CaesuraArena;

typedef struct CaesuraArenaMarker {
    CaesuraArenaBlock* block;
    size_t used;
} CaesuraArenaMarker;

/* Lifecycle functions */
CaesuraArena* caesura_arena_create(size_t default_block_size);
void caesura_arena_destroy(CaesuraArena* arena);
void caesura_arena_reset(CaesuraArena* arena);

/* Allocation functions */
void* caesura_arena_alloc(CaesuraArena* arena, size_t size);
void* caesura_arena_alloc_aligned(CaesuraArena* arena, size_t size, size_t alignment);
void* caesura_arena_alloc_zeroed(CaesuraArena* arena, size_t size);
char* caesura_arena_strdup(CaesuraArena* arena, const char* str);
void* caesura_arena_memdup(CaesuraArena* arena, const void* src, size_t size);

/* Scope markers for localized rollbacks */
CaesuraArenaMarker caesura_arena_mark(CaesuraArena* arena);
void caesura_arena_rewind(CaesuraArena* arena, CaesuraArenaMarker marker);

/* Statistics */
size_t caesura_arena_total_bytes(const CaesuraArena* arena);
size_t caesura_arena_used_bytes(const CaesuraArena* arena);

#ifdef __cplusplus
}
#endif

#endif /* CAESURA_ARENA_H */
