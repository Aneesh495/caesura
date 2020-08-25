/*
 * Caesura Native Core: Robin Hood Hash Map
 *
 * Cache-aligned flat hash table utilizing Robin Hood hashing with
 * backward-shift deletion for integer key-value mapping (pair keys to
 * merge ranks, token IDs to frequencies). Provides O(1) expected lookups
 * with tightly bounded probe sequence lengths.
 */

#ifndef CAESURA_ROBIN_HOOD_H
#define CAESURA_ROBIN_HOOD_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CaesuraPairSlot {
    uint64_t key;       /* Packed pair: (left_id << 32) | right_id */
    uint32_t rank;      /* Merge rank */
    uint32_t token_id;  /* Resulting merged token ID */
    uint16_t psl;       /* Probe sequence length */
    uint8_t occupied;   /* 1 if occupied, 0 otherwise */
    uint8_t _pad;
} CaesuraPairSlot;

typedef struct CaesuraPairMap {
    CaesuraPairSlot* slots;
    size_t capacity;
    size_t size;
    size_t mask;
    size_t max_psl;
} CaesuraPairMap;

CaesuraPairMap* caesura_pair_map_create(size_t initial_capacity);
void caesura_pair_map_destroy(CaesuraPairMap* map);
void caesura_pair_map_clear(CaesuraPairMap* map);

bool caesura_pair_map_insert(CaesuraPairMap* map, uint64_t key, uint32_t rank, uint32_t token_id);
bool caesura_pair_map_get(const CaesuraPairMap* map, uint64_t key, uint32_t* out_rank, uint32_t* out_token_id);
bool caesura_pair_map_contains(const CaesuraPairMap* map, uint64_t key);
bool caesura_pair_map_remove(CaesuraPairMap* map, uint64_t key);

size_t caesura_pair_map_size(const CaesuraPairMap* map);
size_t caesura_pair_map_capacity(const CaesuraPairMap* map);

#ifdef __cplusplus
}
#endif

#endif /* CAESURA_ROBIN_HOOD_H */
