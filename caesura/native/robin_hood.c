/*
 * Caesura Native Core: Robin Hood Hash Map Implementation
 */

#include "robin_hood.h"
#include <stdlib.h>
#include <string.h>

#define CAESURA_INITIAL_CAPACITY 64
#define CAESURA_MAX_LOAD_FACTOR 0.85

static inline uint64_t hash_key(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

static inline size_t next_pow2(size_t n) {
    if (n < CAESURA_INITIAL_CAPACITY) return CAESURA_INITIAL_CAPACITY;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

CaesuraPairMap* caesura_pair_map_create(size_t initial_capacity) {
    CaesuraPairMap* map = (CaesuraPairMap*)malloc(sizeof(CaesuraPairMap));
    if (!map) return NULL;

    size_t cap = next_pow2(initial_capacity);
    map->slots = (CaesuraPairSlot*)calloc(cap, sizeof(CaesuraPairSlot));
    if (!map->slots) {
        free(map);
        return NULL;
    }
    map->capacity = cap;
    map->size = 0;
    map->mask = cap - 1;
    map->max_psl = 0;
    return map;
}

void caesura_pair_map_destroy(CaesuraPairMap* map) {
    if (!map) return;
    free(map->slots);
    free(map);
}

void caesura_pair_map_clear(CaesuraPairMap* map) {
    if (!map) return;
    memset(map->slots, 0, map->capacity * sizeof(CaesuraPairSlot));
    map->size = 0;
    map->max_psl = 0;
}

static bool resize_map(CaesuraPairMap* map, size_t new_cap) {
    CaesuraPairSlot* old_slots = map->slots;
    size_t old_cap = map->capacity;

    CaesuraPairSlot* new_slots = (CaesuraPairSlot*)calloc(new_cap, sizeof(CaesuraPairSlot));
    if (!new_slots) return false;

    map->slots = new_slots;
    map->capacity = new_cap;
    map->mask = new_cap - 1;
    map->size = 0;
    map->max_psl = 0;

    for (size_t i = 0; i < old_cap; i++) {
        if (old_slots[i].occupied) {
            caesura_pair_map_insert(map, old_slots[i].key, old_slots[i].rank, old_slots[i].token_id);
        }
    }

    free(old_slots);
    return true;
}

bool caesura_pair_map_insert(CaesuraPairMap* map, uint64_t key, uint32_t rank, uint32_t token_id) {
    if (!map) return false;

    if ((double)(map->size + 1) / (double)map->capacity > CAESURA_MAX_LOAD_FACTOR) {
        if (!resize_map(map, map->capacity * 2)) {
            return false;
        }
    }

    CaesuraPairSlot incoming;
    incoming.key = key;
    incoming.rank = rank;
    incoming.token_id = token_id;
    incoming.psl = 0;
    incoming.occupied = 1;
    incoming._pad = 0;

    size_t idx = hash_key(key) & map->mask;

    while (1) {
        if (!map->slots[idx].occupied) {
            map->slots[idx] = incoming;
            map->size++;
            if (incoming.psl > map->max_psl) {
                map->max_psl = incoming.psl;
            }
            return true;
        }

        if (map->slots[idx].key == incoming.key) {
            /* Update existing key */
            map->slots[idx].rank = incoming.rank;
            map->slots[idx].token_id = incoming.token_id;
            return true;
        }

        if (incoming.psl > map->slots[idx].psl) {
            /* Robin Hood steal: incoming is poorer than occupant, swap entries */
            CaesuraPairSlot tmp = map->slots[idx];
            map->slots[idx] = incoming;
            incoming = tmp;
            if (map->slots[idx].psl > map->max_psl) {
                map->max_psl = map->slots[idx].psl;
            }
        }

        incoming.psl++;
        idx = (idx + 1) & map->mask;
    }
}

bool caesura_pair_map_get(const CaesuraPairMap* map, uint64_t key, uint32_t* out_rank, uint32_t* out_token_id) {
    if (!map || map->size == 0) return false;

    size_t idx = hash_key(key) & map->mask;
    uint16_t psl = 0;

    while (1) {
        if (!map->slots[idx].occupied || psl > map->slots[idx].psl) {
            return false;
        }

        if (map->slots[idx].key == key) {
            if (out_rank) *out_rank = map->slots[idx].rank;
            if (out_token_id) *out_token_id = map->slots[idx].token_id;
            return true;
        }

        psl++;
        idx = (idx + 1) & map->mask;
    }
}

bool caesura_pair_map_contains(const CaesuraPairMap* map, uint64_t key) {
    return caesura_pair_map_get(map, key, NULL, NULL);
}

bool caesura_pair_map_remove(CaesuraPairMap* map, uint64_t key) {
    if (!map || map->size == 0) return false;

    size_t idx = hash_key(key) & map->mask;
    uint16_t psl = 0;

    while (1) {
        if (!map->slots[idx].occupied || psl > map->slots[idx].psl) {
            return false;
        }

        if (map->slots[idx].key == key) {
            /* Found: apply backward shift deletion */
            map->size--;
            size_t curr = idx;
            size_t next = (curr + 1) & map->mask;

            while (map->slots[next].occupied && map->slots[next].psl > 0) {
                map->slots[curr] = map->slots[next];
                map->slots[curr].psl--;
                curr = next;
                next = (curr + 1) & map->mask;
            }

            map->slots[curr].occupied = 0;
            map->slots[curr].key = 0;
            map->slots[curr].rank = 0;
            map->slots[curr].token_id = 0;
            map->slots[curr].psl = 0;
            return true;
        }

        psl++;
        idx = (idx + 1) & map->mask;
    }
}

size_t caesura_pair_map_size(const CaesuraPairMap* map) {
    return map ? map->size : 0;
}

size_t caesura_pair_map_capacity(const CaesuraPairMap* map) {
    return map ? map->capacity : 0;
}
