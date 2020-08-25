/*
 * Caesura Native Core: MurmurHash3 Fast Hash Functions
 *
 * Implements 32-bit and 128-bit MurmurHash3 variants optimized for
 * subword tokens, byte spans, and packed integer pair keys.
 */

#ifndef CAESURA_MURMUR3_H
#define CAESURA_MURMUR3_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t caesura_murmur3_32(const void* key, size_t len, uint32_t seed);
void caesura_murmur3_128(const void* key, size_t len, uint32_t seed, void* out);

static inline uint64_t caesura_hash_pair(uint32_t left, uint32_t right) {
    uint64_t key = (((uint64_t)left) << 32) | (uint64_t)right;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

#ifdef __cplusplus
}
#endif

#endif /* CAESURA_MURMUR3_H */
