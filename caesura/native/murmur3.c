/*
 * Caesura Native Core: MurmurHash3 Implementation
 */

#include "murmur3.h"

static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

static inline uint64_t rotl64(uint64_t x, int8_t r) {
    return (x << r) | (x >> (64 - r));
}

uint32_t caesura_murmur3_32(const void* key, size_t len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = (int)(len / 4);
    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1;
        /* Handle unaligned reads safely */
        const uint8_t* p = (const uint8_t*)(blocks + i);
        k1 = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;

    switch (len & 3) {
        case 3:
            k1 ^= (uint32_t)tail[2] << 16;
            /* fallthrough */
        case 2:
            k1 ^= (uint32_t)tail[1] << 8;
            /* fallthrough */
        case 1:
            k1 ^= (uint32_t)tail[0];
            k1 *= c1;
            k1 = rotl32(k1, 15);
            k1 *= c2;
            h1 ^= k1;
    }

    h1 ^= (uint32_t)len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    return h1;
}

void caesura_murmur3_128(const void* key, size_t len, uint32_t seed, void* out) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = (int)(len / 16);

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    for (int i = 0; i < nblocks; i++) {
        const uint8_t* p = data + i * 16;
        uint64_t k1 = 0, k2 = 0;
        for (int b = 0; b < 8; b++) k1 |= ((uint64_t)p[b]) << (b * 8);
        for (int b = 0; b < 8; b++) k2 |= ((uint64_t)p[b + 8]) << (b * 8);

        k1 *= c1;
        k1 = rotl64(k1, 31);
        k1 *= c2;
        h1 ^= k1;

        h1 = rotl64(h1, 27);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = rotl64(k2, 33);
        k2 *= c1;
        h2 ^= k2;

        h2 = rotl64(h2, 31);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= ((uint64_t)tail[14]) << 48; /* fallthrough */
        case 14: k2 ^= ((uint64_t)tail[13]) << 40; /* fallthrough */
        case 13: k2 ^= ((uint64_t)tail[12]) << 32; /* fallthrough */
        case 12: k2 ^= ((uint64_t)tail[11]) << 24; /* fallthrough */
        case 11: k2 ^= ((uint64_t)tail[10]) << 16; /* fallthrough */
        case 10: k2 ^= ((uint64_t)tail[9]) << 8;   /* fallthrough */
        case 9:  k2 ^= ((uint64_t)tail[8]);
                 k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
                 /* fallthrough */
        case 8:  k1 ^= ((uint64_t)tail[7]) << 56; /* fallthrough */
        case 7:  k1 ^= ((uint64_t)tail[6]) << 48; /* fallthrough */
        case 6:  k1 ^= ((uint64_t)tail[5]) << 40; /* fallthrough */
        case 5:  k1 ^= ((uint64_t)tail[4]) << 32; /* fallthrough */
        case 4:  k1 ^= ((uint64_t)tail[3]) << 24; /* fallthrough */
        case 3:  k1 ^= ((uint64_t)tail[2]) << 16; /* fallthrough */
        case 2:  k1 ^= ((uint64_t)tail[1]) << 8;  /* fallthrough */
        case 1:  k1 ^= ((uint64_t)tail[0]);
                 k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    }

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdULL;
    h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53ULL;
    h1 ^= h1 >> 33;

    h2 ^= h2 >> 33;
    h2 *= 0xff51afd7ed558ccdULL;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53ULL;
    h2 ^= h2 >> 33;

    h1 += h2;
    h2 += h1;

    uint64_t* out_u64 = (uint64_t*)out;
    out_u64[0] = h1;
    out_u64[1] = h2;
}
