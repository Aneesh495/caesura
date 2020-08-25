/*
 * Caesura Native Core: SIMD / SWAR UTF-8 Implementation
 */

#include "simd_utf8.h"
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define CAESURA_HAS_NEON 1
#endif

size_t caesura_scan_ascii(const uint8_t* ptr, size_t len) {
    size_t i = 0;

#if defined(CAESURA_HAS_NEON)
    /* 16-byte NEON vector fast path */
    while (i + 16 <= len) {
        uint8x16_t v = vld1q_u8(ptr + i);
        /* Check if any high bit is set */
        uint8x16_t high = vcltq_u8(v, vdupq_n_u8(0x80));
        /* If all bytes < 0x80, high is all 0xFF */
        uint64x2_t high64 = vreinterpretq_u64_u8(high);
        if (vgetq_lane_u64(high64, 0) != 0xFFFFFFFFFFFFFFFFULL ||
            vgetq_lane_u64(high64, 1) != 0xFFFFFFFFFFFFFFFFULL) {
            break;
        }
        i += 16;
    }
#endif

    /* 64-bit SWAR (SIMD Within A Register) fast path */
    const uint64_t high_mask = 0x8080808080808080ULL;
    while (i + 8 <= len) {
        uint64_t chunk;
        memcpy(&chunk, ptr + i, 8);
        if ((chunk & high_mask) != 0) {
            break;
        }
        i += 8;
    }

    /* Byte-by-byte scalar tail */
    while (i < len && ptr[i] < 0x80) {
        i++;
    }

    return i;
}

bool caesura_validate_utf8(const uint8_t* ptr, size_t len) {
    size_t i = 0;
    while (i < len) {
        /* Fast skip over ASCII spans */
        size_t ascii_count = caesura_scan_ascii(ptr + i, len - i);
        i += ascii_count;
        if (i >= len) break;

        uint8_t b0 = ptr[i];
        if (b0 >= 0xC2 && b0 <= 0xDF) {
            /* 2-byte sequence */
            if (i + 1 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            if ((b1 & 0xC0) != 0x80) return false;
            i += 2;
        } else if (b0 == 0xE0) {
            /* 3-byte sequence (disallow overlong) */
            if (i + 2 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            if (b1 < 0xA0 || b1 > 0xBF || (b2 & 0xC0) != 0x80) return false;
            i += 3;
        } else if (b0 >= 0xE1 && b0 <= 0xEC) {
            /* 3-byte sequence */
            if (i + 2 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return false;
            i += 3;
        } else if (b0 == 0xED) {
            /* 3-byte sequence (disallow UTF-16 surrogates U+D800..U+DFFF) */
            if (i + 2 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            if (b1 < 0x80 || b1 > 0x9F || (b2 & 0xC0) != 0x80) return false;
            i += 3;
        } else if (b0 >= 0xEE && b0 <= 0xEF) {
            /* 3-byte sequence */
            if (i + 2 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80) return false;
            i += 3;
        } else if (b0 == 0xF0) {
            /* 4-byte sequence (disallow overlong) */
            if (i + 3 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            uint8_t b3 = ptr[i + 3];
            if (b1 < 0x90 || b1 > 0xBF || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) return false;
            i += 4;
        } else if (b0 >= 0xF1 && b0 <= 0xF3) {
            /* 4-byte sequence */
            if (i + 3 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            uint8_t b3 = ptr[i + 3];
            if ((b1 & 0xC0) != 0x80 || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) return false;
            i += 4;
        } else if (b0 == 0xF4) {
            /* 4-byte sequence (disallow > U+10FFFF) */
            if (i + 3 >= len) return false;
            uint8_t b1 = ptr[i + 1];
            uint8_t b2 = ptr[i + 2];
            uint8_t b3 = ptr[i + 3];
            if (b1 < 0x80 || b1 > 0x8F || (b2 & 0xC0) != 0x80 || (b3 & 0xC0) != 0x80) return false;
            i += 4;
        } else {
            /* Invalid lead byte or lone continuation byte */
            return false;
        }
    }
    return true;
}

uint32_t caesura_utf8_next_codepoint(const uint8_t** ptr, size_t* len) {
    if (!ptr || !*ptr || !len || *len == 0) return 0xFFFD;

    const uint8_t* p = *ptr;
    size_t remaining = *len;
    uint8_t b0 = p[0];

    if (b0 < 0x80) {
        *ptr = p + 1;
        *len = remaining - 1;
        return (uint32_t)b0;
    }

    if (b0 >= 0xC2 && b0 <= 0xDF && remaining >= 2) {
        uint8_t b1 = p[1];
        if ((b1 & 0xC0) == 0x80) {
            *ptr = p + 2;
            *len = remaining - 2;
            return ((uint32_t)(b0 & 0x1F) << 6) | (uint32_t)(b1 & 0x3F);
        }
    } else if (b0 >= 0xE0 && b0 <= 0xEF && remaining >= 3) {
        uint8_t b1 = p[1];
        uint8_t b2 = p[2];
        if ((b2 & 0xC0) == 0x80) {
            bool valid = false;
            if (b0 == 0xE0 && b1 >= 0xA0 && b1 <= 0xBF) valid = true;
            else if (b0 >= 0xE1 && b0 <= 0xEC && (b1 & 0xC0) == 0x80) valid = true;
            else if (b0 == 0xED && b1 >= 0x80 && b1 <= 0x9F) valid = true;
            else if (b0 >= 0xEE && b0 <= 0xEF && (b1 & 0xC0) == 0x80) valid = true;

            if (valid) {
                *ptr = p + 3;
                *len = remaining - 3;
                return ((uint32_t)(b0 & 0x0F) << 12) |
                       ((uint32_t)(b1 & 0x3F) << 6) |
                       (uint32_t)(b2 & 0x3F);
            }
        }
    } else if (b0 >= 0xF0 && b0 <= 0xF4 && remaining >= 4) {
        uint8_t b1 = p[1];
        uint8_t b2 = p[2];
        uint8_t b3 = p[3];
        if ((b2 & 0xC0) == 0x80 && (b3 & 0xC0) == 0x80) {
            bool valid = false;
            if (b0 == 0xF0 && b1 >= 0x90 && b1 <= 0xBF) valid = true;
            else if (b0 >= 0xF1 && b0 <= 0xF3 && (b1 & 0xC0) == 0x80) valid = true;
            else if (b0 == 0xF4 && b1 >= 0x80 && b1 <= 0x8F) valid = true;

            if (valid) {
                *ptr = p + 4;
                *len = remaining - 4;
                return ((uint32_t)(b0 & 0x07) << 18) |
                       ((uint32_t)(b1 & 0x3F) << 12) |
                       ((uint32_t)(b2 & 0x3F) << 6) |
                       (uint32_t)(b3 & 0x3F);
            }
        }
    }

    /* Fallback on 1 invalid byte */
    *ptr = p + 1;
    *len = remaining - 1;
    return 0xFFFD;
}

size_t caesura_utf8_encode_codepoint(uint32_t cp, uint8_t* out_buf) {
    if (!out_buf) return 0;

    if (cp <= 0x7F) {
        out_buf[0] = (uint8_t)cp;
        return 1;
    } else if (cp <= 0x7FF) {
        out_buf[0] = (uint8_t)(0xC0 | (cp >> 6));
        out_buf[1] = (uint8_t)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp <= 0xFFFF) {
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            /* Surrogate code point invalid in UTF-8 */
            return 0;
        }
        out_buf[0] = (uint8_t)(0xE0 | (cp >> 12));
        out_buf[1] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out_buf[2] = (uint8_t)(0x80 | (cp & 0x3F));
        return 3;
    } else if (cp <= 0x10FFFF) {
        out_buf[0] = (uint8_t)(0xF0 | (cp >> 18));
        out_buf[1] = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
        out_buf[2] = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        out_buf[3] = (uint8_t)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}
