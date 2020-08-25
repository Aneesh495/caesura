/*
 * Caesura Native Core: SIMD / SWAR UTF-8 Bitstream Validation and Fast ASCII Scanner
 *
 * Scans byte streams in 64-bit and 128-bit chunks to quickly detect pure ASCII
 * spans (the dominant sub-component of English code and natural language text),
 * and validates multi-byte UTF-8 sequences adhering strictly to RFC 3629.
 */

#ifndef CAESURA_SIMD_UTF8_H
#define CAESURA_SIMD_UTF8_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scans forward over pure ASCII bytes (all high bits clear).
 * Returns the number of consecutive ASCII bytes found from ptr. */
size_t caesura_scan_ascii(const uint8_t* ptr, size_t len);

/* Validates whether the given byte buffer is valid RFC 3629 UTF-8.
 * Returns true if valid, false if invalid sequence, overlong, or surrogate. */
bool caesura_validate_utf8(const uint8_t* ptr, size_t len);

/* Decodes the next Unicode code point from ptr, advancing *ptr and decrementing *len.
 * Returns decoded uint32_t code point, or 0xFFFD on invalid byte. */
uint32_t caesura_utf8_next_codepoint(const uint8_t** ptr, size_t* len);

/* Encodes a Unicode code point to UTF-8 into out_buf (must have at least 4 bytes).
 * Returns the number of bytes written (1 to 4), or 0 on invalid code point. */
size_t caesura_utf8_encode_codepoint(uint32_t cp, uint8_t* out_buf);

/* Returns the expected UTF-8 sequence length from the leading byte (1 to 4, or 0 if invalid). */
static inline size_t caesura_utf8_lead_len(uint8_t byte) {
    if ((byte & 0x80) == 0x00) return 1;
    if ((byte & 0xE0) == 0xC0) return 2;
    if ((byte & 0xF0) == 0xE0) return 3;
    if ((byte & 0xF8) == 0xF0) return 4;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* CAESURA_SIMD_UTF8_H */
