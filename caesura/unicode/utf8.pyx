# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
High-Performance UTF-8 Scanning and Byte-Level Unicode Encoding
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from caesura.core.types cimport (
    caesura_scan_ascii,
    caesura_validate_utf8,
    caesura_utf8_next_codepoint,
    caesura_utf8_encode_codepoint,
    caesura_utf8_lead_len,
)

cdef class Utf8Scanner:
    """
    Zero-copy UTF-8 forward scanner and code point iterator.
    """

    def __cinit__(self, bytes text):
        self._data = <const uint8_t*>text
        self._len = len(text)
        self._pos = 0

    cdef uint32_t c_next(self) noexcept nogil:
        if self._pos >= self._len:
            return 0
        cdef const uint8_t* p = self._data + self._pos
        cdef size_t remaining = self._len - self._pos
        cdef uint32_t cp = caesura_utf8_next_codepoint(&p, &remaining)
        self._pos = <size_t>(p - self._data)
        return cp

    cdef bint c_has_more(self) noexcept nogil:
        return self._pos < self._len

    cdef size_t c_pos(self) noexcept nogil:
        return self._pos

    cdef void c_reset(self) noexcept nogil:
        self._pos = 0

    def next_codepoint(self):
        cdef uint32_t cp
        with nogil:
            cp = self.c_next()
        if cp == 0 and not self.has_more():
            return None
        return cp

    def has_more(self) -> bool:
        return bool(self.c_has_more())

    @property
    def position(self) -> int:
        return self._pos


cdef class ByteFallback:
    """
    Bi-directional mapping between raw bytes (0..255) and printable Unicode code points.
    Implements the standard GPT-2 ByteLevel BPE character mapping.
    """

    def __cinit__(self):
        memset(self._unicode_to_byte, 0, 65536 * sizeof(uint8_t))
        memset(self._has_byte_mapping, 0, 65536 * sizeof(bint))

        cdef int b, n = 0
        cdef uint32_t cp

        # Populate standard printable ranges
        for b in range(256):
            if (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255):
                cp = <uint32_t>b
            else:
                cp = <uint32_t>(256 + n)
                n += 1

            self._byte_to_unicode[b] = cp
            if cp < 65536:
                self._unicode_to_byte[cp] = <uint8_t>b
                self._has_byte_mapping[cp] = 1

    cdef uint32_t c_byte_to_cp(self, uint8_t b) noexcept nogil:
        return self._byte_to_unicode[b]

    cdef bint c_cp_to_byte(self, uint32_t cp, uint8_t* out_byte) noexcept nogil:
        if cp < 65536 and self._has_byte_mapping[cp]:
            if out_byte != NULL:
                out_byte[0] = self._unicode_to_byte[cp]
            return 1
        return 0

    def byte_to_char(self, int b) -> str:
        if b < 0 or b > 255:
            raise ValueError("Byte value must be in range 0..255")
        cdef uint32_t cp = self._byte_to_unicode[b]
        return chr(cp)

    def char_to_byte(self, str ch) -> int:
        if len(ch) != 1:
            raise ValueError("Expected single character")
        cdef uint32_t cp = ord(ch)
        cdef uint8_t b
        if self.c_cp_to_byte(cp, &b):
            return b
        raise ValueError(f"Character '{ch}' (U+{cp:04X}) has no byte mapping")

    def encode_bytes_to_string(self, bytes raw_bytes) -> str:
        """Transform raw binary bytes to the printable Unicode representation."""
        cdef size_t l = len(raw_bytes)
        cdef const uint8_t* p = <const uint8_t*>raw_bytes
        cdef list chars = []
        cdef size_t i
        for i in range(l):
            chars.append(chr(self._byte_to_unicode[p[i]]))
        return "".join(chars)

    def decode_string_to_bytes(self, str text) -> bytes:
        """Transform printable Unicode representation back into raw bytes."""
        cdef bytearray out = bytearray()
        cdef uint32_t cp
        cdef uint8_t b
        for ch in text:
            cp = ord(ch)
            if self.c_cp_to_byte(cp, &b):
                out.append(b)
            else:
                # UTF-8 encode character as fallback
                out.extend(ch.encode("utf-8"))
        return bytes(out)


def is_valid_utf8(bytes data) -> bool:
    """Validate whether input bytes adhere to RFC 3629 UTF-8."""
    cdef const uint8_t* p = <const uint8_t*>data
    cdef size_t l = len(data)
    cdef bint res
    with nogil:
        res = caesura_validate_utf8(p, l)
    return bool(res)


def scan_ascii_prefix(bytes data) -> int:
    """Return count of consecutive ASCII bytes at start of buffer."""
    cdef const uint8_t* p = <const uint8_t*>data
    cdef size_t l = len(data)
    cdef size_t res
    with nogil:
        res = caesura_scan_ascii(p, l)
    return res
