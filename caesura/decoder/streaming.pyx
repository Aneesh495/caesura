# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Incremental Streaming Detokenizer for Real-Time LLM Token Generation
"""

from libc.stdint cimport uint8_t
from caesura.unicode.utf8 import ByteFallback

cdef class IncrementalDecoder:
    """
    Streaming text reconstructor that buffers incomplete multi-byte UTF-8 sequences
    across sequential token generation steps, guaranteeing clean character boundaries.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._bf = ByteFallback()

    cpdef str decode_step(self, str token_piece):
        """Processes the next token string piece, buffering trailing incomplete bytes."""
        if not token_piece:
            return ""

        # Decode token_piece to raw bytes using ByteFallback
        cdef bytes raw = self._bf.decode_string_to_bytes(token_piece)
        return self.decode_bytes_step(raw)

    cpdef str decode_bytes_step(self, bytes raw_bytes):
        """Processes raw bytes, yielding complete UTF-8 strings and buffering partial bytes."""
        if not raw_bytes:
            return ""

        self._buffer.extend(raw_bytes)
        cdef bytes buf = bytes(self._buffer)
        cdef size_t n = len(buf)
        if n == 0:
            return ""

        # Check how many bytes at the end form an incomplete sequence
        cdef size_t complete_len = n
        cdef size_t trail_check = min(n, <size_t>4)
        cdef size_t i, lead_idx
        cdef uint8_t lead_byte
        cdef size_t expected_len

        # Walk backwards looking for a lead byte
        for i in range(trail_check):
            lead_idx = n - 1 - i
            lead_byte = buf[lead_idx]

            if lead_byte < 0x80:
                # ASCII byte: sequence ends here
                break
            elif (lead_byte & 0xE0) == 0xC0:
                expected_len = 2
                if n - lead_idx < expected_len:
                    complete_len = lead_idx
                break
            elif (lead_byte & 0xF0) == 0xE0:
                expected_len = 3
                if n - lead_idx < expected_len:
                    complete_len = lead_idx
                break
            elif (lead_byte & 0xF8) == 0xF0:
                expected_len = 4
                if n - lead_idx < expected_len:
                    complete_len = lead_idx
                break

        if complete_len == 0:
            return ""

        cdef bytes to_emit = buf[:complete_len]
        self._buffer = bytearray(buf[complete_len:])
        return to_emit.decode("utf-8", errors="replace")

    cpdef str flush(self):
        """Flush any remaining buffered bytes, replacing incomplete sequences."""
        if not self._buffer:
            return ""
        cdef bytes remaining = bytes(self._buffer)
        self._buffer.clear()
        return remaining.decode("utf-8", errors="replace")

    @property
    def buffer_len(self) -> int:
        return len(self._buffer)
