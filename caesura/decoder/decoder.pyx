# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Subword Detokenizers and Text Reconstructors
"""

from caesura.unicode.utf8 import ByteFallback

cdef class Decoder:
    """Base class for all text decoders."""

    cpdef str decode(self, str text):
        return text

    def __call__(self, str text) -> str:
        return self.decode(text)


cdef class ByteLevelDecoder(Decoder):
    """
    Decodes GPT-2 / HuggingFace ByteLevel tokens back to original UTF-8 bytes and text.
    """

    def __init__(self):
        self._bf = ByteFallback()

    cpdef str decode(self, str text):
        if not text:
            return ""
        cdef bytes raw_bytes = self._bf.decode_string_to_bytes(text)
        return raw_bytes.decode("utf-8", errors="replace")


cdef class WordPieceDecoder(Decoder):
    """
    Strips WordPiece continuation prefixes ('##') and normalizes whitespace around punctuation.
    """

    def __init__(self, str prefix="##", bint cleanup_spaces=True):
        self._prefix = prefix
        self._cleanup_spaces = cleanup_spaces

    cpdef str decode(self, str text):
        if not text:
            return ""

        # Remove prefix
        cdef str cleaned = text.replace(" " + self._prefix, "").replace(self._prefix, "")
        if self._cleanup_spaces:
            # Clean spaces before punctuation
            cleaned = cleaned.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
            cleaned = cleaned.replace(" '", "'").replace(" :", ":").replace(" ;", ";")
        return cleaned


cdef class MetaspaceDecoder(Decoder):
    """
    Reverses Metaspace tokenization by replacing metaspace characters (U+2581) with standard spaces.
    """

    def __init__(self, str replacement="\u2581", bint add_prefix_space=True):
        self._replacement = replacement
        self._add_prefix_space = add_prefix_space

    cpdef str decode(self, str text):
        if not text:
            return ""
        cdef str out = text.replace(self._replacement, " ")
        if self._add_prefix_space and out.startswith(" "):
            out = out[1:]
        return out


cdef class SequenceDecoder(Decoder):
    """Chains multiple decoders sequentially."""

    def __init__(self, list decoders):
        self._decoders = list(decoders)

    cpdef str decode(self, str text):
        cdef Decoder dec
        cdef str current = text
        for dec in self._decoders:
            current = dec.decode(current)
        return current
