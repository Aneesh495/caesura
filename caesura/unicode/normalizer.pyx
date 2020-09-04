# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Text Normalization Pipeline and Concrete Normalizers
"""

import unicodedata

cdef class Normalizer:
    """Base class for all text normalizers in Caesura."""

    cpdef str normalize(self, str text):
        return text

    def __call__(self, str text) -> str:
        return self.normalize(text)


cdef class LowercaseNormalizer(Normalizer):
    """Folds characters to lower case."""

    cpdef str normalize(self, str text):
        if text is None:
            return ""
        return text.lower()


cdef class StripAccentsNormalizer(Normalizer):
    """Strips combining diacritical marks using NFD decomposition."""

    cpdef str normalize(self, str text):
        if text is None:
            return ""
        cdef str decomposed = unicodedata.normalize("NFD", text)
        cdef list chars = []
        for ch in decomposed:
            if unicodedata.category(ch) != "Mn":
                chars.append(ch)
        return "".join(chars)


cdef class NfcNormalizer(Normalizer):
    """Canonical Decomposition followed by Canonical Composition."""

    cpdef str normalize(self, str text):
        if text is None:
            return ""
        return unicodedata.normalize("NFC", text)


cdef class NfkdNormalizer(Normalizer):
    """Compatibility Decomposition."""

    cpdef str normalize(self, str text):
        if text is None:
            return ""
        return unicodedata.normalize("NFKD", text)


cdef class MetaspaceNormalizer(Normalizer):
    """
    Replaces whitespace with a visible meta-space character (e.g. U+2581).
    Used across SentencePiece, LLaMA, and modern subword models.
    """

    def __init__(self, str replacement="\u2581", bint prepend_space=True):
        self._replacement = replacement
        self._prepend_space = prepend_space

    cpdef str normalize(self, str text):
        if not text:
            return self._replacement if self._prepend_space else ""

        cdef str replaced = text.replace(" ", self._replacement)
        if self._prepend_space and not replaced.startswith(self._replacement):
            return self._replacement + replaced
        return replaced


cdef class SequenceNormalizer(Normalizer):
    """Chains multiple normalizers in deterministic sequential order."""

    def __init__(self, list normalizers):
        self._normalizers = list(normalizers)

    cpdef str normalize(self, str text):
        cdef Normalizer norm
        cdef str current = text
        for norm in self._normalizers:
            current = norm.normalize(current)
        return current

    def add(self, Normalizer norm):
        self._normalizers.append(norm)

    def __len__(self) -> int:
        return len(self._normalizers)
