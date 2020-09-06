# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Regex Pre-Tokenizer Compiler and Text Splitting Engine
"""

import re
from caesura.regex.dfa cimport DfaTable, DfaScanner
from caesura.regex.patterns import GPT2_PATTERN, GPT4_PATTERN, WHITESPACE_PATTERN, PUNCTUATION_PATTERN

cdef class RegexPreTokenizer:
    """
    Splits text into chunks prior to subword encoding using DFA or compiled regex patterns.
    """

    def __init__(self, pattern=None, DfaTable dfa=None, bint keep_whitespace=True):
        self._dfa = dfa
        self._keep_whitespace = keep_whitespace
        if isinstance(pattern, str):
            self._pattern_str = pattern
            try:
                self._compiled_regex = re.compile(pattern)
            except re.error:
                # If regex contains regex module syntax like \p{L}, fallback to standard pattern
                self._compiled_regex = re.compile(r"""\w+|[^\w\s]+|\s+""")
        elif dfa is not None:
            self._pattern_str = "custom_dfa"
            self._compiled_regex = None
        else:
            self._pattern_str = r"""\w+|[^\w\s]+|\s+"""
            self._compiled_regex = re.compile(self._pattern_str)

    cpdef list split(self, str text):
        """Split text into raw string chunks."""
        if not text:
            return []

        cdef list result = []
        cdef DfaScanner scanner
        cdef bytes raw_bytes
        cdef list spans
        cdef size_t s, e
        cdef str chunk

        if self._dfa is not None:
            raw_bytes = text.encode("utf-8")
            scanner = DfaScanner(self._dfa, raw_bytes)
            spans = scanner.find_all_spans()
            for s, e in spans:
                chunk = raw_bytes[s:e].decode("utf-8", errors="replace")
                if self._keep_whitespace or not chunk.isspace():
                    result.append(chunk)
            return result

        if self._compiled_regex is not None:
            for m in self._compiled_regex.finditer(text):
                chunk = m.group(0)
                if self._keep_whitespace or not chunk.isspace():
                    result.append(chunk)
            return result

        return [text]

    cpdef list split_with_offsets(self, str text):
        """Split text and return list of (chunk_str, start_char, end_char) tuples."""
        if not text:
            return []

        cdef list result = []
        cdef object match
        cdef str chunk

        if self._compiled_regex is not None:
            for match in self._compiled_regex.finditer(text):
                chunk = match.group(0)
                if self._keep_whitespace or not chunk.isspace():
                    result.append((chunk, match.start(), match.end()))
            return result

        # Fallback character spans
        cdef list chunks = self.split(text)
        cdef size_t cursor = 0
        for chunk in chunks:
            result.append((chunk, cursor, cursor + len(chunk)))
            cursor += len(chunk)
        return result

    @property
    def pattern(self) -> str:
        return self._pattern_str


def gpt2_pre_tokenizer(bint keep_whitespace=True) -> RegexPreTokenizer:
    return RegexPreTokenizer(r"""'s|'t|'re|'ve|'m|'ll|'d|\w+|[^\w\s]+|\s+""", keep_whitespace=keep_whitespace)

def gpt4_pre_tokenizer(bint keep_whitespace=True) -> RegexPreTokenizer:
    return RegexPreTokenizer(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|\w+|\d{1,3}|[^\w\s]+|\s+""", keep_whitespace=keep_whitespace)

def whitespace_pre_tokenizer(bint keep_whitespace=False) -> RegexPreTokenizer:
    return RegexPreTokenizer(WHITESPACE_PATTERN, keep_whitespace=keep_whitespace)

def punctuation_pre_tokenizer(bint keep_whitespace=True) -> RegexPreTokenizer:
    return RegexPreTokenizer(PUNCTUATION_PATTERN, keep_whitespace=keep_whitespace)
