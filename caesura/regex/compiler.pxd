# cython: language_level=3
from caesura.regex.dfa cimport DfaTable, DfaScanner

cdef class RegexPreTokenizer:
    cdef object _compiled_regex
    cdef str _pattern_str
    cdef DfaTable _dfa
    cdef bint _keep_whitespace

    cpdef list split(self, str text)
    cpdef list split_with_offsets(self, str text)
