# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Deterministic Finite Automaton (DFA) Table and Scanning Engine
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from caesura.core.types cimport OffsetSpan

cdef class DfaTable:
    """
    Flat 2D state transition table and rule acceptance array for a DFA.
    Provides O(1) state transitions per byte with zero branching or backtracking.
    """

    def __cinit__(self, size_t num_states, int32_t start_state=0):
        if num_states == 0:
            raise ValueError("Number of states must be greater than zero")
        self._num_states = num_states
        self._start_state = start_state

        cdef size_t trans_size = num_states * 256 * sizeof(int32_t)
        self._transitions = <int32_t*>malloc(trans_size)
        if self._transitions == NULL:
            raise MemoryError("Failed to allocate DFA transition table")
        memset(self._transitions, 0xFF, trans_size)  # initialize all to -1

        self._accept_ids = <int32_t*>malloc(num_states * sizeof(int32_t))
        if self._accept_ids == NULL:
            free(self._transitions)
            self._transitions = NULL
            raise MemoryError("Failed to allocate DFA accept table")
        memset(self._accept_ids, 0xFF, num_states * sizeof(int32_t))  # initialize to -1

    def __dealloc__(self):
        if self._transitions != NULL:
            free(self._transitions)
            self._transitions = NULL
        if self._accept_ids != NULL:
            free(self._accept_ids)
            self._accept_ids = NULL

    cdef int32_t c_step(self, int32_t state, uint8_t byte_val) noexcept nogil:
        if state < 0 or <size_t>state >= self._num_states:
            return -1
        return self._transitions[state * 256 + byte_val]

    cdef bint c_is_accepting(self, int32_t state) noexcept nogil:
        if state < 0 or <size_t>state >= self._num_states:
            return 0
        return 1 if self._accept_ids[state] >= 0 else 0

    cdef int32_t c_accept_rule(self, int32_t state) noexcept nogil:
        if state < 0 or <size_t>state >= self._num_states:
            return -1
        return self._accept_ids[state]

    def set_transition(self, int from_state, int byte_val, int to_state):
        """Set a transition on byte_val from from_state to to_state."""
        if from_state < 0 or <size_t>from_state >= self._num_states:
            raise IndexError("from_state out of bounds")
        if byte_val < 0 or byte_val > 255:
            raise ValueError("byte_val must be in range 0..255")
        if to_state < -1 or <size_t>to_state >= self._num_states:
            raise IndexError("to_state out of bounds")
        self._transitions[from_state * 256 + byte_val] = to_state

    def set_accept(self, int state, int rule_id):
        """Mark state as accepting with given rule_id."""
        if state < 0 or <size_t>state >= self._num_states:
            raise IndexError("state out of bounds")
        self._accept_ids[state] = rule_id

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def start_state(self) -> int:
        return self._start_state


cdef class DfaScanner:
    """
    High-throughput non-backtracking scanner over raw UTF-8 byte streams.
    Yields maximal span boundaries based on the underlying DFA.
    """

    def __cinit__(self, DfaTable table, bytes data):
        self._table = table
        self._data = <const uint8_t*>data
        self._len = len(data)
        self._cursor = 0

    cdef bint c_next_span(self, OffsetSpan* span) noexcept nogil:
        if self._cursor >= self._len:
            return 0

        cdef size_t start_pos = self._cursor
        cdef int32_t state = self._table._start_state
        cdef size_t last_accept_pos = start_pos
        cdef bint found_accept = 0
        cdef size_t pos = start_pos
        cdef int32_t next_state
        cdef uint8_t b

        while pos < self._len:
            b = self._data[pos]
            next_state = self._table.c_step(state, b)
            if next_state == -1:
                break
            state = next_state
            pos += 1
            if self._table.c_is_accepting(state):
                last_accept_pos = pos
                found_accept = 1

        if found_accept and last_accept_pos > start_pos:
            span.start = start_pos
            span.end = last_accept_pos
            self._cursor = last_accept_pos
            return 1

        # Fallback single byte advance
        span.start = start_pos
        span.end = start_pos + 1
        self._cursor = start_pos + 1
        return 1

    def find_all_spans(self) -> list:
        """Scan entire buffer and return list of (start, end) byte tuples."""
        cdef list spans = []
        cdef OffsetSpan span
        cdef bint ok

        while self._cursor < self._len:
            with nogil:
                ok = self.c_next_span(&span)
            if ok:
                spans.append((span.start, span.end))
            else:
                break
        return spans

    def find_all_slices(self, bytes original_bytes) -> list:
        """Scan buffer and return list of matched byte slices."""
        cdef list spans = self.find_all_spans()
        cdef list result = []
        cdef size_t s, e
        for s, e in spans:
            result.append(original_bytes[s:e])
        return result
