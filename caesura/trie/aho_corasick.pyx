# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Aho-Corasick Multi-Pattern String Search Automaton
"""

from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memset
from collections import deque

cdef class AhoCorasick:
    """
    Aho-Corasick string matching automaton.
    Finds all occurrences of multiple patterns (such as special tokens)
    in arbitrary text streams in O(N) time with zero backtracking.
    """

    def __cinit__(self, size_t initial_capacity=1024):
        if initial_capacity < 16:
            initial_capacity = 16
        self._capacity = initial_capacity
        self._nodes = <AcNode*>malloc(initial_capacity * sizeof(AcNode))
        if self._nodes == NULL:
            raise MemoryError("Failed to allocate AhoCorasick nodes")
        self._num_nodes = 0
        self._compiled = 0

        # Allocate root at index 0
        self.c_alloc_node()

    def __dealloc__(self):
        if self._nodes != NULL:
            free(self._nodes)
            self._nodes = NULL

    cdef int32_t c_alloc_node(self) noexcept nogil:
        cdef size_t new_cap
        cdef AcNode* new_nodes
        cdef int32_t idx

        if self._num_nodes >= self._capacity:
            new_cap = self._capacity * 2
            new_nodes = <AcNode*>realloc(self._nodes, new_cap * sizeof(AcNode))
            if new_nodes == NULL:
                return -1
            self._nodes = new_nodes
            self._capacity = new_cap

        idx = <int32_t>self._num_nodes
        self._num_nodes += 1
        memset(self._nodes[idx].transitions, 0xFF, 256 * sizeof(int32_t))  # -1
        self._nodes[idx].fail = 0
        self._nodes[idx].token_id = 0
        self._nodes[idx].pattern_len = 0
        self._nodes[idx].is_match = 0
        return idx

    cdef void c_add_pattern(self, const uint8_t* pattern, size_t pat_len, uint32_t token_id) noexcept nogil:
        if pattern == NULL or pat_len == 0:
            return

        cdef int32_t curr = 0
        cdef size_t i
        cdef uint8_t b
        cdef int32_t next_idx

        for i in range(pat_len):
            b = pattern[i]
            next_idx = self._nodes[curr].transitions[b]
            if next_idx == -1:
                next_idx = self.c_alloc_node()
                if next_idx == -1:
                    return
                self._nodes[curr].transitions[b] = next_idx
            curr = next_idx

        self._nodes[curr].is_match = 1
        self._nodes[curr].token_id = token_id
        self._nodes[curr].pattern_len = <uint32_t>pat_len

    cdef void c_compile(self):
        """Build failure transitions and dictionary links using BFS."""
        cdef object queue = deque()
        cdef int b
        cdef int32_t child, u, v, f

        self._nodes[0].fail = 0

        # Depth 1 nodes fail to root
        for b in range(256):
            child = self._nodes[0].transitions[b]
            if child != -1:
                self._nodes[child].fail = 0
                queue.append(child)

        while len(queue) > 0:
            u = queue.popleft()

            for b in range(256):
                v = self._nodes[u].transitions[b]
                if v != -1:
                    f = self._nodes[u].fail
                    while f > 0 and self._nodes[f].transitions[b] == -1:
                        f = self._nodes[f].fail

                    if self._nodes[f].transitions[b] != -1:
                        self._nodes[v].fail = self._nodes[f].transitions[b]
                    else:
                        self._nodes[v].fail = 0

                    # Inherit match from failure state if v is not its own terminal
                    if self._nodes[self._nodes[v].fail].is_match and not self._nodes[v].is_match:
                        self._nodes[v].is_match = 1
                        self._nodes[v].token_id = self._nodes[self._nodes[v].fail].token_id
                        self._nodes[v].pattern_len = self._nodes[self._nodes[v].fail].pattern_len

                    queue.append(v)

        self._compiled = 1

    def add_pattern(self, object pattern, uint32_t token_id):
        """Register a pattern string or bytes with its token_id."""
        cdef bytes bpat
        if isinstance(pattern, str):
            bpat = pattern.encode("utf-8")
        elif isinstance(pattern, bytes):
            bpat = pattern
        else:
            raise TypeError("Pattern must be str or bytes")

        cdef const uint8_t* p = <const uint8_t*>bpat
        cdef size_t l = len(bpat)
        with nogil:
            self.c_add_pattern(p, l, token_id)
        self._compiled = 0

    def compile(self):
        """Finalize the automaton state transitions."""
        self.c_compile()

    def find_all(self, object text) -> list:
        """Scan text and return all matches as (start_pos, end_pos, token_id) tuples."""
        if not self._compiled:
            self.compile()

        cdef bytes btext
        if isinstance(text, str):
            btext = text.encode("utf-8")
        elif isinstance(text, bytes):
            btext = text
        else:
            return []

        cdef const uint8_t* data = <const uint8_t*>btext
        cdef size_t n = len(btext)
        cdef list matches = []
        cdef int32_t curr = 0
        cdef size_t i
        cdef uint8_t b
        cdef int32_t next_idx

        for i in range(n):
            b = data[i]

            while curr > 0 and self._nodes[curr].transitions[b] == -1:
                curr = self._nodes[curr].fail

            next_idx = self._nodes[curr].transitions[b]
            if next_idx != -1:
                curr = next_idx
            else:
                curr = 0

            if self._nodes[curr].is_match:
                pat_len = self._nodes[curr].pattern_len
                token_id = self._nodes[curr].token_id
                matches.append((i + 1 - pat_len, i + 1, token_id))

        return matches

    def split_with_tokens(self, object text) -> list:
        """
        Splits text on all special token matches, returning alternating
        tuples of ('text', text_chunk) and ('special', (token_id, special_token_str)).
        """
        if not self._compiled:
            self.compile()

        cdef bytes btext
        if isinstance(text, str):
            btext = text.encode("utf-8")
        elif isinstance(text, bytes):
            btext = text
        else:
            return []

        cdef list matches = self.find_all(btext)
        if not matches:
            return [("text", btext.decode("utf-8", errors="replace"))]

        # Filter overlapping matches greedily from left to right
        cdef list non_overlapping = []
        cdef size_t last_end = 0
        for start, end, tid in matches:
            if start >= last_end:
                non_overlapping.append((start, end, tid))
                last_end = end

        cdef list segments = []
        cdef size_t cursor = 0
        for start, end, tid in non_overlapping:
            if start > cursor:
                segments.append(("text", btext[cursor:start].decode("utf-8", errors="replace")))
            segments.append(("special", (tid, btext[start:end].decode("utf-8", errors="replace"))))
            cursor = end

        if cursor < len(btext):
            segments.append(("text", btext[cursor:].decode("utf-8", errors="replace")))

        return segments
