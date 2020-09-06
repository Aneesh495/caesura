# cython: language_level=3
from libc.stdint cimport uint8_t, uint32_t, int32_t
from libc.stddef cimport size_t
from caesura.core.types cimport OffsetSpan

cdef class DfaTable:
    cdef int32_t* _transitions  # flattened 2D array [num_states * 256]
    cdef int32_t* _accept_ids   # [num_states], -1 if non-accepting
    cdef size_t _num_states
    cdef int32_t _start_state

    cdef int32_t c_step(self, int32_t state, uint8_t byte_val) noexcept nogil
    cdef bint c_is_accepting(self, int32_t state) noexcept nogil
    cdef int32_t c_accept_rule(self, int32_t state) noexcept nogil

cdef class DfaScanner:
    cdef DfaTable _table
    cdef const uint8_t* _data
    cdef size_t _len
    cdef size_t _cursor

    cdef bint c_next_span(self, OffsetSpan* span) noexcept nogil
