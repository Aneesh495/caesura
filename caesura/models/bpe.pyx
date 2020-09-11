# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
High-Throughput Byte-Pair Encoding (BPE) Subword Engine
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from caesura.core.types cimport (
    TokenNode,
    caesura_pair_map_create,
    caesura_pair_map_destroy,
    caesura_pair_map_insert,
    caesura_pair_map_get,
    caesura_pair_map_size,
    caesura_pair_map_capacity,
)
from caesura.core.memory cimport MemoryArena
from caesura.vocab.vocab cimport Vocabulary
from caesura.vocab.vocab import create_byte_vocab
from caesura.unicode.utf8 import ByteFallback

cdef class BpeModel:
    """
    Byte-Pair Encoding model executing pair reduction directly in raw C structures.
    Releases the Python GIL during compute loops for true multicore scaling.
    """

    def __cinit__(self, Vocabulary vocab=None, bint byte_fallback=True):
        self._vocab = vocab if vocab is not None else create_byte_vocab()
        self._pair_map = caesura_pair_map_create(1024)
        if self._pair_map == NULL:
            raise MemoryError("Failed to allocate BPE pair hash map")
        self._arena = MemoryArena(65536)
        self._unk_id = 0
        self._byte_fallback = byte_fallback

    def __dealloc__(self):
        if self._pair_map != NULL:
            caesura_pair_map_destroy(self._pair_map)
            self._pair_map = NULL

    def add_merge(self, object left, object right, uint32_t rank, object merged_token=None):
        """Register a BPE merge rule between left and right tokens."""
        cdef uint32_t left_id, right_id, merged_id
        if isinstance(left, str):
            left_id = self._vocab[left]
        else:
            left_id = <uint32_t>left

        if isinstance(right, str):
            right_id = self._vocab[right]
        else:
            right_id = <uint32_t>right

        if merged_token is not None:
            if isinstance(merged_token, str):
                merged_id = self._vocab.add_token(merged_token)
            else:
                merged_id = <uint32_t>merged_token
        else:
            merged_str = self._vocab.get_token(left_id) + self._vocab.get_token(right_id)
            merged_id = self._vocab.add_token(merged_str)

        cdef uint64_t key = ((<uint64_t>left_id) << 32) | (<uint64_t>right_id)
        caesura_pair_map_insert(self._pair_map, key, rank, merged_id)

    cdef size_t c_encode_chunk(
        self,
        const uint8_t* chunk_bytes,
        size_t chunk_len,
        size_t chunk_offset,
        uint32_t* out_ids,
        size_t* out_offsets_start,
        size_t* out_offsets_end,
        size_t max_out
    ) noexcept nogil:
        if chunk_bytes == NULL or chunk_len == 0 or max_out == 0:
            return 0

        # Stack allocation for short chunks; fallback to arena for large chunks
        cdef TokenNode stack_nodes[256]
        cdef TokenNode* nodes
        cdef bint heap_allocated = 0

        if chunk_len <= 256:
            nodes = stack_nodes
        else:
            nodes = <TokenNode*>malloc(chunk_len * sizeof(TokenNode))
            if nodes == NULL:
                return 0
            heap_allocated = 1

        cdef size_t i
        for i in range(chunk_len):
            # In byte-level BPE, each byte maps to its token id directly (0..255)
            nodes[i].token_id = <uint32_t>chunk_bytes[i]
            nodes[i].byte_len = 1
            nodes[i].orig_offset = chunk_offset + i
            nodes[i].prev = <int32_t>(i - 1)
            nodes[i].next = <int32_t>(i + 1) if (i + 1 < chunk_len) else -1
            nodes[i].flags = 0

        # Pair reduction loop
        cdef int32_t best_node
        cdef uint32_t best_rank
        cdef uint32_t best_merged_id
        cdef int32_t curr, right_node
        cdef uint64_t pair_key
        cdef uint32_t rank, merged_id

        while 1:
            best_rank = 0xFFFFFFFF  # max uint32
            best_node = -1
            best_merged_id = 0

            curr = 0
            # Scan current linked list for lowest merge rank
            while curr != -1:
                right_node = nodes[curr].next
                if right_node != -1:
                    pair_key = ((<uint64_t>nodes[curr].token_id) << 32) | (<uint64_t>nodes[right_node].token_id)
                    if caesura_pair_map_get(self._pair_map, pair_key, &rank, &merged_id):
                        if rank < best_rank:
                            best_rank = rank
                            best_node = curr
                            best_merged_id = merged_id
                curr = nodes[curr].next

            if best_node == -1:
                # No more merges found
                break

            # Execute merge on best_node
            right_node = nodes[best_node].next
            nodes[best_node].token_id = best_merged_id
            nodes[best_node].byte_len += nodes[right_node].byte_len
            nodes[best_node].next = nodes[right_node].next
            if nodes[right_node].next != -1:
                nodes[nodes[right_node].next].prev = best_node

        # Collect final tokens
        cdef size_t out_count = 0
        curr = 0
        # Find start head
        while curr != -1 and nodes[curr].prev != -1:
            curr = nodes[curr].prev

        while curr != -1 and out_count < max_out:
            out_ids[out_count] = nodes[curr].token_id
            if out_offsets_start != NULL:
                out_offsets_start[out_count] = nodes[curr].orig_offset
            if out_offsets_end != NULL:
                out_offsets_end[out_count] = nodes[curr].orig_offset + nodes[curr].byte_len
            out_count += 1
            curr = nodes[curr].next

        if heap_allocated:
            free(nodes)

        return out_count

    cpdef list encode(self, str text):
        """Encode text string into list of integer token IDs."""
        if not text:
            return []

        cdef bytes raw_bytes = text.encode("utf-8")
        cdef const uint8_t* p = <const uint8_t*>raw_bytes
        cdef size_t l = len(raw_bytes)
        cdef size_t max_out = l * 2 + 16
        cdef uint32_t* out_ids = <uint32_t*>malloc(max_out * sizeof(uint32_t))
        if out_ids == NULL:
            raise MemoryError("Failed to allocate token output buffer")

        cdef size_t count
        with nogil:
            count = self.c_encode_chunk(p, l, 0, out_ids, NULL, NULL, max_out)

        cdef list res = []
        cdef size_t i
        for i in range(count):
            res.append(out_ids[i])

        free(out_ids)
        return res

    cpdef list encode_with_offsets(self, str text):
        """Encode text returning list of (token_id, start_byte, end_byte) tuples."""
        if not text:
            return []

        cdef bytes raw_bytes = text.encode("utf-8")
        cdef const uint8_t* p = <const uint8_t*>raw_bytes
        cdef size_t l = len(raw_bytes)
        cdef size_t max_out = l * 2 + 16
        cdef uint32_t* out_ids = <uint32_t*>malloc(max_out * sizeof(uint32_t))
        cdef size_t* out_s = <size_t*>malloc(max_out * sizeof(size_t))
        cdef size_t* out_e = <size_t*>malloc(max_out * sizeof(size_t))

        if out_ids == NULL or out_s == NULL or out_e == NULL:
            if out_ids: free(out_ids)
            if out_s: free(out_s)
            if out_e: free(out_e)
            raise MemoryError("Failed to allocate token offset buffers")

        cdef size_t count
        with nogil:
            count = self.c_encode_chunk(p, l, 0, out_ids, out_s, out_e, max_out)

        cdef list res = []
        cdef size_t i
        for i in range(count):
            res.append((out_ids[i], out_s[i], out_e[i]))

        free(out_ids)
        free(out_s)
        free(out_e)
        return res

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def num_merges(self) -> int:
        return caesura_pair_map_size(self._pair_map)
