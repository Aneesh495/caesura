# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Unigram Language Model with Dynamic Programming Lattice Decoder
"""

import math
from libc.stdint cimport uint32_t
from caesura.vocab.vocab cimport Vocabulary

cdef class UnigramModel:
    """
    Unigram subword model using Viterbi dynamic programming trellis decoding.
    """

    def __init__(self, Vocabulary vocab, dict scores=None, str unk_token="<unk>", double unk_score=-10.0):
        self._vocab = vocab
        self._scores = dict(scores) if scores is not None else {}
        self._unk_token = unk_token
        self._unk_score = unk_score
        self._unk_id = vocab.get_id(unk_token) if unk_token in vocab else 0

        # Fill default scores if missing
        cdef str token
        cdef uint32_t tid
        for token, tid in vocab.to_dict().items():
            if tid not in self._scores:
                self._scores[tid] = -1.0 * len(token)

    cpdef double score(self, str token):
        cdef object tid = self._vocab.get_id(token)
        if tid is None:
            return self._unk_score
        return self._scores.get(<uint32_t>tid, self._unk_score)

    cpdef double score_sequence(self, list ids):
        """Compute cumulative log probability score for a sequence of token IDs."""
        cdef double total = 0.0
        cdef object tid
        for tid in ids:
            total += self._scores.get(<uint32_t>tid, self._unk_score)
        return total

    cpdef list encode(self, str text):
        """Find globally optimal subword segmentation using Viterbi lattice decoding."""
        cdef list offsets = self.encode_with_offsets(text)
        cdef list tokens = []
        cdef tuple item
        for item in offsets:
            tokens.append(item[0])
        return tokens

    cpdef list encode_with_offsets(self, str text):
        """Find optimal subword segmentation with character start and end offsets."""
        if not text:
            return []

        cdef size_t n = len(text)
        cdef list best_score = [-float("inf")] * (n + 1)
        cdef list best_prev = [-1] * (n + 1)
        cdef list best_token = [0] * (n + 1)

        best_score[0] = 0.0

        cdef size_t i, j
        cdef str sub
        cdef object tid
        cdef double sc, cand_score

        for i in range(n):
            if best_score[i] == -float("inf"):
                continue

            for j in range(i + 1, min(n + 1, i + 32)):  # Max subword span
                sub = text[i:j]
                tid = self._vocab.get_id(sub)
                if tid is not None:
                    sc = self._scores.get(<uint32_t>tid, self._unk_score)
                    cand_score = best_score[i] + sc
                    if cand_score > best_score[j]:
                        best_score[j] = cand_score
                        best_prev[j] = i
                        best_token[j] = <uint32_t>tid

            # Fallback 1-character transition to prevent dead ends
            if best_score[i + 1] == -float("inf"):
                sub = text[i:i + 1]
                tid = self._vocab.get_id(sub)
                if tid is None:
                    tid = self._unk_id
                    sc = self._unk_score
                else:
                    sc = self._scores.get(<uint32_t>tid, self._unk_score)
                best_score[i + 1] = best_score[i] + sc
                best_prev[i + 1] = i
                best_token[i + 1] = <uint32_t>tid

        # Viterbi backtrace
        cdef list result = []
        cdef int curr = <int>n
        cdef int prev
        while curr > 0:
            prev = best_prev[curr]
            result.append((best_token[curr], prev, curr))
            curr = prev

        result.reverse()
        return result

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def scores(self) -> dict:
        return self._scores
