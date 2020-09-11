# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Unigram Subword Vocabulary Trainer
"""

import math
from collections import Counter
from libc.stdint cimport uint32_t
from caesura.vocab.vocab import Vocabulary
from caesura.models.unigram import UnigramModel

cdef class UnigramTrainer:
    """
    Trains a Unigram Language Model vocabulary and log-probability weights.
    """

    def __init__(self, size_t vocab_size=1000, list special_tokens=None, str unk_token="<unk>"):
        self._vocab_size = vocab_size
        self._special_tokens = list(special_tokens) if special_tokens else ["<unk>", "<s>", "</s>", "<pad>"]
        self._unk_token = unk_token

    cpdef UnigramModel train(self, list texts):
        cdef Vocabulary vocab = Vocabulary()
        cdef str spec
        for spec in self._special_tokens:
            vocab.add_token(spec)

        # Count character and subword frequencies
        cdef object subword_counts = Counter()
        cdef str text
        cdef size_t n, i, j

        for text in texts:
            n = len(text)
            for i in range(n):
                for j in range(i + 1, min(n + 1, i + 8)):
                    subword_counts[text[i:j]] += 1

        # Select most frequent tokens up to vocab_size
        cdef size_t capacity = self._vocab_size - len(vocab)
        cdef list most_common = [sub for sub, _ in subword_counts.most_common(capacity)]

        # Always ensure single characters are included to prevent out-of-vocab dead ends
        cdef set single_chars = set()
        for text in texts:
            for ch in text:
                single_chars.add(ch)

        for ch in sorted(single_chars):
            if ch not in vocab:
                vocab.add_token(ch)

        for sub in most_common:
            if sub not in vocab:
                vocab.add_token(sub)

        # Compute log-probability scores
        cdef double total_count = sum(subword_counts.values()) if subword_counts else 1.0
        cdef dict scores = {}
        cdef str token
        cdef uint32_t tid
        cdef double count, prob

        for token, tid in vocab.to_dict().items():
            count = <double>subword_counts.get(token, 1)
            prob = count / total_count
            scores[tid] = math.log(max(prob, 1e-9))

        return UnigramModel(vocab=vocab, scores=scores, unk_token=self._unk_token)
