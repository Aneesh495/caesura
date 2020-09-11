# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
High-Performance Byte-Pair Encoding (BPE) Vocabulary Trainer
"""

from collections import Counter, defaultdict
from caesura.vocab.vocab import Vocabulary, create_byte_vocab
from caesura.models.bpe import BpeModel
from caesura.regex.compiler import whitespace_pre_tokenizer
from caesura.unicode.utf8 import ByteFallback

cdef class BpeTrainer:
    """
    Trains a Byte-Pair Encoding subword vocabulary and merge table from raw text.
    """

    def __init__(
        self,
        size_t vocab_size=1000,
        size_t min_frequency=2,
        list special_tokens=None,
        bint show_progress=False
    ):
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._special_tokens = list(special_tokens) if special_tokens else []
        self._show_progress = show_progress

    cpdef BpeModel train(self, list texts):
        """Train BPE merge rules and vocabulary from list of text strings."""
        cdef Vocabulary vocab = create_byte_vocab()
        cdef str spec_tok
        for spec_tok in self._special_tokens:
            vocab.add_token(spec_tok)

        cdef BpeModel model = BpeModel(vocab=vocab)
        cdef object bf = ByteFallback()
        cdef object pre_tok = whitespace_pre_tokenizer(keep_whitespace=True)

        # Count word frequencies across the corpus
        cdef object word_counts = Counter()
        cdef str text, chunk
        for text in texts:
            for chunk in pre_tok.split(text):
                # Represent chunk as string of byte-level unicode characters
                b_str = bf.encode_bytes_to_string(chunk.encode("utf-8"))
                word_counts[b_str] += 1

        # Represent words as lists of individual characters
        cdef dict word_splits = {}
        for w in word_counts:
            word_splits[w] = list(w)

        cdef size_t max_merges
        if self._vocab_size > len(vocab):
            max_merges = self._vocab_size - len(vocab)
        else:
            max_merges = 0

        cdef size_t merge_idx = 0
        cdef object pair_counts
        cdef tuple best_pair, pair
        cdef int best_freq, count
        cdef str p1, p2, merged
        cdef list symbols, new_symbols
        cdef size_t k, slen

        while merge_idx < max_merges:
            # Count all adjacent pairs
            pair_counts = defaultdict(int)
            for w, freq in word_counts.items():
                symbols = word_splits[w]
                slen = len(symbols)
                for k in range(slen - 1):
                    pair_counts[(symbols[k], symbols[k + 1])] += freq

            if not pair_counts:
                break

            best_pair = None
            best_freq = -1
            for pair, count in pair_counts.items():
                if count > best_freq:
                    best_freq = count
                    best_pair = pair

            if best_pair is None or best_freq < self._min_frequency:
                break

            p1, p2 = best_pair
            merged = p1 + p2

            # Register merge in model and vocabulary
            model.add_merge(p1, p2, rank=<uint32_t>merge_idx, merged_token=merged)
            merge_idx += 1

            # Update words with the newly merged pair
            for w in word_counts:
                symbols = word_splits[w]
                slen = len(symbols)
                if slen < 2:
                    continue

                new_symbols = []
                k = 0
                while k < slen:
                    if k < slen - 1 and symbols[k] == p1 and symbols[k + 1] == p2:
                        new_symbols.append(merged)
                        k += 2
                    else:
                        new_symbols.append(symbols[k])
                        k += 1
                word_splits[w] = new_symbols

        return model

    cpdef BpeModel train_from_files(self, list file_paths):
        """Train BPE model directly from text files."""
        cdef list lines = []
        cdef str path, line
        for path in file_paths:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    lines.append(line.strip())
        return self.train(lines)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def min_frequency(self) -> int:
        return self._min_frequency
