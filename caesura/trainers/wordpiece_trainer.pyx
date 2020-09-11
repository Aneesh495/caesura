# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
WordPiece Vocabulary Trainer
"""

from collections import Counter, defaultdict
from caesura.vocab.vocab import Vocabulary
from caesura.models.wordpiece import WordPieceModel
from caesura.regex.compiler import whitespace_pre_tokenizer

cdef class WordPieceTrainer:
    """
    Trains a WordPiece subword vocabulary maximizing mutual information scores.
    """

    def __init__(
        self,
        size_t vocab_size=1000,
        size_t min_frequency=2,
        list special_tokens=None,
        str continuing_prefix="##"
    ):
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._special_tokens = list(special_tokens) if special_tokens else ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self._continuing_prefix = continuing_prefix

    cpdef WordPieceModel train(self, list texts):
        cdef Vocabulary vocab = Vocabulary()
        cdef str spec
        for spec in self._special_tokens:
            vocab.add_token(spec)

        cdef object pre_tok = whitespace_pre_tokenizer(keep_whitespace=False)
        cdef object word_counts = Counter()
        cdef str text, word
        for text in texts:
            for word in pre_tok.split(text):
                word_counts[word] += 1

        # Collect character alphabet
        cdef set alphabet = set()
        for word in word_counts:
            for i, ch in enumerate(word):
                if i == 0:
                    alphabet.add(ch)
                else:
                    alphabet.add(self._continuing_prefix + ch)

        for ch in sorted(alphabet):
            vocab.add_token(ch)

        # Word representation
        cdef dict word_splits = {}
        for word in word_counts:
            split = []
            for i, ch in enumerate(word):
                if i == 0:
                    split.append(ch)
                else:
                    split.append(self._continuing_prefix + ch)
            word_splits[word] = split

        # Iteratively merge highest scored pair
        cdef size_t target_merges
        if self._vocab_size > len(vocab):
            target_merges = self._vocab_size - len(vocab)
        else:
            target_merges = 0

        cdef size_t merge_idx = 0
        cdef object pair_counts, piece_counts
        cdef tuple best_pair
        cdef double best_score, score
        cdef str p1, p2, merged
        cdef list symbols, new_symbols
        cdef size_t k, slen
        cdef int freq

        while merge_idx < target_merges:
            pair_counts = defaultdict(int)
            piece_counts = defaultdict(int)

            for word, freq in word_counts.items():
                symbols = word_splits[word]
                slen = len(symbols)
                for k in range(slen):
                    piece_counts[symbols[k]] += freq
                    if k < slen - 1:
                        pair_counts[(symbols[k], symbols[k + 1])] += freq

            if not pair_counts:
                break

            best_pair = None
            best_score = -1.0

            for pair, count in pair_counts.items():
                if count < self._min_frequency:
                    continue
                score = <double>count / (<double>piece_counts[pair[0]] * <double>piece_counts[pair[1]])
                if score > best_score:
                    best_score = score
                    best_pair = pair

            if best_pair is None:
                break

            p1, p2 = best_pair
            if p2.startswith(self._continuing_prefix):
                merged = p1 + p2[len(self._continuing_prefix):]
            else:
                merged = p1 + p2

            vocab.add_token(merged)
            merge_idx += 1

            # Update words
            for word in word_counts:
                symbols = word_splits[word]
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
                word_splits[word] = new_symbols

        return WordPieceModel(vocab=vocab, unk_token="[UNK]", continuing_prefix=self._continuing_prefix)
