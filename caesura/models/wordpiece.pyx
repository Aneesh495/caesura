# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
WordPiece Subword Model for BERT-style Architectures
"""

from caesura.vocab.vocab cimport Vocabulary
from caesura.trie.trie cimport Trie

cdef class WordPieceModel:
    """
    Greedy longest-match WordPiece tokenizer with continuation prefixes ('##').
    """

    def __init__(
        self,
        Vocabulary vocab,
        str unk_token="[UNK]",
        str continuing_prefix="##",
        size_t max_input_chars_per_word=100
    ):
        self._vocab = vocab
        self._unk_token = unk_token
        self._continuing_prefix = continuing_prefix
        self._max_input_chars_per_word = max_input_chars_per_word

        self._unk_id = vocab.get_id(unk_token) if unk_token in vocab else 0

        # Build Trie from vocabulary for fast prefix matching
        self._trie = Trie(initial_capacity=len(vocab) * 2 + 64)
        cdef str token
        cdef uint32_t tid
        for token, tid in vocab.to_dict().items():
            self._trie.insert(token, tid)

    cpdef list encode_word(self, str word):
        """Greedily encode a single word into subwords using WordPiece."""
        if not word:
            return []

        cdef size_t word_len = len(word)
        if word_len > self._max_input_chars_per_word:
            return [self._unk_id]

        cdef list output_tokens = []
        cdef size_t start = 0
        cdef size_t end
        cdef str cur_substr
        cdef object match_id
        cdef bint is_bad = 0

        while start < word_len:
            end = word_len
            cur_substr = ""
            match_id = None

            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self._continuing_prefix + substr

                match_id = self._vocab.get_id(substr)
                if match_id is not None:
                    cur_substr = substr
                    break
                end -= 1

            if match_id is None:
                is_bad = 1
                break

            output_tokens.append(<uint32_t>match_id)
            start = end

        if is_bad:
            return [self._unk_id]
        return output_tokens

    cpdef list encode(self, list words):
        """Encode a pre-tokenized sequence of words."""
        cdef list result = []
        cdef str word
        for word in words:
            result.extend(self.encode_word(word))
        return result

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def unk_token(self) -> str:
        return self._unk_token
