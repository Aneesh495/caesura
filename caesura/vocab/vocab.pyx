# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Bi-Directional Vocabulary Mapping Engine
"""

from caesura.unicode.utf8 import ByteFallback

cdef class Vocabulary:
    """
    Two-way mapping between textual subword tokens and integer token IDs.
    """

    def __init__(self, dict initial_vocab=None):
        self._token_to_id = {}
        self._id_to_token = []
        self._size = 0

        if initial_vocab:
            # Sort by ID to ensure contiguous indexing
            sorted_items = sorted(initial_vocab.items(), key=lambda x: x[1])
            for token, tid in sorted_items:
                self.add_token(token, tid)

    cpdef int add_token(self, str token, object token_id=None):
        """Add a token to the vocabulary, returning its assigned integer ID."""
        if token in self._token_to_id:
            return self._token_to_id[token]

        cdef uint32_t assigned_id
        if token_id is not None:
            assigned_id = <uint32_t>token_id
            while len(self._id_to_token) <= assigned_id:
                self._id_to_token.append(None)
            self._id_to_token[assigned_id] = token
        else:
            assigned_id = <uint32_t>len(self._id_to_token)
            self._id_to_token.append(token)

        self._token_to_id[token] = assigned_id
        self._size = len(self._token_to_id)
        return assigned_id

    cpdef object get_id(self, str token):
        """Look up the integer ID for a token string, returning None if absent."""
        return self._token_to_id.get(token, None)

    cpdef str get_token(self, uint32_t token_id):
        """Look up the string token corresponding to an integer ID."""
        if token_id < len(self._id_to_token):
            return self._id_to_token[token_id]
        return None

    cpdef bint contains(self, str token):
        return token in self._token_to_id

    def to_dict(self) -> dict:
        return dict(self._token_to_id)

    def to_list(self) -> list:
        return list(self._id_to_token)

    @classmethod
    def from_dict(cls, dict d):
        return cls(initial_vocab=d)

    @classmethod
    def from_list(cls, list token_list):
        cdef dict d = {tok: idx for idx, tok in enumerate(token_list) if tok is not None}
        return cls(initial_vocab=d)

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, str token) -> bool:
        return token in self._token_to_id

    def __getitem__(self, object key):
        if isinstance(key, str):
            res = self.get_id(key)
            if res is None:
                raise KeyError(f"Token '{key}' not in vocabulary")
            return res
        elif isinstance(key, int):
            res = self.get_token(<uint32_t>key)
            if res is None:
                raise IndexError(f"Token ID {key} out of range")
            return res
        raise TypeError("Key must be str or int")

    def __repr__(self) -> str:
        return f"<Vocabulary size={len(self)}>"


def create_byte_vocab() -> Vocabulary:
    """Creates a vocabulary containing all 256 byte-level tokens."""
    bf = ByteFallback()
    v = Vocabulary()
    for b in range(256):
        v.add_token(bf.byte_to_char(b), b)
    return v
