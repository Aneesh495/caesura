# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Special Token Registry and Multi-Pattern Infill Dispatcher
"""

from caesura.trie.aho_corasick import AhoCorasick

cdef class SpecialTokenRegistry:
    """
    Manages reserved control tokens (BOS, EOS, UNK, PAD, MASK, FIM)
    backed by an Aho-Corasick automaton for single-pass linear text splitting.
    """

    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = {}
        self._role_to_token = {}
        self._matcher = AhoCorasick()
        self._dirty = False

    cpdef void register(self, str token, uint32_t token_id, str role=None):
        """Register a special token with an assigned ID and optional role."""
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        if role:
            self._role_to_token[role] = token
        self._matcher.add_pattern(token, token_id)
        self._dirty = True

    cpdef bint is_special_token(self, str token):
        return token in self._token_to_id

    cpdef bint is_special_id(self, uint32_t token_id):
        return token_id in self._id_to_token

    cpdef object get_id(self, str token):
        return self._token_to_id.get(token, None)

    cpdef str get_token(self, uint32_t token_id):
        return self._id_to_token.get(token_id, None)

    cpdef object get_role_id(self, str role):
        cdef str token = self._role_to_token.get(role, None)
        if token:
            return self._token_to_id.get(token, None)
        return None

    def get_role_token(self, str role) -> str:
        return self._role_to_token.get(role, None)

    def split(self, str text) -> list:
        """
        Splits input text on all registered special tokens.
        Returns alternating segments of ('text', chunk) and ('special', (token_id, token_str)).
        """
        if not self._token_to_id:
            return [("text", text)]
        if self._dirty:
            self._matcher.compile()
            self._dirty = False
        return self._matcher.split_with_tokens(text)

    @property
    def token_to_id(self) -> dict:
        return dict(self._token_to_id)

    @property
    def role_to_token(self) -> dict:
        return dict(self._role_to_token)

    @property
    def bos_token_id(self):
        return self.get_role_id("bos")

    @property
    def eos_token_id(self):
        return self.get_role_id("eos")

    @property
    def unk_token_id(self):
        return self.get_role_id("unk")

    @property
    def pad_token_id(self):
        return self.get_role_id("pad")

    @property
    def mask_token_id(self):
        return self.get_role_id("mask")

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, str token) -> bool:
        return token in self._token_to_id

    def __repr__(self) -> str:
        return f"<SpecialTokenRegistry count={len(self)} roles={list(self._role_to_token.keys())}>"
