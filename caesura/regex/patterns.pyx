# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Pre-Compiled Regex Patterns and Built-in DFA Generators
"""

from caesura.regex.dfa cimport DfaTable
from caesura.unicode.tables cimport (
    c_is_whitespace,
    c_is_punctuation,
    c_is_alphanumeric,
)

# Standard regex strings used across leading LLM architectures
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
LLAMA3_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
WHITESPACE_PATTERN = r"""\S+|\s+"""
PUNCTUATION_PATTERN = r"""\w+|[^\w\s]+|\s+"""

cdef DfaTable create_word_dfa():
    """
    Constructs a 4-state minimal DFA recognizing:
    Rule 1: Alphanumeric words [a-zA-Z0-9]+
    Rule 2: Whitespace runs [ \\t\\r\\n]+
    Rule 3: Punctuation sequences
    """
    cdef DfaTable dfa = DfaTable(4, start_state=0)
    cdef int b

    # State 0: start state
    for b in range(256):
        if (b >= 0x30 and b <= 0x39) or (b >= 0x41 and b <= 0x5A) or (b >= 0x61 and b <= 0x7A):
            dfa.set_transition(0, b, 1)  # go to word state
            dfa.set_transition(1, b, 1)  # loop in word state
        elif b == 0x20 or b == 0x09 or b == 0x0A or b == 0x0D:
            dfa.set_transition(0, b, 2)  # go to whitespace state
            dfa.set_transition(2, b, 2)  # loop in whitespace state
        elif (b >= 33 and b <= 47) or (b >= 58 and b <= 64) or (b >= 91 and b <= 96) or (b >= 123 and b <= 126):
            dfa.set_transition(0, b, 3)  # go to punctuation state
            dfa.set_transition(3, b, 3)  # loop in punctuation state

    # Mark accepting states
    dfa.set_accept(1, 1)  # Rule 1: Word
    dfa.set_accept(2, 2)  # Rule 2: Whitespace
    dfa.set_accept(3, 3)  # Rule 3: Punctuation

    return dfa

def get_word_dfa() -> DfaTable:
    return create_word_dfa()
