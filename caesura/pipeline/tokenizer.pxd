# cython: language_level=3
from caesura.unicode.normalizer cimport Normalizer
from caesura.regex.compiler cimport RegexPreTokenizer
from caesura.vocab.special cimport SpecialTokenRegistry
from caesura.pipeline.encoding cimport Encoding
from caesura.pipeline.batch cimport BatchEncoding

cdef class Tokenizer:
    cdef object _model
    cdef Normalizer _normalizer
    cdef RegexPreTokenizer _pre_tokenizer
    cdef SpecialTokenRegistry _special_tokens
    cdef object _decoder

    cpdef Encoding encode(self, str text, bint add_special_tokens=*)
    cpdef BatchEncoding encode_batch(self, list texts, bint add_special_tokens=*, object padding=*, object max_length=*)
    cpdef str decode(self, list ids, bint skip_special_tokens=*)
    cpdef list decode_batch(self, list batch_ids, bint skip_special_tokens=*)
