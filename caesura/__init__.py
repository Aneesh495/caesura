"""
Caesura: High-throughput subword tokenization and lexical analysis engine.

Built with high-performance Cython and pure C99 data structures for zero-copy
subword encoding, deterministic finite automata regex splitting, and SIMD
UTF-8 parsing.
"""

from caesura.core.memory import MemoryArena
from caesura.core.buffer import ByteBuffer, TokenBuffer
from caesura.unicode.utf8 import (
    Utf8Scanner,
    scan_ascii_prefix,
    is_valid_utf8,
    ByteFallback,
)
from caesura.unicode.normalizer import (
    Normalizer,
    LowercaseNormalizer,
    StripAccentsNormalizer,
    NfcNormalizer,
    NfkdNormalizer,
    MetaspaceNormalizer,
    SequenceNormalizer,
)
from caesura.regex.dfa import DfaTable, DfaScanner
from caesura.regex.compiler import (
    RegexPreTokenizer,
    whitespace_pre_tokenizer,
    punctuation_pre_tokenizer,
    gpt2_pre_tokenizer,
    gpt4_pre_tokenizer,
)
from caesura.trie.trie import Trie
from caesura.trie.double_array import DoubleArrayTrie
from caesura.trie.aho_corasick import AhoCorasick
from caesura.vocab.vocab import Vocabulary, create_byte_vocab
from caesura.vocab.special import SpecialTokenRegistry
from caesura.models.bpe import BpeModel
from caesura.models.wordpiece import WordPieceModel
from caesura.models.unigram import UnigramModel
from caesura.trainers.bpe_trainer import BpeTrainer
from caesura.trainers.wordpiece_trainer import WordPieceTrainer
from caesura.trainers.unigram_trainer import UnigramTrainer
from caesura.pipeline.encoding import Encoding
from caesura.pipeline.batch import BatchEncoding
from caesura.pipeline.tokenizer import Tokenizer
from caesura.decoder.decoder import (
    Decoder,
    ByteLevelDecoder,
    WordPieceDecoder,
    MetaspaceDecoder,
    SequenceDecoder,
)
from caesura.decoder.streaming import IncrementalDecoder
from caesura.io.format import BinarySerializer
from caesura.io.tiktoken_io import TiktokenLoader
from caesura.io.huggingface_io import load_huggingface_json, save_huggingface_json

__version__ = "0.1.0"

__all__ = [
    "MemoryArena",
    "ByteBuffer",
    "TokenBuffer",
    "Utf8Scanner",
    "scan_ascii_prefix",
    "is_valid_utf8",
    "ByteFallback",
    "Normalizer",
    "LowercaseNormalizer",
    "StripAccentsNormalizer",
    "NfcNormalizer",
    "NfkdNormalizer",
    "MetaspaceNormalizer",
    "SequenceNormalizer",
    "DfaTable",
    "DfaScanner",
    "RegexPreTokenizer",
    "whitespace_pre_tokenizer",
    "punctuation_pre_tokenizer",
    "gpt2_pre_tokenizer",
    "gpt4_pre_tokenizer",
    "Trie",
    "DoubleArrayTrie",
    "AhoCorasick",
    "Vocabulary",
    "SpecialTokenRegistry",
    "create_byte_vocab",
    "BpeModel",
    "WordPieceModel",
    "UnigramModel",
    "BpeTrainer",
    "WordPieceTrainer",
    "UnigramTrainer",
    "Encoding",
    "BatchEncoding",
    "Tokenizer",
    "Decoder",
    "ByteLevelDecoder",
    "WordPieceDecoder",
    "MetaspaceDecoder",
    "SequenceDecoder",
    "IncrementalDecoder",
    "BinarySerializer",
    "TiktokenLoader",
    "load_huggingface_json",
    "save_huggingface_json",
    "__version__",
]
