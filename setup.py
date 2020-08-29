import os
import sys
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

extra_compile_args = ["-O3", "-Wall", "-Wextra", "-Wno-unused-function", "-std=c11"]
extra_link_args = []

if sys.platform == "darwin":
    extra_compile_args.extend(["-fPIC"])
elif sys.platform.startswith("linux"):
    extra_compile_args.extend(["-fPIC"])

native_sources = [
    "caesura/native/arena.c",
    "caesura/native/murmur3.c",
    "caesura/native/robin_hood.c",
    "caesura/native/simd_utf8.c",
]

include_dirs = ["caesura/native", "."]

extension_modules = [
    # Core memory and buffer abstractions
    Extension(
        "caesura.core.memory",
        ["caesura/core/memory.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.core.buffer",
        ["caesura/core/buffer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Unicode and normalization
    Extension(
        "caesura.unicode.utf8",
        ["caesura/unicode/utf8.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.unicode.tables",
        ["caesura/unicode/tables.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.unicode.normalizer",
        ["caesura/unicode/normalizer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Regex DFA engine
    Extension(
        "caesura.regex.dfa",
        ["caesura/regex/dfa.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.regex.patterns",
        ["caesura/regex/patterns.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.regex.compiler",
        ["caesura/regex/compiler.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Trie and Aho-Corasick
    Extension(
        "caesura.trie.trie",
        ["caesura/trie/trie.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.trie.double_array",
        ["caesura/trie/double_array.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.trie.aho_corasick",
        ["caesura/trie/aho_corasick.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Subword models
    Extension(
        "caesura.models.bpe",
        ["caesura/models/bpe.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.models.wordpiece",
        ["caesura/models/wordpiece.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.models.unigram",
        ["caesura/models/unigram.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Vocabulary and special tokens
    Extension(
        "caesura.vocab.vocab",
        ["caesura/vocab/vocab.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.vocab.special",
        ["caesura/vocab/special.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Trainers
    Extension(
        "caesura.trainers.bpe_trainer",
        ["caesura/trainers/bpe_trainer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.trainers.wordpiece_trainer",
        ["caesura/trainers/wordpiece_trainer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.trainers.unigram_trainer",
        ["caesura/trainers/unigram_trainer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Pipeline, batching, tokenizer
    Extension(
        "caesura.pipeline.encoding",
        ["caesura/pipeline/encoding.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.pipeline.batch",
        ["caesura/pipeline/batch.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.pipeline.tokenizer",
        ["caesura/pipeline/tokenizer.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Decoder and streaming
    Extension(
        "caesura.decoder.decoder",
        ["caesura/decoder/decoder.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.decoder.streaming",
        ["caesura/decoder/streaming.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    # Binary I/O and format
    Extension(
        "caesura.io.format",
        ["caesura/io/format.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "caesura.io.tiktoken_io",
        ["caesura/io/tiktoken_io.pyx"] + native_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

# Only build extensions whose primary .pyx file exists
available_extensions = [
    ext for ext in extension_modules if os.path.exists(ext.sources[0])
]

directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "nonecheck": False,
    "initializedcheck": False,
    "embedsignature": True,
}

if USE_CYTHON and available_extensions:
    ext_modules = cythonize(available_extensions, compiler_directives=directives)
else:
    ext_modules = available_extensions

setup(
    ext_modules=ext_modules,
)
