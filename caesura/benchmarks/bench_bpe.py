"""
Micro-benchmarks for Byte-Pair Encoding pair reduction throughput.
"""

import time
from typing import Dict, Any
from caesura.vocab.vocab import create_byte_vocab
from caesura.models.bpe import BpeModel


def benchmark_bpe_reduction(num_words: int = 50000, word_len: int = 16) -> Dict[str, Any]:
    """Measures BPE pair reduction speed on repeated subword patterns."""
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)

    # Add systematic multi-stage merges: a+b -> ab, c+d -> cd, ab+cd -> abcd
    bpe.add_merge("a", "b", rank=0, merged_token="ab")
    bpe.add_merge("c", "d", rank=1, merged_token="cd")
    bpe.add_merge("ab", "cd", rank=2, merged_token="abcd")
    bpe.add_merge("e", "f", rank=3, merged_token="ef")
    bpe.add_merge("g", "h", rank=4, merged_token="gh")
    bpe.add_merge("ef", "gh", rank=5, merged_token="efgh")
    bpe.add_merge("abcd", "efgh", rank=6, merged_token="abcdefgh")

    unit_word = "abcdefgh" * (word_len // 8)
    total_bytes = len(unit_word.encode("utf-8")) * num_words

    # Warmup pass
    for _ in range(100):
        _ = bpe.encode(unit_word)

    t0 = time.perf_counter()
    tokens_produced = 0
    for _ in range(num_words):
        res = bpe.encode(unit_word)
        tokens_produced += len(res)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    throughput_mb = (total_bytes / (1024 * 1024)) / elapsed
    words_per_sec = num_words / elapsed

    return {
        "num_words": num_words,
        "word_len": word_len,
        "elapsed_sec": elapsed,
        "throughput_mb_s": throughput_mb,
        "words_per_sec": words_per_sec,
        "tokens_produced": tokens_produced,
    }


def benchmark_bpe_offsets(num_words: int = 50000) -> Dict[str, Any]:
    """Measures overhead of offset tracking during BPE reduction."""
    vocab = create_byte_vocab()
    bpe = BpeModel(vocab=vocab)
    bpe.add_merge("u", "n", rank=0, merged_token="un")
    bpe.add_merge("un", "k", rank=1, merged_token="unk")
    bpe.add_merge("n", "o", rank=2, merged_token="no")
    bpe.add_merge("w", "n", rank=3, merged_token="wn")

    word = "unknown"
    total_bytes = len(word.encode("utf-8")) * num_words

    t0 = time.perf_counter()
    for _ in range(num_words):
        _ = bpe.encode_with_offsets(word)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    throughput_mb = (total_bytes / (1024 * 1024)) / elapsed

    return {
        "num_words": num_words,
        "elapsed_sec": elapsed,
        "throughput_mb_s": throughput_mb,
        "words_per_sec": num_words / elapsed,
    }


if __name__ == "__main__":
    res = benchmark_bpe_reduction()
    print(f"BPE Reduction Throughput: {res['throughput_mb_s']:.2f} MB/s ({res['words_per_sec']:,.0f} words/s)")
    res_off = benchmark_bpe_offsets()
    print(f"BPE Offsets Throughput: {res_off['throughput_mb_s']:.2f} MB/s ({res_off['words_per_sec']:,.0f} words/s)")
