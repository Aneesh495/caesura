"""
Benchmarks for batch tokenization throughput, padding, and truncation overhead.
"""

import time
from typing import Dict, Any, List
from caesura.vocab.vocab import create_byte_vocab
from caesura.models.bpe import BpeModel
from caesura.pipeline.tokenizer import Tokenizer
from caesura.regex.compiler import whitespace_pre_tokenizer


def benchmark_batch_throughput(
    num_docs: int = 5000,
    batch_sizes: List[int] = None,
) -> Dict[str, Any]:
    """Measures batch tokenization throughput and alignment performance."""
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000, 5000]

    vocab = create_byte_vocab()
    vocab.add_token(" ", 256)
    bpe = BpeModel(vocab=vocab)

    # Setup standard subword merges
    bpe.add_merge("t", "h", rank=0, merged_token="th")
    bpe.add_merge("th", "e", rank=1, merged_token="the")
    bpe.add_merge("a", "n", rank=2, merged_token="an")
    bpe.add_merge("i", "n", rank=3, merged_token="in")
    bpe.add_merge("e", "r", rank=4, merged_token="er")

    tok = Tokenizer(model=bpe, pre_tokenizer=whitespace_pre_tokenizer(keep_whitespace=True))

    corpus_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Deterministic finite state automata enable linear time lexical scanning.",
        "Cython bridges high-performance C algorithms with Python developer ergonomics.",
        "Zero-copy token buffers and memory arenas eliminate allocation thrashing.",
    ]

    docs = [corpus_samples[i % len(corpus_samples)] for i in range(num_docs)]
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)

    # Warmup
    _ = tok.encode_batch(docs[:50])

    results_by_batch = {}

    for bs in batch_sizes:
        subset = docs[:bs]
        subset_bytes = sum(len(d.encode("utf-8")) for d in subset)

        t0 = time.perf_counter()
        batch_enc = tok.encode_batch(subset, padding=True)
        t1 = time.perf_counter()

        elapsed = max(t1 - t0, 1e-6)
        total_tokens = sum(len(enc) for enc in batch_enc)
        throughput_tok_s = total_tokens / elapsed
        throughput_mb_s = (subset_bytes / (1024 * 1024)) / elapsed

        results_by_batch[bs] = {
            "elapsed_sec": elapsed,
            "total_tokens": total_tokens,
            "tokens_per_sec": throughput_tok_s,
            "throughput_mb_s": throughput_mb_s,
            "num_docs": bs,
        }

    return {
        "num_docs": num_docs,
        "total_bytes": total_bytes,
        "results_by_batch": results_by_batch,
    }


if __name__ == "__main__":
    res = benchmark_batch_throughput(num_docs=5000, batch_sizes=[100, 500, 1000, 5000])
    for bs, data in res["results_by_batch"].items():
        print(
            f"Batch Size: {bs:5d} | "
            f"Throughput: {data['tokens_per_sec']:,.0f} tok/s ({data['throughput_mb_s']:.2f} MB/s) | "
            f"Time: {data['elapsed_sec'] * 1000:.2f} ms"
        )
