"""
Caesura Benchmarks Module
"""

from caesura.benchmarks.suite import run_all_benchmarks
from caesura.benchmarks.bench_bpe import benchmark_bpe_reduction, benchmark_bpe_offsets
from caesura.benchmarks.bench_dfa import benchmark_dfa_scanning
from caesura.benchmarks.bench_trie import benchmark_trie_lookups, benchmark_aho_corasick
from caesura.benchmarks.bench_batch import benchmark_batch_throughput

__all__ = [
    "run_all_benchmarks",
    "benchmark_bpe_reduction",
    "benchmark_bpe_offsets",
    "benchmark_dfa_scanning",
    "benchmark_trie_lookups",
    "benchmark_aho_corasick",
    "benchmark_batch_throughput",
]
