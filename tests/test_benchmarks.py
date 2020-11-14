"""
Smoke and regression tests for benchmark suite.
"""

from caesura.benchmarks.bench_bpe import benchmark_bpe_reduction, benchmark_bpe_offsets
from caesura.benchmarks.bench_dfa import benchmark_dfa_scanning
from caesura.benchmarks.bench_trie import benchmark_trie_lookups, benchmark_aho_corasick
from caesura.benchmarks.bench_batch import benchmark_batch_throughput


def test_bpe_benchmarks_smoke():
    res = benchmark_bpe_reduction(num_words=100, word_len=8)
    assert res["throughput_mb_s"] > 0
    assert res["words_per_sec"] > 0
    assert res["tokens_produced"] > 0

    res_off = benchmark_bpe_offsets(num_words=100)
    assert res_off["throughput_mb_s"] > 0
    assert res_off["words_per_sec"] > 0


def test_dfa_benchmark_smoke():
    res = benchmark_dfa_scanning(payload_size_kb=8, iterations=2)
    assert res["caesura_mb_s"] > 0
    assert res["caesura_spans_per_sec"] > 0
    assert res["payload_kb"] > 0


def test_trie_benchmarks_smoke():
    t_res = benchmark_trie_lookups(num_keys=50, num_queries=200)
    assert t_res["exact_qps"] > 0
    assert t_res["prefix_qps"] > 0

    ac_res = benchmark_aho_corasick(num_patterns=5, iterations=10)
    assert ac_res["throughput_mb_s"] > 0
    assert ac_res["matches_per_sec"] > 0


def test_batch_benchmark_smoke():
    b_res = benchmark_batch_throughput(num_docs=20, batch_sizes=[10, 20])
    assert 10 in b_res["results_by_batch"]
    assert 20 in b_res["results_by_batch"]
    assert b_res["results_by_batch"][10]["tokens_per_sec"] > 0
