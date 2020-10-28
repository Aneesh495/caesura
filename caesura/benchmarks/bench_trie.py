"""
Micro-benchmarks for Trie, DoubleArrayTrie, and Aho-Corasick Automaton.
"""

import time
from typing import Dict, Any
from caesura.trie.trie import Trie
from caesura.trie.aho_corasick import AhoCorasick


def benchmark_trie_lookups(num_keys: int = 5000, num_queries: int = 100000) -> Dict[str, Any]:
    """Measures exact search and longest prefix matching on Trie."""
    trie = Trie()
    keys = [f"subword_token_prefix_{i}" for i in range(num_keys)]
    for i, k in enumerate(keys):
        trie.insert(k, i)

    query_key = keys[num_keys // 2]

    # Exact search
    t0 = time.perf_counter()
    for _ in range(num_queries):
        _ = trie.find(query_key)
    t1 = time.perf_counter()

    exact_elapsed = max(t1 - t0, 1e-6)
    exact_qps = num_queries / exact_elapsed

    # Longest prefix search
    extended_query = query_key + "_suffix_extra_chars"
    t2 = time.perf_counter()
    for _ in range(num_queries):
        _ = trie.longest_prefix(extended_query)
    t3 = time.perf_counter()

    prefix_elapsed = max(t3 - t2, 1e-6)
    prefix_qps = num_queries / prefix_elapsed

    return {
        "num_keys": num_keys,
        "num_queries": num_queries,
        "exact_qps": exact_qps,
        "exact_latency_ns": (exact_elapsed / num_queries) * 1e9,
        "prefix_qps": prefix_qps,
        "prefix_latency_ns": (prefix_elapsed / num_queries) * 1e9,
    }


def benchmark_aho_corasick(num_patterns: int = 100, iterations: int = 500) -> Dict[str, Any]:
    """Measures Aho-Corasick multi-pattern search throughput."""
    ac = AhoCorasick()
    patterns = [f"<|special_{i}|>" for i in range(num_patterns)]
    for i, pat in enumerate(patterns):
        ac.add_pattern(pat, 50000 + i)
    ac.compile()

    text = "Lorem ipsum dolor sit amet, " + " ".join(patterns) + " consectetur adipiscing elit."
    text_bytes_len = len(text.encode("utf-8"))

    t0 = time.perf_counter()
    total_matches = 0
    for _ in range(iterations):
        matches = ac.find_all(text)
        total_matches += len(matches)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    total_mb = (text_bytes_len * iterations) / (1024 * 1024)

    return {
        "num_patterns": num_patterns,
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "throughput_mb_s": total_mb / elapsed,
        "matches_per_sec": total_matches / elapsed,
    }


if __name__ == "__main__":
    t_res = benchmark_trie_lookups()
    print(f"Trie Exact Match: {t_res['exact_qps']:,.0f} queries/s ({t_res['exact_latency_ns']:.1f} ns/op)")
    print(f"Trie Longest Prefix: {t_res['prefix_qps']:,.0f} queries/s ({t_res['prefix_latency_ns']:.1f} ns/op)")
    ac_res = benchmark_aho_corasick()
    print(f"Aho-Corasick Multi-Pattern: {ac_res['throughput_mb_s']:.2f} MB/s ({ac_res['matches_per_sec']:,.0f} matches/s)")
