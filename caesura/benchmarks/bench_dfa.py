"""
Micro-benchmarks for Deterministic Finite Automata (DFA) scanning.
"""

import re
import time
from typing import Dict, Any
from caesura.regex.dfa import DfaScanner
from caesura.regex.patterns import get_word_dfa


def benchmark_dfa_scanning(payload_size_kb: int = 512, iterations: int = 20) -> Dict[str, Any]:
    """Measures scanning speed of DfaScanner over repeated text."""
    sample_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Cython engines provide deterministic execution speeds "
        "without garbage collection pauses.\n"
    )
    multiplier = (payload_size_kb * 1024) // len(sample_text)
    full_text = (sample_text * multiplier).encode("utf-8")
    actual_kb = len(full_text) / 1024

    dfa = get_word_dfa()

    # Warmup
    _ = DfaScanner(dfa, full_text).find_all_spans()

    t0 = time.perf_counter()
    total_spans = 0
    for _ in range(iterations):
        scanner = DfaScanner(dfa, full_text)
        spans = scanner.find_all_spans()
        total_spans += len(spans)
    t1 = time.perf_counter()

    elapsed = max(t1 - t0, 1e-6)
    total_mb = (len(full_text) * iterations) / (1024 * 1024)
    throughput_mb_s = total_mb / elapsed

    # Baseline comparison with Python re
    py_pattern = re.compile(rb"[A-Za-z0-9_]+")
    t2 = time.perf_counter()
    py_spans = 0
    for _ in range(iterations):
        matches = list(py_pattern.finditer(full_text))
        py_spans += len(matches)
    t3 = time.perf_counter()

    py_elapsed = max(t3 - t2, 1e-6)
    py_throughput_mb_s = total_mb / py_elapsed

    return {
        "payload_kb": actual_kb,
        "iterations": iterations,
        "caesura_elapsed": elapsed,
        "caesura_mb_s": throughput_mb_s,
        "caesura_spans_per_sec": total_spans / elapsed,
        "python_re_elapsed": py_elapsed,
        "python_re_mb_s": py_throughput_mb_s,
        "speedup": py_elapsed / elapsed,
    }


if __name__ == "__main__":
    res = benchmark_dfa_scanning()
    print(f"Caesura DFA Scanner Throughput: {res['caesura_mb_s']:.2f} MB/s")
    print(f"Python re Scanner Throughput:  {res['python_re_mb_s']:.2f} MB/s")
    print(f"Speedup Factor: {res['speedup']:.2f}x")
