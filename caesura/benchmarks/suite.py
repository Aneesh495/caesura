"""
Unified Benchmark Suite for Caesura Subword Engine.
"""

from typing import Dict, Any
from rich.console import Console
from rich.table import Table

from caesura.benchmarks.bench_bpe import benchmark_bpe_reduction, benchmark_bpe_offsets
from caesura.benchmarks.bench_dfa import benchmark_dfa_scanning
from caesura.benchmarks.bench_trie import benchmark_trie_lookups, benchmark_aho_corasick
from caesura.benchmarks.bench_batch import benchmark_batch_throughput


def run_all_benchmarks(console: Console = None) -> Dict[str, Any]:
    """Execute all engine micro-benchmarks and return aggregated report."""
    if console is None:
        console = Console()

    console.print("\n[bold cyan]==================================================[/bold cyan]")
    console.print("[bold cyan]       CAESURA ENGINE BENCHMARK SUITE             [/bold cyan]")
    console.print("[bold cyan]==================================================[/bold cyan]\n")

    report = {}

    # 1. BPE Reduction
    console.print("[yellow]Benchmarking BPE Pair Reduction...[/yellow]")
    bpe_res = benchmark_bpe_reduction(num_words=20000)
    bpe_off = benchmark_bpe_offsets(num_words=20000)
    report["bpe_reduction"] = bpe_res
    report["bpe_offsets"] = bpe_off

    # 2. DFA Scanning
    console.print("[yellow]Benchmarking DFA Lexical Scanner...[/yellow]")
    dfa_res = benchmark_dfa_scanning(payload_size_kb=256, iterations=10)
    report["dfa_scanner"] = dfa_res

    # 3. Trie Lookups
    console.print("[yellow]Benchmarking Trie & Aho-Corasick Automata...[/yellow]")
    trie_res = benchmark_trie_lookups(num_keys=2000, num_queries=50000)
    ac_res = benchmark_aho_corasick(num_patterns=50, iterations=200)
    report["trie"] = trie_res
    report["aho_corasick"] = ac_res

    # 4. Batch Tokenization
    console.print("[yellow]Benchmarking Batch Tokenizer Throughput...[/yellow]")
    batch_res = benchmark_batch_throughput(num_docs=2000, batch_sizes=[100, 500, 2000])
    report["batch"] = batch_res

    # Present Summary Table
    table = Table(title="Caesura Subsystem Performance Summary")
    table.add_column("Subsystem / Micro-Benchmark", style="cyan")
    table.add_column("Throughput", style="green")
    table.add_column("Latency / Rate", style="magenta")

    table.add_row(
        "BPE Pair Reduction",
        f"{bpe_res['throughput_mb_s']:.2f} MB/s",
        f"{bpe_res['words_per_sec']:,.0f} words/s",
    )
    table.add_row(
        "BPE With Offset Tracking",
        f"{bpe_off['throughput_mb_s']:.2f} MB/s",
        f"{bpe_off['words_per_sec']:,.0f} words/s",
    )
    table.add_row(
        "DFA Lexical Scanner",
        f"{dfa_res['caesura_mb_s']:.2f} MB/s",
        f"{dfa_res['caesura_spans_per_sec']:,.0f} spans/s",
    )
    table.add_row(
        "Trie Exact Key Lookup",
        f"{trie_res['exact_qps']:,.0f} ops/s",
        f"{trie_res['exact_latency_ns']:.1f} ns/op",
    )
    table.add_row(
        "Trie Longest Prefix Match",
        f"{trie_res['prefix_qps']:,.0f} ops/s",
        f"{trie_res['prefix_latency_ns']:.1f} ns/op",
    )
    table.add_row(
        "Aho-Corasick Multi-Pattern",
        f"{ac_res['throughput_mb_s']:.2f} MB/s",
        f"{ac_res['matches_per_sec']:,.0f} matches/s",
    )

    batch_500 = batch_res["results_by_batch"][500]
    table.add_row(
        "Batch Tokenizer (Batch Size 500)",
        f"{batch_500['throughput_mb_s']:.2f} MB/s",
        f"{batch_500['tokens_per_sec']:,.0f} tokens/s",
    )

    console.print("\n")
    console.print(table)
    console.print("\n[bold green]Benchmark suite completed successfully.[/bold green]\n")

    return report


if __name__ == "__main__":
    run_all_benchmarks()
