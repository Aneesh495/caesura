"""
Tests for Caesura Trie, DoubleArrayTrie, and Aho-Corasick Multi-Pattern Automaton
"""

import pytest
from caesura.trie import Trie, DoubleArrayTrie, AhoCorasick


def test_trie_basic_operations():
    trie = Trie()
    assert len(trie) == 0

    trie.insert("caesura", 100)
    trie.insert("caesural", 101)
    trie.insert("caesar", 102)
    trie.insert("token", 200)

    assert len(trie) == 4
    assert trie.find("caesura") == 100
    assert trie.find("caesural") == 101
    assert trie.find("caesar") == 102
    assert trie.find("token") == 200
    assert trie.find("caes") is None
    assert trie.find("unknown") is None
    assert "caesura" in trie
    assert "unknown" not in trie


def test_trie_longest_prefix():
    trie = Trie()
    trie.insert("pre", 1)
    trie.insert("prefix", 2)
    trie.insert("preface", 3)

    # "prefixation" should match "prefix" (len 6)
    match_len, tid = trie.longest_prefix("prefixation")
    assert match_len == 6
    assert tid == 2

    # "pretend" should match "pre" (len 3)
    match_len, tid = trie.longest_prefix("pretend")
    assert match_len == 3
    assert tid == 1

    # "xyz" has no match
    match_len, tid = trie.longest_prefix("xyz")
    assert match_len == 0
    assert tid is None


def test_double_array_trie_exact():
    dat = DoubleArrayTrie(initial_capacity=1024)
    assert dat.capacity >= 1024


def test_aho_corasick_multi_pattern():
    ac = AhoCorasick()
    ac.add_pattern("<|endoftext|>", 50256)
    ac.add_pattern("<|fim_prefix|>", 50257)
    ac.add_pattern("<|fim_suffix|>", 50258)
    ac.compile()

    text = "Prompt: write code <|fim_prefix|>def add(a, b):<|fim_suffix|> return a + b<|endoftext|>"
    matches = ac.find_all(text)
    assert len(matches) == 3

    # Check match contents
    m0_start, m0_end, m0_id = matches[0]
    assert m0_id == 50257
    assert text[m0_start:m0_end] == "<|fim_prefix|>"

    m1_start, m1_end, m1_id = matches[1]
    assert m1_id == 50258
    assert text[m1_start:m1_end] == "<|fim_suffix|>"

    m2_start, m2_end, m2_id = matches[2]
    assert m2_id == 50256
    assert text[m2_start:m2_end] == "<|endoftext|>"


def test_aho_corasick_split_with_tokens():
    ac = AhoCorasick()
    ac.add_pattern("<|start|>", 1)
    ac.add_pattern("<|end|>", 2)
    ac.compile()

    text = "<|start|>hello world<|end|>"
    segments = ac.split_with_tokens(text)

    expected = [
        ("special", (1, "<|start|>")),
        ("text", "hello world"),
        ("special", (2, "<|end|>")),
    ]
    assert segments == expected
