"""
Advanced tests for Trie prefix enumeration and Levenshtein fuzzy search.
"""

import pytest
from caesura.trie.trie import Trie


def test_trie_keys_with_prefix():
    trie = Trie()
    trie.insert("apple", 1)
    trie.insert("application", 2)
    trie.insert("apply", 3)
    trie.insert("banana", 4)

    matches = trie.keys_with_prefix("app")
    keys = {m[0] for m in matches}
    assert keys == {"apple", "application", "apply"}

    # Non-existent prefix
    assert trie.keys_with_prefix("orange") == []


def test_trie_fuzzy_search():
    trie = Trie()
    trie.insert("caesura", 100)
    trie.insert("caesar", 101)
    trie.insert("capsule", 102)

    # 1 edit distance from "caesura": "caesar" has distance 2, "caesur" has 1
    # Query "caesur" (1 deletion from caesura)
    res = trie.fuzzy_search("caesur", max_edits=1)
    res_keys = [r[0] for r in res]
    assert "caesura" in res_keys

    # Query "caesura" with 0 edits
    res0 = trie.fuzzy_search("caesura", max_edits=0)
    assert len(res0) == 1
    assert res0[0][0] == "caesura"
    assert res0[0][2] == 0


def test_trie_large_scale_resizing():
    trie = Trie(initial_capacity=16)
    for i in range(2000):
        trie.insert(f"token_{i}", i)

    assert len(trie) == 2000
    for i in range(2000):
        assert trie.find(f"token_{i}") == i
