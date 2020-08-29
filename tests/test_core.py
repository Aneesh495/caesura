"""
Tests for Caesura Core Memory and Buffer Primitives
"""

import pytest
from caesura.core.memory import MemoryArena
from caesura.core.buffer import ByteBuffer, TokenBuffer


def test_memory_arena_lifecycle():
    arena = MemoryArena(default_block_size=4096)
    assert arena.used_bytes == 0
    assert arena.total_bytes >= 4096

    with arena.scope():
        # Scoped operations
        pass

    assert arena.used_bytes == 0
    arena.reset()
    assert arena.used_bytes == 0


def test_byte_buffer_basic():
    buf = ByteBuffer(b"hello world")
    assert len(buf) == 11
    assert buf.to_bytes() == b"hello world"
    assert buf[0] == ord("h")
    assert buf[10] == ord("d")

    with pytest.raises(IndexError):
        _ = buf[11]


def test_byte_buffer_append():
    buf = ByteBuffer(initial_capacity=16)
    assert len(buf) == 0
    buf.append(b"caesura")
    assert len(buf) == 7
    assert buf.to_bytes() == b"caesura"

    buf.append(b"_token")
    assert len(buf) == 13
    assert buf.to_bytes() == b"caesura_token"


def test_token_buffer():
    tbuf = TokenBuffer(initial_capacity=8)
    assert len(tbuf) == 0
    tbuf.append(101)
    tbuf.append(202)
    tbuf.append(303)
    assert len(tbuf) == 3
    assert tbuf.to_list() == [101, 202, 303]
    assert tbuf[1] == 202

    tbuf.extend([404, 505, 606])
    assert len(tbuf) == 6
    assert tbuf.to_list() == [101, 202, 303, 404, 505, 606]

    tbuf.clear()
    assert len(tbuf) == 0
    assert tbuf.to_list() == []
