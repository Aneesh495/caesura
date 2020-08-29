"""
Caesura Core Module

Provides foundational memory management, native struct definitions,
and zero-copy buffer abstractions for the tokenization engine.
"""

from caesura.core.memory import MemoryArena
from caesura.core.buffer import ByteBuffer, TokenBuffer

__all__ = ["MemoryArena", "ByteBuffer", "TokenBuffer"]
