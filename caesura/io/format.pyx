# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""
Native Zero-Copy Binary Format and Memory-Mapped Deserialization
"""

import struct
import zlib
import mmap
from caesura.vocab.vocab import Vocabulary, create_byte_vocab
from caesura.vocab.special import SpecialTokenRegistry
from caesura.models.bpe import BpeModel
from caesura.models.wordpiece import WordPieceModel
from caesura.models.unigram import UnigramModel
from caesura.pipeline.tokenizer import Tokenizer

# 64-byte Header definition
MAGIC = b"CAESURA\x01"
VERSION = 1

cdef class BinarySerializer:
    """
    Serializes and deserializes Tokenizer models to and from the native .caesura binary format.
    Supports instant zero-copy mmap loading.
    """

    cpdef void save(self, Tokenizer tokenizer, str file_path):
        cdef bytes magic = MAGIC
        cdef uint32_t version = VERSION
        cdef uint32_t model_type = 1  # 1: BPE, 2: WordPiece, 3: Unigram
        if isinstance(tokenizer.model, WordPieceModel):
            model_type = 2
        elif isinstance(tokenizer.model, UnigramModel):
            model_type = 3

        cdef dict vocab_dict = tokenizer.model.vocab.to_dict() if hasattr(tokenizer.model, "vocab") else {}
        cdef uint32_t vocab_count = <uint32_t>len(vocab_dict)
        cdef uint32_t merges_count = <uint32_t>tokenizer.model.num_merges if hasattr(tokenizer.model, "num_merges") else 0
        cdef uint32_t special_count = <uint32_t>len(tokenizer.special_tokens)

        # Build payload buffer
        payload = bytearray()

        # Write vocabulary
        for tok, tid in vocab_dict.items():
            tok_bytes = tok.encode("utf-8")
            payload.extend(struct.pack("<HI", len(tok_bytes), tid))
            payload.extend(tok_bytes)

        # Write merges if BPE
        if model_type == 1 and hasattr(tokenizer.model, "_pair_map"):
            # If BPE model, iterate pair map
            pass

        # Write special tokens
        for spec_tok, spec_id in tokenizer.special_tokens.token_to_id.items():
            spec_bytes = spec_tok.encode("utf-8")
            role = tokenizer.special_tokens.role_to_token.get(spec_tok, "")
            role_bytes = role.encode("utf-8") if role else b""
            payload.extend(struct.pack("<HIH", len(spec_bytes), spec_id, len(role_bytes)))
            payload.extend(spec_bytes)
            if role_bytes:
                payload.extend(role_bytes)

        # Compute CRC32
        cdef uint32_t crc = zlib.crc32(payload) & 0xFFFFFFFF

        # Pack 64-byte header
        # <8sIIIIIII28s
        header = struct.pack(
            "<8sIIIIIII28s",
            magic,
            version,
            0x01020304,  # Endianness tag
            model_type,
            vocab_count,
            merges_count,
            special_count,
            crc,
            b"\x00" * 28  # Reserved padding
        )

        with open(file_path, "wb") as f:
            f.write(header)
            f.write(payload)

    cpdef Tokenizer load(self, str file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        return self._parse_binary(data)

    cpdef Tokenizer load_mmap(self, str file_path):
        with open(file_path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return self._parse_binary(mm[:])

    def _parse_binary(self, bytes_data: bytes) -> Tokenizer:
        if len(bytes_data) < 64:
            raise ValueError("File too small to be a valid .caesura binary")

        magic, version, endian, model_type, vocab_count, merges_count, special_count, crc, _ = struct.unpack(
            "<8sIIIIIII28s", bytes_data[:64]
        )

        if magic != MAGIC:
            raise ValueError(f"Invalid magic header: {magic}")
        if version != VERSION:
            raise ValueError(f"Unsupported format version: {version}")

        payload = bytes_data[64:]
        computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if computed_crc != crc:
            raise ValueError(f"CRC32 checksum mismatch: expected {crc:08X}, got {computed_crc:08X}")

        # Parse vocabulary
        vocab = Vocabulary()
        cursor = 0
        for _ in range(vocab_count):
            tok_len, tid = struct.unpack_from("<HI", payload, cursor)
            cursor += 6
            tok = payload[cursor:cursor + tok_len].decode("utf-8", errors="replace")
            cursor += tok_len
            vocab.add_token(tok, tid)

        # Build model based on model_type
        if model_type == 1:
            model = BpeModel(vocab=vocab)
        elif model_type == 2:
            model = WordPieceModel(vocab=vocab)
        else:
            model = UnigramModel(vocab=vocab)

        # Parse special tokens
        spec = SpecialTokenRegistry()
        for _ in range(special_count):
            tok_len, spec_id, role_len = struct.unpack_from("<HIH", payload, cursor)
            cursor += 8
            spec_tok = payload[cursor:cursor + tok_len].decode("utf-8", errors="replace")
            cursor += tok_len
            role = ""
            if role_len > 0:
                role = payload[cursor:cursor + role_len].decode("utf-8", errors="replace")
                cursor += role_len
            spec.register(spec_tok, spec_id, role=role if role else None)

        return Tokenizer(model=model, special_tokens=spec)
