"""
Streaming and Incremental Detokenization Demo.

Demonstrates real-time incremental token generation decoding, including
handling split multi-byte UTF-8 sequences across token boundaries without
rendering replacement characters or corrupted glyphs.
"""

import time
from caesura import (
    IncrementalDecoder,
    ByteLevelDecoder,
    create_byte_vocab,
)


def main():
    print("==================================================")
    print("      CAESURA STREAMING DETOKENIZER DEMO          ")
    print("==================================================")

    stream_decoder = IncrementalDecoder()

    # Unicode phrase containing multi-byte characters and emoji
    # "Hello 🌍 World 🚀"
    phrase = "Hello 🌍 World 🚀"
    vocab = create_byte_vocab()

    # Simulate tokens arriving token-by-token from an inference engine
    print(f"\nTarget Text: {phrase}")
    print("Simulating token stream output with incremental buffering:\n")

    # Encode into byte tokens
    from caesura.unicode.utf8 import ByteFallback
    bf = ByteFallback()
    encoded_chars = [bf.byte_to_char(b) for b in phrase.encode("utf-8")]

    # Feed chunks of 1 to 3 characters to test boundary recombination
    reconstructed = []
    chunk_size = 2
    for i in range(0, len(encoded_chars), chunk_size):
        chunk = "".join(encoded_chars[i:i + chunk_size])
        text_piece = stream_decoder.decode_step(chunk)
        if text_piece:
            reconstructed.append(text_piece)
            print(f"  Chunk received: {repr(chunk):10s} -> Emitted: {repr(text_piece)}")
        else:
            print(f"  Chunk received: {repr(chunk):10s} -> [Buffered incomplete UTF-8 bytes]")

    # Flush remaining bytes
    final_piece = stream_decoder.flush()
    if final_piece:
        reconstructed.append(final_piece)
        print(f"  Flushed remaining: {repr(final_piece)}")

    result = "".join(reconstructed)
    print(f"\nFinal Reconstructed Text: '{result}'")
    assert result == phrase
    print("Incremental streaming validation successful.")


if __name__ == "__main__":
    main()
