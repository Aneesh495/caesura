"""
Caesura Quickstart: Basic tokenization, offset tracking, and decoding.
"""

from caesura import (
    Tokenizer,
    BpeModel,
    create_byte_vocab,
    SpecialTokenRegistry,
    whitespace_pre_tokenizer,
    ByteLevelDecoder,
)


def main():
    print("==================================================")
    print("           CAESURA ENGINE QUICKSTART              ")
    print("==================================================")

    # 1. Initialize Vocabulary and BPE Model
    vocab = create_byte_vocab()
    vocab.add_token(" ", 256)
    bpe = BpeModel(vocab=vocab)

    # 2. Add subword merge rules
    bpe.add_merge("h", "e", rank=0, merged_token="he")
    bpe.add_merge("l", "l", rank=1, merged_token="ll")
    bpe.add_merge("he", "ll", rank=2, merged_token="hell")
    bpe.add_merge("hell", "o", rank=3, merged_token="hello")
    bpe.add_merge("w", "o", rank=4, merged_token="wo")
    bpe.add_merge("r", "l", rank=5, merged_token="rl")
    bpe.add_merge("rl", "d", rank=6, merged_token="rld")
    bpe.add_merge("wo", "rld", rank=7, merged_token="world")

    # 3. Configure Special Tokens
    spec = SpecialTokenRegistry()
    spec.register("<|bos|>", 500, role="bos")
    spec.register("<|eos|>", 501, role="eos")

    # 4. Construct Master Tokenizer
    tok = Tokenizer(
        model=bpe,
        pre_tokenizer=whitespace_pre_tokenizer(keep_whitespace=True),
        decoder=ByteLevelDecoder(),
        special_tokens=spec,
    )

    # 5. Encode Sample Prompt
    prompt = "<|bos|>hello world<|eos|>"
    encoding = tok.encode(prompt)

    print(f"\nInput Prompt: {prompt}")
    print(f"Token Count:  {len(encoding)}")
    print(f"Token IDs:    {encoding.ids}")
    print(f"Token Pieces: {encoding.tokens}")

    print("\nToken Offset Alignment:")
    for i in range(len(encoding)):
        s, e = encoding.offsets[i]
        span_slice = prompt[s:e]
        is_spec = encoding.special_tokens_mask[i]
        print(f"  [{i}] ID: {encoding.ids[i]:4d} | Span: {s:2d}..{e:2d} | Slice: {repr(span_slice):12s} | Special: {is_spec}")

    # 6. Decode back to text
    decoded = tok.decode(encoding.ids, skip_special_tokens=True)
    print(f"\nDecoded (skip special): '{decoded}'")
    assert decoded == "hello world"

    decoded_all = tok.decode(encoding.ids, skip_special_tokens=False)
    print(f"Decoded (with special): '{decoded_all}'")
    assert decoded_all == prompt

    print("\nQuickstart walkthrough completed successfully.")


if __name__ == "__main__":
    main()
