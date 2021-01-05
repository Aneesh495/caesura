"""
Custom BPE Vocabulary Training Walkthrough.

Demonstrates end-to-end training of a subword vocabulary from text files,
exporting the compiled model to native .caesura binary format, and reloading
via zero-copy memory mapping.
"""

import os
import tempfile
from caesura import (
    BpeTrainer,
    Tokenizer,
    BinarySerializer,
    SpecialTokenRegistry,
    whitespace_pre_tokenizer,
    ByteLevelDecoder,
)


def main():
    print("==================================================")
    print("       CAESURA CUSTOM BPE MODEL TRAINER           ")
    print("==================================================")

    corpus_text = (
        "Cython bridges high-performance C data structures with Python developer ergonomics.\n"
        "Deterministic finite state automata enable linear time lexical scanning.\n"
        "Byte-pair encoding iteratively merges frequent character pairs into subword tokens.\n"
        "Subword tokenization avoids out-of-vocabulary anomalies while preserving vocabulary efficiency.\n"
        "Zero-copy memory mapped files enable instantaneous model loading without serialization overhead.\n"
    ) * 20

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        model_path = os.path.join(tmpdir, "custom_engine.caesura")

        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(corpus_text)

        print(f"Generated synthetic training corpus ({len(corpus_text):,} bytes).")

        # Configure trainer
        trainer = BpeTrainer(
            vocab_size=320,
            min_frequency=2,
            special_tokens=["<|pad|>", "<|bos|>", "<|eos|>"],
        )

        print("Training BPE model...")
        model = trainer.train_from_files([corpus_path])

        spec = SpecialTokenRegistry()
        spec.register("<|pad|>", 0, role="pad")
        spec.register("<|bos|>", 1, role="bos")
        spec.register("<|eos|>", 2, role="eos")

        tokenizer = Tokenizer(
            model=model,
            pre_tokenizer=whitespace_pre_tokenizer(keep_whitespace=True),
            decoder=ByteLevelDecoder(),
            special_tokens=spec,
        )

        print(f"Training completed successfully:")
        print(f"  Vocabulary size: {len(model.vocab)}")
        print(f"  Learned merges:  {model.num_merges}")

        # Save to native binary format
        serializer = BinarySerializer()
        serializer.save(tokenizer, model_path)
        print(f"Model serialized to binary format: {model_path} ({os.path.getsize(model_path):,} bytes)")

        # Reload with zero-copy mmap
        loaded_tok = serializer.load_mmap(model_path)
        print("Model reloaded via zero-copy mmap.")

        sample_input = "Deterministic subword tokenization enables high-performance inference."
        encoding = loaded_tok.encode(sample_input)

        print(f"\nSample Input:  '{sample_input}'")
        print(f"Token IDs:     {encoding.ids}")
        print(f"Tokens:        {encoding.tokens}")

        decoded = loaded_tok.decode(encoding.ids, skip_special_tokens=True)
        print(f"Decoded Text:  '{decoded}'")
        assert decoded == sample_input
        print("\nTrainer walkthrough completed with zero defects.")


if __name__ == "__main__":
    main()
