# Caesura API Reference Manual

Exhaustive API documentation for public Python and Cython extension types in the Caesura engine.

---

## 1. Pipeline and Orchestration

### `caesura.pipeline.Tokenizer`

The master orchestration class executing normalization, pre-tokenization, subword modeling, special token injection, and decoding.

```python
class Tokenizer:
    def __init__(
        self,
        model: object,
        normalizer: Optional[Normalizer] = None,
        pre_tokenizer: Optional[RegexPreTokenizer] = None,
        post_processor: Optional[object] = None,
        decoder: Optional[Decoder] = None,
        special_tokens: Optional[SpecialTokenRegistry] = None,
    ): ...
```

#### Methods

- **`encode(text: str, add_special_tokens: bool = True) -> Encoding`**
  Encodes a single text string into an `Encoding` container with token IDs, string pieces, character offsets, attention mask, and special token masks.
- **`encode_batch(texts: List[str], add_special_tokens: bool = True, padding: bool = False, max_length: Optional[int] = None) -> BatchEncoding`**
  Encodes multiple text strings in a batch with optional uniform padding and truncation.
- **`decode(ids: List[int], skip_special_tokens: bool = True) -> str`**
  Decodes a sequence of numerical token IDs back into natural text.

---

### `caesura.pipeline.Encoding`

Container storing aligned token representations for a single sequence.

#### Properties
- **`ids: List[int]`**: Integer token IDs.
- **`tokens: List[str]`**: String representation of each subword piece.
- **`offsets: List[Tuple[int, int]]`**: Start and end character offsets in the original text.
- **`attention_mask: List[int]`**: Binary mask (1 for active tokens, 0 for padding).
- **`special_tokens_mask: List[int]`**: Binary mask (1 for special tokens, 0 for regular content).

#### Methods
- **`pad(target_length: int, pad_id: int = 0, pad_token: str = "<pad>", direction: str = "right")`**
  Pads the sequence to `target_length`.
- **`truncate(max_length: int, direction: str = "right")`**
  Truncates sequence to `max_length`.

---

### `caesura.pipeline.BatchEncoding`

Container managing multiple `Encoding` objects with tensor export capabilities.

#### Methods
- **`pad(length: Optional[int] = None, pad_id: int = 0, pad_token: str = "<pad>", direction: str = "right")`**
  Pads all sequences in the batch to the maximum sequence length or explicitly provided target length.
- **`truncate(max_length: int, direction: str = "right")`**
  Truncates all sequences in the batch to `max_length`.
- **`as_dict() -> Dict[str, List[List[int]]]`**
  Exports the batch as standard dictionary format (`input_ids`, `attention_mask`, `token_type_ids`).

---

## 2. Subword Models

### `caesura.models.BpeModel`

Byte-Pair Encoding model executing pair reduction via a Robin Hood hash map and a C doubly-linked token node array.

```python
class BpeModel:
    def __init__(self, vocab: Optional[Vocabulary] = None, byte_fallback: bool = True): ...
```

#### Methods
- **`add_merge(left: Union[str, int], right: Union[str, int], rank: int, merged_token: Optional[Union[str, int]] = None)`**
  Registers a prioritized merge rule.
- **`encode(text: str) -> List[int]`**
  Encodes word or text string into subword IDs.
- **`encode_with_offsets(text: str) -> List[Tuple[int, int, int]]`**
  Returns tuples of `(token_id, start_offset, end_offset)`.
- **`merges -> List[Tuple[int, int, int, int]]`**
  List of learned merge rules.

---

### `caesura.models.WordPieceModel`

Greedy longest-match WordPiece model with continuation prefix support (e.g., `##`).

```python
class WordPieceModel:
    def __init__(
        self,
        vocab: Vocabulary,
        unk_token: str = "[UNK]",
        continuing_prefix: str = "##",
        max_input_chars_per_word: int = 100,
    ): ...
```

#### Methods
- **`encode_word(word: str) -> List[int]`**
- **`encode_word_with_offsets(word: str, word_offset: int = 0) -> List[Tuple[int, int, int]]`**
- **`encode(words: Union[str, List[str]]) -> List[int]`**
- **`encode_with_offsets(words: Union[str, List[str]]) -> List[Tuple[int, int, int]]`**

---

### `caesura.models.UnigramModel`

Unigram language model finding globally optimal token segmentations via dynamic programming Viterbi trellis decoding.

```python
class UnigramModel:
    def __init__(
        self,
        vocab: Vocabulary,
        scores: Optional[Dict[int, float]] = None,
        unk_token: str = "<unk>",
        unk_score: float = -10.0,
    ): ...
```

#### Methods
- **`encode(text: str) -> List[int]`**
- **`encode_with_offsets(text: str) -> List[Tuple[int, int, int]]`**
- **`score(token: str) -> float`**
- **`score_sequence(ids: List[int]) -> float`**

---

## 3. Data Structures and Lexical Automata

### `caesura.trie.Trie`

Pointer-based trie with support for longest-prefix matching and Levenshtein edit tolerance.

#### Methods
- **`insert(key: Union[str, bytes], token_id: int)`**
- **`find(key: Union[str, bytes]) -> Optional[int]`**
- **`longest_prefix(key: Union[str, bytes]) -> Tuple[int, Optional[int]]`**
- **`keys_with_prefix(prefix: Union[str, bytes], max_results: int = 100) -> List[Tuple[str, int]]`**
- **`fuzzy_search(query: str, max_edits: int = 1) -> List[Tuple[str, int, int]]`**

---

### `caesura.trie.AhoCorasick`

Multi-pattern automaton searching dictionaries in a single linear pass using failure links.

#### Methods
- **`add_pattern(pattern: str, pattern_id: int)`**
- **`compile()`**
- **`find_all(text: str) -> List[Tuple[int, int, int]]`**
- **`split_with_tokens(text: str) -> List[Tuple[str, Optional[int]]]`**

---

## 4. Input / Output and Serialization

### `caesura.io.BinarySerializer`

Native `.caesura` binary format serializer with CRC32 integrity verification and zero-copy `mmap` loading.

#### Methods
- **`save(tokenizer: Tokenizer, file_path: str)`**
- **`load(file_path: str) -> Tokenizer`**
- **`load_mmap(file_path: str) -> Tokenizer`**
