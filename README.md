# BPE Tokenizer from Scratch

Train a simple **Byte-Pair Encoding** tokenizer on English sentences using HuggingFace's `tokenizers` library.

BPE is the subword tokenization algorithm behind GPT-2 and friends — it iteratively merges the most frequent token pairs until a target vocabulary size is reached.

## Usage

**1. Build the corpus** — downloads the dataset and writes one sentence per line to `corpus.txt`:

```bash
python download_dataset.py
python prepare_dataset.py
```

**2. Train the tokenizer** — trains a 16k-vocab BPE tokenizer with ByteLevel pre-tokenization and saves `tokenizer.json`:

```bash
python train_tokenizer.py
```

**3. Encode text:**

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
print(tokenizer.encode("Hello world").tokens)
# ['Hello', 'Ġworld']
```

The `Ġ` prefix represents a leading space — that's how ByteLevel pre-tokenization preserves whitespace info.

## Credits

Training data: [agentlans/high-quality-english-sentences](https://huggingface.co/datasets/agentlans/high-quality-english-sentences)