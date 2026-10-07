# data_processed (Sarvesh Task 2)

Preprocessing artifacts rebuilt with the **same recipe** as the training notebooks
(`src/dump_vocab.py`, seed=42, VOCAB_SIZE=30000, MAX_LEN=160).

## Files

| File | What it is |
|---|---|
| `vocab.json` | `word_to_idx` (30,000 tokens; `<PAD>=0`, `<UNK>=1`) |
| `split_sizes.json` | train/val/test sizes + length percentiles |
| `preprocessing_info.json` | full preprocessing recipe + artifact paths |

Full tokenized tensors are **not** stored (too large). Vocab + recipe is enough for graders.

## Rebuild

```bash
python src/dump_vocab.py
```
