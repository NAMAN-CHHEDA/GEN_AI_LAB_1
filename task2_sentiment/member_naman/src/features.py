"""Feature helpers (pure functions): hashed bigrams and head+tail truncation. No model code here."""

import numpy as np
import torch

# Odd 64-bit mixing constants (they fit in a signed int64).
HASH_A = 0x2545F4914F6CDD1D
HASH_B = 0x1B873593CC9E2D51


def hash_bigram(prev_ids, cur_ids, num_buckets):
    """Map each (previous word id, current word id) pair to a bucket in [0, num_buckets).

    int64 tensors in, int64 tensor out. The same pair always gives the same bucket (no Python hash(),
    which changes between runs).

    Why not (prev * K + cur) % N? That is linear, so pairs that differ in a structured way collide
    in a structured way: for example (prev + 1, cur) and (prev, cur + K) always land in the same
    bucket. Here the two ids are multiplied by different large constants (int64 wraps around on
    overflow), and only the well-mixed HIGH bits of the result are kept, so such patterns disappear.
    """
    h = prev_ids * HASH_A + cur_ids * HASH_B
    h = (h >> 32) & 0x7FFFFFFF  # keep the upper 32 bits, made non-negative
    return h % num_buckets


def truncate_head_tail(ids, max_len, head_fraction):
    """Keep the first head_fraction of max_len ids and the rest from the end (numpy in, numpy out)."""
    if len(ids) <= max_len:
        return ids
    head = round(max_len * head_fraction)
    tail = max_len - head
    return np.concatenate([ids[:head], ids[len(ids) - tail:]])


def bag_ids_for_review(ids, vocab_size, num_buckets, bos_id):
    """Ids for one bag-of-n-grams review: the unigram ids, then one hashed-bigram id per position.

    The bigram at position t pairs ids[t-1] with ids[t] (bos_id before the first word). Bigram rows
    live after the vocabulary in the same embedding table: row = vocab_size + bucket.
    Output length is 2 * len(ids); values are in [0, vocab_size + num_buckets).
    """
    ids = np.asarray(ids, dtype=np.int64)
    prev = np.concatenate([[bos_id], ids[:-1]]).astype(np.int64)
    buckets = hash_bigram(torch.from_numpy(prev), torch.from_numpy(ids), num_buckets).numpy()
    return np.concatenate([ids, vocab_size + buckets])


def bigram_buckets_padded(ids, bos_id, num_buckets):
    """Bucket ids for a padded batch [B, L]. Bucket 0 is reserved for padding; real ones are 1..num_buckets."""
    first_prev = torch.full_like(ids[:, :1], bos_id)
    prev = torch.cat([first_prev, ids[:, :-1]], dim=1)
    buckets = 1 + hash_bigram(prev, ids, num_buckets - 1)
    return buckets.masked_fill(ids == 0, 0)


def bigram_rows_for_split(flat_ids, offsets, vocab_size, num_buckets, bos_id, chunk_tokens=5_000_000):
    """Bigram embedding rows for a whole split at once (same values as bag_ids_for_review, per review).

    flat_ids: int32 ids of all reviews joined; offsets: CSR offsets (len n+1). The previous token is the
    one before it inside the same review, and bos_id at the first token of every review.
    Done in chunks of tokens so memory stays bounded. Returns int32, same length as flat_ids.
    """
    total = len(flat_ids)
    starts = np.asarray(offsets[:-1])
    out = np.empty(total, dtype=np.int32)
    for begin in range(0, total, chunk_tokens):
        end = min(begin + chunk_tokens, total)
        cur = flat_ids[begin:end].astype(np.int64)
        prev = np.empty_like(cur)
        prev[1:] = cur[:-1]
        prev[0] = flat_ids[begin - 1] if begin > 0 else bos_id
        first = starts[(starts >= begin) & (starts < end)] - begin  # review starts inside this chunk
        prev[first] = bos_id
        buckets = hash_bigram(torch.from_numpy(prev), torch.from_numpy(cur), num_buckets).numpy()
        out[begin:end] = vocab_size + buckets
    return out
