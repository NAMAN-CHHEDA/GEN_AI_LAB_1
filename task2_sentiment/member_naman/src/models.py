"""The three models trained from scratch (no pretrained embeddings or language models).

Every model returns one logit per review (shape [B]); the sigmoid is applied inside the loss.
input_kind says what the dataloader must give the model:
  "bag"    -> forward(flat_ids, offsets)   variable-length bags, no padding
  "padded" -> forward(ids, lengths)        ids is [B, L] with PAD id 0
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import bigram_buckets_padded
from utils import count_parameters

PAD_ID, UNK_ID, BOS_ID = 0, 1, 2


def masked_mean(x, valid):
    """Mean over time of x [B, L, D], counting only valid positions (valid is a [B, L] bool)."""
    valid = valid.unsqueeze(-1)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def masked_max(x, valid):
    """Max over time of x [B, L, D] over valid positions. finfo.min (not -inf) is safe under autocast."""
    return x.masked_fill(~valid.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values


class NGramBag(nn.Module):
    """fastText-style classifier: the mean of unigram and hashed-bigram embeddings, then one linear layer.

    Each review is a variable-length bag of ids, so nothing is padded or truncated. It is a very
    fast, strong baseline; hashing the bigrams into buckets keeps the table size fixed.
    """

    input_kind = "bag"

    def __init__(self, vocab_size, cfg_model):
        super().__init__()
        self.bag = nn.EmbeddingBag(vocab_size + cfg_model["bigram_buckets"], cfg_model["embed_dim"], mode="mean")
        self.dropout = nn.Dropout(cfg_model["dropout"])
        self.out = nn.Linear(cfg_model["embed_dim"], 1)
        nn.init.normal_(self.bag.weight, mean=0.0, std=0.1)

    def forward(self, flat_ids, offsets):
        x = self.dropout(self.bag(flat_ids, offsets))
        return self.out(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    """A small Transformer encoder (pre-norm) trained from scratch, with mean+max pooling on top.

    Learned position embeddings give word order; word dropout (random tokens -> UNK) during training
    regularises it; the padding mask stops attention and pooling from seeing PAD positions.
    """

    input_kind = "padded"

    def __init__(self, vocab_size, cfg_model, max_len):
        super().__init__()
        d_model = cfg_model["d_model"]
        self.max_len = max_len
        self.token_dropout = cfg_model["token_dropout"]
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.embed_dropout = nn.Dropout(cfg_model["dropout"])
        layer = nn.TransformerEncoderLayer(d_model, cfg_model["heads"], cfg_model["ffn"], cfg_model["dropout"],
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg_model["layers"], enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(0.3)
        self.out = nn.Linear(2 * d_model, 1)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[PAD_ID].zero_()  # keep the PAD vector at zero after re-initialising

    def forward(self, ids, lengths=None):  # lengths is not needed: the mask is built from the PAD ids
        if ids.size(1) > self.max_len:
            raise ValueError(f"batch length {ids.size(1)} is longer than max_len={self.max_len}; truncate first")
        pad_mask = ids == PAD_ID
        if self.training and self.token_dropout > 0:  # word dropout: replace some real tokens by UNK
            drop = (torch.rand(ids.shape, device=ids.device) < self.token_dropout) & ~pad_mask
            ids = ids.masked_fill(drop, UNK_ID)
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.embed_dropout(self.embed(ids) + self.pos_embed(positions))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.final_norm(x)
        valid = ~pad_mask
        pooled = torch.cat([masked_mean(x, valid), masked_max(x, valid)], dim=-1)
        return self.out(self.head_dropout(pooled)).squeeze(-1)


class BiGRUClassifier(nn.Module):
    """Bidirectional GRU over word + hashed-bigram embeddings, pooled by max, mean and attention.

    The GRU reads in both directions and the bigram embedding gives it phrase information up front.
    Three poolings are concatenated: max (strongest cue), mean (overall tone) and additive attention
    (a learned weighted average, whose weights can be inspected later).
    """

    input_kind = "padded"

    def __init__(self, vocab_size, cfg_model):
        super().__init__()
        hidden = cfg_model["hidden"]
        self.bigram_buckets = cfg_model["bigram_buckets"]
        self.word_embed = nn.Embedding(vocab_size, cfg_model["word_dim"], padding_idx=PAD_ID)
        self.bigram_embed = nn.Embedding(self.bigram_buckets + 1, cfg_model["bigram_dim"], padding_idx=0)
        self.emb_dropout = nn.Dropout(cfg_model["emb_dropout"])
        self.gru = nn.GRU(cfg_model["word_dim"] + cfg_model["bigram_dim"], hidden, num_layers=cfg_model["layers"],
                          batch_first=True, bidirectional=True, dropout=cfg_model["rnn_dropout"])
        self.attn_proj = nn.Linear(2 * hidden, hidden)  # W
        self.attn_vector = nn.Linear(hidden, 1, bias=False)  # v
        self.head_dropout = nn.Dropout(cfg_model["head_dropout"])
        self.out = nn.Linear(6 * hidden, 1)
        for table in (self.word_embed, self.bigram_embed):
            nn.init.normal_(table.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.word_embed.weight[PAD_ID].zero_()
            self.bigram_embed.weight[0].zero_()

    def forward(self, ids, lengths, return_attention=False):
        bigram_ids = bigram_buckets_padded(ids, BOS_ID, self.bigram_buckets)
        x = torch.cat([self.word_embed(ids), self.bigram_embed(bigram_ids)], dim=-1)
        x = self.emb_dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=ids.size(1))  # [B, L, 2*hidden]

        valid = ids != PAD_ID
        scores = self.attn_vector(torch.tanh(self.attn_proj(h))).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=1)  # sums to 1 over the real positions, 0 on PAD
        context = (attn.unsqueeze(-1) * h).sum(dim=1)

        pooled = torch.cat([masked_max(h, valid), masked_mean(h, valid), context], dim=-1)
        logits = self.out(self.head_dropout(pooled)).squeeze(-1)
        return (logits, attn) if return_attention else logits


def build_model(name, cfg, vocab_size):
    """Create a model by name from the config."""
    if name == "ngram_bag":
        return NGramBag(vocab_size, cfg["models"]["ngram_bag"])
    if name == "transformer":
        return TransformerClassifier(vocab_size, cfg["models"]["transformer"], cfg["data"]["max_len"])
    if name == "bigru":
        return BiGRUClassifier(vocab_size, cfg["models"]["bigru"])
    raise ValueError(f"unknown model '{name}'; choose from ngram_bag, transformer, bigru")


def parameter_breakdown(model):
    """(total, embedding parameters, everything else) for printing."""
    total = count_parameters(model)
    embeddings = sum(p.numel() for m in model.modules() if isinstance(m, (nn.Embedding, nn.EmbeddingBag))
                     for p in m.parameters())
    return total, embeddings, total - embeddings


# What each model expects from the dataloader (same as the input_kind class attributes).
INPUT_KINDS = {"ngram_bag": NGramBag.input_kind, "transformer": TransformerClassifier.input_kind,
               "bigru": BiGRUClassifier.input_kind}
