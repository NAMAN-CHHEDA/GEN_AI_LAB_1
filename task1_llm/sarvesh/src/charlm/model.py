"""
From-scratch decoder-only Transformer for character LM.

Design notes (intentionally distinct from a fused-QKV / GELU / 128-d baseline):
  - Separate Linear projections for Q, K, and V (not one fused 3d matrix)
  - Causal mask built with torch.triu (upper triangle = future = blocked)
  - Classic Transformer FFN: Linear -> ReLU -> Dropout -> Linear
  - Optional embedding scaling by sqrt(d_model) (Vaswani et al.)
  - Optional weight tying between token embedding and LM head
  - Pre-LayerNorm residual blocks + final LayerNorm before the LM head

Forbidden APIs are not used: no nn.Transformer*, no nn.MultiheadAttention,
no F.scaled_dot_product_attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    """Multi-head self-attention with an explicit causal mask."""

    def __init__(self, d_model: int, n_heads: int, block_size: int, attn_dropout: float, resid_dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        # Separate projections — pedagogical clarity and a deliberate structural difference.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

        # Upper-triangular ones mark FUTURE positions that must be blocked.
        future = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer("future_mask", future.view(1, 1, block_size, block_size), persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        # (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.future_mask[:, :, :T, :T], float("-inf"))
        # Softmax in float32 for fp16 stability
        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        weights = self.attn_drop(weights)

        ctx = torch.matmul(weights, v)  # (B, H, T, d_head)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.W_o(ctx))


class PositionwiseFFN(nn.Module):
    """Original Transformer feed-forward: d_model -> d_ff -> ReLU -> d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str = "relu"):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff)
        self.fc_out = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        act = activation.lower()
        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc_out(self.act(self.fc_in(x))))


class PreNormDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        block_size: int,
        dropout: float,
        attn_dropout: float,
        activation: str,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, block_size, attn_dropout, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class CharGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 192,
        n_layers: int = 5,
        n_heads: int = 6,
        d_ff: int = 768,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        activation: str = "relu",
        tie_embeddings: bool = True,
        emb_scale: bool = True,
        init_std: float = 0.02,
        scale_residual_proj: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.init_std = init_std
        self.emb_scale = emb_scale
        self.scale_factor = math.sqrt(d_model) if emb_scale else 1.0

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = nn.Embedding(block_size, d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                PreNormDecoderBlock(
                    d_model, n_heads, d_ff, block_size, dropout, attn_dropout, activation
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        if scale_residual_proj:
            resid_std = init_std / math.sqrt(2 * n_layers)
            for name, param in self.named_parameters():
                if name.endswith("W_o.weight") or name.endswith("fc_out.weight"):
                    nn.init.normal_(param, mean=0.0, std=resid_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"T={T} exceeds block_size={self.block_size}")

        positions = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) * self.scale_factor + self.position_embed(positions)
        x = self.embed_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            cond = idx[:, -self.block_size :]
            logits, _ = self(cond)
            logits = logits[:, -1, :].float()

            if greedy:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k).values[:, -1, None]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    mask = cumulative > top_p
                    # Keep at least the first token
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat([idx, next_id], dim=1)
        self.train(was_training)
        return idx


def build_model(cfg: dict, vocab_size: int) -> CharGPT:
    m = cfg["model"]
    return CharGPT(
        vocab_size=vocab_size,
        block_size=cfg["data"]["block_size"],
        d_model=m["d_model"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        d_ff=m["d_ff"],
        dropout=m["dropout"],
        attn_dropout=m.get("attn_dropout", m["dropout"]),
        activation=m.get("activation", "relu"),
        tie_embeddings=m.get("tie_embeddings", True),
        emb_scale=m.get("emb_scale", True),
        init_std=m.get("init_std", 0.02),
        scale_residual_proj=m.get("scale_residual_proj", True),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
