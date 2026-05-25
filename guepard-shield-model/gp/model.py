"""SyscallTransformer — pre-norm decoder-only Transformer (LLaMA-style).

Architecture:
  Embedding → 4× [RMSNorm → CausalSelfAttn(RoPE) → residual
                  RMSNorm → FFN(GELU)             → residual]
  → RMSNorm → Linear(d_model, vocab_size)

Phase 3 hook: encode(x) returns the last-token hidden state [B, d_model].
Called with stride-1 dense windows (one window per syscall position) so that
every h_t has a full W-token context — used for K-Means state clustering.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


# ── RoPE ─────────────────────────────────────────────────────────────────────

def _rope_freqs(d_head: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Return cos/sin rotation tensors of shape [seq_len, d_head]."""
    theta = 1.0 / (10000 ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)          # [seq_len, d_head//2]
    freqs = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_head]
    return freqs


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: [B, n_heads, seq_len, d_head]  freqs: [seq_len, d_head]."""
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# ── building blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_heads, self.d_head

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]

        Q = _apply_rope(split_heads(self.q(x)), freqs)
        K = _apply_rope(split_heads(self.k(x)), freqs)
        V = split_heads(self.v(x))

        # Flash attention with causal mask (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs)
        x = x + self.ffn(self.norm2(x))
        return x


# ── main model ────────────────────────────────────────────────────────────────

class SyscallTransformer(LightningModule):
    """Decoder-only Transformer trained on next-token prediction (normal syscalls).

    At inference, anomaly score = -log P(s_W | s_1…s_{W-1}): last-token NLL.
    """

    def __init__(
        self,
        vocab_size: int = 102,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        window_size: int = 64,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.05,
        max_steps: int = 10_000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.drop    = nn.Dropout(dropout)
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = RMSNorm(d_model)
        self.head     = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying: embedding ↔ lm head
        self.head.weight = self.embed.weight

        d_head = d_model // n_heads
        self.register_buffer(
            "_rope_freqs",
            _rope_freqs(d_head, window_size, device=torch.device("cpu")),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _get_freqs(self, seq_len: int) -> torch.Tensor:
        return self._rope_freqs[:seq_len].to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] int64 → logits [B, T, vocab_size]."""
        h = self.drop(self.embed(x))
        freqs = self._get_freqs(x.shape[1])
        for block in self.blocks:
            h = block(h, freqs)
        h = self.norm_out(h)
        return self.head(h)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return last-token hidden state [B, d_model].

        For Phase 3: call with stride-1 dense windows so each h has full W context.
        """
        h = self.drop(self.embed(x))
        freqs = self._get_freqs(x.shape[1])
        for block in self.blocks:
            h = block(h, freqs)
        h = self.norm_out(h)
        return h[:, -1, :]  # [B, d_model]

    # ── Lightning methods ─────────────────────────────────────────────────────

    def _loss(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        logits = self(x)                                    # [B, T, V]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, self.hparams.vocab_size),
            x[:, 1:].reshape(-1),
            ignore_index=0,
        )
        return loss

    def training_step(self, batch, _):
        loss = self._loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, _):
        """Return (last_nll, max_nll, labels) on CPU for a batch."""
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            x = batch
            y = torch.zeros(x.shape[0], dtype=torch.long)

        logits = self(x)                             # [B, T, V]
        B, T, V = logits.shape
        nll_all = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            x[:, 1:].reshape(-1),
            reduction="none",
        ).view(B, T - 1)                             # [B, T-1]

        # Use the last *real* (non-PAD) token position.
        # For full windows (no padding) this is always T-1, same as nll_all[:, -1].
        # For short recordings padded with 0, this avoids scoring a PAD token.
        seq_lens = (x != 0).sum(dim=1).clamp(min=1)  # [B]
        score_idx = (seq_lens - 2).clamp(min=0)       # nll_all[b, i] predicts x[b, i+1]

        last_nll = nll_all[torch.arange(B, device=x.device), score_idx]

        return (
            last_nll.cpu(),                          # last real-token NLL
            nll_all.max(dim=1).values.cpu(),         # max-window NLL
            y.cpu(),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup_steps = int(self.hparams.max_steps * self.hparams.warmup_ratio)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, self.hparams.max_steps - warmup_steps)
            min_lr_ratio = 1e-5 / self.hparams.lr
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
