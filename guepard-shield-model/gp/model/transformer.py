import math
import torch
import torch.nn as nn
import lightning.pytorch as pl

class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x is batch_first=True, so swap to [seq_len, batch_size, embedding_dim] for PE
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.transpose(0, 1) # back to [batch_size, seq_len, embedding_dim]

class SyscallTransformer(pl.LightningModule):
    causal_mask: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        lr: float = 1e-3,
        pad_idx: int = 0,
        max_len: int = 1024
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.lr = lr
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # Cache causal mask
        self.register_buffer("causal_mask", self._generate_square_subsequent_mask(max_len))

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        sz = src.size(1)
        mask = self.causal_mask[:sz, :sz]
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        output = self.decoder(output)
        return output

    def training_step(self, batch, batch_idx):
        x = batch
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        
        logits = self(input_seq)
        loss = self.criterion(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1))
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        
        logits = self(input_seq)
        loss = self.criterion(logits.reshape(-1, self.vocab_size), target_seq.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # Use CosineAnnealing as a simpler alternative to OneCycle if total steps unknown
        max_epochs = self.trainer.max_epochs if self.trainer and self.trainer.max_epochs else 30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }

    def compute_anomaly_score(self, batch: torch.Tensor, aggregation: str = "mean") -> torch.Tensor:
        """
        High-level interface for Anomaly Detection.
        Calculates Negative Log-Likelihood per window with configurable aggregation.

        Args:
            batch: [batch_size, seq_len] tensor of syscall IDs.
            aggregation: "mean", "max", or "p95" for how to aggregate token-level NLL.
        Returns:
            [batch_size] tensor of anomaly scores.
        """
        # input: x[:, :-1], target: x[:, 1:]
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]

        logits = self(input_seq)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Calculate NLL for targets
        target_nll = -torch.gather(log_probs, -1, target_seq.unsqueeze(-1)).squeeze(-1)

        pad_mask = (target_seq != self.pad_idx).float()

        if aggregation == "mean":
            window_nll = (target_nll * pad_mask).sum(dim=1) / (pad_mask.sum(dim=1) + 1e-8)
        elif aggregation == "max":
            # Ignore padding for max
            target_nll_masked = target_nll.masked_fill(pad_mask == 0, float("-inf"))
            window_nll = target_nll_masked.max(dim=1).values
            # Fallback for all-pad windows (should not happen)
            window_nll = torch.where(window_nll.isneginf(), torch.zeros_like(window_nll), window_nll)
        elif aggregation == "p95":
            # 95th percentile ignoring padding
            target_nll_masked = target_nll.masked_fill(pad_mask == 0, float("-inf"))
            sorted_nll, _ = torch.sort(target_nll_masked, dim=1, descending=False)
            valid_counts = pad_mask.sum(dim=1).long()
            idx = torch.clamp((valid_counts.float() * 0.95).ceil().long() - 1, min=0)
            window_nll = sorted_nll.gather(1, idx.unsqueeze(1)).squeeze(1)
            window_nll = torch.where(window_nll.isneginf(), torch.zeros_like(window_nll), window_nll)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Choose from mean, max, p95.")

        return window_nll
