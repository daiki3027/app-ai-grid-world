from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """Small CNN encoder for 5x5 patches."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 5 * 5, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)


class HistoryEncoder(nn.Module):
    """
    Shared encoder that turns (obs/action/reward) history into a memory embedding.
    Returns the last valid token after Transformer encoding.
    """

    def __init__(
        self,
        action_dim: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_pad_idx: int = 4,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.action_pad_idx = action_pad_idx
        self.patch_encoder = PatchEncoder(d_model)
        self.action_embed = nn.Embedding(action_dim + 1, d_model, padding_idx=action_pad_idx)
        self.reward_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(max_len=seq_len, d_model=d_model)

    def encode(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs_seq: (B, L, 5, 5)
            action_seq: (B, L)
            reward_seq: (B, L)
            lengths: (B,)
        Returns:
            memory: (B, d_model) last valid token embedding
        """
        bsz, seq_len, _, _ = obs_seq.shape
        x = obs_seq.view(bsz * seq_len, 1, 5, 5)
        patch_emb = self.patch_encoder(x).view(bsz, seq_len, self.d_model)
        action_emb = self.action_embed(action_seq)
        reward_emb = self.reward_mlp(reward_seq.unsqueeze(-1))
        tokens = patch_emb + action_emb + reward_emb

        positions = torch.arange(seq_len, device=obs_seq.device).unsqueeze(0).expand(bsz, seq_len)
        tokens = tokens + self.positional(positions)

        mask = torch.arange(seq_len, device=obs_seq.device).expand(bsz, seq_len) >= lengths.unsqueeze(1)
        encoded = self.transformer(tokens, src_key_padding_mask=mask)

        idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2).expand(bsz, 1, self.d_model)
        return encoded.gather(1, idx).squeeze(1)
