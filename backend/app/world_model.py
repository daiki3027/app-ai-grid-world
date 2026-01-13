from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory_encoder import HistoryEncoder

DONE_LOSS_WEIGHT = 3.0  # trap=done を最優先で覚えさせるため重みを上げる


@dataclass
class WMTransition:
    obs_seq: List[List[List[float]]]
    action_seq: List[int]
    reward_seq: List[float]
    action: int
    next_patch: List[List[float]]
    reward: float
    done: bool


class WMReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[WMTransition] = deque(maxlen=capacity)

    def add(self, transition: WMTransition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[WMTransition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def prepare_history_tensors(
    obs_seq: Sequence[Sequence[Sequence[float]]],
    action_seq: Sequence[int],
    reward_seq: Sequence[float],
    seq_len: int,
    action_pad_idx: int,
    normalize_factor: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    truncated_obs = list(obs_seq)[-seq_len:]
    truncated_actions = list(action_seq)[-seq_len:]
    truncated_rewards = list(reward_seq)[-seq_len:]
    pad_count = seq_len - len(truncated_obs)

    length = torch.tensor([len(truncated_obs)], dtype=torch.long, device=device)

    obs_list = truncated_obs + [[[1 for _ in range(5)] for _ in range(5)] for _ in range(pad_count)]
    obs_tensor = torch.tensor(obs_list, dtype=torch.float32, device=device) / normalize_factor

    action_list = truncated_actions + [action_pad_idx for _ in range(pad_count)]
    action_tensor = torch.tensor(action_list, dtype=torch.long, device=device)

    reward_list = truncated_rewards + [0.0 for _ in range(pad_count)]
    reward_tensor = torch.tensor(reward_list, dtype=torch.float32, device=device)
    return obs_tensor.unsqueeze(0), action_tensor.unsqueeze(0), reward_tensor.unsqueeze(0), length


class WorldModel(nn.Module):
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
        self.encoder = HistoryEncoder(
            action_dim=action_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            action_pad_idx=action_pad_idx,
        )
        self.action_embed = nn.Embedding(action_dim, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )
        self.next_patch_head = nn.Linear(d_model, 25)
        self.reward_head = nn.Linear(d_model, 1)
        self.done_head = nn.Linear(d_model, 1)

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor,
        lengths: torch.Tensor,
        candidate_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_seq: (B, L, 5, 5)
            action_seq: (B, L)
            reward_seq: (B, L)
            lengths: (B,)
            candidate_actions: (B,)
        Returns:
            next_patch: (B, 25) normalized patch prediction
            reward: (B,) predicted reward
            done_logit: (B,) predicted done logit
        """
        memory = self.encoder.encode(obs_seq, action_seq, reward_seq, lengths)
        action_emb = self.action_embed(candidate_actions)
        hidden = self.fuse(torch.cat([memory, action_emb], dim=-1))
        next_patch = self.next_patch_head(hidden)
        reward = self.reward_head(hidden).squeeze(-1)
        done_logit = self.done_head(hidden).squeeze(-1)
        return next_patch, reward, done_logit


class WorldModelTrainer:
    def __init__(
        self,
        actions: Sequence[int],
        seq_len: int,
        device: torch.device,
        buffer_size: int = 8000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        train_every: int = 4,
        min_train_size: int = 200,
        normalize_factor: float = 8.0,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        lambda_patch: float = 1.0,
        lambda_reward: float = 1.0,
        lambda_done: float = DONE_LOSS_WEIGHT,
    ):
        self.actions = list(actions)
        self.seq_len = seq_len
        self.device = device
        self.batch_size = batch_size
        self.train_every = train_every
        self.min_train_size = min_train_size
        self.normalize_factor = normalize_factor
        self.action_pad_idx = len(self.actions)
        self.buffer = WMReplayBuffer(buffer_size)
        self.model = WorldModel(
            action_dim=len(self.actions),
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            action_pad_idx=self.action_pad_idx,
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lambda_patch = lambda_patch
        self.lambda_reward = lambda_reward
        self.lambda_done = lambda_done
        self.loss_window: deque[float] = deque(maxlen=200)
        self.last_loss: float | None = None
        self.train_step_count = 0

    def add_transition(
        self,
        obs_seq: List[List[List[float]]],
        action_seq: List[int],
        reward_seq: List[float],
        action: int,
        next_patch: List[List[float]],
        reward: float,
        done: bool,
    ) -> None:
        transition = WMTransition(
            obs_seq=list(obs_seq),
            action_seq=list(action_seq),
            reward_seq=list(reward_seq),
            action=action,
            next_patch=list(next_patch),
            reward=reward,
            done=done,
        )
        self.buffer.add(transition)

    def _prep_batch(self, transitions: List[WMTransition]) -> Tuple[torch.Tensor, ...]:
        obs_tensors = []
        action_tensors = []
        reward_tensors = []
        lengths = []
        candidate_actions = []
        target_patches = []
        target_rewards = []
        target_dones = []

        for t in transitions:
            obs, act, rew, length = prepare_history_tensors(
                t.obs_seq,
                t.action_seq,
                t.reward_seq,
                self.seq_len,
                self.action_pad_idx,
                self.normalize_factor,
                self.device,
            )
            obs_tensors.append(obs)
            action_tensors.append(act)
            reward_tensors.append(rew)
            lengths.append(length)
            candidate_actions.append(torch.tensor([t.action], dtype=torch.long, device=self.device))
            target_patches.append(
                torch.tensor(t.next_patch, dtype=torch.float32, device=self.device).view(1, -1)
                / self.normalize_factor
            )
            target_rewards.append(torch.tensor([t.reward], dtype=torch.float32, device=self.device))
            target_dones.append(torch.tensor([float(t.done)], dtype=torch.float32, device=self.device))

        batch_obs = torch.cat(obs_tensors, dim=0)
        batch_actions = torch.cat(action_tensors, dim=0)
        batch_rewards = torch.cat(reward_tensors, dim=0)
        batch_lengths = torch.cat(lengths, dim=0)
        batch_candidate_actions = torch.cat(candidate_actions, dim=0)
        batch_target_patches = torch.cat(target_patches, dim=0)
        batch_target_rewards = torch.cat(target_rewards, dim=0)
        batch_target_dones = torch.cat(target_dones, dim=0)
        return (
            batch_obs,
            batch_actions,
            batch_rewards,
            batch_lengths,
            batch_candidate_actions,
            batch_target_patches,
            batch_target_rewards,
            batch_target_dones,
        )

    def train_step(self) -> float | None:
        if len(self.buffer) < self.min_train_size:
            return None
        self.train_step_count += 1
        if self.train_step_count % self.train_every != 0:
            return None
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        (
            obs,
            action_seq,
            reward_seq,
            lengths,
            candidate_actions,
            target_patches,
            target_rewards,
            target_dones,
        ) = self._prep_batch(transitions)

        pred_patch, pred_reward, pred_done = self.model(obs, action_seq, reward_seq, lengths, candidate_actions)
        loss_patch = F.mse_loss(pred_patch, target_patches)
        loss_reward = F.smooth_l1_loss(pred_reward, target_rewards)
        loss_done = F.binary_cross_entropy_with_logits(pred_done, target_dones)
        loss = self.lambda_patch * loss_patch + self.lambda_reward * loss_reward + self.lambda_done * loss_done

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        self.last_loss = loss_value
        self.loss_window.append(loss_value)
        return loss_value

    @property
    def avg_recent_loss(self) -> float | None:
        if not self.loss_window:
            return None
        return float(sum(self.loss_window) / len(self.loss_window))
