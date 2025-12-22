from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvQNetwork(nn.Module):
    def __init__(self, input_channels: int, action_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class Transition:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        actions: Sequence[int],
        seed: int | None = 42,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 5000,
        batch_size: int = 64,
        learning_start: int = 100,
        train_every: int = 4,
        target_sync_interval: int = 100,
        huber_loss: bool = True,
        normalize_factor: float = 8.0,
        device: str = "cpu",
    ):
        self.seed = seed
        _set_global_seeds(seed)
        self.device = torch.device(device)
        self.actions = list(actions)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learning_start = learning_start
        self.train_every = train_every
        self.target_sync_interval = target_sync_interval
        self.huber_loss = huber_loss
        self.normalize_factor = normalize_factor
        self.random = random.Random(seed)
        self.step_count = 0
        self.buffer_size = buffer_size

        self.replay_buffer = ReplayBuffer(buffer_size)
        self._init_networks(lr)

    def _init_networks(self, lr: float) -> None:
        self.policy_net = ConvQNetwork(1, len(self.actions)).to(self.device)
        self.target_net = ConvQNetwork(1, len(self.actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def reset(self) -> None:
        _set_global_seeds(self.seed)
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.epsilon = 1.0
        self.step_count = 0
        self._init_networks(self.optimizer.param_groups[0]["lr"])

    def _to_tensor(self, patch: List[List[int]]) -> torch.Tensor:
        arr = torch.tensor(patch, dtype=torch.float32, device=self.device)
        arr = arr / self.normalize_factor
        return arr.unsqueeze(0).unsqueeze(0)

    def choose_action(self, observation_patch: List[List[int]]) -> Tuple[int, List[float]]:
        state_tensor = self._to_tensor(observation_patch)
        q_values = self.policy_net(state_tensor).detach().cpu().tolist()[0]
        if self.random.random() < self.epsilon:
            action = self.random.choice(self.actions)
        else:
            max_q = max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            action = self.random.choice(best_actions)
        return action, q_values

    def learn(
        self,
        observation_patch: List[List[int]],
        action: int,
        reward: float,
        next_observation_patch: List[List[int]],
        done: bool,
    ) -> None:
        state_tensor = self._to_tensor(observation_patch)
        next_state_tensor = self._to_tensor(next_observation_patch)
        self.replay_buffer.add(Transition(state_tensor, action, reward, next_state_tensor, done))
        self.step_count += 1

        if len(self.replay_buffer) < self.learning_start:
            return
        if self.step_count % self.train_every != 0:
            return
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        batch_state = torch.cat([t.state for t in transitions], dim=0)
        batch_next_state = torch.cat([t.next_state for t in transitions], dim=0)
        batch_action = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        batch_reward = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        batch_done = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device)

        current_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(batch_next_state).max(1)[0]
            target_q = batch_reward + (1 - batch_done) * self.gamma * next_q

        if self.huber_loss:
            loss = F.smooth_l1_loss(current_q, target_q)
        else:
            loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()

        if self.step_count % self.target_sync_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
