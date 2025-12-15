from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

State = Tuple[int, int]


class QLearningAgent:
    def __init__(
        self,
        actions: List[int],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.q_table: Dict[State, List[float]] = defaultdict(lambda: [0.0 for _ in actions])

    def reset(self) -> None:
        self.q_table.clear()
        self.epsilon = 1.0

    def choose_action(self, state: State) -> int:
        if self.random.random() < self.epsilon:
            return self.random.choice(self.actions)
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return self.random.choice(best_actions)

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        target = reward + (0 if done else self.gamma * max_next_q)
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

