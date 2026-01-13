from __future__ import annotations

import random
from collections import deque
from typing import Callable, Deque, List, Sequence, Tuple

import torch

from .world_model import WorldModelTrainer, prepare_history_tensors

# すべて調整可能な評価定数
GOAL_BONUS = 4.0  # goal は強く優遇する（曖昧判定を避けるため、確定情報を加点）
TRAP_PENALTY = -4.0  # trap は強く回避
WALL_PENALTY = -0.75  # 壁方向を選び続けないよう即時ペナルティ
DONE_PENALTY = -3.0  # 不明な done は安全側で強めの罰
DONE_THRESHOLD = 0.8  # WMのdone予測がこの確率を超えたら終端扱い

class MPCPlanner:
    def __init__(
        self,
        world_model_trainer: WorldModelTrainer,
        actions: Sequence[int],
        seq_len: int,
        action_pad_idx: int,
        normalize_factor: float = 8.0,
        horizon: int = 5,
        num_samples: int = 16,
        gamma: float = 0.95,
        min_wm_samples: int = 300,
        done_threshold: float = DONE_THRESHOLD,
        first_step_top_k: int = 1,
        epsilon_mpc: float = 0.0,
        softmax_temp: float = 0.8,
        done_penalty: float = DONE_PENALTY,
        get_q_values: Callable[[List[List[List[float]]], List[int], List[float]], List[float]] | None = None,
        seed: int | None = 42,
    ):
        self.wm_trainer = world_model_trainer
        self.actions = list(actions)
        self.seq_len = seq_len
        self.action_pad_idx = action_pad_idx
        self.normalize_factor = normalize_factor
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.min_wm_samples = min_wm_samples
        self.done_threshold = done_threshold
        self.first_step_top_k = max(1, first_step_top_k)
        self.epsilon_mpc = epsilon_mpc
        self.softmax_temp = softmax_temp
        self.done_penalty = done_penalty
        self.get_q_values = get_q_values
        self.random = random.Random(seed)

    def is_ready(self) -> bool:
        return len(self.wm_trainer.buffer) >= self.min_wm_samples and self.wm_trainer.last_loss is not None

    def plan(
        self,
        obs_history: Deque[List[List[float]]],
        action_history: Deque[int],
        reward_history: Deque[float],
        current_patch: List[List[int]] | None,
        fallback_policy,
    ) -> Tuple[int, None]:
        """
        Returns:
            action, q_values_like (None for MPC)
        """
        if not self.is_ready() or self.get_q_values is None:
            return fallback_policy()

        device = self.wm_trainer.device
        best_return = float("-inf")
        best_action = self.actions[0] if self.actions else 0

        base_q_values = self.get_q_values(list(obs_history), list(action_history), list(reward_history))

        for _ in range(self.num_samples):
            predicted_return, first_action = self._rollout_return(
                obs_history,
                action_history,
                reward_history,
                base_q_values,
                current_patch,
                device,
            )
            if predicted_return > best_return:
                best_return = predicted_return
                best_action = first_action

        return best_action, None

    def _rollout_return(
        self,
        obs_history: Deque[List[List[float]]],
        action_history: Deque[int],
        reward_history: Deque[float],
        base_q_values: List[float],
        current_patch: List[List[int]] | None,
        device: torch.device,
    ) -> Tuple[float, int]:
        obs_hist = deque(obs_history, maxlen=self.seq_len)
        act_hist = deque(action_history, maxlen=self.seq_len)
        rew_hist = deque(reward_history, maxlen=self.seq_len)

        total = 0.0
        discount = 1.0
        model = self.wm_trainer.model
        model.eval()

        with torch.no_grad():
            # first action: Transformer記憶から得た現在のQに基づくソフトマックスサンプリング
            first_action = self._sample_action_from_q(base_q_values)

            for step in range(self.horizon):
                if step > 0:
                    q_values = self.get_q_values(list(obs_hist), list(act_hist), list(rew_hist))
                    action = self._sample_action_from_q(q_values)
                else:
                    action = first_action

                # 1手目は確定情報（現在の5x5パッチ）から安全性を判定する
                if step == 0 and current_patch is not None:
                    tile = self._tile_in_direction(current_patch, action)
                    if tile == 1:
                        total += WALL_PENALTY
                        # 壁ヒットは位置が変わらない想定なので同じパッチを履歴に入れて続行
                        obs_hist.append(current_patch)
                        act_hist.append(action)
                        rew_hist.append(WALL_PENALTY)
                        discount *= self.gamma
                        continue
                    if tile == 3:
                        total += GOAL_BONUS  # goal は即時ボーナスを付与して終了
                        return total, first_action
                    if tile == 4:
                        total += TRAP_PENALTY  # trap は即時ペナルティで終了
                        return total, first_action

                obs_tensor, act_tensor, rew_tensor, length = prepare_history_tensors(
                    obs_hist,
                    act_hist,
                    rew_hist,
                    self.seq_len,
                    self.action_pad_idx,
                    self.normalize_factor,
                    device,
                )
                pred_patch, pred_reward, pred_done = model(
                    obs_tensor, act_tensor, rew_tensor, length, torch.tensor([action], device=device)
                )
                step_reward = float(pred_reward.squeeze().cpu().item())
                total += discount * step_reward
                discount *= self.gamma

                done_prob = float(torch.sigmoid(pred_done).cpu().item())
                if done_prob > self.done_threshold:
                    # goal/trapを混線させず、未知の終端は安全側（ペナルティ）に倒す
                    total += self.done_penalty
                    break

                patch = (pred_patch.view(5, 5) * self.normalize_factor).detach().cpu().tolist()
                obs_hist.append(patch)
                act_hist.append(action)
                rew_hist.append(step_reward)
        return total, first_action

    def _sample_action_from_q(self, q_values: List[float]) -> int:
        if self.random.random() < self.epsilon_mpc:
            return self.random.choice(self.actions)
        q_tensor = torch.tensor(q_values, dtype=torch.float32)
        probs = torch.softmax(q_tensor / max(1e-6, self.softmax_temp), dim=0).cpu().numpy().tolist()
        return self.random.choices(self.actions, weights=probs, k=1)[0]

    @staticmethod
    def _tile_in_direction(patch: List[List[int]], action: int) -> int | None:
        """Return tile value for 1-step in the given direction based on current 5x5 patch."""
        if len(patch) != 5 or len(patch[0]) != 5:
            return None
        center = (2, 2)
        offsets = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dx, dy = offsets.get(action, (0, 0))
        x, y = center[0] + dx, center[1] + dy
        if 0 <= x < 5 and 0 <= y < 5:
            return patch[y][x]
        return None
