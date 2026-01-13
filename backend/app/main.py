from __future__ import annotations

import asyncio
import contextlib
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gridworld import GridWorld
from .dqn import DQNAgent
from .mpc import MPCPlanner
from .world_model import WorldModelTrainer


class TrainingSession:
    def __init__(self, seed: int = 42, maze_mode: str = "random", reward_shaping: bool = True):
        self.random = random.Random(seed)
        self.maze_mode = maze_mode  # "random", "fixed", "memory"
        self.reward_shaping = reward_shaping
        self.env = self._make_env()
        self.agent = DQNAgent(actions=[0, 1, 2, 3], seed=seed)
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.learning_enabled = True
        self.speed = 1.0
        self.max_steps = 200
        self.success_window = deque(maxlen=50)
        self._stop = False
        self.last_action = None
        self.last_q_values: List[float] | None = None
        self.last_planner_used: str | None = None
        self.distance_reward_weight = 0.02
        self.seq_len = self.agent.seq_len
        self.obs_history: Deque[List[List[int]]] = deque(maxlen=self.seq_len)
        self.action_history: Deque[int] = deque(maxlen=self.seq_len)
        self.reward_history: Deque[float] = deque(maxlen=self.seq_len)
        self.last_loss: float | None = None
        self.loss_window: Deque[float] = deque(maxlen=200)
        self.episode_rewards: Deque[float] = deque(maxlen=50)
        self.planner_mode = "dqn"  # "dqn" or "mpc"
        self.wm_trainer = self._init_world_model_trainer()
        self.mpc_planner = self._init_mpc_planner()

    def _make_env(self) -> GridWorld:
        if self.maze_mode == "memory":
            return GridWorld.memory_challenge()
        if self.maze_mode == "random":
            return GridWorld.random(seed=self.random.randint(0, 1_000_000))
        return GridWorld.default()

    def _init_world_model_trainer(self) -> WorldModelTrainer:
        return WorldModelTrainer(
            actions=self.agent.actions,
            seq_len=self.seq_len,
            device=self.agent.device,
            buffer_size=self.agent.buffer_size,
            batch_size=self.agent.batch_size,
            learning_rate=self.agent.optimizer.param_groups[0]["lr"],
            train_every=4,
            min_train_size=200,
            normalize_factor=self.agent.normalize_factor,
            d_model=self.agent.d_model,
            nhead=self.agent.nhead,
            num_layers=self.agent.num_layers,
            dropout=0.1,
        )

    def _init_mpc_planner(self) -> MPCPlanner:
        return MPCPlanner(
            world_model_trainer=self.wm_trainer,
            actions=self.agent.actions,
            seq_len=self.seq_len,
            action_pad_idx=self.agent.action_pad_idx,
            normalize_factor=self.agent.normalize_factor,
            horizon=5,
            num_samples=16,
            gamma=0.95,
            min_wm_samples=300,
            done_threshold=0.6,
            first_step_top_k=1,
            epsilon_mpc=0.0,
            softmax_temp=0.8,
            get_q_values=self.agent.evaluate_q_values,
            seed=self.random.randint(0, 1_000_000),
        )

    def reset(self) -> None:
        self.env = self._make_env()
        self.agent.reset()
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.last_action = None
        self.last_q_values = None
        self.last_planner_used = None
        self.success_window.clear()
        self.loss_window.clear()
        self.episode_rewards.clear()
        self._init_histories()
        self.wm_trainer = self._init_world_model_trainer()
        self.mpc_planner = self._init_mpc_planner()

    def toggle_learning(self) -> bool:
        self.learning_enabled = not self.learning_enabled
        return self.learning_enabled

    def set_speed(self, multiplier: float) -> float:
        self.speed = max(0.1, min(multiplier, 50.0))
        return self.speed

    def stop(self) -> None:
        self._stop = True

    def _compute_success_rate(self) -> float:
        if not self.success_window:
            return 0.0
        return sum(self.success_window) / len(self.success_window)

    async def training_loop(self, websocket: WebSocket) -> None:
        self._init_histories()
        await self._send_state(websocket, done=False, info={"status": "init"})
        while not self._stop:
            if not self.running:
                await asyncio.sleep(0.05)
                continue

            obs_seq, action_seq, reward_seq = self._get_sequences()
            current_patch = self.env.get_local_patch(5)
            prev_distance = self.env.shortest_path_length()
            if not self.learning_enabled:
                action = self.agent.random.choice(self.agent.actions)
                _, q_values = self.agent.choose_action(obs_seq, action_seq, reward_seq)
                self.last_planner_used = "dqn"
            elif self.planner_mode == "mpc":
                action, q_values = self.mpc_planner.plan(
                    self.obs_history,
                    self.action_history,
                    self.reward_history,
                    current_patch,
                    fallback_policy=lambda: self.agent.choose_action(obs_seq, action_seq, reward_seq),
                )
                self.last_planner_used = "mpc" if q_values is None else "dqn"
            else:
                action, q_values = self.agent.choose_action(obs_seq, action_seq, reward_seq)
                self.last_planner_used = "dqn"
            self.last_q_values = q_values
            result = self.env.step(action)
            self.last_action = action
            self.step += 1
            current_distance = self.env.shortest_path_length()
            next_patch = self.env.get_local_patch(5)

            shaped_reward = result.reward
            if self.reward_shaping and prev_distance is not None and current_distance is not None:
                shaped_reward += self.distance_reward_weight * (prev_distance - current_distance)
            self.total_reward += shaped_reward

            done = result.done or self.step >= self.max_steps
            if done and not result.done and self.step >= self.max_steps:
                result.info["status"] = "timeout"

            self.wm_trainer.add_transition(
                obs_seq=obs_seq,
                action_seq=action_seq,
                reward_seq=reward_seq,
                action=action,
                next_patch=next_patch,
                reward=shaped_reward,
                done=done,
            )

            if self.learning_enabled:
                next_obs_seq, next_action_seq, next_reward_seq = self._build_next_sequences(
                    action, shaped_reward, next_patch=next_patch
                )
                loss = self.agent.learn(
                    obs_seq=obs_seq,
                    action_seq=action_seq,
                    reward_seq=reward_seq,
                    action=action,
                    reward=shaped_reward,
                    next_obs_seq=next_obs_seq,
                    next_action_seq=next_action_seq,
                    next_reward_seq=next_reward_seq,
                    done=done,
                )
                if loss is not None:
                    self.last_loss = loss
                    self.loss_window.append(loss)
                self.wm_trainer.train_step()

            self._append_transition(action, shaped_reward, next_patch=next_patch)

            await self._send_state(websocket, done=done, info=result.info)

            if done:
                self.success_window.append(1 if result.info.get("status") == "goal" else 0)
                self.episode_rewards.append(self.total_reward)
                self.env = self._make_env()
                self.agent.decay_epsilon()
                self.episode += 1
                self.step = 0
                self.total_reward = 0.0
                self.last_action = None
                self.last_q_values = None
                self._init_histories()

            await asyncio.sleep(max(0.005, 0.05 / self.speed))

    async def _send_state(self, websocket: WebSocket, done: bool, info: Dict[str, Any]) -> None:
        obs = self.env.get_observation()
        message = {
            "grid": self.env.grid,
            "agent_pos": {"x": self.env.position[0], "y": self.env.position[1]},
            "episode": self.episode,
            "step": self.step,
            "epsilon": round(self.agent.epsilon, 4),
            "total_reward": round(self.total_reward, 3),
            "done": done,
            "info": info,
            "last_action": self.last_action,
            "success_rate": round(self._compute_success_rate(), 3),
            "learning": self.learning_enabled,
            "speed": self.speed,
            "observation": obs,
            "observation_patch": self.env.get_local_patch(5),
            "q_values": self.last_q_values,
            "random_maze": self.maze_mode == "random",
            "maze_mode": self.maze_mode,
            "last_loss": self.last_loss,
            "avg_recent_loss": round(sum(self.loss_window) / len(self.loss_window), 4)
            if self.loss_window
            else None,
            "avg_episode_reward": round(sum(self.episode_rewards) / len(self.episode_rewards), 3)
            if self.episode_rewards
            else None,
            "planner_mode": self.planner_mode,
            "planner_used": self.last_planner_used,
            "wm_ready": self.mpc_planner.is_ready(),
            "wm_buffer_size": len(self.wm_trainer.buffer),
            "wm_min_samples": self.mpc_planner.min_wm_samples,
            "wm_samples_needed": max(0, self.mpc_planner.min_wm_samples - len(self.wm_trainer.buffer)),
            "wm_last_loss": self.wm_trainer.last_loss,
            "wm_avg_recent_loss": round(self.wm_trainer.avg_recent_loss, 4)
            if self.wm_trainer.avg_recent_loss is not None
            else None,
        }
        await websocket.send_json(message)

    def _init_histories(self) -> None:
        self.obs_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        self.obs_history.append(self.env.get_local_patch(5))
        self.action_history.append(self.agent.action_pad_idx)
        self.reward_history.append(0.0)

    def _get_sequences(self) -> Tuple[List[List[List[int]]], List[int], List[float]]:
        return list(self.obs_history), list(self.action_history), list(self.reward_history)

    def _build_next_sequences(
        self, action: int, reward: float, next_patch: List[List[int]] | None = None
    ) -> Tuple[List[List[List[int]]], List[int], List[float]]:
        next_obs_history = deque(self.obs_history, maxlen=self.seq_len)
        next_action_history = deque(self.action_history, maxlen=self.seq_len)
        next_reward_history = deque(self.reward_history, maxlen=self.seq_len)
        next_obs_history.append(next_patch or self.env.get_local_patch(5))
        next_action_history.append(action)
        next_reward_history.append(reward)
        return list(next_obs_history), list(next_action_history), list(next_reward_history)

    def _append_transition(self, action: int, reward: float, next_patch: List[List[int]] | None = None) -> None:
        self.obs_history.append(next_patch or self.env.get_local_patch(5))
        self.action_history.append(action)
        self.reward_history.append(reward)


app = FastAPI()

# Allow local dev in case of different ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
favicon_path = frontend_dir / "favicon.svg"


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(favicon_path, media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = TrainingSession()
    training_task = asyncio.create_task(session.training_loop(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "start":
                session.running = True
            elif msg_type == "pause":
                session.running = False
            elif msg_type == "reset":
                session.reset()
                await session._send_state(websocket, done=False, info={"status": "reset"})
            elif msg_type == "toggle_learning":
                session.toggle_learning()
                await session._send_state(websocket, done=False, info={"status": "learning_toggled"})
            elif msg_type == "speed":
                speed = float(data.get("value", 1.0))
                session.set_speed(speed)
                await session._send_state(websocket, done=False, info={"status": "speed_updated"})
            elif msg_type == "maze_mode":
                mode_value = data.get("value", True)
                if isinstance(mode_value, str) and mode_value.lower() == "memory":
                    session.maze_mode = "memory"
                elif isinstance(mode_value, bool):
                    session.maze_mode = "random" if mode_value else "fixed"
                else:
                    session.maze_mode = "random"
                session.reset()
                await session._send_state(
                    websocket,
                    done=False,
                    info={"status": "maze_mode_updated", "random_maze": session.maze_mode == "random", "maze_mode": session.maze_mode},
                )
            elif msg_type == "planner_mode":
                mode_value = str(data.get("value", "dqn")).lower()
                session.planner_mode = "mpc" if mode_value == "mpc" else "dqn"
                await session._send_state(
                    websocket,
                    done=False,
                    info={"status": "planner_mode_updated", "planner_mode": session.planner_mode},
                )
            else:
                await websocket.send_json({"info": {"status": "unknown_command", "command": msg_type}})
    except WebSocketDisconnect:
        session.stop()
        training_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await training_task
    except Exception:
        session.stop()
        training_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await training_task
        raise
