from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .gridworld import GridWorld
from .qlearning import QLearningAgent


class TrainingSession:
    def __init__(self, seed: int = 42):
        self.env = GridWorld.default()
        self.agent = QLearningAgent(actions=[0, 1, 2, 3], seed=seed)
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

    def reset(self) -> None:
        self.env = GridWorld.default()
        self.agent.reset()
        self.episode = 1
        self.step = 0
        self.total_reward = 0.0
        self.running = False
        self.last_action = None
        self.success_window.clear()

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
        await self._send_state(websocket, done=False, info={"status": "init"})
        while not self._stop:
            if not self.running:
                await asyncio.sleep(0.05)
                continue

            state = self.env.position
            if self.learning_enabled:
                action = self.agent.choose_action(state)
            else:
                action = self.agent.random.choice(self.agent.actions)
            result = self.env.step(action)
            self.last_action = action
            self.step += 1
            self.total_reward += result.reward

            done = result.done or self.step >= self.max_steps
            if done and not result.done and self.step >= self.max_steps:
                result.info["status"] = "timeout"

            if self.learning_enabled:
                self.agent.learn(state, action, result.reward, result.next_state, done)

            await self._send_state(websocket, done=done, info=result.info)

            if done:
                self.success_window.append(1 if result.info.get("status") == "goal" else 0)
                self.env.reset()
                self.agent.decay_epsilon()
                self.episode += 1
                self.step = 0
                self.total_reward = 0.0
                self.last_action = None

            await asyncio.sleep(max(0.005, 0.05 / self.speed))

    async def _send_state(self, websocket: WebSocket, done: bool, info: Dict[str, Any]) -> None:
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
        }
        await websocket.send_json(message)


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


@app.get("/")
async def index():
    return FileResponse(frontend_dir / "index.html")

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
