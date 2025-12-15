from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


Grid = List[List[int]]
Position = Tuple[int, int]


@dataclass
class StepResult:
    next_state: Position
    reward: float
    done: bool
    info: dict


class GridWorld:
    """
    Simple GridWorld environment with walls, goal, and optional traps.
    Cells:
        0: empty
        1: wall
        2: start
        3: goal
        4: trap
    """

    def __init__(self, grid: Grid, start: Position, goal: Position, traps: List[Position] | None = None):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.traps = traps or []
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        self.position = start

    @classmethod
    def default(cls) -> "GridWorld":
        grid = [
            [2, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 0, 0, 3],
        ]
        start = (0, 0)
        goal = (9, 9)
        traps = [(4, 4), (8, 7)]
        grid[start[1]][start[0]] = 2
        grid[goal[1]][goal[0]] = 3
        for tx, ty in traps:
            grid[ty][tx] = 4
        return cls(grid=grid, start=start, goal=goal, traps=traps)

    def reset(self) -> Position:
        self.position = self.start
        return self.position

    def step(self, action: int) -> StepResult:
        # Actions: 0=up,1=down,2=left,3=right
        moves = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dx, dy = moves[action]
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        reward = -0.01  # step penalty
        status = "move"

        if not self._in_bounds(new_x, new_y) or self._is_wall(new_x, new_y):
            # Bump into wall: stay in place
            reward -= 0.05
            new_x, new_y = self.position
            status = "wall"
        elif (new_x, new_y) == self.goal:
            reward += 1.0
            status = "goal"
            self.position = (new_x, new_y)
            return StepResult(next_state=self.position, reward=reward, done=True, info={"status": status})
        elif (new_x, new_y) in self.traps:
            reward -= 1.0
            status = "trap"
            self.position = (new_x, new_y)
            return StepResult(next_state=self.position, reward=reward, done=True, info={"status": status})

        self.position = (new_x, new_y)
        return StepResult(next_state=self.position, reward=reward, done=False, info={"status": status})

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def _is_wall(self, x: int, y: int) -> bool:
        return self.grid[y][x] == 1

