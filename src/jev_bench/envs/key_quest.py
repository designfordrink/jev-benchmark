from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

Position = Tuple[int, int]


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


DELTAS = {
    Action.UP: (0, 1),
    Action.RIGHT: (1, 0),
    Action.DOWN: (0, -1),
    Action.LEFT: (-1, 0),
}


@dataclass(frozen=True)
class Transition:
    state: tuple[int, int, int]
    action: int
    next_state: tuple[int, int, int]
    event: str
    terminated: bool


class KeyQuestEnv:
    """Small deterministic gridworld matching the JEV-RL reference shape."""

    name = "key_quest"

    def __init__(self, width: int = 5, height: int = 5, max_steps: int = 40):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.walls = frozenset({(1, 1), (1, 2), (3, 2), (3, 3)})
        self.lava = frozenset({(2, 3)})
        self.start: Position = (0, 0)
        self.key: Position = (2, 2)
        self.exit: Position = (4, 4)
        self.pos = self.start
        self.has_key = False
        self.steps = 0
        self.done = False

    def reset(self) -> tuple[int, int, int]:
        self.pos, self.has_key, self.steps, self.done = self.start, False, 0, False
        return self.state

    @property
    def state(self) -> tuple[int, int, int]:
        return (self.pos[0], self.pos[1], int(self.has_key))

    def step(self, action: int) -> Transition:
        if self.done:
            raise RuntimeError("episode is terminated")
        try:
            act = Action(int(action))
        except ValueError as exc:
            raise ValueError(f"invalid action: {action}") from exc
        old = self.state
        dx, dy = DELTAS[act]
        candidate = (self.pos[0] + dx, self.pos[1] + dy)
        self.steps += 1
        event = "move"
        if not (0 <= candidate[0] < self.width and 0 <= candidate[1] < self.height):
            candidate = self.pos
            event = "boundary"
        elif candidate in self.walls:
            candidate = self.pos
            event = "wall"
        self.pos = candidate
        if self.pos in self.lava:
            event, self.done = "lava", True
        elif self.pos == self.key and not self.has_key:
            self.has_key = True
            event = "key"
        elif self.pos == self.exit and self.has_key:
            event, self.done = "exit", True
        elif self.steps >= self.max_steps:
            event, self.done = "timeout", True
        return Transition(old, int(act), self.state, event, self.done)
