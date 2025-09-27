from __future__ import annotations

import numpy as np

from auto_router.env.snake_route_env import Actions, Rules, SnakeRouteEnv


def test_boundary_violation() -> None:
    env = SnakeRouteEnv(5, 5, start=(0, 0), goal=(4, 4), max_steps=20)
    env.reset()
    # Move up out of bounds
    _, r, term, trunc, info = env.step(Actions.UP)
    assert r < 0 and not (term or trunc)
    assert "out_of_bounds" in info or "violation" in "".join(info.keys())


def test_forbid_obstacle_cross() -> None:
    obs = np.zeros((6, 6), dtype=bool)
    obs[2, 1:5] = True
    env = SnakeRouteEnv(6, 6, start=(1, 0), goal=(1, 5), obstacles=obs, rules=Rules(clearance=0))
    env.reset()
    # Try to cross obstacle
    env.step(Actions.RIGHT)
    env.step(Actions.RIGHT)
    _, r, _, _, info = env.step(Actions.DOWN)
    assert r < 0
    assert "forbidden" in info


def test_dead_end_and_stop() -> None:
    obs = np.zeros((5, 5), dtype=bool)
    obs[1, 1:4] = True
    env = SnakeRouteEnv(5, 5, start=(0, 0), goal=(4, 4), obstacles=obs, max_steps=10)
    env.reset()
    # Walk into a dead-end corridor
    env.step(Actions.RIGHT)
    env.step(Actions.RIGHT)
    env.step(Actions.DOWN)
    _, _, term, trunc, _ = env.step(Actions.STOP)
    assert term or trunc


def test_corner_penalty_and_shaping() -> None:
    env = SnakeRouteEnv(5, 5, start=(0, 0), goal=(4, 4), max_steps=20)
    env.reset()
    # Turn once to incur corner cost
    _, r1, *_ = env.step(Actions.RIGHT)
    _, r2, *_ = env.step(Actions.DOWN)
    assert r2 < r1
