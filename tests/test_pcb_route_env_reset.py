import numpy as np

from auto_router.env.pcb_route_env import PcbRouteEnv, Rules


def test_reset_clears_paths_and_preserves_obstacles() -> None:
    rules = Rules()
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[1, 1] = True

    env = PcbRouteEnv(
        height=3,
        width=3,
        start=(0, 0),
        goal_bbox=(2, 2, 2, 2),
        obstacles=obstacles,
        rules=rules,
        num_layers=2,
    )

    env.step({"d_heading": 1, "step_len": 0, "layer_change": 0, "commit": 0})
    assert bool(env.occ[0, 0, 1])

    obs, _info = env.reset()

    assert env.y == 0 and env.x == 0
    assert env.steps == 0
    assert env.success is False

    assert env.occ[0, 0, 0] is True
    assert env.occ[0, 0, 1] is False
    assert env.occ[0, 1, 1] is True
    assert env.occ[1, 1, 1] is True

    grid = obs["grid"]
    state = obs["state"]

    assert grid.shape == (env.num_layers + 1, env.height, env.width)
    assert state[0] == 0 and state[1] == 0

    # Occupancy (start + obstacles)
    assert grid[0, 0, 0] == 1
    assert grid[0, 0, 1] == 0
    assert grid[0, 1, 1] == 1
    assert grid[1, 1, 1] == 1

    # Goal mask channel
    assert grid[-1, 2, 2] == 1

    env.close()


def test_random_start_and_goal_not_overlap_and_not_on_obstacles() -> None:
    rules = Rules()
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[1, 1] = True

    env = PcbRouteEnv(
        height=4,
        width=4,
        start=(0, 0),
        goal_bbox=(0, 0, 0, 0),
        obstacles=obstacles,
        rules=rules,
        num_layers=1,
        random_start=True,
        random_goal=True,
    )

    for _ in range(5):
        obs, _info = env.reset()
        sy, sx = env.y, env.x
        y0, x0, y1, x1 = env.goal_bbox

        assert obstacles[sy, sx] is False
        assert not (y0 <= sy <= y1 and x0 <= sx <= x1)

        grid = obs["grid"]
        assert grid[-1, y0 : y1 + 1, x0 : x1 + 1].any()

    env.close()
