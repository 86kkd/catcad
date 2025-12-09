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

    assert obs[0] == 0 and obs[1] == 0

    env.close()

