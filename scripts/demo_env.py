#!/usr/bin/env python
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from auto_router.common.config import load_yaml
from auto_router.env.snake_route_env import Actions, Rules, SnakeRouteEnv

app = typer.Typer(add_completion=False, help="Snake-route environment demo with CLI")
console = Console()


def setup_logging(verbosity: int, log_file: Path | None) -> None:
    """Initialize Loguru logging with Rich-friendly formatting."""
    logger.remove()
    level = "WARNING"
    if verbosity >= 2:
        level = "DEBUG"
    elif verbosity == 1:
        level = "INFO"
    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        enqueue=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
            "<cyan>{message}</cyan>"
        ),
    )
    if log_file is not None:
        logger.add(log_file, level=level, enqueue=True, rotation="5 MB", retention=3)


def _build_env(
    cfg_path: Path,
    height: int,
    width: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    seed: int,
    override_max_steps: int | None,
    override_enable_via: bool | None,
    render_mode: str | None,
) -> SnakeRouteEnv:
    cfg = load_yaml(cfg_path)
    rules_cfg = cfg.get("rules", {})
    env_cfg = cfg.get("env", {})
    rules = Rules.from_dict(rules_cfg)
    max_steps = (
        int(env_cfg.get("max_steps_factor", 1.0) * height * width)
        if override_max_steps is None
        else int(override_max_steps)
    )
    enable_via = (
        bool(env_cfg.get("enable_via", False))
        if override_enable_via is None
        else bool(override_enable_via)
    )
    local_crops = list(env_cfg.get("local_crop_sizes", []))

    # Simple demo obstacles: a horizontal wall with a gap
    obstacles = np.zeros((height, width), dtype=bool)
    gap_x0 = max(2, width // 4)
    gap_x1 = min(width - 2, width - width // 4)
    wall_y = max(1, height // 2)
    obstacles[wall_y, gap_x0:gap_x1] = True

    env = SnakeRouteEnv(
        height=height,
        width=width,
        start=start,
        goal=goal,
        obstacles=obstacles,
        rules=rules,
        max_steps=max_steps,
        enable_via=enable_via,
        local_crop_sizes=local_crops,
        seed=seed,
        render_mode=render_mode,
    )
    return env


@app.command()
def run(
    cfg: Path = typer.Option(  # noqa: B008
        Path("configs/default.yaml"), "--cfg", help="YAML 配置文件路径"
    ),
    height: int = typer.Option(10, help="网格高度 H"),
    width: int = typer.Option(10, help="网格宽度 W"),
    start_y: int = typer.Option(0, help="起点 Y"),
    start_x: int = typer.Option(0, help="起点 X"),
    goal_y: int = typer.Option(-1, help="终点 Y, -1 代表 H-1"),
    goal_x: int = typer.Option(-1, help="终点 X, -1 代表 W-1"),
    steps: int = typer.Option(300, help="贪心策略的最大步数"),
    sleep_s: float = typer.Option(0.01, help="每步之间的 sleep 秒数"),
    seed: int = typer.Option(42, help="随机种子"),
    max_steps: int | None = typer.Option(
        None, help="覆盖环境 max_steps(默认为基于 cfg 的 H*W*factor)"
    ),
    enable_via: bool | None = typer.Option(None, help="覆盖是否启用 VIA(默认使用 cfg 值)"),
    verbosity: int = typer.Option(1, "-v", count=True, help="增加日志详细程度, 可叠加, 例如 -vv"),
    log_file: Path | None = typer.Option(None, help="将日志同时写入该文件"),  # noqa: B008
    render_mode: str = typer.Option(
        "rgb_array", help="渲染模式: rgb_array 或 human(与 Gymnasium 接口一致)"
    ),
) -> None:
    """运行 SnakeRouteEnv 的演示, 并使用简单的朝目标贪心策略。"""
    setup_logging(verbosity=verbosity, log_file=log_file)

    # Resolve dynamic goals
    gy = goal_y if goal_y >= 0 else (height - 1)
    gx = goal_x if goal_x >= 0 else (width - 1)
    start = (int(start_y), int(start_x))
    goal = (int(gy), int(gx))

    console.rule("[bold green]SnakeRoute 环境演示")
    table = Table(title="参数")
    table.add_column("名称")
    table.add_column("值")
    table.add_row("cfg", str(cfg))
    table.add_row("grid", f"{height} x {width}")
    table.add_row("start", str(start))
    table.add_row("goal", str(goal))
    table.add_row("policy_steps", str(steps))
    console.print(table)

    env = _build_env(
        cfg_path=cfg,
        height=height,
        width=width,
        start=start,
        goal=goal,
        seed=seed,
        override_max_steps=max_steps,
        override_enable_via=enable_via,
        render_mode=render_mode,
    )

    obs, _ = env.reset()
    assert "global" in obs
    total_reward = 0.0
    logger.info("Env reset. Start routing...")

    for step_idx in range(int(steps)):
        # 简单朝目标的贪心策略, 优先对角
        y, x = env.pos
        gy, gx = env.goal
        dy = int(np.sign(gy - y))
        dx = int(np.sign(gx - x))
        diag_map = {
            (-1, -1): Actions.UP_LEFT,
            (-1, 1): Actions.UP_RIGHT,
            (1, -1): Actions.DOWN_LEFT,
            (1, 1): Actions.DOWN_RIGHT,
        }
        if dy != 0 and dx != 0:
            action = diag_map[(dy, dx)]
        elif dy < 0:
            action = Actions.UP
        elif dy > 0:
            action = Actions.DOWN
        elif dx < 0:
            action = Actions.LEFT
        elif dx > 0:
            action = Actions.RIGHT
        else:
            action = Actions.STOP

        _, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if step_idx % 10 == 0:
            logger.debug(
                "step={} pos={} reward={:.3f} dist_to_goal={}",
                env.steps,
                env.pos,
                reward,
                info.get("dist_to_goal"),
            )

        # 遵循 Gymnasium 渲染接口
        _ = env.render() if render_mode == "human" else env.render_rgb()
        time.sleep(float(sleep_s))
        if terminated or truncated:
            break

    console.print(
        f"[bold cyan]Success[/]: {env.success}  [bold cyan]Steps[/]: {env.steps}  "
        f"[bold cyan]Reward[/]: {total_reward:.2f}"
    )
    logger.info(
        "Routing finished. success={} steps={} reward={:.2f}", env.success, env.steps, total_reward
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
