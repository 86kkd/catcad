#!/usr/bin/env python
from __future__ import annotations

# extras [viz]: pygame
# 安装: uv pip install -e .[viz]
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

from auto_router.common.config import load_yaml
from auto_router.env.snake_route_env import Actions, Rules, SnakeRouteEnv

app = typer.Typer(
    add_completion=False,
    help="Human viewer: pygame 控制 Env, 终端以 Rich Live 刷新奖励表",
)
console = Console()


def setup_logging(verbosity: int) -> None:
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


@dataclass
class ViewerState:
    paused: bool = False
    target_fps: int = 30
    last_action: int = Actions.STOP
    total_reward: float = 0.0


def _build_env(
    cfg_path: Path,
    height: int,
    width: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    seed: int,
    max_steps: int | None,
    enable_via: bool,
) -> SnakeRouteEnv:
    cfg = load_yaml(cfg_path)
    rules_cfg = cfg.get("rules", {})
    env_cfg = cfg.get("env", {})
    rules = Rules.from_dict(rules_cfg)
    if max_steps is None:
        max_steps = int(env_cfg.get("max_steps_factor", 1.0) * height * width)

    env = SnakeRouteEnv(
        height=height,
        width=width,
        start=start,
        goal=goal,
        obstacles=np.zeros((height, width), dtype=bool),
        rules=rules,
        max_steps=max_steps,
        enable_via=enable_via,
        local_crop_sizes=list(env_cfg.get("local_crop_sizes", [])),
        seed=seed,
        render_mode="human",
    )
    return env


def _update_table(
    env: SnakeRouteEnv, state: ViewerState, step_reward: float, info: dict[str, float]
) -> Table:
    table = Table(title="Reward Live Stats", expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("paused", str(state.paused))
    table.add_row("fps_target", str(state.target_fps))
    table.add_row("steps", str(env.steps))
    table.add_row("pos", str(env.pos))
    table.add_row("last_action", str(state.last_action))
    table.add_row("step_reward", f"{step_reward:.3f}")
    table.add_row("total_reward", f"{state.total_reward:.3f}")
    if "dist_to_goal" in info:
        table.add_row("dist_to_goal", f"{info['dist_to_goal']}")
    table.add_row("success", str(env.success))
    return table


def _keymap_to_action(pressed: Sequence[bool], enable_via: bool) -> int | None:
    # 基础方向键
    import pygame

    # 对角: Q/E/Z/C; 方向: 箭头或 WASD
    up = pressed[pygame.K_UP] or pressed[pygame.K_w]
    down = pressed[pygame.K_DOWN] or pressed[pygame.K_s]
    left = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]

    ul = pressed[pygame.K_q]
    ur = pressed[pygame.K_e]
    dl = pressed[pygame.K_z]
    dr = pressed[pygame.K_c]

    if ul or (up and left):
        return Actions.UP_LEFT
    if ur or (up and right):
        return Actions.UP_RIGHT
    if dl or (down and left):
        return Actions.DOWN_LEFT
    if dr or (down and right):
        return Actions.DOWN_RIGHT
    if up and not down:
        return Actions.UP
    if down and not up:
        return Actions.DOWN
    if left and not right:
        return Actions.LEFT
    if right and not left:
        return Actions.RIGHT
    if enable_via and pressed[pygame.K_v]:
        return Actions.VIA
    return None


@app.command()
def run(
    cfg: Path = typer.Option(  # noqa: B008
        Path("configs/default.yaml"), "--cfg", help="YAML 配置文件路径"
    ),
    height: int = typer.Option(100, help="网格高度 H"),
    width: int = typer.Option(100, help="网格宽度 W"),
    start_y: int = typer.Option(0, help="起点 Y"),
    start_x: int = typer.Option(0, help="起点 X"),
    goal_y: int = typer.Option(-1, help="终点 Y, -1 表示 H-1"),
    goal_x: int = typer.Option(-1, help="终点 X, -1 表示 W-1"),
    seed: int = typer.Option(42, help="随机种子"),
    enable_via: bool = typer.Option(False, help="启用 VIA 动作(演示用途)"),
    verbosity: int = typer.Option(1, "-v", count=True, help="日志详细程度"),
) -> None:
    """基于 pygame 的人类控制器: 非阻塞键盘映射 Env 动作, Rich Live 实时表格。"""
    setup_logging(verbosity)

    # 仅在需要时导入 pygame, 给出清晰的安装与无显示环境指引
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API.*",
                category=UserWarning,
                module=r"^pygame\.pkgdata$",
            )
            import pygame
    except ImportError as exc:
        raise RuntimeError("需要安装 pygame; 执行 uv pip install -e .[viz]") from exc

    gy = goal_y if goal_y >= 0 else (height - 1)
    gx = goal_x if goal_x >= 0 else (width - 1)
    start = (int(start_y), int(start_x))
    goal = (int(gy), int(gx))

    env = _build_env(
        cfg_path=cfg,
        height=height,
        width=width,
        start=start,
        goal=goal,
        seed=seed,
        max_steps=None,
        enable_via=enable_via,
    )
    env.reset()
    # 先进行一次渲染, 确保 pygame 显卡子系统和窗口已初始化,
    # 防止事件获取时报 "video system not initialized"
    env.render()

    state = ViewerState(paused=False, target_fps=30, last_action=Actions.STOP, total_reward=0.0)
    clock = pygame.time.Clock()

    console.rule("[bold green]Human Viewer — Controls")
    console.print(
        "ESC 退出 | R 重置 | SPACE 暂停/继续 | +/- 调整速度 | "
        "方向键/WASD/QEZC 控制方向 | V 经由(VIA)"
    )

    step_reward = 0.0
    info: dict[str, float] = {}
    with Live(
        _update_table(env, state, step_reward, info),
        refresh_per_second=20,
        console=console,
        screen=False,
    ) as live:
        running = True
        while running:
            # 非阻塞事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset()
                        state.total_reward = 0.0
                        step_reward = 0.0
                        info = {}
                    elif event.key == pygame.K_SPACE:
                        state.paused = not state.paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        state.target_fps = min(240, state.target_fps + 5)
                    elif event.key == pygame.K_MINUS:
                        state.target_fps = max(1, state.target_fps - 5)

            pressed = pygame.key.get_pressed()
            action = _keymap_to_action(pressed, enable_via)
            if action is None:
                action = state.last_action  # 保持上一次动作
            else:
                state.last_action = action

            if not state.paused:
                _, r, terminated, truncated, info = env.step(int(action))
                step_reward = float(r)
                state.total_reward += step_reward
                if terminated or truncated:
                    # 结束后保持当前画面, 并进入暂停
                    state.paused = True

            # 渲染(交给 env), 刷新终端表
            env.render()
            live.update(_update_table(env, state, step_reward, info))

            # 速度控制: 尽量靠近目标 FPS
            try:
                clock.tick(state.target_fps)
            except Exception:
                time.sleep(max(0.0, 1.0 / float(state.target_fps)))

    try:
        env.close()
    finally:
        pass


def main() -> None:
    app()


if __name__ == "__main__":
    main()
