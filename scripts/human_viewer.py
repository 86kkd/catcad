#!/usr/bin/env python
from __future__ import annotations

# extras [viz]: pygame
# Install: uv pip install -e .[viz]
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
from auto_router.env import PcbRouteEnv, Rules

app = typer.Typer(
    add_completion=False,
    help="Human viewer: Control Env with pygame, refresh reward table in terminal with Rich Live",
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
    total_reward: float = 0.0
    d_heading_idx: int = 1  # Corresponds to Δheading=0 ([-45,0,45])
    step_len: int = 1  # Actual step length (1..max)
    layer_target: int | None = None  # None means no switch; otherwise target layer index (0..L-1)
    commit_next: bool = False


def _build_env(
    cfg_path: Path,
    height: int,
    width: int,
    start: tuple[int, int],
    goal_bbox: tuple[int, int, int, int],
    seed: int,
    max_steps: int | None,
) -> PcbRouteEnv:
    cfg = load_yaml(cfg_path)
    rules_cfg = cfg.get("rules", {})
    env_cfg = cfg.get("env", {})
    rules = Rules.from_dict(rules_cfg)
    if max_steps is None:
        max_steps = int(env_cfg.get("max_steps_factor", 1.0) * height * width)

    num_layers = int(env_cfg.get("num_layers", 2))
    via_budget = int(env_cfg.get("via_budget", 8))
    via_cooldown = int(env_cfg.get("via_cooldown", 1))
    max_step_len = int(env_cfg.get("max_step_len", 5))

    env = PcbRouteEnv(
        height=height,
        width=width,
        start=start,
        goal_bbox=goal_bbox,
        obstacles=np.zeros((height, width), dtype=bool),
        rules=rules,
        num_layers=num_layers,
        via_budget=via_budget,
        via_cooldown_steps=via_cooldown,
        max_steps=max_steps,
        max_step_len=max_step_len,
        seed=seed,
        render_mode="human",
    )
    return env


def _update_table(
    env: PcbRouteEnv, state: ViewerState, step_reward: float, info: dict[str, float]
) -> Table:
    table = Table(title="Reward Live Stats", expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("paused", str(state.paused))
    table.add_row("fps_target", str(state.target_fps))
    table.add_row("steps", str(env.steps))
    table.add_row("pos", str(env.pos))
    table.add_row("heading", str(getattr(env, "heading", 0)))
    table.add_row("layer", str(getattr(env, "layer", 0)))
    table.add_row("via_budget", str(getattr(env, "via_budget", 0)))
    table.add_row("via_cooldown", str(getattr(env, "via_cooldown", 0)))
    table.add_row("step_reward", f"{step_reward:.3f}")
    table.add_row("total_reward", f"{state.total_reward:.3f}")
    if "dist_to_goal" in info:
        table.add_row("dist_to_goal", f"{info['dist_to_goal']}")
    table.add_row("success", str(env.success))
    return table


def _gather_head_choices(
    pressed: Sequence[bool],
    state: ViewerState,
    max_step_len: int,
    num_layers: int,
) -> tuple[int, int, int, int]:
    import pygame

    # Δheading selection: left/right for -45/+45, up or W for 0
    # Momentary control: default to straight each frame unless a key is pressed
    d_idx = 1  # 0:-45, 1:0, 2:+45
    if pressed[pygame.K_LEFT]:
        d_idx = 0  # -45 corresponds to index 0 ([-45,0,45])
    elif pressed[pygame.K_RIGHT]:
        d_idx = 2  # +45 corresponds to index 2 ([-45,0,45])
    elif pressed[pygame.K_UP] or pressed[pygame.K_w]:
        d_idx = 1  # 0 corresponds to index 1 ([-45,0,45])

    # Step length: number keys 1..9, limited to max_step_len
    step_len = state.step_len
    for key_num, val in (
        (pygame.K_1, 1),
        (pygame.K_2, 2),
        (pygame.K_3, 3),
        (pygame.K_4, 4),
        (pygame.K_5, 5),
        (pygame.K_6, 6),
        (pygame.K_7, 7),
        (pygame.K_8, 8),
        (pygame.K_9, 9),
    ):
        if pressed[key_num]:
            step_len = min(val, max_step_len)
            break

    # Layer switch: state.layer_target is set in KEYDOWN event loop
    layer_sel = 0
    if state.layer_target is not None:
        layer_sel = state.layer_target + 1  # env encoding: 1..L

    # commit: set once on KEYDOWN Enter
    commit = 1 if state.commit_next else 0

    return d_idx, step_len, layer_sel, commit


@app.command()
def run(
    cfg: Path = typer.Option(  # noqa: B008
        Path("configs/default.yaml"), "--cfg", help="Path to YAML config file"
    ),
    height: int = typer.Option(100, help="Grid height H"),
    width: int = typer.Option(100, help="Grid width W"),
    start_y: int = typer.Option(0, help="Start point Y"),
    start_x: int = typer.Option(0, help="Start point X"),
    goal_y: int = typer.Option(-1, help="Goal Y, -1 means H-1"),
    goal_x: int = typer.Option(-1, help="Goal X, -1 means W-1"),
    seed: int = typer.Option(42, help="Random seed"),
    enable_via: bool = typer.Option(False, help="(Reserved parameter)"),
    verbosity: int = typer.Option(1, "-v", count=True, help="Log verbosity level"),
) -> None:
    """Human viewer using pygame.

    Non-blocking keyboard controls and Rich Live stats table.
    """
    setup_logging(verbosity)

    # Only import pygame when needed, provide clear installation and headless environment guidance
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
        raise RuntimeError("pygame required; run: uv pip install -e .[viz]") from exc

    gy = goal_y if goal_y >= 0 else (height - 1)
    gx = goal_x if goal_x >= 0 else (width - 1)
    start = (int(start_y), int(start_x))
    goal_bbox = (int(gy), int(gx), int(gy), int(gx))

    env = _build_env(
        cfg_path=cfg,
        height=height,
        width=width,
        start=start,
        goal_bbox=goal_bbox,
        seed=seed,
        max_steps=None,
    )
    env.reset()
    # Render once first to ensure pygame video system and window are initialized,
    # preventing "video system not initialized" error during event handling
    env.render()

    state = ViewerState(paused=False, target_fps=30, total_reward=0.0)
    clock = pygame.time.Clock()

    console.rule("[bold green]Human Viewer — Controls")
    console.print(
        "ESC to quit | R to reset | SPACE to pause/resume | +/- adjust speed\n"
        "Direction: ← -45 / ↑(or W) straight / → +45 | Numbers 1-9 select step length\n"
        "TAB cycle target layer(if legal) | ENTER commit(only available at goal)\n"
        "Tip: 90° turns can be achieved by two consecutive 45° turns"
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
            # Non-blocking event handling
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
                        state.layer_target = None
                        state.commit_next = False
                    elif event.key == pygame.K_SPACE:
                        state.paused = not state.paused
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        state.target_fps = min(240, state.target_fps + 5)
                    elif event.key == pygame.K_MINUS:
                        state.target_fps = max(1, state.target_fps - 5)
                    elif event.key == pygame.K_TAB:
                        # Cycle through target layers (not applied immediately)
                        current = getattr(env, "layer", 0)
                        L = int(getattr(env, "num_layers", 2))
                        if state.layer_target is None:
                            state.layer_target = (current + 1) % L
                        else:
                            state.layer_target = (state.layer_target + 1) % L
                        if state.layer_target == current:
                            # Avoid selecting current layer, advance one more
                            state.layer_target = (state.layer_target + 1) % L
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        state.commit_next = True

            pressed = pygame.key.get_pressed()
            # Read masks, select by head parts and fallback if illegal
            masks = env.get_action_masks()
            d_idx, step_len, layer_sel, commit = _gather_head_choices(
                pressed, state, getattr(env, "max_step_len", 5), getattr(env, "num_layers", 2)
            )

            # Correct to legal mask values
            # Legalize Δheading: only allow 0(-45), 1(straight), 2(+45)
            if not (0 <= int(d_idx) < len(masks["d_heading"])):
                d_idx = 1
            elif masks["d_heading"][int(d_idx)] != 1:
                # Fallback to straight if chosen turn masked (shouldn't happen; all ones now)
                d_idx = 1

            step_len = int(step_len)
            step_mask = masks["step_len"]
            if step_len >= len(step_mask) or step_mask[step_len] == 0:
                # Selection illegal, find nearest available <= target step length
                legal = [i for i in range(1, len(step_mask)) if step_mask[i] == 1]
                step_len = legal[-1] if legal else 1

            layer_sel = int(layer_sel)
            layer_mask = masks["layer_change"]
            if layer_sel >= len(layer_mask) or layer_mask[layer_sel] == 0:
                layer_sel = 0

            commit_mask = masks["commit"]
            if commit == 1 and commit_mask[1] != 1:
                commit = 0

            # Execute action
            if not state.paused:
                _, r, terminated, truncated, info = env.step(
                    (int(d_idx), int(step_len - 1), int(layer_sel), int(commit))
                )
                step_reward = float(r)
                state.total_reward += step_reward
                state.d_heading_idx = d_idx
                state.step_len = step_len
                # commit is one-time
                state.commit_next = False
                if terminated or truncated:
                    state.paused = True

            # Render (handled by env), refresh terminal table
            env.render()
            live.update(_update_table(env, state, step_reward, info))

            # Speed control: try to match target FPS
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
