from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

Action = int


class Actions:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    VIA = 4  # placeholder for multi-layer; disabled in single-layer maps
    STOP = 5
    UP_LEFT = 6
    UP_RIGHT = 7
    DOWN_LEFT = 8
    DOWN_RIGHT = 9

    ALL = (UP, DOWN, LEFT, RIGHT, VIA, STOP, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT)


def _dilation_radius(cells_clearance: int, cells_line_width: int) -> int:
    # Ensure at least Manhattan radius for conservative DRC approximation
    return max(0, math.ceil((cells_clearance + cells_line_width - 1) / 2))


def _manhattan_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[1]], dtype=np.uint8)
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if abs(i - radius) + abs(j - radius) <= radius:
                kernel[i, j] = 1
    return kernel


def _binary_dilation(mask: np.ndarray, radius: int) -> np.ndarray:
    """Fast binary dilation using convolution with a Manhattan kernel (uint8)."""
    if radius <= 0:
        return mask.astype(bool)
    kernel = _manhattan_kernel(radius)
    # Convolve via FFT only for larger kernels to reduce overhead
    from numpy.fft import fft2, ifft2

    kernel = _manhattan_kernel(radius)
    h, w = mask.shape
    kh, kw = kernel.shape
    pad_h, pad_w = h + kh - 1, w + kw - 1
    fa = fft2(mask.astype(np.float32), s=(pad_h, pad_w))
    fb = fft2(kernel.astype(np.float32), s=(pad_h, pad_w))
    conv = np.real(ifft2(fa * fb))
    conv = conv[(kh - 1) // 2 : (kh - 1) // 2 + h, (kw - 1) // 2 : (kw - 1) // 2 + w]
    return conv > 0.5


@dataclass(frozen=True)
class Rules:
    """
    Design rules interpreted on a unit grid.

    All distances are expressed in grid cells. Map real units (mm) to grid externally.
    """

    line_width: int = 1
    clearance: int = 1
    via_cost: float = 3.0
    corner_cost: float = 0.2
    step_cost: float = 0.01
    violation_cost: float = 1.0
    success_reward: float = 10.0

    @staticmethod
    def from_dict(cfg: dict[str, float | int]) -> Rules:
        return Rules(
            line_width=int(cfg.get("line_width", 1)),
            clearance=int(cfg.get("clearance", 1)),
            via_cost=float(cfg.get("via_cost", 3.0)),
            corner_cost=float(cfg.get("corner_cost", 0.2)),
            step_cost=float(cfg.get("step_cost", 0.01)),
            violation_cost=float(cfg.get("violation_cost", 1.0)),
            success_reward=float(cfg.get("success_reward", 10.0)),
        )


class SnakeRouteEnv(gym.Env):
    """
    Single-net, single-layer grid routing environment (Gym-like).

    Observation (global): H x W x C tensor with channels:
      0: obstacles mask (binary)
      1: goal field (Manhattan distance normalized)
      2: layer one-hot (always 1 for single-layer)
      3: congestion/occupancy (routed path so far)
      4: DRC margin mask (1 = safe region, 0 = forbidden by dilation)

    Actions: UP, DOWN, LEFT, RIGHT, VIA (disabled), STOP
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        height: int,
        width: int,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: np.ndarray | None = None,
        rules: Rules | None = None,
        max_steps: int | None = None,
        enable_via: bool = False,
        local_crop_sizes: Sequence[int] | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.h = int(height)
        self.w = int(width)
        self.start: tuple[int, int] = (int(start[0]), int(start[1]))
        self.goal: tuple[int, int] = (int(goal[0]), int(goal[1]))
        self.rules = rules or Rules()
        self.max_steps = max_steps or (self.h * self.w)
        self.enable_via = enable_via
        self.local_crop_sizes: list[int] = list(local_crop_sizes or [])
        self._rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        # Lazy pygame resources for human rendering (initialized on first use)
        from typing import Any

        self._pg_screen: Any | None = None
        self._pg_clock: Any | None = None
        # Default scale heuristics for human window
        self._pg_scale: int = max(4, min(32, 640 // max(self.h, self.w)))

        if obstacles is None:
            self.obstacles: np.ndarray = np.zeros((self.h, self.w), dtype=bool)
        else:
            assert obstacles.shape == (self.h, self.w)
            self.obstacles = obstacles.astype(bool)

        # Derived masks
        self.forbidden: np.ndarray = self._compute_forbidden()

        # Define action and observation spaces
        self.action_space = spaces.Discrete(10)
        low = np.zeros((self.h, self.w, 5), dtype=np.float32)
        high = np.ones((self.h, self.w, 5), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=low, high=high, dtype=np.float32),
                "local": spaces.Sequence(
                    spaces.Box(low=0.0, high=1.0, shape=(1, 1, 5), dtype=np.float32)
                ),
            }
        )

        # Predeclare stateful attributes for type-checkers
        self.pos: tuple[int, int] = (0, 0)
        self.prev_dir: tuple[int, int] | None = None
        self.occupancy: np.ndarray = np.zeros((self.h, self.w), dtype=bool)
        self.steps: int = 0
        self.success: bool = False

        self._reset_state()

    # Gymnasium API
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[dict[str, np.ndarray | list[np.ndarray]], dict[str, float]]:
        # Reset state
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        return self._observation(), {}

    def step(self, action: Action) -> tuple[
        dict[str, np.ndarray | list[np.ndarray]],
        float,
        bool,
        bool,
        dict[str, float],
    ]:
        # Take a step
        terminated = False
        truncated = False
        reward = 0.0
        info: dict[str, float] = {}

        if action == Actions.STOP:
            # Stop action
            terminated = True
            info["stopped"] = 1.0
            return self._observation(), reward, terminated, truncated, info

        if action == Actions.VIA and not self.enable_via:
            # Disallowed in single-layer phase; treat as no-op with a small penalty
            reward -= self.rules.step_cost
            return self._tick(reward, terminated, truncated, info)

        dxy = self._action_to_delta(action)
        if dxy is None:
            # Invalid action id
            reward -= self.rules.violation_cost
            return self._tick(reward, terminated, truncated, info)

        new_pos = (self.pos[0] + dxy[0], self.pos[1] + dxy[1])

        # Bounds check
        if not (0 <= new_pos[0] < self.h and 0 <= new_pos[1] < self.w):
            # Out of bounds
            reward -= self.rules.violation_cost
            return self._tick(reward, terminated, truncated, {**info, "out_of_bounds": 1.0})

        # Disallow diagonal corner-cutting: both adjacent orthogonal cells must be free
        if dxy[0] != 0 and dxy[1] != 0:
            inter1 = (self.pos[0] + dxy[0], self.pos[1])
            inter2 = (self.pos[0], self.pos[1] + dxy[1])
            if (
                not (0 <= inter1[0] < self.h and 0 <= inter1[1] < self.w)
                or not (0 <= inter2[0] < self.h and 0 <= inter2[1] < self.w)
                or self.obstacles[inter1]
                or self.obstacles[inter2]
                or self.forbidden[inter1]
                or self.forbidden[inter2]
                or self.occupancy[inter1]
                or self.occupancy[inter2]
            ):
                # Corner cut
                reward -= self.rules.violation_cost
                return self._tick(reward, terminated, truncated, {**info, "corner_cut": 1.0})

        # Forbidden (DRC dilation) or obstacle
        if self.forbidden[new_pos] or self.obstacles[new_pos]:
            reward -= self.rules.violation_cost
            return self._tick(reward, terminated, truncated, {**info, "forbidden": 1.0})

        # Self-intersection not allowed
        if self.occupancy[new_pos]:
            reward -= self.rules.violation_cost
            return self._tick(reward, terminated, truncated, {**info, "self_intersect": 1.0})

        # Corner penalty scaled by turn angle (penalize 90° more than 45°)
        if self.prev_dir is not None:
            pdx, pdy = self.prev_dir
            ndx, ndy = dxy
            prev_len = math.hypot(pdx, pdy)
            new_len = math.hypot(ndx, ndy)
            if prev_len > 0 and new_len > 0:
                cosang = max(-1.0, min(1.0, (pdx * ndx + pdy * ndy) / (prev_len * new_len)))
                angle = math.acos(cosang)
                # Normalize: 0 -> 0, 90deg -> 1; cap at 1
                angle_ratio = min(angle / (math.pi / 2.0), 1.0)
                if angle_ratio > 1e-6:
                    reward -= self.rules.corner_cost * angle_ratio

        # Step penalty proportional to step length (diagonal ≈ sqrt(2))
        step_len = math.hypot(dxy[0], dxy[1])
        reward -= self.rules.step_cost * float(step_len)

        # Move
        self.pos = new_pos
        self.occupancy[self.pos] = True
        self.prev_dir = dxy

        # Shaping: proximity to goal (negative distance)
        dist = self._manhattan(self.pos, self.goal)
        reward += -0.01 * float(dist)
        info["dist_to_goal"] = float(dist)

        # Success check
        if self.pos == self.goal:
            reward += self.rules.success_reward
            terminated = True
            self.success = True

        return self._tick(reward, terminated, truncated, info)

    # Internal helpers
    def _tick(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, float],
    ) -> tuple[dict[str, np.ndarray | list[np.ndarray]], float, bool, bool, dict[str, float]]:
        # Increment step count and check for truncation
        self.steps += 1
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            info = {**info, "max_steps": 1.0}
        return self._observation(), float(reward), bool(terminated), bool(truncated), info

    def _reset_state(self) -> None:
        # Reset state
        self.pos = self.start
        self.prev_dir = None
        self.occupancy = np.zeros((self.h, self.w), dtype=bool)
        self.occupancy[self.pos] = True
        self.steps = 0
        self.success = False

    def _action_to_delta(self, action: Action) -> tuple[int, int] | None:
        """Convert an action to a delta."""
        if action == Actions.UP:
            return (-1, 0)
        if action == Actions.DOWN:
            return (1, 0)
        if action == Actions.LEFT:
            return (0, -1)
        if action == Actions.RIGHT:
            return (0, 1)
        if action == Actions.UP_LEFT:
            return (-1, -1)
        if action == Actions.UP_RIGHT:
            return (-1, 1)
        if action == Actions.DOWN_LEFT:
            return (1, -1)
        if action == Actions.DOWN_RIGHT:
            return (1, 1)
        return None

    def _compute_forbidden(self) -> np.ndarray:
        """Compute the forbidden mask."""
        radius = _dilation_radius(self.rules.clearance, self.rules.line_width)
        return _binary_dilation(self.obstacles.astype(np.uint8), radius)

    def _goal_field(self) -> np.ndarray:
        yy, xx = np.mgrid[0 : self.h, 0 : self.w]
        dist = np.abs(yy - self.goal[0]) + np.abs(xx - self.goal[1])
        maxd = max(1, self.h + self.w - 2)
        return 1.0 - (dist / maxd)

    def _drc_margin_mask(self) -> np.ndarray:
        # 1 where safe to route (not forbidden), else 0
        return (~self.forbidden).astype(np.float32)

    def _observation(self) -> dict[str, np.ndarray | list[np.ndarray]]:
        """Return a dictionary of observations."""
        C = 5
        obs = np.zeros((self.h, self.w, C), dtype=np.float32)
        obs[:, :, 0] = self.obstacles.astype(np.float32)
        obs[:, :, 1] = self._goal_field()
        obs[:, :, 2] = 1.0
        obs[:, :, 3] = self.occupancy.astype(np.float32)
        obs[:, :, 4] = self._drc_margin_mask()
        out: dict[str, np.ndarray | list[np.ndarray]] = {"global": obs}
        if self.local_crop_sizes:
            # Return a list of crops at multiple scales; shapes differ by size
            out["local"] = [self._crop(obs, size) for size in self.local_crop_sizes]
        return out

    def _crop(self, obs: np.ndarray, size: int) -> np.ndarray:
        """Crop a square region around the current position."""
        r = size // 2
        y, x = self.pos
        y0, y1 = max(0, y - r), min(self.h, y + r + 1)
        x0, x1 = max(0, x - r), min(self.w, x + r + 1)
        crop = np.zeros((size, size, obs.shape[-1]), dtype=obs.dtype)
        cy0, cx0 = r - (y - y0), r - (x - x0)
        crop[cy0 : cy0 + (y1 - y0), cx0 : cx0 + (x1 - x0)] = obs[y0:y1, x0:x1]
        return crop

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Headless render (rgb_array) or human terminal rendering via Rich
    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self.render_rgb()
        if self.render_mode == "human":
            img = self.render_rgb()
            self._render_human(img)
            return None
        return None

    def render_rgb(self) -> np.ndarray:
        """Return an RGB image (H, W, 3) representing the current state. No plotting."""
        img = np.zeros((self.h, self.w, 3), dtype=np.float32)
        img[self.obstacles] = (0.2, 0.2, 0.2)
        img[self.forbidden & (~self.obstacles)] = (0.6, 0.6, 0.6)
        oy, ox = np.nonzero(self.occupancy)
        img[oy, ox] = (0.1, 0.6, 1.0)
        img[self.start] = (0.3, 1.0, 0.3)
        img[self.goal] = (1.0, 0.3, 0.3)
        return img

    def _render_human(self, img: np.ndarray) -> None:
        """使用 pygame 进行人类可视化渲染(仅在 render_mode == "human" 时启用)。"""
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Human render 需要 'pygame'. 请安装: uv pip install -e .[viz] 或 uv add pygame"
            ) from exc

        # 初始化窗口(首帧或被关闭后)
        if self._pg_screen is None:
            try:
                pygame.init()
                win_size = (int(self.w * self._pg_scale), int(self.h * self._pg_scale))
                screen = pygame.display.set_mode(win_size)
                pygame.display.set_caption("SnakeRoute - Human Render")
                self._pg_screen = screen
                self._pg_clock = pygame.time.Clock()
            except pygame.error as exc:
                raise RuntimeError(
                    "无法初始化 pygame 显示. 若在无显示环境, 请设置环境变量 "
                    "SDL_VIDEODRIVER=dummy, 或改用 render_mode='rgb_array'/None."
                ) from exc

        # 处理事件队列, 保持窗口响应
        from contextlib import suppress

        with suppress(pygame.error):
            pygame.event.pump()

        # 将 (H,W,3) float32 [0,1] 转为 surface, 并缩放到窗口大小
        arr8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        # pygame 的像素数组是 (W,H,3)
        arr8_wh = np.transpose(arr8, (1, 0, 2))
        surface = pygame.surfarray.make_surface(arr8_wh)

        screen = self._pg_screen
        if screen is not None:
            win_w, win_h = screen.get_size()
            if (win_w, win_h) != (self.w * self._pg_scale, self.h * self._pg_scale):
                # 窗口可能被用户缩放, 尊重当前窗口尺寸
                scaled = pygame.transform.smoothscale(surface, (win_w, win_h))
            else:
                scaled = pygame.transform.scale(
                    surface, (self.w * self._pg_scale, self.h * self._pg_scale)
                )
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            if self._pg_clock is not None:
                # 适度限制帧率, 避免占用过高
                self._pg_clock.tick(60)

    def close(self) -> None:
        """释放环境资源(包含 pygame 资源)。"""
        if self._pg_screen is not None:
            try:
                import pygame
            except ImportError:
                # 未安装 pygame, 无需清理
                self._pg_screen = None
                self._pg_clock = None
                return
            try:
                pygame.quit()
            except pygame.error:
                # 忽略关闭时的环境错误
                pass
            finally:
                self._pg_screen = None
                self._pg_clock = None
