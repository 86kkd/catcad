from __future__ import annotations

from collections.abc import Mapping

# extras [viz]: pygame (only imported when render("human") is needed)
from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HEADING_DEGREES: list[int] = [-45, 0, 45]


def _deg_to_step(d: int) -> tuple[int, int]:
    # Map to 8-neighborhood unit step (allows diagonal)
    # Uses screen coordinates with y increasing downward, supports any multiple of 45°
    dm = int(d) % 360
    mapping: dict[int, tuple[int, int]] = {
        0: (0, 1),
        45: (-1, 1),
        90: (-1, 0),
        135: (-1, -1),
        180: (0, -1),
        225: (1, -1),
        270: (1, 0),
        315: (1, 1),
    }
    if dm in mapping:
        return mapping[dm]
    # Default to 0° if not at 45° increment
    return (0, 1)


@dataclass
class Rules:
    line_width: int = 1
    clearance: int = 1
    via_cost: float = 3.0
    corner_cost: float = 0.2
    step_cost: float = 0.01
    violation_cost: float = 1.0
    success_reward: float = 10.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Rules:
        kwargs: dict[str, Any] = {}
        for f in (
            "line_width",
            "clearance",
            "via_cost",
            "corner_cost",
            "step_cost",
            "violation_cost",
            "success_reward",
        ):
            if f in data:
                kwargs[f] = data[f]
        return cls(**kwargs)


class PcbRouteEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str | None]]] = {"render_modes": ["human", None]}

    def __init__(
        self,
        height: int,
        width: int,
        start: tuple[int, int],
        goal_bbox: tuple[int, int, int, int],  # (y0, x0, y1, x1) inclusive
        obstacles: np.ndarray | None,
        rules: Rules,
        num_layers: int = 2,
        via_budget: int = 8,
        via_cooldown_steps: int = 1,
        max_steps: int | None = None,
        max_step_len: int = 5,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.height = int(height)
        self.width = int(width)
        self.start = (int(start[0]), int(start[1]))
        self.goal_bbox = goal_bbox
        self.rules = rules
        self.num_layers = int(num_layers)
        self.via_budget_init = int(via_budget)
        self.via_cooldown_init = int(via_cooldown_steps)
        self.max_steps = int(max_steps) if max_steps is not None else height * width
        self.max_step_len = max(1, int(max_step_len))
        self.render_mode = render_mode

        # Grid occupancy: (layers, H, W) boolean array
        if obstacles is None:
            base_occ = np.zeros((self.num_layers, self.height, self.width), dtype=bool)
        else:
            assert obstacles.shape == (self.height, self.width)
            base_occ = np.repeat(obstacles[None, :, :], self.num_layers, axis=0)
        # Keep a pristine copy of static obstacles to rebuild occupancy on reset
        self._static_occ = base_occ.copy()
        self.occ = base_occ.copy()
        # Track sites where a via has been created to avoid double-charging budget
        self.via_sites = np.zeros((self.height, self.width), dtype=bool)

        # Gym spaces (for training alignment). Note: step accepts tuple form
        self.action_space = spaces.Dict(
            {
                "d_heading": spaces.Discrete(len(HEADING_DEGREES)),
                "step_len": spaces.Discrete(
                    self.max_step_len
                ),  # 0..max_step_len-1 maps to 1..max_step_len
                "layer_change": spaces.Discrete(
                    self.num_layers + 1
                ),  # 0 means no change, 1..L target layer
                "commit": spaces.Discrete(2),
            }
        )
        # Simplified observation space: only core scalars, grid view handled by renderer
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
        high = np.array(
            [
                self.height - 1,
                self.width - 1,
                360,
                self.num_layers - 1,
                self.via_budget_init,
                self.via_cooldown_init,
            ],
            dtype=np.int32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self._pygame: Any | None = None  # Lazy import

        self.reset(seed=seed)

    # ------- Public properties for human viewer -------
    @property
    def pos(self) -> tuple[int, int]:
        return (self.y, self.x)

    @property
    def success(self) -> bool:
        return bool(self._success)

    # ------- Core logic -------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Rebuild occupancy: keep static obstacles, drop previous trajectories
        self.occ = self._static_occ.copy()
        self.y, self.x = self.start
        self.heading = 0  # Initial heading is right
        self.layer = 0
        self.steps = 0
        self.via_budget = self.via_budget_init
        self.via_cooldown = 0
        self._success = False
        self._terminated = False
        self._truncated = False
        # Path occupancy: Mark starting point
        self.occ[self.layer, self.y, self.x] = True
        # Reset via sites
        self.via_sites[:, :] = False
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: Mapping[str, int] | tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, tuple):
            d_idx, step_len_raw, layer_sel, commit = action
            action = {
                "d_heading": d_idx,
                "step_len": step_len_raw,
                "layer_change": layer_sel,
                "commit": commit,
            }
        else:
            # Compatible with gym's ndarray/int
            action = {
                "d_heading": int(action["d_heading"]),
                "step_len": int(action["step_len"]),
                "layer_change": int(action["layer_change"]),
                "commit": int(action["commit"]),
            }

        if self._terminated or self._truncated:
            return self._get_obs(), 0.0, self._terminated, self._truncated, self._get_info()

        reward = 0.0
        self.steps += 1

        # Masks and legality
        masks = self.get_action_masks()

        # Rotation
        d_idx = int(action["d_heading"]) % len(HEADING_DEGREES)
        d_deg = HEADING_DEGREES[d_idx]
        new_heading = (self.heading + d_deg) % 360
        turned = d_deg != 0

        # Layer change (via), 0 means no change
        layer_sel = int(action["layer_change"])
        if layer_sel != 0 and masks["layer_change"][layer_sel] == 1:
            target_layer = layer_sel - 1
            # Only charge budget when creating a new via at this (y,x)
            if target_layer != self.layer:
                creating_new_via = not bool(self.via_sites[self.y, self.x])
                self.layer = target_layer
                if creating_new_via:
                    self.via_sites[self.y, self.x] = True
                    self.via_budget -= 1
                    self.via_cooldown = self.via_cooldown_init
                    reward -= float(self.rules.via_cost)

        # Update heading before movement (rotation before movement)
        self.heading = new_heading

        # Forward step length: action space encoded as 0..max-1 → actual 1..max
        step_len = int(action["step_len"]) + 1
        if step_len > self.max_step_len:
            step_len = self.max_step_len

        step_mask = masks["step_len"]
        if step_len >= len(step_mask) or step_mask[step_len] == 0:
            # Illegal move, apply violation cost, stay in place
            reward -= float(self.rules.violation_cost)
        else:
            dy, dx = _deg_to_step(self.heading)
            # Step by step, check for goal snapping
            for _k in range(step_len):
                ny = self.y + dy
                nx = self.x + dx
                # Goal snapping: if entering bounding box, snap to center and terminate
                if self._in_goal(ny, nx):
                    gyc = (self.goal_bbox[0] + self.goal_bbox[2]) // 2
                    gxc = (self.goal_bbox[1] + self.goal_bbox[3]) // 2
                    self.y, self.x = int(gyc), int(gxc)
                    self.occ[self.layer, self.y, self.x] = True
                    self._success = True
                    reward += float(self.rules.success_reward)
                    break

                # DRC check during movement (includes OOB)
                if not self._is_cell_legal(ny, nx, self.layer):
                    reward -= float(self.rules.violation_cost)
                    break

                self.y, self.x = ny, nx
                self.occ[self.layer, self.y, self.x] = True

            # Step length cost
            reward -= float(self.rules.step_cost) * float(step_len)

        # Turn cost
        if turned:
            reward -= float(self.rules.corner_cost)

        # Cooldown decay
        if self.via_cooldown > 0:
            self.via_cooldown -= 1

        # Truncation/termination
        if self._success:
            self._terminated = True
        if self.steps >= self.max_steps:
            self._truncated = True

        # commit: only allowed when on target
        if int(action["commit"]) == 1 and masks["commit"][1] == 1:
            self._terminated = True

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), self._terminated, self._truncated, info

    # ------- Mask calculation -------
    def get_action_masks(self) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {}
        # d_heading: all allowed (all legal), encoded 0..4
        masks["d_heading"] = np.ones((len(HEADING_DEGREES),), dtype=np.int8)

        # step_len: 0 is placeholder (unused), we return 0..max mapping: index == actual step length
        step_mask = np.zeros((self.max_step_len + 1,), dtype=np.int8)
        step_mask[0] = 0
        for L in range(1, self.max_step_len + 1):
            if self._can_move_L(L):
                step_mask[L] = 1
        masks["step_len"] = step_mask

        # layer_change: 0 means no change; if cooling down or budget is 0, all 0 (except index 0)
        layer_mask = np.zeros((self.num_layers + 1,), dtype=np.int8)
        layer_mask[0] = 1  # always allow no-op
        if self.via_cooldown == 0 and self.via_budget > 0:
            for li in range(self.num_layers):
                if li != self.layer:
                    layer_mask[li + 1] = 1
        masks["layer_change"] = layer_mask

        # commit: only 1 when in goal box
        commit_mask = np.zeros((2,), dtype=np.int8)
        if self._in_goal(self.y, self.x):
            commit_mask[1] = 1
        masks["commit"] = commit_mask
        return masks

    # ------- Utilities -------
    def _in_goal(self, y: int, x: int) -> bool:
        y0, x0, y1, x1 = self.goal_bbox
        return (y0 <= y <= y1) and (x0 <= x <= x1)

    def _is_oob(self, y: int, x: int) -> bool:
        return not (0 <= y < self.height and 0 <= x < self.width)

    def _clearance_ok(self, y: int, x: int, layer: int) -> bool:
        # Simplified: check existing occupancy within Manhattan/chess distance clearance
        cl = int(self.rules.clearance)
        y0 = max(0, y - cl)
        y1 = min(self.height - 1, y + cl)
        x0 = max(0, x - cl)
        x1 = min(self.width - 1, x + cl)
        region = self.occ[layer, y0 : y1 + 1, x0 : x1 + 1]
        return not bool(region.any())

    def _is_cell_legal(self, y: int, x: int, layer: int) -> bool:
        if self._is_oob(y, x):
            return False
        # Line width occupancy: expand by line width radius, simplified and merged with clearance
        lw = int(self.rules.line_width)
        # Relax to larger clearance radius when line width > 1
        saved_clearance = int(self.rules.clearance)
        tmp_clearance = max(saved_clearance, lw - 1)

        # Quick local check (ignore current cell occupancy to allow forward extension)
        cl = tmp_clearance
        y0 = max(0, y - cl)
        y1 = min(self.height - 1, y + cl)
        x0 = max(0, x - cl)
        x1 = min(self.width - 1, x + cl)
        region = self.occ[layer, y0 : y1 + 1, x0 : x1 + 1].copy()
        # Clear current position if it lies within the checked region
        if y0 <= self.y <= y1 and x0 <= self.x <= x1:
            region[self.y - y0, self.x - x0] = False
        return not bool(region.any())

    def _can_move_L(self, L: int) -> bool:
        # Predict if moving L steps with current heading is legal
        heading = self.heading
        dy, dx = _deg_to_step(heading)
        y, x = self.y, self.x
        for _ in range(L):
            y += dy
            x += dx
            if self._in_goal(y, x):
                return True
            if not self._is_cell_legal(y, x, self.layer):
                return False
        return True

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.y, self.x, self.heading, self.layer, self.via_budget, self.via_cooldown],
            dtype=np.int32,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "pos": (self.y, self.x),
            "layer": self.layer,
            "via_budget": self.via_budget,
            "via_cooldown": self.via_cooldown,
            "success": self._success,
        }

    # ------- Rendering -------
    def render(self) -> None:
        if self.render_mode != "human":
            return
        try:
            if self._pygame is None:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pygame
                self._pygame = pygame
                self._pg_screen = None
                self._pg_cell = 6
                self._pg_margin = 8
                self._pg_gap = 12
            assert self._pygame is not None
            pygame = self._pygame
            # Compute near-square layout for panels
            import math

            cols = math.ceil(math.sqrt(self.num_layers))
            rows = math.ceil(self.num_layers / float(cols))

            panel_w = self._pg_margin * 2 + self.width * self._pg_cell
            panel_h = self._pg_margin * 2 + self.height * self._pg_cell
            total_w = cols * panel_w + (cols - 1) * self._pg_gap
            total_h = rows * panel_h + (rows - 1) * self._pg_gap

            if (
                self._pg_screen is None
                or self._pg_screen.get_width() != total_w
                or self._pg_screen.get_height() != total_h
            ):
                self._pg_screen = pygame.display.set_mode((total_w, total_h))
                pygame.display.set_caption("PCB Router Viewer — Multi-layer")

            screen = self._pg_screen
            screen.fill((20, 20, 20))

            # Per-layer color function (distinct hues)
            def _layer_rgb(idx: int) -> tuple[int, int, int]:
                import colorsys

                h = (idx % max(1, self.num_layers)) / float(max(1, self.num_layers))
                s = 0.65
                v = 0.9
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                return int(r * 255), int(g * 255), int(b * 255)

            # Draw panels
            for li in range(self.num_layers):
                row = li // cols
                col = li % cols
                ox = col * (panel_w + self._pg_gap)
                oy = row * (panel_h + self._pg_gap)

                # Create panel and semi-transparent overlay
                panel = pygame.Surface((panel_w, panel_h))
                panel.fill((24, 24, 28))
                overlay = pygame.Surface((panel_w, panel_h), flags=pygame.SRCALPHA)

                # Other layers (semi-transparent)
                for lj in range(self.num_layers):
                    if lj == li:
                        continue
                    cr, cg, cb = _layer_rgb(lj)
                    ys, xs = np.where(self.occ[lj])
                    for yy, xx in zip(ys.tolist(), xs.tolist()):
                        rx = self._pg_margin + xx * self._pg_cell
                        ry = self._pg_margin + yy * self._pg_cell
                        pygame.draw.rect(
                            overlay, (cr, cg, cb, 60), (rx, ry, self._pg_cell, self._pg_cell)
                        )

                # Current layer (opaque)
                cr, cg, cb = _layer_rgb(li)
                ys, xs = np.where(self.occ[li])
                for yy, xx in zip(ys.tolist(), xs.tolist()):
                    rx = self._pg_margin + xx * self._pg_cell
                    ry = self._pg_margin + yy * self._pg_cell
                    pygame.draw.rect(panel, (cr, cg, cb), (rx, ry, self._pg_cell, self._pg_cell))

                # Goal bbox
                y0, x0, y1, x1 = self.goal_bbox
                x0p = self._pg_margin + x0 * self._pg_cell
                y0p = self._pg_margin + y0 * self._pg_cell
                x1p = self._pg_margin + (x1 + 1) * self._pg_cell
                y1p = self._pg_margin + (y1 + 1) * self._pg_cell
                pygame.draw.rect(panel, (255, 165, 0), (x0p, y0p, x1p - x0p, y1p - y0p), 2)

                # Start/current markers
                cy = self._pg_margin + self.y * self._pg_cell
                cx = self._pg_margin + self.x * self._pg_cell
                pygame.draw.circle(panel, (235, 235, 235), (cx, cy), 3)

                # Composite overlay onto panel
                panel.blit(overlay, (0, 0))

                # Highlight active layer
                if li == self.layer:
                    pygame.draw.rect(panel, (cr, cg, cb), (1, 1, panel_w - 2, panel_h - 2), 2)
                else:
                    pygame.draw.rect(panel, (60, 60, 70), (1, 1, panel_w - 2, panel_h - 2), 1)

                # Blit to main screen
                screen.blit(panel, (ox, oy))

            pygame.display.flip()
        except Exception:
            # Silently fail in headless environment: viewer will catch and notify
            return

    def close(self) -> None:
        if self._pygame is not None:
            from contextlib import suppress

            with suppress(Exception):
                self._pygame.display.quit()
            self._pygame = None
