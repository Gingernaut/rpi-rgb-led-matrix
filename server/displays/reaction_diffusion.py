#!/usr/bin/env python
"""Reaction-diffusion: the Gray-Scott model growing coral, mazes, spots and
dividing cells.

Two virtual chemicals spread across a wrap-around board and react; depending
on the feed and kill rates they settle into strikingly organic patterns. The
display drifts between rate presets every PRESET_SECONDS, so one pattern
morphs into the next, and every few seconds it drops in a seed or wipes a
patch for the pattern to regrow into, so it never freezes.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random

import numpy as np

from matrix_common import gradient, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps
STEPS_PER_FRAME = 6
WARMUP_STEPS = 2500  # grow a pattern before the first frame

# Diffusion rates of the two chemicals (dt = 1). Half of Karl Sims' usual
# 1.0/0.5, which shrinks features to ~5 px so several fit on a 64x32 panel.
DIFFUSE_U = 0.5
DIFFUSE_V = 0.25

# (name, feed, kill, palette). All checked to form a pattern at these
# diffusion rates; chaos keeps churning, the rest settle and rely on the
# disturbances (and morphs) for motion.
PRESETS = [
    ("coral", 0.0545, 0.062, [(0, 0, 0), (0, 30, 70), (0, 130, 170), (80, 240, 230), (220, 255, 255)]),
    ("mitosis", 0.0367, 0.0649, [(0, 0, 0), (50, 0, 60), (170, 0, 140), (255, 60, 120), (255, 200, 200)]),
    ("maze", 0.029, 0.057, [(0, 0, 0), (40, 10, 0), (170, 50, 0), (255, 150, 0), (255, 240, 160)]),
    ("chaos", 0.018, 0.051, [(0, 0, 0), (0, 40, 10), (20, 150, 40), (150, 255, 60), (240, 255, 200)]),
    ("spots", 0.03, 0.062, [(0, 0, 0), (20, 0, 80), (90, 30, 220), (200, 120, 255), (255, 230, 255)]),
]
PALETTES = [np.array(gradient(stops), dtype=np.float32) for _, _, _, stops in PRESETS]

PRESET_SECONDS = 75.0
MORPH_SECONDS = 15.0
# Seconds between disturbances: a new seed, or a wiped patch that regrows
DISTURB_EVERY = (5.0, 10.0)

# V levels mapped onto the palette: below V_LOW is background, V_HIGH is full
V_LOW, V_HIGH = 0.05, 0.42


class ReactionDiffusion:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.u = np.ones((height, width), dtype=np.float32)
        self.v = np.zeros((height, width), dtype=np.float32)
        self.order = random.sample(range(len(PRESETS)), len(PRESETS))
        self.frame_count = 0
        self.next_disturb = 0
        for _ in range(4):
            self.seed()
        feed, kill = self.rates()
        for _ in range(WARMUP_STEPS):
            self.step(feed, kill)

    def disk(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """Index arrays for a disk at a random spot, wrapping around the edges."""
        cy, cx = random.randrange(self.height), random.randrange(self.width)
        r = int(radius)
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        inside = dx * dx + dy * dy <= radius * radius
        return (cy + dy[inside]) % self.height, (cx + dx[inside]) % self.width

    def seed(self) -> None:
        ys, xs = self.disk(random.uniform(2.5, 4.0))  # smaller seeds die out
        self.u[ys, xs] = 0.5
        self.v[ys, xs] = 0.25 + np.random.uniform(0, 0.05, len(ys))

    def wipe(self) -> None:
        ys, xs = self.disk(random.uniform(3.0, 6.0))
        self.u[ys, xs] = 1.0
        self.v[ys, xs] = 0.0

    def step(self, feed: float, kill: float) -> None:
        u, v = self.u, self.v
        lap_u = laplacian(u)
        lap_v = laplacian(v)
        uvv = u * v * v
        u += DIFFUSE_U * lap_u - uvv + feed * (1 - u)
        v += DIFFUSE_V * lap_v + uvv - (kill + feed) * v

    def preset_position(self) -> tuple[int, int, float]:
        """(current preset, next preset, how far morphed toward next 0–1)."""
        seconds = self.frame_count * FRAME_SECONDS
        n, into = divmod(seconds, PRESET_SECONDS)
        n = int(n)
        current = self.order[n % len(self.order)]
        following = self.order[(n + 1) % len(self.order)]
        f = max(0.0, (into - (PRESET_SECONDS - MORPH_SECONDS)) / MORPH_SECONDS)
        return current, following, f * f * (3 - 2 * f)

    def rates(self) -> tuple[float, float]:
        current, following, f = self.preset_position()
        _, feed_a, kill_a, _ = PRESETS[current]
        _, feed_b, kill_b, _ = PRESETS[following]
        return feed_a + (feed_b - feed_a) * f, kill_a + (kill_b - kill_a) * f

    def frame(self) -> np.ndarray:
        self.frame_count += 1
        if self.frame_count >= self.next_disturb:
            # Keep it alive: reseed if the pattern died out, else disturb it
            if self.v.max() < V_LOW:
                for _ in range(4):
                    self.seed()
            elif random.random() < 0.5:
                self.seed()
            else:
                self.wipe()
            self.next_disturb = self.frame_count + round(random.uniform(*DISTURB_EVERY) / FRAME_SECONDS)

        feed, kill = self.rates()
        for _ in range(STEPS_PER_FRAME):
            self.step(feed, kill)

        current, following, f = self.preset_position()
        palette = PALETTES[current] * (1 - f) + PALETTES[following] * f
        level = np.clip((self.v - V_LOW) * (255 / (V_HIGH - V_LOW)), 0, 255).astype(np.uint8)
        return palette.astype(np.uint8)[level].reshape(-1)


def laplacian(a: np.ndarray) -> np.ndarray:
    """3x3 Laplacian on a wrap-around grid: 0.2 per side neighbor, 0.05 per
    diagonal, -1 center."""
    vertical = np.roll(a, 1, 0) + np.roll(a, -1, 0)
    sides = np.roll(a, 1, 1) + np.roll(a, -1, 1)
    diagonals = np.roll(vertical, 1, 1) + np.roll(vertical, -1, 1)
    return 0.2 * (vertical + sides) + 0.05 * diagonals - a


SCENE = ReactionDiffusion


def main() -> None:
    run_scene(ReactionDiffusion, "Reaction-diffusion LED matrix display")


if __name__ == "__main__":
    main()
