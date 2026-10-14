#!/usr/bin/env python
"""Reaction-diffusion: the Gray-Scott model growing coral, mazes, holes and
dividing cells.

Two virtual chemicals spread across a wrap-around board and react; depending
on the feed and kill rates they form strikingly organic patterns. The display
drifts between rate presets every PRESET_SECONDS, so one pattern morphs into
the next. Within a preset a slow wave of higher kill rate rolls across the
board, so the pattern keeps re-forming as it passes, and every few seconds a
seed drops in or a patch is wiped for the pattern to regrow into.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from matrix_common import gradient, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps
STEPS_PER_FRAME = 6
WARMUP_STEPS = 2500  # grow a pattern before the first frame

TAU = 2 * math.pi

# Diffusion rates of the two chemicals (dt = 1). Half of Karl Sims' usual
# 1.0/0.5, which shrinks features to ~5 px so several fit on a 64x32 panel.
DIFFUSE_U = 0.5
DIFFUSE_V = 0.25

# (name, feed, kill, kill drift, palette). The kill rate swings between kill
# and kill + drift as the wave passes: coral's worms bud into spots and back,
# mitosis cells die back and divide to refill, the maze breaks into spots.
# Lower kill rates fill in to a flat wash, so the wave only raises them. All
# checked to form a pattern at these diffusion rates across the whole swing;
# chaos keeps churning on its own.
PRESETS = [
    ("coral", 0.0545, 0.062, 0.006, [(0, 0, 0), (0, 30, 70), (0, 130, 170), (80, 240, 230), (220, 255, 255)]),
    ("mitosis", 0.0367, 0.0649, 0.006, [(0, 0, 0), (50, 0, 60), (170, 0, 140), (255, 60, 120), (255, 200, 200)]),
    ("maze", 0.029, 0.057, 0.004, [(0, 0, 0), (40, 10, 0), (170, 50, 0), (255, 150, 0), (255, 240, 160)]),
    ("chaos", 0.018, 0.051, 0.0, [(0, 0, 0), (0, 40, 10), (20, 150, 40), (150, 255, 60), (240, 255, 200)]),
    ("holes", 0.039, 0.058, 0.002, [(0, 0, 0), (20, 0, 80), (90, 30, 220), (200, 120, 255), (255, 230, 255)]),
]

PRESET_SECONDS = 75.0
MORPH_SECONDS = 15.0
# Seconds for the kill-rate wave to roll once across the board (one
# wavelength wide), right to left
DRIFT_SECONDS = 40.0
# Seconds between disturbances: a new seed, or a wiped patch that regrows
DISTURB_EVERY = (5.0, 10.0)

# V levels mapped onto the palette: below V_LOW is background, V_HIGH is full.
# Colors are looked up by V itself, in steps of 1 / LUT_STEPS.
V_LOW, V_HIGH = 0.05, 0.42
LUT_STEPS = 1024
LUT_V = np.arange(int(LUT_STEPS * V_HIGH) + 1, dtype=np.float32) / LUT_STEPS

# A morph floods the next palette in from the patterns' edges (low V) to
# their cores behind a soft edge this wide in V, rather than blending whole
# palettes, which goes muddy halfway between unrelated ones
FLOOD_EDGE = 0.1


def color_lut(stops: list[tuple[int, int, int]]) -> np.ndarray:
    """Palette colors indexed by int(V * LUT_STEPS), as uint8 (len(LUT_V), 3)."""
    palette = np.array(gradient(stops), dtype=np.uint8)
    return palette[np.clip((LUT_V - V_LOW) * (255 / (V_HIGH - V_LOW)), 0, 255).astype(np.intp)]


COLOR_LUTS = [color_lut(stops) for *_, stops in PRESETS]


class ReactionDiffusion:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # On the Pi each numpy call costs ~15 us however small its array, so
        # a step is a few whole-buffer calls on preallocated arrays rather
        # than np.roll and temporaries (13 ms a frame down to under 2).
        #
        # u and v are the two rows of cells, each the board flattened
        # row-major between borders of width + 1 cells that every step
        # refills from the other end of the row. Every neighbor is then a
        # fixed offset (±1, ±width, ±width ± 1) along one contiguous buffer.
        # Off the right edge comes back on the left one row down, which on
        # a wrap-around board looks no different.
        n = width * height
        pad = width + 1
        self.n, self.pad = n, pad
        self.cells = np.zeros((2, n + 2 * pad), dtype=np.float32)
        self.u = self.cells[0, pad:pad + n]
        self.v = self.cells[1, pad:pad + n]
        self.u[:] = 1

        # The stencil runs over both rows in one pass, from u's first cell to
        # v's last. What it computes in the borders between them is junk that
        # the next refill overwrites.
        flat = self.cells.reshape(-1)
        span = flat.size - 2 * pad
        self.center = flat[pad:pad + span]
        self.left = flat[pad - 1:pad - 1 + span]
        self.right = flat[pad + 1:pad + 1 + span]
        # One cell wider at each end, for the diagonals
        self.above = flat[pad - width - 1:pad - width + 1 + span]
        self.below = flat[pad + width - 1:pad + width + 1 + span]
        self.vertical = np.empty(span + 2, dtype=np.float32)
        self.near = np.empty(span, dtype=np.float32)
        self.diagonal = np.empty(span, dtype=np.float32)
        self.uvv = np.empty(n, dtype=np.float32)

        # Per-cell factors along the span: the Laplacian's side and diagonal
        # weights times each chemical's diffusion rate, and how much of its
        # own value each cell keeps (set each frame by set_rates)
        def per_chemical(for_u: float, for_v: float) -> np.ndarray:
            rows = np.empty_like(self.cells)
            rows[0], rows[1] = for_u, for_v
            return rows.reshape(-1)[pad:pad + span]

        self.side = per_chemical(0.2 * DIFFUSE_U, 0.2 * DIFFUSE_V)
        self.corner = per_chemical(0.05 * DIFFUSE_U, 0.05 * DIFFUSE_V)
        self.keep_rows = np.empty_like(self.cells)
        self.keep = self.keep_rows.reshape(-1)[pad:pad + span]
        # An array, as the Pi adds one faster than it adds a Python float
        self.feed = np.empty(n, dtype=np.float32)
        # Each cell's column, for spreading the kill-rate wave along a row
        self.column = (np.arange(n + 2 * pad) - pad) % width
        self.wave_x = np.arange(width) * (TAU / width)

        self.order = random.sample(range(len(PRESETS)), len(PRESETS))
        self.frame_count = 0
        self.next_disturb = 0
        for _ in range(4):
            self.seed()
        self.set_rates()
        for _ in range(WARMUP_STEPS):
            self.step()

    def disk(self, radius: float) -> np.ndarray:
        """Cell indexes of a disk at a random spot, wrapping like the board."""
        cy, cx = random.randrange(self.height), random.randrange(self.width)
        r = int(radius)
        dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
        inside = dx * dx + dy * dy <= radius * radius
        return ((cy + dy[inside]) * self.width + cx + dx[inside]) % self.n

    def seed(self) -> None:
        cells = self.disk(random.uniform(2.5, 4.0))  # smaller seeds die out
        self.u[cells] = 0.5
        self.v[cells] = 0.25 + np.random.uniform(0, 0.05, len(cells))

    def wipe(self) -> None:
        cells = self.disk(random.uniform(3.0, 6.0))
        self.u[cells] = 1.0
        self.v[cells] = 0.0

    def step(self) -> None:
        """One Gray-Scott step:
            u += DIFFUSE_U * laplacian(u) - u * v * v + feed * (1 - u)
            v += DIFFUSE_V * laplacian(v) + u * v * v - (feed + kill) * v
        with a 3x3 Laplacian of 0.2 per side neighbor, 0.05 per diagonal and
        -1 for the center. keep folds in each cell's own terms: 1 - DIFFUSE_U
        - feed for u, 1 - DIFFUSE_V - feed - kill for v."""
        cells, n, pad = self.cells, self.n, self.pad
        cells[:, :pad] = cells[:, n:n + pad]
        cells[:, n + pad:] = cells[:, pad:2 * pad]

        vertical, near, diagonal = self.vertical, self.near, self.diagonal
        np.add(self.above, self.below, out=vertical)
        np.add(vertical[1:-1], self.left, out=near)
        near += self.right
        np.add(vertical[:-2], vertical[2:], out=diagonal)
        near *= self.side
        diagonal *= self.corner
        near += diagonal

        u, v, uvv = self.u, self.v, self.uvv
        np.multiply(u, v, out=uvv)
        uvv *= v
        self.center *= self.keep
        self.center += near
        u -= uvv
        u += self.feed
        v += uvv

    def preset_position(self) -> tuple[int, int, float]:
        """(current preset, next preset, how far morphed toward next 0–1)."""
        seconds = self.frame_count * FRAME_SECONDS
        n, into = divmod(seconds, PRESET_SECONDS)
        n = int(n)
        current = self.order[n % len(self.order)]
        following = self.order[(n + 1) % len(self.order)]
        f = max(0.0, (into - (PRESET_SECONDS - MORPH_SECONDS)) / MORPH_SECONDS)
        return current, following, f * f * (3 - 2 * f)

    def set_rates(self) -> None:
        """Set feed and keep for this frame's rates and kill-rate wave."""
        current, following, f = self.preset_position()
        _, feed_a, kill_a, drift_a, _ = PRESETS[current]
        _, feed_b, kill_b, drift_b, _ = PRESETS[following]
        feed = feed_a + (feed_b - feed_a) * f
        kill = kill_a + (kill_b - kill_a) * f
        drift = drift_a + (drift_b - drift_a) * f
        # The wave's phase is wrapped in float64 before it meets the float32
        # cells (see matrix_common's Scenes notes)
        phase = (self.frame_count * FRAME_SECONDS / DRIFT_SECONDS) % 1 * TAU
        wave = (0.5 + 0.5 * np.sin(self.wave_x + phase)) * drift
        self.feed[:] = feed
        self.keep_rows[0] = 1 - DIFFUSE_U - feed
        np.subtract(1 - DIFFUSE_V - feed - kill, wave.take(self.column), out=self.keep_rows[1])

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

        self.set_rates()
        for _ in range(STEPS_PER_FRAME):
            self.step()

        current, following, f = self.preset_position()
        lut = COLOR_LUTS[current]
        if f > 0:
            level = V_LOW - FLOOD_EDGE / 2 + f * (V_HIGH - V_LOW + FLOOD_EDGE)
            w = np.clip((level - LUT_V) / FLOOD_EDGE + 0.5, 0, 1)[:, np.newaxis]
            lut = (lut + (COLOR_LUTS[following] - lut.astype(np.float32)) * w).astype(np.uint8)
        # mode="clip" gives V past V_HIGH the top color (and any V below 0
        # the background)
        return lut.take((self.v * LUT_STEPS).astype(np.intp), axis=0, mode="clip").reshape(-1)


SCENE = ReactionDiffusion


def main() -> None:
    run_scene(ReactionDiffusion, "Reaction-diffusion LED matrix display")


if __name__ == "__main__":
    main()
