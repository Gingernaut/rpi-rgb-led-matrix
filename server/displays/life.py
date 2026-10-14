#!/usr/bin/env python
"""Game of Life with two competing species and glowing trails.

Each board seeds one small patch of random cells per species on a
wrap-around board, and the two grow into each other. Standard Life rules
count all live neighbors regardless of species; a newborn cell takes the
majority species of its three parents. Newborns fade in over their
generation and dead cells fade out, so the board flows rather than stepping.
When the board dies down, settles into a repeating loop, or runs too long,
it fades out and a new one fades in with new colors.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random

import numpy as np

from matrix_common import fade_lut, hsv, run_scene

# One frame per panel refresh (~75 Hz), so fades are as smooth as the panel
# allows. Slightly shorter than a refresh, so the swap sets the pace: at
# exactly 1/75 s the schedule and the refresh drift apart and about 1% of
# refreshes repeat a frame, a visible hitch.
FRAME_SECONDS = 1 / 76
FRAMES_PER_GENERATION = 5  # ~15 generations per second

# Each species starts as an elliptical patch of random cells with radii
# PATCH_RADII (x, y) before a random size jitter, the two centers
# PATCH_SPACING of the board width apart, so they nearly touch. Together they
# cover ~19% of a 64x32 board. (Patches further apart or smaller often settle
# down before the species ever meet.)
PATCH_RADII = (9.0, 7.0)
PATCH_SPACING = 0.3
SEED_DENSITY = 0.4

# Brightness kept per frame by cells that died (or a board fading out), and
# the level below which a channel goes black: very dim LEDs look muddy
TRAIL_FADE = 0.93
TRAIL_CUTOFF = 8

# When to start over: fewer live cells than this, a repeating pattern (still
# lifes, oscillators, a lone glider on the torus), or MAX_GENERATIONS
MIN_POPULATION = 12
MAX_GENERATIONS = 3000
# Generations to keep showing a looping board before fading out, and frames
# to fade the old board out and the new one in
STUCK_LINGER = 30
FADE_OUT_FRAMES = 60
FADE_IN_FRAMES = 40

# Neighbors are counted for both species at once: species 1 is 1 and
# species 2 is 16, so a 3x3 sum (at most 9 * 16) fits a byte with each
# species' count in its own 4 bits. These tables turn that packed sum into
# the total count and the majority species of three parents.
SPECIES_BITS = np.array([0, 1, 16], dtype=np.uint8)
_packed = np.arange(256)
NEIGHBORS = ((_packed & 15) + (_packed >> 4)).astype(np.uint8)
MAJORITY = np.where(_packed >> 4 >= 2, 2, 1).astype(np.uint8)

TRAIL_LUT = np.frombuffer(fade_lut(TRAIL_FADE, TRAIL_CUTOFF), dtype=np.uint8)
# Newborn brightness on each frame of their generation, reaching full at the end
NEWBORN_RAMP = [(i + 1) / FRAMES_PER_GENERATION for i in range(FRAMES_PER_GENERATION)]


def species_palette() -> np.ndarray:
    """Two well-separated vivid colors, as (3, 3) uint8 indexed by species
    (0 is empty: black)."""
    base = random.random()
    hues = (base, base + random.uniform(0.35, 0.65))
    return np.array([(0, 0, 0)] + [hsv(h, random.uniform(0.85, 1.0)) for h in hues], dtype=np.uint8)


class Life:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.glow = np.zeros((height, width, 3), dtype=np.uint8)
        self.reseed()

    def reseed(self) -> None:
        h, w = self.height, self.width
        self.palette = species_palette()

        # A patch per species side by side, so they grow into each other;
        # distances are measured around the wrap-around edges
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = random.uniform(0, w), random.uniform(0, h)
        centers = [(cx, cy), (cx + w * PATCH_SPACING, cy + random.uniform(-h / 4, h / 4))]
        self.grid = np.zeros((h, w), dtype=np.uint8)
        for species, (px, py) in enumerate(centers, start=1):
            rx, ry = (r * random.uniform(0.8, 1.2) for r in PATCH_RADII)
            dx = np.minimum((x - px) % w, (px - x) % w)
            dy = np.minimum((y - py) % h, (py - y) % h)
            patch = (dx / rx) ** 2 + (dy / ry) ** 2 <= 1
            self.grid[patch & (np.random.random((h, w)) < SEED_DENSITY)] = species
        self.draw_layers(self.grid > 0, np.zeros((h, w), dtype=bool))

        self.generation = 0
        # Hashes of every live pattern this board has shown. Life rules ignore
        # species, so once a pattern repeats the board loops forever.
        self.seen: set[int] = set()
        self.looping_since: int | None = None
        self.fading_in = FADE_IN_FRAMES
        self.fading_out = 0
        self.frame_count = 0  # frames this board has been evolving

    def draw_layers(self, survivors: np.ndarray, born: np.ndarray) -> None:
        """Precompute this generation's frames: survivors at full color with
        newborns ramping up, and where the old glow may show through."""
        solid = np.take(self.palette, self.grid * survivors, axis=0)
        newborn = np.take(self.palette, self.grid * born, axis=0)
        self.layers = [np.maximum(solid, (newborn * f).astype(np.uint8)) for f in NEWBORN_RAMP]
        # 1 per channel where the trail shows (everywhere but survivors);
        # newborns light up over any trail already there
        self.trail_mask = np.repeat(~survivors[..., None], 3, axis=2).view(np.uint8)

    def step(self) -> None:
        # Sum each cell's 3x3 block (rows, then columns), less the cell itself
        cells = SPECIES_BITS[self.grid]
        rows = cells + np.roll(cells, 1, axis=1) + np.roll(cells, -1, axis=1)
        packed = rows + np.roll(rows, 1, axis=0) + np.roll(rows, -1, axis=0) - cells
        total = NEIGHBORS[packed]

        alive = self.grid > 0
        survivors = alive & ((total == 2) | (total == 3))
        born = ~alive & (total == 3)
        # Newborns take the majority species of their three parents
        self.grid = np.where(born, MAJORITY[packed], self.grid * survivors)
        self.generation += 1
        self.draw_layers(survivors, born)

        state = hash((survivors | born).tobytes())
        if self.looping_since is None and state in self.seen:
            self.looping_since = self.generation
        self.seen.add(state)

    def finished(self) -> bool:
        return (
            (self.looping_since is not None and self.generation - self.looping_since > STUCK_LINGER)
            or self.generation > MAX_GENERATIONS
            or np.count_nonzero(self.grid) < MIN_POPULATION
        )

    def frame(self) -> np.ndarray:
        if self.fading_in:
            # New board brightening from black; it starts evolving after
            self.fading_in -= 1
            f = 1 - self.fading_in / FADE_IN_FRAMES
            self.glow = (self.layers[-1] * (f * f)).astype(np.uint8)
            return self.glow.reshape(-1)

        self.glow = TRAIL_LUT[self.glow]
        if self.fading_out:
            # Old board fading away; nothing new is drawn
            self.fading_out -= 1
            if not self.fading_out:
                self.reseed()
            return self.glow.reshape(-1)

        phase = self.frame_count % FRAMES_PER_GENERATION
        self.frame_count += 1
        if phase == 0:
            self.step()
            if self.finished():
                self.fading_out = FADE_OUT_FRAMES
                return self.glow.reshape(-1)
        self.glow = np.maximum(self.glow * self.trail_mask, self.layers[phase])
        return self.glow.reshape(-1)


SCENE = Life


def main() -> None:
    run_scene(Life, "Game of Life LED matrix display")


if __name__ == "__main__":
    main()
