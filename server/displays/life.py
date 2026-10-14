#!/usr/bin/env python
"""Game of Life with 2–3 competing species and glowing trails.

Each run seeds 2 or 3 species, each in its own territory, on a wrap-around
board. Standard Life rules count all live neighbors regardless of species; a
newborn cell takes the majority species of its three parents (a random one
when all three differ). Newborns flash pale, live cells glow in their species
color, and dead cells fade out slowly. When the board dies down, settles into
a repeating loop, or runs too long, it fades out and a new one fades in with
new colors.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random

import numpy as np

from matrix_common import fade_lut, hsv, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps; trails fade every frame
FRAMES_PER_GENERATION = 4  # ~9 generations per second

SEED_DENSITY = 0.35
# Brightness kept per frame by cells that died (or a board fading out).
# Channels that fade below TRAIL_CUTOFF go black: very dim LEDs look muddy.
TRAIL_FADE = 0.86
TRAIL_CUTOFF = 8
# Newborn cells start this far toward white, easing to their species color
NEWBORN_WHITE = 0.55
NEWBORN_GENERATIONS = 3
SHADES = NEWBORN_GENERATIONS + 1  # ages 0 to NEWBORN_GENERATIONS

# When to start over: fewer live cells than this, a repeating pattern (still
# lifes, oscillators, a lone glider on the torus), or MAX_GENERATIONS
MIN_POPULATION = 12
MAX_GENERATIONS = 2500
# Generations to keep showing a looping board before fading out, and frames
# to fade the old board out and the new one in
STUCK_LINGER = 40
FADE_OUT_FRAMES = 45
FADE_IN_FRAMES = 20

# Neighbors are counted for all species at once: species s is the bit
# 1 << 4 * (s - 1), so summing a cell's neighbors leaves each species' count
# (at most 8) in its own 4 bits. These tables turn that packed sum into the
# total count and the majority species (0 when no species has 2 or more).
SPECIES_BITS = np.array([0, 1, 1 << 4, 1 << 8], dtype=np.uint16)


def _count_tables() -> tuple[np.ndarray, np.ndarray]:
    packed = np.arange(0x889)  # three species with up to 8 neighbors each
    counts = np.stack([(packed >> 4 * s) & 15 for s in range(3)])
    majority = np.where(counts.max(axis=0) >= 2, counts.argmax(axis=0) + 1, 0)
    return counts.sum(axis=0).astype(np.uint8), majority.astype(np.uint8)


NEIGHBORS, MAJORITY = _count_tables()
TRAIL_LUT = np.frombuffer(fade_lut(TRAIL_FADE, TRAIL_CUTOFF), dtype=np.uint8)


def species_palette(count: int) -> np.ndarray:
    """count well-separated vivid colors, as (count + 1, SHADES, 3) uint8
    indexed by species (0 is empty: black) and age in generations (capped
    at NEWBORN_GENERATIONS), so newborns start pale."""
    base = random.random()
    hues = [base + i / count + random.uniform(-0.04, 0.04) for i in range(count)]
    colors = np.array([(0, 0, 0)] + [hsv(h, random.uniform(0.85, 1.0)) for h in hues], dtype=np.float32)
    white = NEWBORN_WHITE * (1 - np.arange(SHADES) / NEWBORN_GENERATIONS)
    palette = colors[:, None] + (255 - colors[:, None]) * white[:, None]
    palette[0] = 0
    return palette.astype(np.uint8)


class Life:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.glow = np.zeros((height, width, 3), dtype=np.uint8)
        self.reseed()

    def reseed(self) -> None:
        h, w = self.height, self.width
        self.species_count = random.choice((2, 3))
        # Flat, so a cell's color is row species * SHADES + age
        self.palette = species_palette(self.species_count).reshape(-1, 3)

        # Territories: each cell belongs to the species of its nearest random
        # center, measured around the wrap-around edges
        y, x = np.mgrid[0:h, 0:w]
        distances = []
        for _ in range(self.species_count):
            cy, cx = random.randrange(h), random.randrange(w)
            dy = np.minimum(abs(y - cy), h - abs(y - cy))
            dx = np.minimum(abs(x - cx), w - abs(x - cx))
            distances.append(dx * dx + dy * dy)
        territory = np.argmin(distances, axis=0).astype(np.uint8) + 1
        alive = np.random.random((h, w)) < SEED_DENSITY
        self.grid = territory * alive
        # The seed shows in its species colors, not as newborns
        self.age = np.full((h, w), NEWBORN_GENERATIONS, dtype=np.uint8)
        self.draw_layer()

        self.generation = 0
        # Hashes of every live pattern this board has shown. Life rules ignore
        # species, so once a pattern repeats the board loops forever.
        self.seen: set[int] = set()
        self.looping_since: int | None = None
        self.fading_in = FADE_IN_FRAMES
        self.fading_out = 0
        self.frame_count = 0  # frames this board has been evolving

    def draw_layer(self) -> None:
        """Color the live cells (black elsewhere), once per generation."""
        self.live = np.take(self.palette, self.grid * SHADES + self.age, axis=0)
        # 1 per channel of an empty cell, where the trail shows; 0 where live
        self.empty = np.repeat((self.grid == 0)[..., None], 3, axis=2).view(np.uint8)

    def step(self) -> None:
        # Sum each cell's 3x3 block (rows, then columns), less the cell itself
        cells = SPECIES_BITS[self.grid]
        rows = cells + np.roll(cells, 1, axis=1) + np.roll(cells, -1, axis=1)
        packed = rows + np.roll(rows, 1, axis=0) + np.roll(rows, -1, axis=0) - cells
        total = NEIGHBORS[packed]

        alive = self.grid > 0
        survives = alive & ((total == 2) | (total == 3))
        born = ~alive & (total == 3)

        # Newborns take the majority species of their three parents; with one
        # parent of each of three species, pick one at random
        newborn = MAJORITY[packed[born]]
        tie = newborn == 0
        if tie.any():
            newborn[tie] = np.random.randint(1, 4, size=int(tie.sum()))

        self.grid = self.grid * survives
        self.grid[born] = newborn
        self.age = np.where(survives, np.minimum(self.age + 1, NEWBORN_GENERATIONS), 0)
        self.generation += 1
        self.draw_layer()

        state = hash((survives | born).tobytes())
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
            self.glow = (self.live * (f * f)).astype(np.uint8)
            return self.glow.reshape(-1)

        self.glow = TRAIL_LUT[self.glow]
        if self.fading_out:
            # Old board fading away; nothing new is drawn
            self.fading_out -= 1
            if not self.fading_out:
                self.reseed()
        else:
            self.frame_count += 1
            if self.frame_count % FRAMES_PER_GENERATION == 0:
                self.step()
                if self.finished():
                    self.fading_out = FADE_OUT_FRAMES
            if not self.fading_out:
                self.glow = np.maximum(self.glow * self.empty, self.live)

        return self.glow.reshape(-1)


SCENE = Life


def main() -> None:
    run_scene(Life, "Game of Life LED matrix display")


if __name__ == "__main__":
    main()
