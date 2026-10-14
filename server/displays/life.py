#!/usr/bin/env python
"""Game of Life with 2–3 competing species and glowing trails.

Each run seeds 2 or 3 species, each in its own territory, on a wrap-around
board. Standard Life rules count all live neighbors regardless of species; a
newborn cell takes the majority species of its three parents (a random one
when all three differ). Newborns flash pale, live cells glow in their species
color, and dead cells fade out slowly. When the board dies down, settles into
a repeating loop, or runs too long, it fades out and reseeds with new colors.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random
from collections import deque

import numpy as np

from matrix_common import hsv, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps; trails fade every frame
FRAMES_PER_GENERATION = 4  # ~9 generations per second

SEED_DENSITY = 0.35
# Brightness kept per frame by cells that died (or a board fading out)
TRAIL_FADE = 0.86
# Newborn cells start this far toward white, easing to their species color
NEWBORN_WHITE = 0.55
NEWBORN_GENERATIONS = 3

# When to start over: fewer live cells than this, a board state repeating
# within HISTORY generations (still lifes, oscillators, a lone glider on the
# torus), or MAX_GENERATIONS
MIN_POPULATION = 12
HISTORY = 300
MAX_GENERATIONS = 2500
# Generations to keep showing a stuck board before fading out, and frames
# to fade out
STUCK_LINGER = 40
FADE_OUT_FRAMES = 45

NEIGHBOR_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


def species_colors(count: int) -> np.ndarray:
    """count well-separated vivid colors, as (count + 1, 3) float32 with a
    black row 0 for empty cells."""
    base = random.random()
    hues = [base + i / count + random.uniform(-0.04, 0.04) for i in range(count)]
    return np.array([(0, 0, 0)] + [hsv(h, random.uniform(0.85, 1.0)) for h in hues], dtype=np.float32)


class Life:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.glow = np.zeros((height, width, 3), dtype=np.float32)
        self.frame_count = 0
        self.reseed()

    def reseed(self) -> None:
        h, w = self.height, self.width
        self.species_count = random.choice((2, 3))
        self.colors = species_colors(self.species_count)

        # Territories: each cell belongs to the species of its nearest random
        # center, measured around the wrap-around edges
        y, x = np.mgrid[0:h, 0:w]
        distances = []
        for _ in range(self.species_count):
            cy, cx = random.randrange(h), random.randrange(w)
            dy = np.minimum(abs(y - cy), h - abs(y - cy))
            dx = np.minimum(abs(x - cx), w - abs(x - cx))
            distances.append(dx * dx + dy * dy)
        territory = np.argmin(distances, axis=0).astype(np.int8) + 1
        alive = np.random.random((h, w)) < SEED_DENSITY
        self.grid = np.where(alive, territory, 0).astype(np.int8)
        self.age = np.zeros((h, w), dtype=np.int16)

        self.generation = 0
        self.history: deque[bytes] = deque(maxlen=HISTORY)
        self.seen: set[bytes] = set()
        self.stuck_for = 0
        self.fading_out = 0

    def neighbor_counts(self) -> np.ndarray:
        """Live neighbors of each species, as (species, height, width)."""
        counts = np.zeros((self.species_count, self.height, self.width), dtype=np.int8)
        for s in range(self.species_count):
            cells = (self.grid == s + 1).astype(np.int8)
            for dy, dx in NEIGHBOR_OFFSETS:
                counts[s] += np.roll(cells, (dy, dx), axis=(0, 1))
        return counts

    def step(self) -> None:
        counts = self.neighbor_counts()
        total = counts.sum(axis=0)
        alive = self.grid > 0
        survives = alive & ((total == 2) | (total == 3))
        born = ~alive & (total == 3)

        # Majority species among the three parents; with three species a
        # three-way tie (one parent each) picks at random
        majority = counts.argmax(axis=0).astype(np.int8) + 1
        if self.species_count == 3:
            tie = born & (counts.max(axis=0) == 1)
            majority[tie] = np.random.randint(1, 4, size=int(tie.sum()))

        self.grid = np.where(survives, self.grid, np.where(born, majority, 0)).astype(np.int8)
        self.age = np.where(survives, self.age + 1, 0).astype(np.int16)
        self.generation += 1

        state = self.grid.tobytes()
        if state in self.seen:
            self.stuck_for += 1
        if len(self.history) == HISTORY:
            self.seen.discard(self.history[0])
        self.history.append(state)
        self.seen.add(state)

    def finished(self) -> bool:
        return (
            self.stuck_for > STUCK_LINGER
            or self.generation > MAX_GENERATIONS
            or np.count_nonzero(self.grid) < MIN_POPULATION
        )

    def frame(self) -> np.ndarray:
        self.frame_count += 1
        self.glow *= TRAIL_FADE

        if self.fading_out:
            # Old board fading away; nothing new is drawn
            self.fading_out -= 1
            if not self.fading_out:
                self.reseed()
        else:
            if self.frame_count % FRAMES_PER_GENERATION == 0:
                self.step()
                if self.finished():
                    self.fading_out = FADE_OUT_FRAMES
            if not self.fading_out:
                alive = self.grid > 0
                color = self.colors[self.grid[alive]]
                # Newborns start pale and settle into their species color
                white = np.clip(1 - self.age[alive] / NEWBORN_GENERATIONS, 0, 1)[:, None] * NEWBORN_WHITE
                self.glow[alive] = color + (255 - color) * white

        return self.glow.astype(np.uint8).reshape(-1)


SCENE = Life


def main() -> None:
    run_scene(Life, "Game of Life LED matrix display")


if __name__ == "__main__":
    main()
