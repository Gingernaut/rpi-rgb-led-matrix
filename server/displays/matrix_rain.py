#!/usr/bin/env python
"""Matrix rain: green code streams falling down the panel, as in The Matrix.

Each stream has a pale, flickering head and a green tail that fades behind it.
Streams fall at different speeds with different tail lengths, and tail cells
shimmer as their "glyphs" change.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random

import numpy as np

from matrix_common import gradient, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps

# Pixels per frame, and tail length in pixels (to ~3% brightness)
SPEED = (0.2, 0.75)
TAIL = (5, 22)
# Chance per frame that an empty column starts a stream
SPAWN_CHANCE = 0.015

# Fraction of tail cells whose glyph changes each frame, and how dim a glyph
# can be relative to its tail brightness
GLYPH_CHANGE = 0.04
GLYPH_MIN = 0.45

HEAD_COLOR = (200, 255, 210)

# Tail brightness to color: dark green through green to pale green at the top
PALETTE = np.array(
    gradient([(0, 0, 0), (0, 35, 8), (0, 110, 25), (20, 200, 50), (90, 255, 110), (170, 255, 180)]),
    dtype=np.uint8,
)


class Stream:
    __slots__ = ("x", "y", "speed", "decay")

    def __init__(self, x: int) -> None:
        self.x = x
        self.y = random.uniform(-3.0, 0.0)
        self.speed = random.uniform(*SPEED)
        tail = random.uniform(*TAIL)
        # Per-frame fade so the tail reaches ~3% about `tail` pixels behind
        self.decay = 0.03 ** (self.speed / tail)


class MatrixRain:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.energy = np.zeros((height, width), dtype=np.float32)
        self.glyphs = np.random.uniform(GLYPH_MIN, 1.0, (height, width)).astype(np.float32)
        self.decay = np.full(width, 0.9, dtype=np.float32)
        self.streams: dict[int, Stream] = {}
        # Start mid-rain rather than on an empty screen
        for _ in range(round(4 / FRAME_SECONDS)):
            self.step()

    def step(self) -> list[tuple[int, int]]:
        """Advance the rain one frame; returns the head cells to draw."""
        self.energy *= self.decay

        for x in range(self.width):
            if x not in self.streams and random.random() < SPAWN_CHANCE:
                stream = self.streams[x] = Stream(x)
                self.decay[x] = stream.decay

        heads = []
        for x, stream in list(self.streams.items()):
            top = int(stream.y)
            stream.y += stream.speed
            bottom = int(stream.y)
            # Light every cell the head passed this frame
            if bottom >= 0:
                self.energy[max(top, 0):min(bottom, self.height - 1) + 1, x] = 1.0
            if bottom < self.height:
                heads.append((x, bottom))
            elif self.energy[:, x].max() < 0.02:
                del self.streams[x]  # tail gone; the column can start again
        return heads

    def frame(self) -> np.ndarray:
        heads = self.step()

        # Glyphs changing: a few random cells get a new brightness each frame
        changed = np.random.random(self.glyphs.shape) < GLYPH_CHANGE
        self.glyphs[changed] = np.random.uniform(GLYPH_MIN, 1.0, int(changed.sum()))

        index = (self.energy * self.glyphs * 255).astype(np.uint8)
        rgb = PALETTE[index]
        for x, y in heads:
            if y >= 0:
                rgb[y, x] = HEAD_COLOR if random.random() > 0.15 else PALETTE[200]
        return rgb.reshape(-1)


SCENE = MatrixRain


def main() -> None:
    run_scene(MatrixRain, "Matrix rain LED matrix display")


if __name__ == "__main__":
    main()
