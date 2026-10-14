#!/usr/bin/env python
"""Matrix rain: green code streams falling down the panel, as in The Matrix.

Each stream has a pale, flickering head and a green tail that fades behind it.
Streams fall at different speeds with different tail lengths, and tail cells
shimmer as their "glyphs" change.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from matrix_common import REFRESH_HZ, gradient, run_scene

FRAME_SECONDS = 2 / REFRESH_HZ  # every 2nd refresh, ~37.5 fps

# Pixels per frame, and how far behind the head a tail's brightness reaches 0
# (it goes dark at TAIL_CUTOFF, about a third sooner)
SPEED = (0.2, 0.75)
TAIL = (10, 40)
# Chance per frame that a free column starts a stream. A column is free once
# the newest streams in it and in both neighbors have tails GAP pixels clear of
# the top, so a column can hold more than one stream but streams don't start
# side by side and merge into a wide bar.
SPAWN_CHANCE = 0.015
GAP = 4

# Fraction of tail cells whose glyph changes each frame, and how dim a glyph
# can be relative to its tail brightness
GLYPH_CHANGE = 0.04
GLYPH_MIN = 0.35

HEAD_COLOR = (200, 255, 210)

# Tail brightness to color: dark green through green to pale green at the top.
# Indexes below TAIL_CUTOFF are off, so a tail ends cleanly instead of
# smearing out in values too dim for LEDs to show well.
TAIL_CUTOFF = 16
PALETTE = np.array(
    gradient([(0, 0, 0), (0, 60, 14), (0, 120, 28), (20, 200, 50), (90, 255, 110), (170, 255, 180)]),
    dtype=np.uint8,
)
PALETTE[:TAIL_CUTOFF] = 0


class Stream:
    __slots__ = ("x", "y", "speed", "tail", "fade")

    def __init__(self, x: int) -> None:
        self.x = x
        self.y = random.uniform(-3.0, 0.0)  # head row; starts just above the top
        self.speed = random.uniform(*SPEED)
        self.tail = random.uniform(*TAIL)
        # Brightness lost per frame, so the tail fades out `tail` pixels behind
        self.fade = self.speed / self.tail


class MatrixRain:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # Per cell: brightness left from the last head to pass, and how fast
        # it fades (set by that head's stream)
        self.energy = np.zeros((height, width), dtype=np.float32)
        self.fade = np.zeros((height, width), dtype=np.float32)
        self.glyphs = np.random.uniform(GLYPH_MIN, 1.0, (height, width)).astype(np.float32)
        self.streams: list[Stream] = []
        self.newest: list[Stream | None] = [None] * width  # last stream started per column
        # Start mid-rain rather than on an empty screen
        for _ in range(round(4 / FRAME_SECONDS)):
            self.step()

    def free(self, x: int) -> bool:
        """Whether column x can start a stream now."""
        for stream in self.newest[max(x - 1, 0):x + 2]:
            if stream is not None and stream.y - stream.tail < GAP:
                return False
        return True

    def step(self) -> list[tuple[int, int]]:
        """Advance the rain one frame; returns the on-screen head cells."""
        self.energy -= self.fade
        np.maximum(self.energy, 0, out=self.energy)

        for x in range(self.width):
            if random.random() < SPAWN_CHANCE and self.free(x):
                stream = self.newest[x] = Stream(x)
                self.streams.append(stream)

        heads = []
        falling = []
        for stream in self.streams:
            top = math.floor(stream.y)
            stream.y += stream.speed
            bottom = math.floor(stream.y)
            if bottom >= 0:
                # Light every cell the head reached this frame
                rows = slice(max(top, 0), min(bottom, self.height - 1) + 1)
                self.energy[rows, stream.x] = 1.0
                self.fade[rows, stream.x] = stream.fade
            if bottom < self.height:
                falling.append(stream)
                if bottom >= 0:
                    heads.append((stream.x, bottom))
            elif self.newest[stream.x] is stream:
                # Off the bottom; its tail fades out on its own
                self.newest[stream.x] = None
        self.streams = falling
        return heads

    def frame(self) -> np.ndarray:
        heads = self.step()

        # Glyphs changing: a few random cells get a new brightness each frame
        changed = np.random.random(self.glyphs.shape) < GLYPH_CHANGE
        self.glyphs[changed] = np.random.uniform(GLYPH_MIN, 1.0, int(changed.sum()))

        # Squared, so tails fade quickly from the head and then linger dimly
        index = (self.energy * self.energy * self.glyphs * 255).astype(np.uint8)
        rgb = PALETTE[index]
        for x, y in heads:
            rgb[y, x] = HEAD_COLOR if random.random() > 0.15 else PALETTE[200]
        return rgb.reshape(-1)


SCENE = MatrixRain


def main() -> None:
    run_scene(MatrixRain, "Matrix rain LED matrix display")


if __name__ == "__main__":
    main()
