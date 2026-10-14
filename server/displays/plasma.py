#!/usr/bin/env python
"""Plasma: the classic demoscene effect, a slowly flowing field of summed sine
waves colored through a palette that also rotates, so color bands stream
through the shapes. Crossfades to a new palette every PALETTE_SECONDS.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import random

import numpy as np

from matrix_common import gradient, hsv, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps

# Palette steps per frame (256 steps make one full cycle) and how fast the
# field itself flows, in radians per second
PALETTE_SPEED = 1.2
FLOW_SPEED = 0.6

PALETTE_SECONDS = 40.0
CROSSFADE_SECONDS = 4.0

# Cyclic palettes, so rotating one has no seam. Darker stops give the bands
# some contrast; LEDs look washed out when everything is bright.
PALETTES = [
    gradient([hsv(i / 6) for i in range(6)], cyclic=True),  # rainbow
    gradient([(20, 0, 40), (160, 0, 120), (255, 80, 0), (255, 200, 40), (160, 0, 120)], cyclic=True),  # sunset
    gradient([(0, 10, 40), (0, 80, 160), (0, 220, 220), (180, 255, 255), (0, 80, 160)], cyclic=True),  # ocean
    gradient([(0, 0, 0), (120, 0, 0), (255, 60, 0), (255, 200, 0), (255, 60, 0), (120, 0, 0)], cyclic=True),  # fire
    gradient([(10, 0, 40), (120, 0, 200), (255, 0, 140), (0, 200, 255), (120, 0, 200)], cyclic=True),  # synthwave
    gradient([(0, 20, 10), (0, 160, 60), (0, 220, 180), (120, 60, 255), (0, 160, 60)], cyclic=True),  # aurora
]


class Plasma:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        self.x = x
        self.y = y
        self.frame_count = 0
        self.palettes = [np.array(p, dtype=np.float32) for p in PALETTES]
        self.order = random.sample(range(len(PALETTES)), len(PALETTES))
        # Each run starts somewhere different along the flow
        self.t0 = random.uniform(0, 1000)

    def palette(self) -> np.ndarray:
        """The current palette as (256, 3) uint8, crossfading at switches."""
        seconds = self.frame_count * FRAME_SECONDS
        index, into = divmod(seconds, PALETTE_SECONDS)
        index = int(index)
        current = self.palettes[self.order[index % len(self.order)]]
        fade_start = PALETTE_SECONDS - CROSSFADE_SECONDS
        if into > fade_start:
            following = self.palettes[self.order[(index + 1) % len(self.order)]]
            f = (into - fade_start) / CROSSFADE_SECONDS
            f = f * f * (3 - 2 * f)  # ease in and out
            current = current * (1 - f) + following * f
        return current.astype(np.uint8)

    def frame(self) -> np.ndarray:
        self.frame_count += 1
        t = self.t0 + self.frame_count * FRAME_SECONDS * FLOW_SPEED
        x, y = self.x, self.y
        # A moving center for the ripple term, wandering on a Lissajous path
        cx = self.width * (0.5 + 0.4 * np.sin(t * 0.37))
        cy = self.height * (0.5 + 0.4 * np.cos(t * 0.29))
        v = (
            np.sin(x * 0.15 + t)
            + np.sin(y * 0.21 - t * 0.8)
            + np.sin((x * 0.09 + y * 0.13) + t * 0.6)
            + np.sin(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) * 0.3 - t * 1.4)
        )
        # v is in [-4, 4]; map to palette indices (one sweep is 64 palette steps
        # per unit, so the four waves span the palette twice) and rotate
        offset = int(self.frame_count * PALETTE_SPEED)
        index = ((v * 64).astype(np.int32) + offset) & 255
        return self.palette()[index].reshape(-1)


SCENE = Plasma


def main() -> None:
    run_scene(Plasma, "Plasma LED matrix display")


if __name__ == "__main__":
    main()
