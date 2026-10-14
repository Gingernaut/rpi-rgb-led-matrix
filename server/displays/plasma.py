#!/usr/bin/env python
"""Plasma: the classic demoscene effect, a slowly flowing field of summed sine
waves colored through a palette that also rotates, so color bands stream
through the shapes. Every PALETTE_SECONDS a new palette floods in along the
field's contours.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from matrix_common import REFRESH_HZ, gradient, hsv, run_scene

FRAME_SECONDS = 2 / REFRESH_HZ  # every 2nd refresh, ~37.5 fps

TAU = 2 * math.pi

# Palette steps per frame (256 steps make one full cycle) and how fast the
# field itself flows, in radians per second
PALETTE_SPEED = 1.2
FLOW_SPEED = 0.6

PALETTE_SECONDS = 40.0
CROSSFADE_SECONDS = 6.0

# The crossfade sweeps a threshold steadily up through the field's values,
# switching pixels below it to the new palette behind a soft edge this wide
# (in field units). Most pixels sit near the middle of the field's [-4, 4]
# range, so the change eases in and out by itself.
FLOOD_EDGE = 0.8

# Cyclic palettes, so rotating one has no seam. Darker stops give the bands
# some contrast; LEDs look washed out when everything is bright, and the
# brightest stops stay saturated rather than pastel.
PALETTES = [
    gradient([hsv(i / 6) for i in range(6)] + [(0, 0, 0)], cyclic=True),  # rainbow
    gradient([(20, 0, 40), (160, 0, 120), (255, 60, 20), (255, 190, 40), (220, 30, 80)], cyclic=True),  # sunset
    gradient([(0, 10, 40), (0, 80, 180), (0, 220, 220), (60, 255, 170), (0, 80, 180)], cyclic=True),  # ocean
    gradient([(0, 0, 0), (120, 0, 0), (255, 60, 0), (255, 200, 0), (255, 60, 0), (120, 0, 0)], cyclic=True),  # fire
    gradient([(10, 0, 40), (120, 0, 200), (255, 0, 140), (0, 200, 255), (120, 0, 200)], cyclic=True),  # synthwave
    gradient([(0, 20, 10), (0, 160, 60), (0, 220, 180), (140, 0, 255), (0, 160, 60)], cyclic=True),  # aurora
]


class Plasma:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # A row and a column, so the terms that depend on only one axis are
        # computed once per column/row and broadcast
        self.x = np.arange(width, dtype=np.float32)[np.newaxis, :]
        self.y = np.arange(height, dtype=np.float32)[:, np.newaxis]
        self.diagonal = self.x * 0.09 + self.y * 0.13
        self.frame_count = 0
        self.palettes = [np.array(p, dtype=np.uint8) for p in PALETTES]
        self.order = random.sample(range(len(PALETTES)), len(PALETTES))
        # Each run starts somewhere different along the flow
        self.t0 = random.uniform(0, 1000)

    def field(self) -> np.ndarray:
        """The plasma value per pixel, (height, width) float32 in [-4, 4]."""
        t = self.t0 + self.frame_count * FRAME_SECONDS * FLOW_SPEED
        x, y = self.x, self.y
        # A moving center for the ripple term, wandering on a Lissajous path
        cx = self.width * (0.5 + 0.4 * math.sin(t * 0.37))
        cy = self.height * (0.5 + 0.4 * math.cos(t * 0.29))
        # Phases are wrapped in float64 before meeting the float32 grids;
        # unwrapped, float32 rounding of t makes the flow stutter after about
        # a week of uptime and lurch in jumps after a month
        v = np.sin(x * 0.15 + t % TAU) + np.sin(y * 0.21 - (t * 0.8) % TAU)
        v += np.sin(self.diagonal + (t * 0.6) % TAU)
        v += np.sin(np.hypot(x - cx, y - cy) * 0.3 - (t * 1.4) % TAU)
        return v

    def frame(self) -> np.ndarray:
        self.frame_count += 1
        v = self.field()
        # One unit of v is 64 palette steps, so the four waves span the palette
        # twice. The rotation is added before rounding down, so its fractional
        # steps spread across pixels instead of jumping the whole frame at once;
        # the +256 keeps the sum positive, so astype rounds down.
        rotation = (self.frame_count * PALETTE_SPEED) % 256
        index = (v * 64 + (rotation + 256)).astype(np.intp) & 255

        number, into = divmod(self.frame_count * FRAME_SECONDS, PALETTE_SECONDS)
        rgb = self.palette(int(number))[index]
        fade_start = PALETTE_SECONDS - CROSSFADE_SECONDS
        if into > fade_start:
            # Flood the next palette in from the field's low values up, rather
            # than blending whole palettes, which goes muddy halfway between
            # unrelated ones
            following = self.palette(int(number) + 1)[index]
            f = (into - fade_start) / CROSSFADE_SECONDS
            level = -4 - FLOOD_EDGE / 2 + f * (8 + FLOOD_EDGE)
            w = np.clip((level - v) / FLOOD_EDGE + 0.5, 0, 1)[..., np.newaxis]
            rgb = (rgb + (following - rgb.astype(np.float32)) * w).astype(np.uint8)
        return rgb.reshape(-1)

    def palette(self, number: int) -> np.ndarray:
        """The number-th palette shown, as (256, 3) uint8."""
        return self.palettes[self.order[number % len(self.order)]]


SCENE = Plasma


def main() -> None:
    run_scene(Plasma, "Plasma LED matrix display")


if __name__ == "__main__":
    main()
