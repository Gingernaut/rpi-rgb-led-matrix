#!/usr/bin/env python
"""Meteor wave animation display for the RGB LED matrix.

Renders colored meteors (vertical streaks of light) that travel horizontally
across the display with gradient trailing effects. Features fade-trail streaks,
slowly rotating hue palette, and variable meteor speeds.

Requires the rgbmatrix Python bindings (make build-python).
"""

import argparse
import colorsys
import random
import time
from enum import Enum

from matrix_common import FramePacer, create_matrix, parse_matrix_args, run_display


class Direction(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


DIMMED_PCTS = [95, 90, 80, 75, 70, 50, 30, 0]

# Hue rotation: full cycle over this many seconds
HUE_CYCLE_SECONDS = 60.0

# Fade factor applied to every pixel each frame for streak trails (0.0–1.0).
# Lower = longer trails. 0.85 gives a nice medium-length glow.
FADE_FACTOR = 0.85

# Per-channel fade lookup table: FADE_LUT[v] == int(v * FADE_FACTOR), applied to
# the whole framebuffer at C speed with bytearray.translate().
FADE_LUT = bytes(int(v * FADE_FACTOR) for v in range(256))

# Frame period. Meteor speeds and the fade are per frame, and were tuned at the
# old cadence: ~5 ms of work + sleep(0.02), rounded up to two refreshes of the
# panel's 75 Hz refresh = 26.67 ms (37.5 fps). Frames follow a fixed deadline
# schedule, so the average rate stays 37.5 fps at other refresh rates too.
FRAME_SECONDS = 2 / 75


def hue_to_rgb(hue: float, saturation: float = 1.0, value: float = 1.0) -> tuple[int, int, int]:
    """Convert HSV (hue 0–1, s 0–1, v 0–1) to RGB (0–255)."""
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _build_pixels(base_rgb: tuple[int, int, int], direction: Direction) -> list[tuple[int, int, int]]:
    """Pre-compute the pixel strip for a meteor. Returns a flat list of RGB tuples."""
    pixels = []
    for pct in DIMMED_PCTS:
        factor = 1 - pct / 100
        pixels.append((
            max(int(base_rgb[0] * factor), 0),
            max(int(base_rgb[1] * factor), 0),
            max(int(base_rgb[2] * factor), 0),
        ))
    if direction == Direction.LEFT:
        pixels.reverse()
    return pixels


class Meteor:
    __slots__ = ("base_rgb", "x", "y", "direction", "speed", "accum", "pixels", "length")

    def __init__(self, x: int, y: int, direction: Direction,
                 base_rgb: tuple[int, int, int], speed: float = 0.5):
        self.base_rgb = base_rgb
        self.x = x
        self.y = y
        self.direction = direction
        self.speed = speed
        self.accum = 0.0
        self.pixels = _build_pixels(base_rgb, direction)
        self.length = len(self.pixels)


def _random_meteor_color(base_hue: float) -> tuple[int, int, int]:
    """Generate a random color near the current base hue.

    Adds up to +/-0.08 hue jitter so meteors aren't all identical,
    with high saturation and brightness for vivid LED colors.
    """
    hue = (base_hue + random.uniform(-0.08, 0.08)) % 1.0
    saturation = random.uniform(0.7, 1.0)
    value = random.uniform(0.7, 1.0)
    return hue_to_rgb(hue, saturation, value)


class Wave1Display:
    def __init__(self, args: argparse.Namespace) -> None:
        self.matrix = create_matrix(args)

    def new_random_meteor(self, base_hue: float, random_x: bool = False) -> Meteor:
        y = random.randint(0, self.matrix.height - 1)
        go_left = bool(random.getrandbits(1))
        speed = random.uniform(0.4, 1.8)
        color = _random_meteor_color(base_hue)

        if random_x:
            x = random.randint(0, self.matrix.width // 2)
        elif go_left:
            x = self.matrix.width
        else:
            x = -8

        return Meteor(
            x=x,
            y=y,
            direction=Direction.LEFT if go_left else Direction.RIGHT,
            base_rgb=color,
            speed=speed,
        )

    def run(self) -> None:
        canvas = self.matrix.CreateFrameCanvas()
        met_count = 16
        width = self.matrix.width
        height = self.matrix.height
        start_time = time.monotonic()
        pacer = FramePacer(self.matrix, FRAME_SECONDS, first_deadline=start_time + FRAME_SECONDS)

        # Framebuffer for the fade-trail effect: packed RGB bytes, row-major
        fb = bytearray(width * height * 3)

        base_hue = 0.55  # start in the cyan/blue range
        meteors = [self.new_random_meteor(base_hue, random_x=True) for _ in range(met_count)]

        while True:
            elapsed = time.monotonic() - start_time
            base_hue = (0.55 + elapsed / HUE_CYCLE_SECONDS) % 1.0

            # Fade the entire framebuffer for streak trails
            fb = fb.translate(FADE_LUT)

            # Draw meteors into the framebuffer
            for i, meteor in enumerate(meteors):
                meteor.accum += meteor.speed
                steps = int(meteor.accum)
                if steps > 0:
                    meteor.accum -= steps
                    if meteor.direction == Direction.RIGHT:
                        meteor.x += steps
                    else:
                        meteor.x -= steps

                mx = meteor.x
                my = meteor.y
                if 0 <= my < height:
                    row_base = my * width
                    for dx in range(max(0, -mx), min(meteor.length, width - mx)):
                        r, g, b = meteor.pixels[dx]
                        # Brightest-wins blend so overlapping meteors glow
                        p = (row_base + mx + dx) * 3
                        if r > fb[p]:
                            fb[p] = r
                        if g > fb[p + 1]:
                            fb[p + 1] = g
                        if b > fb[p + 2]:
                            fb[p + 2] = b

                if meteor.direction == Direction.RIGHT and mx >= width:
                    meteors[i] = self.new_random_meteor(base_hue)
                elif meteor.direction == Direction.LEFT and mx + meteor.length <= 0:
                    meteors[i] = self.new_random_meteor(base_hue)

            canvas.SetPixelsRGB(fb, width, height)
            canvas = pacer.swap(canvas)


def main() -> None:
    run_display(Wave1Display(parse_matrix_args("Wave1 meteor LED matrix display")))


if __name__ == "__main__":
    main()
