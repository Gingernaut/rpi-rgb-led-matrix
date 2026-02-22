#!/usr/bin/env python
"""Meteor wave animation display for the RGB LED matrix.

Renders colored meteors (vertical streaks of light) that travel horizontally
across the display with gradient trailing effects. Features fade-trail streaks,
slowly rotating hue palette, and variable meteor speeds.

Requires rgbmatrix Python bindings installed (make build-python && make install-python).
"""

import argparse
import colorsys
import math
import random
import signal
import sys
import time
from enum import Enum

# Lazy import — rgbmatrix only exists on the Pi after make install-python
rgbmatrix = None


def import_rgbmatrix():
    global rgbmatrix
    from rgbmatrix import RGBMatrix, RGBMatrixOptions, FrameCanvas

    rgbmatrix = type(sys)("rgbmatrix")
    rgbmatrix.RGBMatrix = RGBMatrix
    rgbmatrix.RGBMatrixOptions = RGBMatrixOptions
    rgbmatrix.FrameCanvas = FrameCanvas


class Direction(Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


DIMMED_PCTS = [95, 90, 80, 75, 70, 50, 30, 0]

# Hue rotation: full cycle over this many seconds
HUE_CYCLE_SECONDS = 60.0

# Fade factor applied to every pixel each frame for streak trails (0.0–1.0).
# Lower = longer trails. 0.85 gives a nice medium-length glow.
FADE_FACTOR = 0.85


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
        self.args = args

        import_rgbmatrix()
        options = rgbmatrix.RGBMatrixOptions()
        options.rows = args.rows
        options.cols = args.cols
        options.hardware_mapping = args.gpio_mapping
        options.brightness = args.brightness
        options.pwm_lsb_nanoseconds = args.pwm_lsb_nanoseconds
        options.limit_refresh_rate_hz = args.limit_refresh_rate_hz
        options.drop_privileges = False
        options.gpio_slowdown = args.slowdown_gpio

        self.matrix = rgbmatrix.RGBMatrix(options=options)

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
        delay = 0.02
        met_count = 16
        width = self.matrix.width
        height = self.matrix.height
        start_time = time.monotonic()

        # Framebuffer for fade-trail effect: stores (r, g, b) per pixel
        fb = [[(0, 0, 0) for _ in range(width)] for _ in range(height)]

        base_hue = 0.55  # start in the cyan/blue range
        meteors = [self.new_random_meteor(base_hue, random_x=True) for _ in range(met_count)]

        while True:
            elapsed = time.monotonic() - start_time
            base_hue = (0.55 + elapsed / HUE_CYCLE_SECONDS) % 1.0

            # Fade the entire framebuffer for streak trails
            fade = FADE_FACTOR
            for row in range(height):
                fb_row = fb[row]
                for col in range(width):
                    r, g, b = fb_row[col]
                    fb_row[col] = (int(r * fade), int(g * fade), int(b * fade))

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
                for dx, (r, g, b) in enumerate(meteor.pixels):
                    px = mx + dx
                    if 0 <= px < width and 0 <= my < height:
                        # Brightest-wins blend so overlapping meteors glow
                        er, eg, eb = fb[my][px]
                        fb[my][px] = (max(er, r), max(eg, g), max(eb, b))

                if meteor.direction == Direction.RIGHT and mx >= width:
                    meteors[i] = self.new_random_meteor(base_hue)
                elif meteor.direction == Direction.LEFT and mx + meteor.length <= 0:
                    meteors[i] = self.new_random_meteor(base_hue)

            # Blit framebuffer to canvas
            for row in range(height):
                fb_row = fb[row]
                for col in range(width):
                    r, g, b = fb_row[col]
                    canvas.SetPixel(col, row, r, g, b)

            time.sleep(delay)
            canvas = self.matrix.SwapOnVSync(canvas)

    def cleanup(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave1 meteor LED matrix display")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--gpio-mapping", default="adafruit-hat")
    parser.add_argument("--brightness", type=int, default=50)
    parser.add_argument("--slowdown-gpio", type=int, default=4)
    parser.add_argument("--pwm-lsb-nanoseconds", type=int, default=300)
    parser.add_argument("--limit-refresh-rate-hz", type=int, default=150)
    args = parser.parse_args()

    display = Wave1Display(args)

    def handle_signal(signum, frame):
        display.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        display.run()
    finally:
        display.cleanup()


if __name__ == "__main__":
    main()
