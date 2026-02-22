#!/usr/bin/env python
"""Meteor wave animation display for the RGB LED matrix.

Renders colored meteors (vertical streaks of light) that travel horizontally
across the display with gradient trailing effects.

Requires rgbmatrix Python bindings installed (make build-python && make install-python).
"""

import argparse
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
    __slots__ = ("base_rgb", "x", "y", "direction", "pixels", "length")

    def __init__(self, x: int, y: int, direction: Direction, base_rgb: tuple[int, int, int]):
        self.base_rgb = base_rgb
        self.x = x
        self.y = y
        self.direction = direction
        self.pixels = _build_pixels(base_rgb, direction)
        self.length = len(self.pixels)


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

    def new_random_meteor(self, random_x: bool = False) -> Meteor:
        y = random.randint(0, self.matrix.height - 1)
        green = random.randint(160, 255)
        blue = random.randint(160, 255)
        go_left = bool(random.getrandbits(1))

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
            base_rgb=(0, green, blue),
        )

    def run(self) -> None:
        canvas = self.matrix.CreateFrameCanvas()
        delay = 0.02
        met_count = 16
        width = self.matrix.width

        meteors = [self.new_random_meteor(random_x=True) for _ in range(met_count)]

        while True:
            canvas.Fill(0, 0, 0)

            for i, meteor in enumerate(meteors):
                if meteor.direction == Direction.RIGHT:
                    meteor.x += 1
                else:
                    meteor.x -= 1

                mx = meteor.x
                my = meteor.y
                for dx, (r, g, b) in enumerate(meteor.pixels):
                    canvas.SetPixel(mx + dx, my, r, g, b)

                if meteor.direction == Direction.RIGHT and mx >= width:
                    meteors[i] = self.new_random_meteor()
                elif meteor.direction == Direction.LEFT and mx + meteor.length <= 0:
                    meteors[i] = self.new_random_meteor()

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
