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
from typing import Tuple, List

from pydantic import BaseModel
from cachetools import cached

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


class Meteor(BaseModel):
    base_rgb: Tuple[int, int, int]
    top_left: Tuple[int, int]
    direction: Direction

    def __hash__(self):
        return hash((self.base_rgb, self.top_left, str(self.direction)))

    @property
    def x(self):
        return self.top_left[0]

    @property
    def y(self):
        return self.top_left[1]

    @cached(cache={})
    def dimmed_color(self, pct: int) -> Tuple[int, int, int]:
        percent = float(pct / 100)
        return (
            max(int(self.base_rgb[0] * (1 - percent)), 0),
            max(int(self.base_rgb[1] * (1 - percent)), 0),
            max(int(self.base_rgb[2] * (1 - percent)), 0),
        )

    def move_left(self):
        self.top_left = (self.top_left[0] - 1, self.top_left[1])

    def move_right(self):
        self.top_left = (self.top_left[0] + 1, self.top_left[1])

    def move_up(self):
        self.top_left = (self.top_left[0], self.top_left[1] - 1)

    def move_down(self):
        self.top_left = (self.top_left[0], self.top_left[1] + 1)

    def move_to(self, coords: Tuple[int, int]):
        self.top_left = coords

    @cached(cache={})
    def get_pixels(self) -> List[List[int]]:
        """Returns 2D array of RGB colors representing the meteor's pixels."""
        base_shape = [
            [self.dimmed_color(95)],
            [self.dimmed_color(90)],
            [self.dimmed_color(80)],
            [self.dimmed_color(75)],
            [self.dimmed_color(70)],
            [self.dimmed_color(50)],
            [self.dimmed_color(30)],
            [self.base_rgb],
        ]
        if self.direction == Direction.RIGHT:
            return base_shape
        elif self.direction == Direction.LEFT:
            return list(reversed(base_shape))

        raise Exception("self direction not set!")

    @property
    def constrained_x_size(self):
        @cached(cache={})
        def size():
            return len(self.get_pixels())

        return size()

    @property
    def constrained_y_size(self):
        @cached(cache={})
        def size():
            return len(self.get_pixels()[0])

        return size()


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

    def new_random_meteor(self) -> Meteor:
        start_y = random.randint(0, self.matrix.height)
        green = random.randint(160, 255)
        blue = random.randint(160, 255)
        go_left = bool(random.getrandbits(1))

        if go_left:
            return Meteor(
                top_left=(self.matrix.width, start_y),
                direction=Direction.LEFT,
                base_rgb=(0, green, blue),
            )
        else:
            return Meteor(
                top_left=(-4, start_y),
                direction=Direction.RIGHT,
                base_rgb=(0, green, blue),
            )

    def run(self) -> None:
        canvas = self.matrix.CreateFrameCanvas()
        delay = 0.02
        met_count = 16

        meteors = []
        for _ in range(met_count):
            start_y = random.randint(0, self.matrix.height)
            start_x = random.randint(0, self.matrix.width // 2)
            green = random.randint(160, 255)
            blue = random.randint(160, 255)
            go_left = bool(random.getrandbits(1))

            if go_left:
                meteors.append(
                    Meteor(
                        top_left=(start_x, start_y),
                        direction=Direction.LEFT,
                        base_rgb=(0, green, blue),
                    )
                )
            else:
                meteors.append(
                    Meteor(
                        top_left=(start_x, start_y),
                        direction=Direction.RIGHT,
                        base_rgb=(0, green, blue),
                    )
                )

        while True:
            canvas.Fill(0, 0, 0)
            tmp_meteors = []

            for meteor in meteors:
                destroy = False

                if meteor.direction == Direction.RIGHT:
                    meteor.move_right()
                elif meteor.direction == Direction.LEFT:
                    meteor.move_left()

                pixels = meteor.get_pixels()
                for x in range(meteor.constrained_x_size):
                    for y in range(meteor.constrained_y_size):
                        R = pixels[x][y][0]
                        G = pixels[x][y][1]
                        B = pixels[x][y][2]

                        display_x = x + meteor.x
                        display_y = y + meteor.y

                        canvas.SetPixel(display_x, display_y, R, G, B)

                        if (
                            meteor.x >= self.matrix.width
                            and meteor.direction == Direction.RIGHT
                        ):
                            destroy = True

                        if (
                            meteor.x <= (0 - meteor.constrained_x_size)
                            and meteor.direction == Direction.LEFT
                        ):
                            destroy = True

                if destroy:
                    tmp_meteors.append(self.new_random_meteor())
                else:
                    tmp_meteors.append(meteor)

            meteors = tmp_meteors
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
