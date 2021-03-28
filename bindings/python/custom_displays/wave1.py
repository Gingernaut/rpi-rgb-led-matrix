#!/usr/bin/env python
# Display a Wave1 with double-buffering.
from samplebase import SampleBase
from rgbmatrix import graphics
import time
from pydantic import BaseModel
from typing import Tuple, List
from cachetools import cached
import random
from enum import Enum


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
        """
        Returns 2D array of RGBA colors: Represents the pixels for the meteor.
        If going to be cut off, return a smaller array.
        (0, 0, 0)
        """
        base_shape = [
            [
                self.dimmed_color(95),
            ],
            [
                self.dimmed_color(90),
            ],
            [
                self.dimmed_color(80),
            ],
            [
                self.dimmed_color(75),
            ],
            [
                self.dimmed_color(70),
            ],
            [
                self.dimmed_color(50),
            ],
            [
                self.dimmed_color(30),
            ],
            [self.base_rgb],
        ]
        if self.direction == Direction.RIGHT:
            return base_shape
        elif self.direction == Direction.LEFT:
            return list(reversed(base_shape))

        raise Exception(f"self direction not set!")

    @property
    def constrained_x_size(self):
        # TODO: non-quare sizes
        @cached(cache={})
        def size():
            return len(self.get_pixels())

        return size()

    @property
    def constrained_y_size(self):
        # TODO: non-quare sizes
        @cached(cache={})
        def size():
            return len(self.get_pixels()[0])

        return size()


class Wave1(SampleBase):
    def new_random_meteor(self) -> Meteor:
        start_y = random.randint(0, 32)

        green = random.randint(160, 255)
        blue = random.randint(160, 255)
        goLeft = bool(random.getrandbits(1))

        if goLeft:

            return Meteor(
                top_left=(64, start_y),
                direction=Direction.LEFT,
                base_rgb=(0, green, blue),
            )

        else:
            return Meteor(
                top_left=(-4, start_y),
                direction=Direction.RIGHT,
                base_rgb=(0, green, blue),
            )

    def run(self):
        double_buffer = self.matrix.CreateFrameCanvas()

        delay = 0.02

        meteors = []

        met_count = 16

        for i in range(met_count):
            start_y = random.randint(0, 32)
            start_x = random.randint(0, 32)

            green = random.randint(160, 255)
            blue = random.randint(160, 255)
            goLeft = bool(random.getrandbits(1))

            if goLeft:
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

            # comment to do streaks
            double_buffer.Fill(0, 0, 0)

            tmp_meteors = []

            for meteor in meteors:

                destroyAfterwards = False

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

                        double_buffer.SetPixel(display_x, display_y, R, G, B)

                        if (
                            meteor.x >= self.matrix.width
                            and meteor.direction == Direction.RIGHT
                        ):
                            destroyAfterwards = True

                        if (
                            meteor.x <= (0 - meteor.constrained_x_size)
                            and meteor.direction == Direction.LEFT
                        ):
                            destroyAfterwards = True

                if destroyAfterwards:
                    tmp_meteors.append(self.new_random_meteor())
                else:
                    tmp_meteors.append(meteor)

            meteors = tmp_meteors

            time.sleep(delay)
            double_buffer = self.matrix.SwapOnVSync(double_buffer)


# Main function
if __name__ == "__main__":
    wave = Wave1()
    if not wave.process():
        wave.print_help()
