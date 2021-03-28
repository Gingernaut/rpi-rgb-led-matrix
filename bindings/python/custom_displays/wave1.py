#!/usr/bin/env python
# Display a Wave1 with double-buffering.
from samplebase import SampleBase
from rgbmatrix import graphics
import time
from pydantic import BaseModel
from typing import Tuple, List
from cachetools import cached
import random


class Meteor(BaseModel):
    base_rgb: Tuple[int, int, int]
    top_left: Tuple[int, int]

    @property
    def x(self):
        return self.top_left[0]

    @property
    def y(self):
        return self.top_left[1]

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

    def get_constrained_pixels(self) -> List[List[int]]:
        """
        Returns 2D array of RGBA colors: Represents the pixels for the meteor.
        If going to be cut off, return a smaller array.
        (0, 0, 0)
        """
        return [
            [self.base_rgb, self.base_rgb],
            [self.base_rgb, self.base_rgb],
            [self.base_rgb, self.base_rgb],
            [self.base_rgb, self.base_rgb],
        ]

    @property
    def constrained_x_size(self):
        x = len(self.get_constrained_pixels())
        print(f"size of xaxis for meteor: {x}")
        return x

    @property
    def constrained_y_size(self):
        y = len(self.get_constrained_pixels()[0])
        print(f"size of y axis for meteor: {y}")
        return y


class Wave1(SampleBase):
    def run(self):
        double_buffer = self.matrix.CreateFrameCanvas()

        delay = 0.035

        meteors = []

        meteors.append(Meteor(top_left=(0, 16), base_rgb=(0, 255, 255)))
        while True:

            # comment to do streaks
            double_buffer.Fill(0, 0, 0)

            for meteor in meteors:

                meteor.move_right()

                pixels = meteor.get_constrained_pixels()
                for x in range(meteor.constrained_x_size):
                    for y in range(meteor.constrained_y_size):

                        print(f"getting constrained arr at [{x}][{y}]")
                        R = pixels[x][y][0]
                        G = pixels[x][y][1]
                        B = pixels[x][y][2]

                        display_x = x + meteor.x
                        display_y = y + meteor.y

                        print(
                            f"setting pixel at [{display_x}][{display_y}] to {pixels[x][y]}"
                        )
                        print("####")
                        double_buffer.SetPixel(display_x, display_y, R, G, B)

                        if display_x > self.matrix.width:
                            meteor.move_to((0, meteor.y))

                        # if display_y >= self.matrix.height:
                        #     meteor.move_down()

                        # elif display_y <= self.matrix.height:
                        #     meteor.move_up()

            time.sleep(delay)
            double_buffer = self.matrix.SwapOnVSync(double_buffer)


# Main function
if __name__ == "__main__":
    wave = Wave1()
    if not wave.process():
        wave.print_help()
