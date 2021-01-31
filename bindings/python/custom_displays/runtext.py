#!/usr/bin/env python
# Display a runtext with double-buffering.
from samplebase import SampleBase
from rgbmatrix import graphics
import time


class RunText(SampleBase):

    font_options = {
        "10x20.bdf",
        "4x6.bdf",
        "5x7.bdf",
        "5x8.bdf",
        "6x10.bdf",
        "6x12.bdf",
        "6x13.bdf",
        "6x13B.bdf",
        "6x13O.bdf",
        "6x9.bdf",
        "7x13.bdf",
        "7x13B.bdf",
        "7x13O.bdf",
        "7x14.bdf",
        "7x14B.bdf",
        "8x13.bdf",
        "8x13B.bdf",
        "8x13O.bdf",
        "9x15.bdf",
        "9x15B.bdf",
        "9x18.bdf",
        "9x18B.bdf",
        "clR6x12.bdf",
        "helvR12.bdf",
        "texgyre-27.bdf",
        "tom-thumb.bdf",
    }

    def __init__(self, *args, **kwargs):
        super(RunText, self).__init__(*args, **kwargs)
        self.parser.add_argument(
            "-t",
            "--text",
            help="The text to scroll on the RGB LED panel",
            default="Hello world!",
        )
        self.parser.add_argument(
            "-c",
            "--color",
            help="The text color",
            default="white",
        )
        self.parser.add_argument(
            "-s",
            "--size",
            help="The Font Size",
            default="medium",
        )

    def get_color(self, color_str):
        if color_str == "white":
            return graphics.Color(255, 255, 255)
        elif color_str == "green":
            return graphics.Color(0, 255, 0)
        elif color_str == "red":
            return graphics.Color(255, 0, 0)
        elif color_str == "blue":
            return graphics.Color(0, 0, 255)

    def get_font_size(self, font_size: str):
        if font_size == "small":
            return "5x8"
        elif font_size == "medium":
            return "7x13"

        else:
            return "9x15"

    def run(self):
        offscreen_canvas = self.matrix.CreateFrameCanvas()
        font = graphics.Font()

        font.LoadFont(f"../../../fonts/{self.get_font_size(self.args.size)}.bdf")

        textColor = self.get_color(self.args.color)

        pos = offscreen_canvas.width
        my_text = self.args.text

        while True:
            offscreen_canvas.Clear()
            len = graphics.DrawText(offscreen_canvas, font, pos, 10, textColor, my_text)
            pos -= 1
            if pos + len < 0:
                pos = offscreen_canvas.width

            time.sleep(0.05)
            offscreen_canvas = self.matrix.SwapOnVSync(offscreen_canvas)


# Main function
if __name__ == "__main__":
    run_text = RunText()
    if not run_text.process():
        run_text.print_help()
