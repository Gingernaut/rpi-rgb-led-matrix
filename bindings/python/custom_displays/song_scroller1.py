#!/usr/bin/env python
import time
from samplebase import SampleBase
from rgbmatrix import graphics
from PIL import Image


class ImageScroller(SampleBase):
    def __init__(self, *args, **kwargs):
        super(ImageScroller, self).__init__(*args, **kwargs)
        self.parser.add_argument(
            "-i",
            "--image",
            help="The image to display",
            default="../../../examples-api-use/runtext.ppm",
        )
        self.parser.add_argument(
            "-t",
            "--text",
            help="The text to scroll on the RGB LED panel",
            default="Hello world!",
        )
        self.parser.add_argument(
            "-d", "--delay", help="how long to pause before scrolling", default=0.035
        )
        self.parser.add_argument(
            "-f", "--font", help="which font size to choose", default="5x7"
        )

    def run(self):
        if not "image" in self.__dict__:
            self.image = Image.open(self.args.image).convert("RGB")
        # print("dimmensions")
        # print(f"matrix: {self.matrix.width}x{self.matrix.height}")
        # print(f"image: {self.image.size}")
        img_size = self.matrix.height - 2
        newimg = self.image.resize((img_size, img_size), Image.LANCZOS)

        font = graphics.Font()
        my_text = self.args.text

        delay = float(self.args.delay)

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

        chosen_font = f"{self.args.font}.bdf"

        if not chosen_font in font_options:
            raise Exception(
                f"Error: invalid font size. Available options: {[x.replace('.bdf', '') for x in font_options]}"
            )

        font.LoadFont(f"../../../fonts/{chosen_font}")

        textColor = graphics.Color(255, 255, 255)

        double_buffer = self.matrix.CreateFrameCanvas()
        img_width, img_height = newimg.size

        pos = double_buffer.width
        while True:
            double_buffer.Clear()

            len = graphics.DrawText(double_buffer, font, pos, 16, textColor, my_text)
            double_buffer.SetImage(newimg, 1, 1)

            for x in range(0, 1):
                for y in range(double_buffer.height):
                    double_buffer.SetPixel(x, y, 0, 0, 0)

            pos -= 1
            if pos + len < 0:
                pos = double_buffer.width

            time.sleep(delay)
            double_buffer = self.matrix.SwapOnVSync(double_buffer)


# Main function
# e.g. call with
#  sudo ./image-scroller.py --chain=4
# if you have a chain of four
if __name__ == "__main__":
    image_scroller = ImageScroller()
    if not image_scroller.process():
        image_scroller.print_help()
