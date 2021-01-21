#!/usr/bin/env python
import time
from samplebase import SampleBase
from rgbmatrix import graphics
from PIL import Image

class ImageScroller(SampleBase):
    def __init__(self, *args, **kwargs):
        super(ImageScroller, self).__init__(*args, **kwargs)
        self.parser.add_argument("-i", "--image", help="The image to display", default="../../../examples-api-use/runtext.ppm")
        self.parser.add_argument("-a", "--artist", help="The musical artist", default="GAMBINO")
        self.parser.add_argument("-s", "--song", help="The Song", default="3005")

        self.parser.add_argument("-d", "--delay", help="how long to pause before scrolling", default=0.035)
        self.parser.add_argument("-f", "--font", help="which font size to choose", default="5x8")

    def run(self):
        if not 'image' in self.__dict__:
            self.image = Image.open(self.args.image).convert('RGB')
        # print("dimmensions")
        # print(f"matrix: {self.matrix.width}x{self.matrix.height}")
        # print(f"image: {self.image.size}")
        img_size = self.matrix.height - 2
        newimg = self.image.resize((img_size, img_size), Image.LANCZOS)
        delay = float(self.args.delay)

        double_buffer = self.matrix.CreateFrameCanvas()
        img_width, img_height = newimg.size


        font = graphics.Font()

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
            "tom-thumb.bdf"
        }


        textColor = graphics.Color(255, 255, 255)

        artist = self.args.artist.upper().strip()
        song = self.args.song.upper().strip()

        combine_text = True
        scroll_title = False

        if len(artist) < 7:
            combine_text = False

        if len(song) > 6:
            scroll_title = True

        chosen_font = f"{self.args.font}.bdf"

        if not chosen_font in font_options:
            raise Exception(f"Error: invalid font size. Available options: {[x.replace('.bdf', '') for x in font_options]}")

        if combine_text:
            chosen_font = "6x9.bdf"

        font.LoadFont(f"../../fonts/{chosen_font}")

        scroll_text_start_x = double_buffer.width
        while True:
            double_buffer.Clear()


            if combine_text:
                vertical_offset = 18

                combined_text = f"{artist} - {song}"
                i = graphics.DrawText(double_buffer, font, scroll_text_start_x, vertical_offset, textColor, combined_text)

                scroll_text_start_x -= 1
                if (scroll_text_start_x + i < 0):
                    scroll_text_start_x = double_buffer.width
            else:
                artist_vertical_offset = 12
                song_vertical_offset = 26

                artist_start_pos = 32

                if len(artist) == 6:
                    artist_start_pos +=1
                else:
                    artist_start_pos += int(24 / len(artist))

                # artist
                i = graphics.DrawText(double_buffer, font, artist_start_pos, artist_vertical_offset, textColor, artist)


                if not scroll_title:
                    i = graphics.DrawText(double_buffer, font, artist_start_pos, song_vertical_offset, textColor, song)
                else:
                    s = graphics.DrawText(double_buffer, font, scroll_text_start_x, song_vertical_offset, textColor, song)

                    scroll_text_start_x -= 1
                    if (scroll_text_start_x + s < 0):
                        scroll_text_start_x = double_buffer.width


            # Image
            double_buffer.SetImage(newimg, 1, 1)

            # left border
            for x in range(0, 1):
                for y in range(double_buffer.height):
                    double_buffer.SetPixel(x, y, 0,0,0)


            time.sleep(delay)
            double_buffer = self.matrix.SwapOnVSync(double_buffer)

# Main function
# e.g. call with
#  sudo ./image-scroller.py --chain=4
# if you have a chain of four
if __name__ == "__main__":
    image_scroller = ImageScroller()
    if (not image_scroller.process()):
        image_scroller.print_help()
