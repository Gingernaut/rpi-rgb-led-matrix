#!/usr/bin/env python
from samplebase import SampleBase
import time


class BlankDisplay(SampleBase):
    def __init__(self, *args, **kwargs):
        super(BlankDisplay, self).__init__(*args, **kwargs)

    def run(self):
        self.offscreen_canvas = self.matrix.CreateFrameCanvas()

        while True:
            time.sleep(1)
            self.offscreen_canvas.Fill(0, 0, 0)
            self.offscreen_canvas = self.matrix.SwapOnVSync(self.offscreen_canvas)


# Main function
if __name__ == "__main__":
    blank_display = BlankDisplay()
    if not blank_display.process():
        blank_display.print_help()
