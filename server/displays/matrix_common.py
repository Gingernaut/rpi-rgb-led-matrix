"""Shared setup for the Python display scripts: options, matrix, frame pacing,
and the scene runner plus color helpers the animations share.

The scripts run by path (see DisplayManager.start_python), so they import this
as a sibling module: `from matrix_common import ...`.
"""

import argparse
import colorsys
import signal
import sys
import time
from collections.abc import Callable

# Wake this long before each frame deadline; SwapOnVSync then waits for the
# next vsync. Keep it under one refresh period (13.3 ms at 75 Hz).
VSYNC_MARGIN_SECONDS = 0.003

RGB = tuple[int, int, int]


def parse_matrix_args(description: str) -> argparse.Namespace:
    """Parse the flags DisplayConfig.to_python_args() passes. The server always
    passes all of them; the defaults (kept equal to DisplayConfig's) are for
    running a script by hand."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--gpio-mapping", default="adafruit-hat")
    parser.add_argument("--brightness", type=int, default=100)
    parser.add_argument("--slowdown-gpio", type=int, default=4)
    parser.add_argument("--pwm-lsb-nanoseconds", type=int, default=300)
    parser.add_argument("--limit-refresh-rate-hz", type=int, default=150)
    return parser.parse_args()


def create_matrix(args: argparse.Namespace):
    # Imported here so the display modules can be imported off the Pi
    from rgbmatrix import RGBMatrix, RGBMatrixOptions

    options = RGBMatrixOptions()
    options.rows = args.rows
    options.cols = args.cols
    options.hardware_mapping = args.gpio_mapping
    options.brightness = args.brightness
    options.pwm_lsb_nanoseconds = args.pwm_lsb_nanoseconds
    options.limit_refresh_rate_hz = args.limit_refresh_rate_hz
    options.drop_privileges = False
    options.gpio_slowdown = args.slowdown_gpio
    return RGBMatrix(options=options)


class FramePacer:
    """Swaps frames on a fixed deadline schedule, so animation speed doesn't
    depend on the panel's refresh rate or on how long a frame took to draw."""

    def __init__(self, matrix, period: float, first_deadline: float | None = None) -> None:
        self.matrix = matrix
        self.period = period
        self.next_frame = time.monotonic() if first_deadline is None else first_deadline

    def reset(self) -> None:
        """Start the schedule over from now, e.g. after an idle wait."""
        self.next_frame = time.monotonic()

    def swap(self, canvas):
        """Wait for the frame's deadline, swap it in, and return the new back buffer."""
        wait = self.next_frame - VSYNC_MARGIN_SECONDS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        canvas = self.matrix.SwapOnVSync(canvas)
        self.next_frame += self.period
        now = time.monotonic()
        if now > self.next_frame:  # fell a whole frame behind; don't try to catch up
            self.next_frame = now + self.period
        return canvas


def run_display(run: Callable[[], None], cleanup: Callable[[], None] | None = None) -> None:
    """Call run() until SIGTERM/SIGINT, then cleanup() if given."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        run()
    except KeyboardInterrupt:
        pass
    finally:
        if cleanup is not None:
            cleanup()


# --- Scenes ------------------------------------------------------------------
#
# A scene is a class built as Scene(width, height) with a FRAME_SECONDS class
# attribute and a frame() method returning the next frame as packed row-major
# RGB bytes (width * height * 3): a bytes/bytearray, or a flat C-contiguous
# uint8 numpy array. Scenes keep time by counting frames, not by the clock, so
# preview.py can render and time them off the Pi.
#
# Scenes run for weeks, so keep ever-growing values bounded: wrap a time-based
# phase (t % TAU) in float64 before it meets a float32 array, or float32
# rounding makes the motion stutter after days of uptime.


def run_scene(scene_class, description: str) -> None:
    """Entry point for a scene script: show scene_class's frames on the matrix."""
    matrix = create_matrix(parse_matrix_args(description))
    scene = scene_class(matrix.width, matrix.height)

    def run() -> None:
        canvas = matrix.CreateFrameCanvas()
        pacer = FramePacer(matrix, scene_class.FRAME_SECONDS)
        width, height = matrix.width, matrix.height
        while True:
            canvas.SetPixelsRGB(scene.frame(), width, height)
            canvas = pacer.swap(canvas)

    run_display(run)


# --- Color helpers -----------------------------------------------------------


def hsv(hue: float, saturation: float = 1.0, value: float = 1.0) -> RGB:
    """HSV (each 0–1; hue wraps) to 0–255 RGB."""
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def scale(color: RGB, factor: float) -> RGB:
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """Linear blend from a (t=0) to b (t=1)."""
    return (
        round(a[0] + (b[0] - a[0]) * t),
        round(a[1] + (b[1] - a[1]) * t),
        round(a[2] + (b[2] - a[2]) * t),
    )


def gradient(colors: list[RGB], n: int = 256, cyclic: bool = False) -> list[RGB]:
    """n colors spread evenly through colors. cyclic wraps the last color back
    to the first, so a palette can be rotated without a seam."""
    stops = list(colors) + [colors[0]] if cyclic else list(colors)
    segments = len(stops) - 1
    out = []
    for i in range(n):
        pos = i * segments / (n if cyclic else max(n - 1, 1))
        j = min(int(pos), segments - 1)
        out.append(mix(stops[j], stops[j + 1], pos - j))
    return out


def fade_lut(factor: float) -> bytes:
    """Byte lookup table that scales by factor, for fading a whole framebuffer
    at C speed: fb = fb.translate(fade_lut(0.85))."""
    return bytes(int(v * factor) for v in range(256))


def put_max(fb: bytearray, width: int, height: int, x: int, y: int, color: RGB) -> None:
    """Draw a pixel into a packed RGB framebuffer, keeping the brighter value
    per channel, so overlapping lights glow instead of covering each other.
    Off-screen pixels are skipped."""
    if 0 <= x < width and 0 <= y < height:
        p = (y * width + x) * 3
        r, g, b = color
        if r > fb[p]:
            fb[p] = r
        if g > fb[p + 1]:
            fb[p + 1] = g
        if b > fb[p + 2]:
            fb[p + 2] = b
