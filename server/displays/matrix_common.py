"""Shared setup for the Python display scripts: options, matrix, refresh timing,
and the scene runner plus color helpers the animations share.

The scripts run by path (see DisplayManager.start_python), so they import this
as a sibling module: `from matrix_common import ...`.
"""

import argparse
import colorsys
import signal
import sys
from collections.abc import Callable
from pathlib import Path

# The panel refreshes about REFRESH_HZ times a second with DisplayConfig's
# settings (pwm_lsb_nanoseconds=300); the real rate drifts a few percent, e.g.
# with the Pi's supply voltage. Frame periods are whole numbers of refreshes,
# passed to SwapOnVSync as its framerate_fraction: the library's refresh
# thread then swaps each frame in exactly that many refreshes after the last,
# so frames stay evenly spaced however late Python gets scheduled. (Sleeping
# to clock deadlines instead slid in and out of step with the refresh, showing
# some frames a refresh longer or shorter.) Animation speed follows the real
# refresh rate.
REFRESH_HZ = 75

RGB = tuple[int, int, int]

FONTS_DIR = Path(__file__).resolve().parents[2] / "fonts"


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


def refreshes_per_frame(frame_seconds: float) -> int:
    """SwapOnVSync's framerate_fraction for a frame period."""
    return max(1, round(frame_seconds * REFRESH_HZ))


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
# attribute (a whole number of refreshes, e.g. 2 / REFRESH_HZ) and a frame() method returning the next frame as packed row-major
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
    refreshes = refreshes_per_frame(scene_class.FRAME_SECONDS)

    def run() -> None:
        canvas = matrix.CreateFrameCanvas()
        width, height = matrix.width, matrix.height
        while True:
            canvas.SetPixelsRGB(scene.frame(), width, height)
            canvas = matrix.SwapOnVSync(canvas, refreshes)

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


def fade_lut(factor: float, cutoff: int = 0) -> bytes:
    """Byte lookup table that scales by factor, for fading a whole framebuffer
    at C speed: fb = fb.translate(fade_lut(0.85)). Results below cutoff go to
    0, so trails end cleanly instead of lingering at levels too dim for LEDs
    to show well. (For numpy: np.frombuffer(fade_lut(...), np.uint8)[a].)"""
    return bytes(0 if int(v * factor) < cutoff else int(v * factor) for v in range(256))


def put(fb: bytearray, width: int, height: int, x: int, y: int, color: RGB) -> None:
    """Draw a pixel into a packed RGB framebuffer, skipping off-screen pixels."""
    if 0 <= x < width and 0 <= y < height:
        p = (y * width + x) * 3
        fb[p:p + 3] = bytes(color)


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


# --- Text --------------------------------------------------------------------


class BitmapFont:
    """A BDF font from the repo's fonts/ directory, for drawing text into numpy
    frames (the rgbmatrix graphics module only draws onto a canvas).

        font = BitmapFont("4x6")
        mask = font.render("12:34")  # (font.height, width) float32, 1 = ink
        frame[y:y + font.height, x:x + mask.shape[1]] += mask[..., None] * color
    """

    def __init__(self, name: str) -> None:
        import numpy as np  # here so the numpy-free scenes needn't load it

        self.glyphs: dict[str, tuple[np.ndarray, int]] = {}  # char: (cell mask, advance)
        ascent = descent = 0
        lines = iter((FONTS_DIR / f"{name}.bdf").read_text(encoding="latin-1").splitlines())
        for line in lines:
            key, _, value = line.partition(" ")
            if key == "FONT_ASCENT":
                ascent = int(value)
            elif key == "FONT_DESCENT":
                descent = int(value)
            elif key == "ENCODING":
                code = int(value)
            elif key == "DWIDTH":
                advance = int(value.split()[0])
            elif key == "BBX":
                w, h, x, y = map(int, value.split())
            elif key == "BITMAP":
                rows = [int(next(lines), 16) for _ in range(h)]
                # Place the glyph in a full-height cell, baseline at row ascent
                cell = np.zeros((ascent + descent, max(advance, x + w)), dtype=np.float32)
                top = ascent - h - y
                for r, bits in enumerate(rows):
                    if 0 <= top + r < cell.shape[0]:
                        nbits = (w + 7) // 8 * 8
                        cell[top + r, x:x + w] = [(bits >> (nbits - 1 - c)) & 1 for c in range(w)]
                self.glyphs[chr(code)] = (cell, advance)
        self.height = ascent + descent
        self.ascent = ascent

    def render(self, text: str, proportional: bool = False):
        """text as a (height, width) float32 mask. proportional trims each glyph
        to its ink plus a 1 px gap (spaces become 2 px), packing more of a
        fixed-width font's text into a narrow panel."""
        import numpy as np

        blank = self.glyphs.get("?") or next(iter(self.glyphs.values()))
        cells = []
        for ch in text:
            cell, advance = self.glyphs.get(ch, blank)
            if proportional:
                inked = np.flatnonzero(cell.any(axis=0))
                if ch == " " or not len(inked):
                    cell = np.zeros((self.height, 2), dtype=np.float32)
                else:
                    cell = np.pad(cell[:, inked[0]:inked[-1] + 1], ((0, 0), (0, 1)))
            else:
                cell = cell[:, :advance]
            cells.append(cell)
        if proportional and cells:
            cells[-1] = cells[-1][:, :-1] if cells[-1].shape[1] > 1 else cells[-1]
        return np.hstack(cells) if cells else np.zeros((self.height, 0), dtype=np.float32)
