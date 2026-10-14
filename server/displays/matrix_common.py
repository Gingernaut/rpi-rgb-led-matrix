"""Shared setup for the Python display scripts: options, matrix, frame pacing.

The scripts run by path (see DisplayManager.start_python), so they import this
as a sibling module: `from matrix_common import ...`.
"""

import argparse
import signal
import sys
import time

# Wake this long before each frame deadline; SwapOnVSync then waits for the
# next vsync. Keep it under one refresh period (13.3 ms at 75 Hz).
VSYNC_MARGIN_SECONDS = 0.003


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


def run_display(display) -> None:
    """Run display.run() until SIGTERM/SIGINT, then display.cleanup() if it has one."""
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        display.run()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup = getattr(display, "cleanup", None)
        if cleanup is not None:
            cleanup()
