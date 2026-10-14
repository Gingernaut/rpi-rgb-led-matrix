#!/usr/bin/env python
"""Render a scene to an animated GIF, or time its frames, without a matrix.

    uv run server/displays/preview.py fireworks --seconds 20 --out /tmp/fireworks.gif
    uv run server/displays/preview.py plasma --bench 500

MODE is a display script in this directory that defines SCENE (see
matrix_common's "Scenes" notes), e.g. plasma or matrix_rain.
"""

import argparse
import importlib
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--seed", type=int, help="seed random (and numpy's) for a repeatable run")
    parser.add_argument("--seconds", type=float, default=10.0, help="GIF length")
    parser.add_argument("--skip", type=float, default=0.0, help="seconds to run before recording or timing")
    parser.add_argument("--zoom", type=int, default=8, help="GIF pixels per LED")
    parser.add_argument("--out", help="GIF path (default: MODE.gif)")
    parser.add_argument("--bench", type=int, metavar="FRAMES", help="time this many frames instead")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy
            numpy.random.seed(args.seed)
        except ImportError:
            pass

    scene_class = importlib.import_module(args.mode.replace("-", "_")).SCENE
    scene = scene_class(args.width, args.height)
    period = scene_class.FRAME_SECONDS
    frame_bytes = args.width * args.height * 3

    for _ in range(round(args.skip / period)):
        scene.frame()

    if args.bench:
        times = []
        for _ in range(args.bench):
            start = time.perf_counter()
            scene.frame()
            times.append(time.perf_counter() - start)
        times.sort()
        mean = sum(times) / len(times)
        print(f"{args.mode}: {args.bench} frames, mean {mean * 1000:.2f} ms, "
              f"p99 {times[int(len(times) * 0.99)] * 1000:.2f} ms, "
              f"max {times[-1] * 1000:.2f} ms; budget {period * 1000:.1f} ms "
              f"({mean / period:.0%} of one core)")
        return

    from PIL import Image

    frames = []
    for _ in range(round(args.seconds / period)):
        data = bytes(scene.frame())
        if len(data) != frame_bytes:
            sys.exit(f"frame() returned {len(data)} bytes, expected {frame_bytes}")
        image = Image.frombytes("RGB", (args.width, args.height), data)
        frames.append(image.resize((args.width * args.zoom, args.height * args.zoom), Image.NEAREST))
    out = args.out or f"{args.mode}.gif"
    # GIF frame delays are in 10 ms steps, so playback speed is approximate
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(period * 1000), loop=0)
    print(f"wrote {out}: {len(frames)} frames")


if __name__ == "__main__":
    main()
