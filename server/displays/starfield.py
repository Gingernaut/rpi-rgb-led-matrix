#!/usr/bin/env python
"""Starfield: flying through space, with periodic jumps to warp speed.

Stars are points in 3D that drift toward the viewer and are projected onto
the panel, so near ones move fast and bright and far ones crawl. Every so
often the ship accelerates to warp: stars stretch into blue-shifted streaks,
then it drops back to cruising. The vanishing point drifts a little, like a
gentle bank.

Requires the rgbmatrix Python bindings (make build-python).
"""

import math
import random

from matrix_common import fade_lut, mix, put_max, run_scene, scale

FRAME_SECONDS = 2 / 75  # 37.5 fps

STAR_COUNT = 110
# Stars spawn at depth 1 and are recycled once closer than NEAR_Z. They spawn
# across x in ±SPREAD_X and y in ±SPREAD_Y, scaled by the panel's half height,
# so they fill the view as they approach.
NEAR_Z = 0.02
SPREAD_X = 2.4
SPREAD_Y = 1.3

# Depth covered per frame while cruising and at full warp
CRUISE_SPEED = 0.005
WARP_SPEED = 0.05

# Flight plan, in seconds: cruise, speed up, warp, slow down, repeat
CRUISE_SECONDS = (8.0, 15.0)
RAMP_SECONDS = 2.5
WARP_SECONDS = (3.0, 5.0)

# Trail persistence: almost none while cruising, a smear at warp
FADE_LUTS = [fade_lut(f) for f in (0.35, 0.42, 0.5, 0.56, 0.62)]

STAR_TINTS = [(255, 255, 255), (255, 255, 255), (190, 210, 255), (255, 225, 190)]
WARP_TINT = (140, 175, 255)
MIN_BRIGHTNESS = 0.18  # far stars stay visible


class Star:
    __slots__ = ("x", "y", "z", "tint", "sx", "sy")

    def __init__(self, z: float) -> None:
        self.x = random.uniform(-SPREAD_X, SPREAD_X)
        self.y = random.uniform(-SPREAD_Y, SPREAD_Y)
        self.z = z
        self.tint = random.choice(STAR_TINTS)
        self.sx: float | None = None  # last projected position, for streaks
        self.sy: float | None = None


def ease(f: float) -> float:
    return f * f * (3 - 2 * f)


class Starfield:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.focal = height / 2
        self.fb = bytearray(width * height * 3)
        self.stars = [Star(random.uniform(NEAR_Z, 1.0)) for _ in range(STAR_COUNT)]
        self.frame_count = 0
        self.plan = self.flight_plan()

    def flight_plan(self):
        """Yield the warp factor (0 cruising, 1 full warp) for each frame, forever."""
        def frames(seconds: float) -> int:
            return max(1, round(seconds / FRAME_SECONDS))

        while True:
            for _ in range(frames(random.uniform(*CRUISE_SECONDS))):
                yield 0.0
            ramp = frames(RAMP_SECONDS)
            for i in range(ramp):
                yield ease(i / ramp)
            for _ in range(frames(random.uniform(*WARP_SECONDS))):
                yield 1.0
            for i in range(ramp):
                yield ease(1 - i / ramp)

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.frame_count += 1
        warp = next(self.plan)
        speed = CRUISE_SPEED + (WARP_SPEED - CRUISE_SPEED) * warp

        fb = self.fb = self.fb.translate(FADE_LUTS[round(warp * (len(FADE_LUTS) - 1))])

        # Vanishing point drifting gently around the middle
        t = self.frame_count * FRAME_SECONDS
        cx = width / 2 + width * 0.08 * math.sin(t * 0.21)
        cy = height / 2 + height * 0.1 * math.sin(t * 0.13 + 1.0)

        for star in self.stars:
            star.z -= speed
            sx = cx + star.x / star.z * self.focal if star.z > NEAR_Z else -1.0
            sy = cy + star.y / star.z * self.focal if star.z > NEAR_Z else -1.0
            if not (0 <= sx < width and 0 <= sy < height):
                # Passed us or left the view: respawn far away
                star.__init__(1.0)
                continue

            # Nearer is brighter; everything brightens a little at warp
            brightness = max(MIN_BRIGHTNESS, min(1.0, (1.0 - star.z) ** 1.6 * (1.3 + warp)))
            color = scale(mix(star.tint, WARP_TINT, warp * 0.7), brightness)

            if star.sx is None:
                put_max(fb, width, height, int(sx), int(sy), color)
            else:
                # Streak from where the star was last frame, dimmer at the tail
                dx = sx - star.sx
                dy = sy - star.sy
                steps = int(max(abs(dx), abs(dy)))
                for i in range(steps + 1):
                    f = i / steps if steps else 1.0
                    put_max(fb, width, height, int(star.sx + dx * f), int(star.sy + dy * f),
                            scale(color, 0.35 + 0.65 * f))
            star.sx = sx
            star.sy = sy

        return fb


SCENE = Starfield


def main() -> None:
    run_scene(Starfield, "Starfield LED matrix display")


if __name__ == "__main__":
    main()
