#!/usr/bin/env python
"""Starfield: flying through space, with periodic jumps to warp speed.

Stars are points in 3D that drift toward the viewer and are projected onto
the panel, so near ones move fast and bright and far ones crawl. Each star is
drawn as a streak from its position a few frames ago to where it is now, so
fast stars leave light trails. Every so often the ship accelerates to warp:
the streaks stretch out and blue-shift, then it drops back to cruising. The
vanishing point drifts a little, like a gentle bank.

Requires the rgbmatrix Python bindings (make build-python).
"""

import math
import random

from matrix_common import fade_lut, mix, run_scene

FRAME_SECONDS = 2 / 75  # 37.5 fps

STAR_COUNT = 100  # for a 2:1 panel; scaled with the aspect ratio
# Stars spawn at depth 1 across SPAWN_SPREAD times the view, so they still fill
# it as the vanishing point drifts, and are recycled once their streak's tail
# passes NEAR_Z or leaves the panel.
NEAR_Z = 0.02
SPAWN_SPREAD = 1.2

# Depth covered per frame while cruising and at full warp
CRUISE_SPEED = 0.007
WARP_SPEED = 0.05

# Streak length, as how many frames of travel the tail lags the head
CRUISE_TRAIL = 1.5
WARP_TRAIL = 5.0

# Flight plan, in seconds: cruise, speed up, warp, slow down, repeat
CRUISE_SECONDS = (8.0, 15.0)
RAMP_UP_SECONDS = 2.5
WARP_SECONDS = (3.0, 5.0)
RAMP_DOWN_SECONDS = 1.5

# Afterglow behind the streaks: a touch while cruising, a bit more at warp.
# Values under DIM_CUTOFF are dropped, since they flicker on the panel.
DIM_CUTOFF = 10
FADE_LUTS = [fade_lut(f) for f in (0.3, 0.35, 0.4, 0.45)]
CUTOFF_LUT = bytes(0 if v < DIM_CUTOFF else v for v in range(256))

STAR_TINTS = [(255, 255, 255), (255, 255, 255), (190, 210, 255), (255, 225, 190)]
# At warp, heads go blue-white and tails deep blue
WARP_HEAD = (200, 220, 255)
WARP_TAIL = (60, 110, 255)
TAIL_LEVEL = 0.3  # tail brightness relative to the head
# Full brightness nearer than this depth; farther stars dim with distance
BRIGHT_Z = 0.3


class Star:
    __slots__ = ("x", "y", "z", "tint")

    def __init__(self, z: float, spread_x: float, spread_y: float) -> None:
        self.x = random.uniform(-spread_x, spread_x)
        self.y = random.uniform(-spread_y, spread_y)
        self.z = z
        self.tint = random.choice(STAR_TINTS)


def ease(f: float) -> float:
    return f * f * (3 - 2 * f)


class Starfield:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.focal = height / 2
        # Half the view's size at depth 1, plus the spawn margin
        self.spread = (width / 2 / self.focal * SPAWN_SPREAD, height / 2 / self.focal * SPAWN_SPREAD)
        self.fb = bytearray(width * height * 3)
        star_count = round(STAR_COUNT * width / height / 2)
        self.stars = [Star(random.uniform(NEAR_Z, 1.0), *self.spread) for _ in range(star_count)]
        self.frame_count = 0
        self.plan = self.flight_plan()

    def flight_plan(self):
        """Yield the warp factor (0 cruising, 1 full warp) for each frame, forever."""
        def frames(seconds: float) -> int:
            return max(1, round(seconds / FRAME_SECONDS))

        while True:
            for _ in range(frames(random.uniform(*CRUISE_SECONDS))):
                yield 0.0
            ramp = frames(RAMP_UP_SECONDS)
            for i in range(ramp):
                yield ease(i / ramp)
            for _ in range(frames(random.uniform(*WARP_SECONDS))):
                yield 1.0
            ramp = frames(RAMP_DOWN_SECONDS)
            for i in range(ramp):
                yield ease(1 - i / ramp)

    def frame(self) -> bytearray:
        width, height, focal = self.width, self.height, self.focal
        self.frame_count += 1
        warp = next(self.plan)
        # Blend speeds geometrically, so the ramp feels like steady acceleration
        # rather than a lurch in the middle
        speed = CRUISE_SPEED * (WARP_SPEED / CRUISE_SPEED) ** warp
        trail = speed * (CRUISE_TRAIL + (WARP_TRAIL - CRUISE_TRAIL) * warp)
        streak_colors = {}  # tint: (head, tail) colors at this warp factor
        ceil, floor = math.ceil, math.floor

        fb = self.fb = self.fb.translate(FADE_LUTS[round(warp * (len(FADE_LUTS) - 1))])

        # Vanishing point drifting gently around the middle
        t = self.frame_count * FRAME_SECONDS
        cx = width / 2 + width * 0.08 * math.sin(t * 0.21)
        cy = height / 2 + height * 0.1 * math.sin(t * 0.13 + 1.0)

        for star in self.stars:
            z = star.z = star.z - speed
            tail_z = z + trail
            fx, fy = star.x * focal, star.y * focal
            tx = cx + fx / tail_z
            ty = cy + fy / tail_z
            if tail_z <= NEAR_Z or not (0 <= tx < width and 0 <= ty < height):
                # Passed us or left the view: respawn far away
                star.__init__(1.0, *self.spread)
                continue
            # The head may be far off the panel; the streak is clipped below
            z = max(z, NEAR_Z)
            hx = cx + fx / z
            hy = cy + fy / z

            # Nearer is brighter; everything brightens at warp
            level = min(1.0, (BRIGHT_Z / z) ** 1.5 * (1 + warp))
            tint = star.tint
            if tint not in streak_colors:
                streak_colors[tint] = (mix(tint, WARP_HEAD, warp), mix(tint, WARP_TAIL, warp))
            head, tail = streak_colors[tint]
            hr, hg, hb = head[0] * level, head[1] * level, head[2] * level
            tail_level = level * TAIL_LEVEL
            tr, tg, tb = tail[0] * tail_level, tail[1] * tail_level, tail[2] * tail_level

            # The head: a solid pixel, so even a slow star is a crisp point
            if 0 <= hx < width and 0 <= hy < height:
                p = (int(hy) * width + int(hx)) * 3
                if hr > fb[p]:
                    fb[p] = int(hr)
                if hg > fb[p + 1]:
                    fb[p + 1] = int(hg)
                if hb > fb[p + 2]:
                    fb[p + 2] = int(hb)

            # The body: an anti-aliased (Wu) line from tail to head, stepping one
            # pixel at a time along its major axis "a" and splitting each step
            # between the two pixels it straddles on the minor axis "b"
            dx = hx - tx
            dy = hy - ty
            if abs(dx) >= abs(dy):
                a0, da, b0, db = tx, dx, ty, dy
                a_size, b_size, a_stride, b_stride = width, height, 3, width * 3
            else:
                a0, da, b0, db = ty, dy, tx, dx
                a_size, b_size, a_stride, b_stride = height, width, width * 3, 3
            if abs(da) < 1:
                continue  # a slow star: just the head
            # Visit the pixel centers (+0.5) the streak covers on the panel,
            # stepping the minor position and the tail-to-head color along
            lo, hi = (a0, a0 + da) if da > 0 else (a0 + da, a0)
            start = max(0, ceil(lo - 0.5))
            end = min(a_size, floor(hi + 0.5))
            if start >= end:
                continue
            slope = db / da
            along = start + 0.5 - a0
            b = b0 + along * slope - 0.5
            f = along / da  # 0 at the tail, 1 at the head
            cr, cg, cb = tr + (hr - tr) * f, tg + (hg - tg) * f, tb + (hb - tb) * f
            r_step, g_step, b_step = (hr - tr) / da, (hg - tg) / da, (hb - tb) / da
            for p_a in range(start * a_stride, end * a_stride, a_stride):
                bi = floor(b)
                # sqrt keeps the pair's light near one whole pixel's, since the
                # panel's brightness response is gamma-like, not linear
                far = (b - bi) ** 0.5
                near = (1.0 - b + bi) ** 0.5
                p = p_a + bi * b_stride
                if 0 <= bi < b_size:
                    v = int(cr * near)
                    if v > fb[p]:
                        fb[p] = v
                    v = int(cg * near)
                    if v > fb[p + 1]:
                        fb[p + 1] = v
                    v = int(cb * near)
                    if v > fb[p + 2]:
                        fb[p + 2] = v
                if -1 <= bi < b_size - 1:
                    p += b_stride
                    v = int(cr * far)
                    if v > fb[p]:
                        fb[p] = v
                    v = int(cg * far)
                    if v > fb[p + 1]:
                        fb[p + 1] = v
                    v = int(cb * far)
                    if v > fb[p + 2]:
                        fb[p + 2] = v
                b += slope
                cr += r_step
                cg += g_step
                cb += b_step

        return fb.translate(CUTOFF_LUT)


SCENE = Starfield


def main() -> None:
    run_scene(Starfield, "Starfield LED matrix display")


if __name__ == "__main__":
    main()
