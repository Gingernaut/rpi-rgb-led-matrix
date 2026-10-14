#!/usr/bin/env python
"""Pixel aquarium: fish swimming over a sandy floor, with swaying seaweed,
rising bubbles, and shimmering light from the surface.

The water and its moving caustic light are pre-rendered as a seamless loop at
startup, so each frame just copies a background and draws the sprites.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from matrix_common import mix, put, run_scene, scale

FRAME_SECONDS = 3 / 75  # 25 fps

# Background loop length in frames (~10 s), seamless because every light
# term completes a whole number of cycles in it
LOOP_FRAMES = 250

WATER_TOP = (0, 60, 120)
WATER_BOTTOM = (0, 14, 45)
LIGHT = (90, 140, 150)  # added by the brightest caustics near the surface
SAND = [(150, 120, 60), (120, 92, 45), (175, 145, 80)]
SAND_ROWS = 3

# Sprites face right; each character is a color from the sprite's palette,
# "." is transparent. Two frames per fish: tail swept up/down.
SPRITES = {
    "clownfish": (
        {"O": (255, 100, 0), "W": (255, 255, 255), "K": (20, 20, 20)},
        [
            ["...OO..", "O.OWOO.", "OOOWOKO", "O.OWOO.", "...OO.."],
            ["...OO..", "..OWOO.", "OOOWOKO", "OOOWOO.", "...OO.."],
        ],
    ),
    "blue_tang": (
        {"B": (20, 70, 255), "D": (0, 20, 110), "Y": (255, 210, 0), "K": (10, 10, 10)},
        [
            ["..BBB...", "Y.BDDBB.", "YYBBBBKB", "Y.BBDDB.", "..BBB..."],
            ["..BBB...", "..BDDBB.", "YYBBBBKB", "YYBBDDB.", "..BBB..."],
        ],
    ),
    "yellow_tang": (
        {"Y": (255, 220, 0), "K": (10, 10, 10)},
        [
            ["...Y..", "..YYY.", "Y.YYYY", "YYYYKY", "Y.YYYY", "..YYY.", "...Y.."],
            ["...Y..", "..YYY.", "..YYYY", "YYYYKY", "YYYYYY", "..YYY.", "...Y.."],
        ],
    ),
    "tetra": (
        {"C": (0, 220, 255), "R": (255, 30, 40)},
        [["CCCC", "RRRC"], [".CCC", "RRRC"]],
    ),
}

# (sprite, how many)
TANK = [("clownfish", 2), ("blue_tang", 1), ("yellow_tang", 1), ("tetra", 3)]

BUBBLE = (150, 210, 255)


def flip(rows: list[str]) -> list[str]:
    return [row[::-1] for row in rows]


def render_sprite(palette: dict, rows: list[str]) -> list[tuple[int, int, tuple]]:
    """Sprite rows to (dx, dy, color) for the opaque pixels."""
    return [
        (dx, dy, palette[ch])
        for dy, row in enumerate(rows)
        for dx, ch in enumerate(row)
        if ch != "."
    ]


class Fish:
    def __init__(self, kind: str, width: int, height: int, floor: int) -> None:
        palette, frames = SPRITES[kind]
        # Pixel lists for [facing right, facing left][tail frame]
        self.pixels = [
            [render_sprite(palette, f) for f in frames],
            [render_sprite(palette, flip(f)) for f in frames],
        ]
        self.w = len(frames[0][0])
        self.h = len(frames[0])
        self.tank_width = width
        self.top = 1
        self.bottom = floor - self.h - 1
        self.small = kind == "tetra"
        self.x = random.uniform(0, width - self.w)
        self.y = random.uniform(self.top, self.bottom)
        self.target_y = self.y
        self.facing = random.choice((1, -1))
        self.cruise = random.uniform(0.25, 0.45) if self.small else random.uniform(0.12, 0.3)
        self.speed = self.cruise
        self.phase = random.uniform(0, 2 * math.pi)
        self.pause = 0

    def swim(self, t: float) -> None:
        # Now and then drift to a new depth, pause, or turn around
        if random.random() < 0.01:
            self.target_y = random.uniform(self.top, self.bottom)
        if self.pause:
            self.pause -= 1
            self.speed *= 0.9
        else:
            self.speed += (self.cruise - self.speed) * 0.05
            if random.random() < 0.003:
                self.pause = random.randint(20, 60)
            elif random.random() < 0.002:
                self.facing = -self.facing
        self.x += self.speed * self.facing
        # Turn back just past the glass so fish don't vanish for long
        if self.x > self.tank_width - self.w + 2:
            self.facing = -1
        elif self.x < -2:
            self.facing = 1
        self.y += (self.target_y - self.y) * 0.02

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        # Tail beats faster when swimming faster
        beat = int((t * (2 + self.speed * 12) + self.phase) % 2)
        bob = math.sin(t * 1.3 + self.phase) * 0.6
        x0 = round(self.x)
        y0 = round(self.y + bob)
        for dx, dy, color in self.pixels[self.facing < 0][beat]:
            put(fb, width, height, x0 + dx, y0 + dy, color)


class Bubble:
    __slots__ = ("x", "y", "speed", "phase")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.speed = random.uniform(0.12, 0.25)
        self.phase = random.uniform(0, 2 * math.pi)


class Seaweed:
    def __init__(self, x: int, floor: int, height: int) -> None:
        self.x = x
        self.base = floor - 1
        self.length = height
        self.phase = random.uniform(0, 2 * math.pi)
        self.color = random.choice([(0, 150, 40), (20, 120, 30), (60, 170, 30)])
        self.tip = mix(self.color, (180, 255, 120), 0.4)

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        for i in range(self.length):
            # Sways more toward the tip, as a wave traveling up the stalk
            sway = math.sin(t * 0.9 + self.phase - i * 0.35) * (i / self.length) * 2.2
            color = self.tip if i >= self.length - 2 else scale(self.color, 0.7 + 0.3 * (i % 2))
            put(fb, width, height, self.x + round(sway), self.base - i, color)


def render_background(width: int, height: int) -> list[bytes]:
    """The water, surface light, and sand for each frame of the loop."""
    floor = height - SAND_ROWS
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    depth = y / max(floor - 1, 1)
    water = np.array(WATER_TOP, np.float32) * (1 - depth[..., None]) + np.array(WATER_BOTTOM, np.float32) * depth[..., None]

    frames = []
    for n in range(LOOP_FRAMES):
        p = 2 * math.pi * n / LOOP_FRAMES
        # Two slanted wave patterns crossing; their bright ridges are the
        # caustic light, fading out with depth
        waves = np.sin(x * 0.45 + y * 0.25 + p * 2) + np.sin(x * 0.3 - y * 0.35 - p * 3 + 1.0)
        light = np.clip(waves - 1.1, 0, 1) * np.clip(1 - depth * 1.1, 0, 1)
        # A shimmer along the surface row
        surface = (y == 0) * (0.25 + 0.2 * np.sin(x * 0.6 - p * 4))
        rgb = water + np.array(LIGHT, np.float32) * (light + surface)[..., None]
        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))

    # Sand: speckled rows along the bottom, the same in every frame
    rng = random.Random(7)
    sand = np.array([[rng.choice(SAND) for _ in range(width)] for _ in range(SAND_ROWS)], np.uint8)
    for frame in frames:
        frame[floor:] = sand
    return [frame.tobytes() for frame in frames]


class Aquarium:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.floor = height - SAND_ROWS
        self.background = render_background(width, height)
        self.fish = [Fish(kind, width, height, self.floor) for kind, count in TANK for _ in range(count)]
        self.seaweed = [
            Seaweed(x, self.floor, random.randint(height // 3, height // 2 + 2))
            for x in sorted(random.sample(range(2, width - 2), 5))
        ]
        self.bubbler_x = random.choice((random.randint(3, width // 4), random.randint(3 * width // 4, width - 4)))
        self.bubbles: list[Bubble] = []
        self.frame_count = 0

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.frame_count += 1
        t = self.frame_count * FRAME_SECONDS
        fb = bytearray(self.background[self.frame_count % LOOP_FRAMES])

        for weed in self.seaweed:
            weed.draw(fb, width, height, t)

        for fish in self.fish:
            fish.swim(t)
            fish.draw(fb, width, height, t)
            # Fish blow the occasional bubble
            if random.random() < 0.002:
                mouth = fish.x + (fish.w if fish.facing > 0 else -1)
                self.bubbles.append(Bubble(mouth, fish.y + fish.h // 2))

        # The bubbler sends up bursts of bubbles
        if random.random() < 0.08:
            self.bubbles.append(Bubble(self.bubbler_x + random.uniform(-0.5, 0.5), self.floor - 1))

        rising = []
        for b in self.bubbles:
            b.y -= b.speed
            if b.y < 0.5:
                continue  # popped at the surface
            rising.append(b)
            wobble = math.sin(t * 4 + b.phase) * 0.6
            put(fb, width, height, round(b.x + wobble), round(b.y), BUBBLE)
        self.bubbles = rising

        return fb


SCENE = Aquarium


def main() -> None:
    run_scene(Aquarium, "Aquarium LED matrix display")


if __name__ == "__main__":
    main()
