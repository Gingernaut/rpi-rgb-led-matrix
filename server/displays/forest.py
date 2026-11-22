#!/usr/bin/env python
"""A green woodland clearing at fern height. Gaps in the canopy let shafts
of sunlight through, with the odd sun shower. Deer and fawn, foxes, rabbits,
squirrels and banana slugs come through one visit at a time.

FOREST_EVENT forces a deer, fox, rabbit, squirrel or slug for previews.
"""

import math
import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from aquarium_common import render_sprite
from matrix_common import REFRESH_HZ, RGB, mix, run_scene

WIDTH, HEIGHT = 64, 32
FRAME_SECONDS = 3 / REFRESH_HZ
TRAIL_Y = 26
FIRST_VISIT = (8.0, 15.0)
NEXT_VISIT = (30.0, 60.0)
SUN_COLOR = np.array((129, 123, 65), dtype=np.float32)
MOTE_COLOR = np.array((72, 65, 35), dtype=np.float32)
RAIN_COLOR = np.array((61, 74, 72), dtype=np.float32)
AMBIENT = np.array((7, 12, 6), dtype=np.float32)
Y, X = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)


def tree(draw: ImageDraw.ImageDraw, rng: random.Random, x: int, base: int, width: int,
         lean: int, color: RGB, detail: bool = False) -> None:
    top = x + lean
    shade = mix(color, (12, 22, 14), 0.29)
    edge = mix(color, (116, 104, 60), 0.24)
    draw.polygon(
        [(top, 0), (top + width - 1, 0), (x + width, base - 4),
         (x + width + 2, base), (x + width - 1, base + 1), (x + 2, base),
         (x - 3, base + 1), (x - 1, base - 2)],
        fill=shade,
    )
    draw.polygon([(top + 1, 0), (top + width - 1, 0), (x + width, base - 4),
                  (x + width, base), (x + 2, base - 1)], fill=color)
    draw.line([(top + width - 1, 0), (x + width, base - 4)], fill=edge)
    if detail:
        for _ in range(12):
            y = rng.randrange(2, base - 2)
            bark_x = round(top - lean * y / base) + rng.randrange(width)
            draw.line([(bark_x, y), (bark_x, y + rng.randrange(1, 4))],
                      fill=rng.choice((shade, edge)))
        draw.line([(top + 2, 9), (top - 2, 7), (top - 7, 6), (top - 9, 4)], fill=color, width=2)


def fern(draw: ImageDraw.ImageDraw, x: int, y: int, size: int, sway: int,
         color: RGB) -> None:
    tip = mix(color, (139, 169, 75), 0.46)
    for side, reach in ((-1, 1.0), (1, 1.0), (-1, 0.55), (1, 0.55)):
        for step in range(size):
            f = step / max(size - 1, 1)
            px = round(x + (side * reach * size * 1.15 + sway) * f)
            py = round(y - size * (1 - reach * 0.4) * (2 * f - f * f))
            draw.point((px, py), fill=color)
            if step and step < size - 1:
                draw.point((px, py - 1), fill=tip)
                draw.point((px + side, py + 1), fill=color)


def forest_floor() -> np.ndarray:
    rng = random.Random(24)
    clearing = np.exp(-((X - 34) / 25) ** 2 - ((Y - 16) / 22) ** 2)
    crown = np.clip((9 - Y) / 9, 0, 1)
    colors = (np.array((27, 54, 33), np.float32)
              + clearing[..., None] * np.array((38, 46, 26), np.float32)
              - crown[..., None] * np.array((13, 20, 9), np.float32))
    image = Image.fromarray(colors.astype(np.uint8))
    draw = ImageDraw.Draw(image)

    for x, foot, width, lean in ((0, 18, 2, 1), (6, 21, 3, -2), (13, 20, 2, 1), (19, 19, 1, 1),
                                 (24, 22, 2, -1), (31, 19, 1, 0), (37, 21, 2, 1), (43, 20, 1, 1),
                                 (48, 21, 2, -2), (54, 20, 2, 1), (61, 19, 2, -1)):
        tree(draw, rng, x, foot, width, lean, rng.choice(((43, 74, 48), (49, 82, 51), (52, 85, 54))))

    draw.polygon([(0, 22), (11, 21), (21, 23), (37, 22), (49, 21), (63, 22),
                  (63, 31), (0, 31)], fill=(52, 77, 37))
    draw.polygon([(0, 25), (11, 24), (27, 25), (41, 24), (54, 23), (63, 24),
                  (63, 27), (52, 27), (37, 28), (24, 27), (12, 28), (0, 29)], fill=(73, 86, 43))
    draw.polygon([(0, 29), (15, 29), (25, 28), (38, 30), (57, 27), (63, 29),
                  (63, 31), (0, 31)], fill=(34, 62, 31))

    for _ in range(270):
        x, y = rng.randrange(WIDTH), rng.randrange(21, HEIGHT)
        colors = ((83, 105, 48), (63, 88, 39), (47, 68, 34), (89, 93, 45))
        draw.point((x, y), fill=rng.choice(colors))

    for x, base, width, lean in ((2, 22, 3, -1), (13, 23, 3, 2), (47, 22, 3, -2), (60, 23, 4, 1)):
        tree(draw, rng, x, base, width, lean, (51, 62, 36), detail=True)

    for x, y, radius in ((2, 13, 8), (6, 19, 7), (17, 20, 5), (23, 21, 4),
                          (45, 21, 4), (53, 19, 7), (62, 15, 8)):
        for _ in range(14):
            px, py = x + rng.randint(-radius, radius), y + rng.randint(-3, 2)
            span = rng.randint(1, 3)
            draw.ellipse((px, py, px + span, py + 1),
                         fill=rng.choice(((39, 76, 35), (49, 85, 42), (57, 95, 42), (62, 99, 46))))

    for x, y, size in ((11, 24, 3), (17, 24, 3), (23, 24, 2), (43, 24, 2), (47, 23, 3), (55, 23, 4)):
        fern(draw, x, y, size, 0, (56, 103, 42))

    draw.polygon([(-1, 29), (6, 28), (16, 28), (21, 29), (22, 30), (20, 31), (0, 31)],
                 fill=(70, 51, 34))
    draw.line([(0, 29), (8, 28), (18, 28), (20, 29)], fill=(85, 105, 44))
    draw.line([(4, 30), (14, 29), (18, 29)], fill=(96, 72, 45))
    draw.line([(14, 31), (16, 30), (21, 30)], fill=(49, 37, 25))
    return np.array(image, dtype=np.float32) + AMBIENT


def foreground(sway: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.new("RGBA", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    rng = random.Random(61)
    tree(draw, rng, 4, 29, 5, -2, (34, 49, 29), detail=True)
    tree(draw, rng, 56, 29, 6, 2, (35, 53, 30), detail=True)
    draw.line([(0, 3), (8, 5), (15, 4), (20, 2)], fill=(26, 49, 27), width=2)
    draw.line([(63, 5), (53, 6), (45, 4), (42, 1)], fill=(27, 51, 29), width=2)
    for x, y, radius, direction in ((1, 1, 8, 1), (9, 3, 5, 1), (20, 0, 5, 1),
                                     (44, 1, 5, -1), (54, 3, 7, -1), (63, 1, 6, -1)):
        for _ in range(9):
            px = x + rng.randint(-radius, radius) + direction * sway
            py = y + rng.randint(-2, 2)
            span = rng.randint(2, 5)
            draw.ellipse((px, py, px + span, py + 1),
                         fill=rng.choice(((26, 57, 27), (35, 69, 31), (47, 91, 36))))

    for x, y, size, direction in ((0, 31, 8, 1), (9, 32, 5, 1), (23, 32, 4, -1),
                                    (49, 32, 4, 1), (58, 31, 7, -1), (63, 32, 6, -1)):
        fern(draw, x, y, size, direction * sway, (37, 93, 39))
    rgba = np.asarray(image)
    mask = rgba[..., 3] != 0
    return mask, rgba[mask, :3].astype(np.float32)


class Sprite:
    def __init__(self, palette: dict[str, RGB], body: list[str], feet: list[list[str]] | None = None) -> None:
        poses = [body + rows for rows in feet] if feet is not None else [body]
        self.width = max(len(row) for pose in poses for row in pose)
        self.height = len(poses[0])
        self.poses = []
        for pose in poses:
            points = render_sprite(palette, pose)
            self.poses.append((np.array([x for x, _, _ in points]), np.array([y for _, y, _ in points]),
                               np.array([color for _, _, color in points], dtype=np.float32)))

    def draw(self, frame: np.ndarray, x: float, ground: float, facing: int, pose: int = 0,
             climbing: bool = False) -> None:
        u, v, colors = self.poses[pose % len(self.poses)]
        if climbing:
            xs, ys = round(x) + v, round(ground) - u
        else:
            xs = round(x) + (u if facing > 0 else self.width - 1 - u)
            ys = round(ground) - self.height + 1 + v
        visible = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        frame[ys[visible], xs[visible]] = colors[visible]


DEER = Sprite(
    {"a": (159, 112, 63), "b": (130, 83, 47), "l": (191, 148, 91),
     "w": (205, 183, 130), "k": (51, 36, 29)},
    ["............l.l", "............ala", ".............lka", "............alaak",
     "............alw", ".w.llllaaaa.baw", ".aalaaaaaaaalab", "..bbaabaabaaab",
     "...bbaa..baab"],
    [["...b.b....b.b", "...b..b...b..b", "...k..k...k..k"],
     ["....bb....b.b", ".....b...b...b", ".....k...k...k"],
     ["...b..b....bb", "..b....b....b", "..k....k....k"]],
)
FAWN = Sprite(
    {"a": (171, 109, 62), "b": (133, 80, 43), "w": (211, 179, 121), "k": (57, 42, 29)},
    ["........w.w", "........aak", ".......aabw", ".awaawwaaab", "..babaaaab"],
    [["..b.b..b.b", "..k.k..k.k"], ["...b..b.b", "....k.k..k"], ["..b..b.bb", ".k....kk"]],
)
FOX = Sprite(
    {"r": (177, 85, 34), "a": (209, 120, 49), "w": (213, 189, 136), "k": (55, 41, 30)},
    ["............a.a", "............rar", "...........arak", ".w........raaww",
     "wwraaaaaaararw", ".wrrrraarrrrk", "...rr..r...r"],
    [[".......k.k.k.k", ".......k.k.k.k"], ["........kk.kk", ".......k....k"],
     ["......k...kk.k", ".....k........k"]],
)
RABBIT = Sprite(
    {"g": (136, 124, 90), "b": (98, 82, 62), "l": (169, 153, 111), "w": (198, 184, 142), "k": (55, 44, 36)},
    [".......l.l", ".......g.l", ".......gl", ".....glgkg", ".w.gggglll", ".wggggggg", "..bbbbgg"],
    [["..bl..bl"], ["...bbll"], [".bb.....bl"]],
)
SQUIRREL = Sprite(
    {"a": (128, 89, 53), "l": (165, 125, 77), "b": (94, 63, 40), "w": (199, 163, 104), "k": (49, 37, 31)},
    ["aaa", "alaa......a", ".ala.....llk", "..ala...law", "...aalaaaaw", ".....baaal"],
    [[".....b...b"], ["....b...bb"], ["......bb..b"]],
)
SLUG = Sprite(
    {"y": (175, 155, 50), "l": (219, 199, 83), "b": (117, 112, 41)},
    ["....l.l", "..lllyl", ".byyyyy", "bbbbbb"],
    [[".bbbbb"], ["bbbbbb"]],
)


@dataclass(frozen=True)
class Walk:
    sprite: Sprite
    travel: float
    pause: float
    stride: float
    bounce: float = 0

    @property
    def duration(self) -> float:
        return self.travel + self.pause

    def draw(self, frame: np.ndarray, age: float, facing: int, ground: float = TRAIL_Y) -> None:
        if not 0 <= age < self.duration:
            return
        halfway = self.travel * 0.49
        moving = age - min(self.pause, max(0.0, age - halfway))
        still = halfway <= age < halfway + self.pause
        x = -self.sprite.width + (WIDTH + self.sprite.width) * (moving / self.travel)
        if facing < 0:
            x = WIDTH - self.sprite.width - x
        phase = (moving / self.stride) % 1.0
        pose = 0 if still else int(phase * len(self.sprite.poses))
        height = 0 if still else math.sin(math.pi * phase) * self.bounce
        self.sprite.draw(frame, x, ground - height, facing, pose)


WALKS = {
    "deer": Walk(DEER, 9.0, 1.6, 0.72),
    "fox": Walk(FOX, 8.0, 0.35, 0.56),
    "rabbit": Walk(RABBIT, 6.5, 0.8, 0.64, 2),
}
FAWN_WALK = Walk(FAWN, 9.2, 0.5, 0.55, 0.8)
VISIT_WEIGHTS = {"deer": 1, "fox": 2, "rabbit": 3, "squirrel": 4, "slug": 2}
FAWN_DELAY = 3.6
SQUIRREL_RUN = 5.8
SQUIRREL_CLIMB = 2.8
SLUG_SECONDS = 18.0
VISIT_LENGTHS = {
    **{name: walk.duration for name, walk in WALKS.items()},
    "deer": FAWN_DELAY + FAWN_WALK.duration,
    "squirrel": SQUIRREL_RUN + SQUIRREL_CLIMB,
    "slug": SLUG_SECONDS,
}


class Forest:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the clearing is drawn for a {WIDTH}x{HEIGHT} panel")
        self.background = forest_floor()
        self.front = [foreground(sway) for sway in (-1, 0, 1)]
        self.rng = random.Random(random.getrandbits(64))
        self.frames = 0
        self.only = os.environ.get("FOREST_EVENT")
        if self.only and self.only not in VISIT_WEIGHTS:
            raise ValueError(f"FOREST_EVENT must be one of {', '.join(VISIT_WEIGHTS)}")
        self.next_visit = 2.0 if self.only else self.rng.uniform(*FIRST_VISIT)
        self.visitor: str | None = None
        self.visited: str | None = None
        self.visit_start = 0.0
        self.facing = 1
        self.open_canopy(self.rng.uniform(3.0, 6.0))
        self.motes = [(self.rng.uniform(1, 30), self.rng.uniform(-3, 3), self.rng.uniform(0.7, 2.1))
                      for _ in range(12)]
        self.rain = np.array([(self.rng.randrange(WIDTH), self.rng.randrange(HEIGHT))
                              for _ in range(36)], dtype=np.float32).T

    def open_canopy(self, start: float) -> None:
        self.sun_start = start
        self.sun_end = start + self.rng.uniform(8, 14)
        self.shower = self.rng.random() < 0.18

    def sunlight(self, t: float) -> np.ndarray | None:
        if t >= self.sun_end:
            self.open_canopy(t + self.rng.uniform(15, 33))
        if t <= self.sun_start:
            return None

        fade = min(1.0, (t - self.sun_start) / 2.5, (self.sun_end - t) / 2)
        fade = fade * fade * (3 - 2 * fade)
        sway = 0.95 * math.sin(t * 0.44) + 0.32 * math.sin(t * 1.13)
        shaft = 15.5 + sway + 0.57 * Y
        main = np.clip(1 - np.abs(X - shaft) / (1.9 + 0.115 * Y), 0, 1) ** 1.6
        second = np.clip(1 - np.abs(X - shaft - 11) / (1.1 + 0.083 * Y), 0, 1) ** 1.6
        pool = np.exp(-((X - 31 - sway) / 11) ** 2 - ((Y - 26) / 2.8) ** 2)
        taper = np.clip((30 - Y) / 7, 0, 1)
        return fade * (taper * (0.58 * main + 0.31 * second) + 0.27 * pool)

    def air(self, frame: np.ndarray, light: np.ndarray, t: float) -> None:
        if self.shower and self.sun_start + 3 < t < self.sun_end - 1:
            rain_x, rain_y = self.rain
            ys = ((rain_y + (t * 24) % HEIGHT) % HEIGHT).astype(np.intp)
            xs = ((rain_x + ys * 0.2) % WIDTH).astype(np.intp)
            frame[ys, xs] += light[ys, xs, None] * RAIN_COLOR
        else:
            for index, (y0, offset, speed) in enumerate(self.motes):
                y = int((y0 - t * speed) % (TRAIL_Y + 1))
                x = round(16 + y * 0.57 + offset + (11 if index % 2 else 0) + 1.1 * math.sin(t + index))
                frame[y, x] += light[y, x] * MOTE_COLOR

    def advance_visitors(self, t: float) -> None:
        if self.visitor is not None:
            if t - self.visit_start >= VISIT_LENGTHS[self.visitor]:
                self.visitor = None
                self.next_visit = t + (4 if self.only else self.rng.uniform(*NEXT_VISIT))
        elif t >= self.next_visit:
            names = [name for name in VISIT_WEIGHTS if name != self.visited]
            self.visitor = self.only or self.rng.choices(names, [VISIT_WEIGHTS[name] for name in names])[0]
            self.visited = self.visitor
            self.visit_start = t
            self.facing = self.rng.choice((-1, 1))

    def draw_visitor(self, frame: np.ndarray, age: float) -> None:
        if self.visitor in WALKS:
            WALKS[self.visitor].draw(frame, age, self.facing)
            if self.visitor == "deer":
                FAWN_WALK.draw(frame, age - FAWN_DELAY, self.facing, TRAIL_Y + 1)
        elif self.visitor == "slug":
            progress = age / SLUG_SECONDS
            x = 5 + 12 * progress if self.facing > 0 else 17 - 12 * progress
            SLUG.draw(frame, x, 28, self.facing, int(age * 1.3))
        elif self.visitor == "squirrel" and age < SQUIRREL_RUN:
            f = min(1.0, age / (SQUIRREL_RUN - 0.45))
            x = -SQUIRREL.width + (47 + SQUIRREL.width) * f
            if self.facing < 0:
                x = WIDTH - SQUIRREL.width - x
            hop = math.sin(math.pi * ((age * 3.1) % 1)) * 1.45 if f < 1 else 0
            SQUIRREL.draw(frame, x, TRAIL_Y - hop, self.facing, int(age * 9.3) if f < 1 else 0)

    def frame(self) -> np.ndarray:
        t = self.frames * FRAME_SECONDS
        self.frames += 1
        self.advance_visitors(t)

        frame = self.background.copy()
        age = t - self.visit_start
        self.draw_visitor(frame, age)
        light = self.sunlight(t)
        if light is not None:
            frame += light[..., None] * SUN_COLOR
            self.air(frame, light, t)

        wind = math.sin(t * 0.76) + 0.34 * math.sin(t * 1.55)
        mask, color = self.front[round(max(-1, min(1, wind))) + 1]
        frame[mask] = color if light is None else color + light[mask, None] * (SUN_COLOR * 0.22)
        if self.visitor == "squirrel" and age >= SQUIRREL_RUN:
            climb = (age - SQUIRREL_RUN) / SQUIRREL_CLIMB
            trunk = 54 if self.facing > 0 else 4
            SQUIRREL.draw(frame, trunk + math.sin(climb * 9) * 1.2, TRAIL_Y * (1 - climb),
                          self.facing, int(age * 9.3), climbing=True)

        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


SCENE = Forest


def main() -> None:
    run_scene(Forest, "Forest clearing LED matrix display")


if __name__ == "__main__":
    main()
