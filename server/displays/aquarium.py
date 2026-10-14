#!/usr/bin/env python
"""Pixel aquarium: fish swimming over a sandy floor among swaying seaweed,
with rising bubbles, a treasure chest, and light rays from the surface.

The water, light rays, sand and ornaments are pre-rendered as a seamless loop
at startup, so each frame copies a background and draws the moving things.
Fish are drawn far to near, the far ones hazed toward the water color, some
seaweed stands in front of them, and the neon tetras swim as a school.

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from matrix_common import mix, put, run_scene

FRAME_SECONDS = 3 / 75  # 25 fps

# Background loop length in frames (20 s), seamless because every light term
# completes a whole number of cycles in it
LOOP_FRAMES = 500

WATER_TOP = (0, 48, 105)
WATER_BOTTOM = (0, 10, 36)
RAYS = (60, 110, 115)  # added to the water by a light ray at full strength
SURFACE = (70, 120, 140)  # the shimmer along the top row
HAZE = (0, 26, 66)  # far fish and the back seaweed fade toward this
SAND = [(150, 120, 60), (120, 92, 45), (175, 145, 80), (135, 105, 52)]
PEBBLES = [(110, 110, 105), (150, 85, 70), (80, 75, 70)]
SAND_ROWS = 3  # on average; the sand's surface rolls a row up and down

# Sprites face right; each character is a color from the sprite's palette,
# "." is transparent. Two frames per fish: tail swept up/down.
SPRITES = {
    "clownfish": (
        {"O": (255, 100, 0), "W": (255, 255, 255), "K": (20, 20, 20)},
        [
            ["...OO...", "O.OWOWO.", "OOOWOWKO", "O.OWOWO.", "...OO..."],
            ["...OO...", "..OWOWO.", "OOOWOWKO", "OOOWOWO.", "...OO..."],
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

# (sprite, how many) swimming on their own
TANK = [("clownfish", 2), ("blue_tang", 1), ("yellow_tang", 1)]
# The tetra school: a leader, and each follower's place relative to it. The
# places are fixed in the tank, not relative to the leader's heading, so when
# the school turns it wheels around in place and the back becomes the front.
SCHOOL = "tetra"
SCHOOL_PLACES = [(-6, -3), (-6, 3), (6, 2), (-12, 0)]

BUBBLE = (150, 210, 255)

# The chest sits on the sand; now and then its lid lifts and lets out a burst
# of bubbles. Closed it's part of the background; open, the lid is drawn a row
# higher over it, with the gold inside showing beneath.
CHEST_PALETTE = {"L": (150, 80, 25), "B": (100, 50, 16), "G": (230, 175, 40), "K": (35, 18, 5)}
CHEST = [".LLLLLL.", "LLLLLLLL", "GGGGGGGG", "BBBGGBBB", "BBBBBBBB"]
CHEST_OPEN = [".LLLLLL.", "LLLLLLLL", "KGKGGKGK"]  # drawn from the row above
CHEST_SECONDS = (15.0, 40.0)  # closed for this long
CHEST_OPEN_FRAMES = 30
STONE = (95, 95, 100)  # the bubbler's air stone


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
    """A fish cruising back and forth, now and then pausing, turning around or
    changing depth. Given a leader, it keeps its place near that fish instead.

    depth is 0 (near) to 1 (far): far fish are hazed and swim a little slower."""

    def __init__(self, kind: str, width: int, top: int, bottom: int, depth: float,
                 leader: "Fish | None" = None, place: tuple[int, int] = (0, 0)) -> None:
        palette, frames = SPRITES[kind]
        palette = {ch: mix(color, HAZE, depth * 0.22) for ch, color in palette.items()}
        # Pixel lists for [facing right, facing left][tail frame]
        self.pixels = [
            [render_sprite(palette, f) for f in frames],
            [render_sprite(palette, flip(f)) for f in frames],
        ]
        self.w = len(frames[0][0])
        self.h = len(frames[0])
        self.tank_width = width
        self.top = top
        self.bottom = max(top, bottom - self.h)
        self.depth = depth
        self.leader = leader
        self.place = place
        self.offset = place
        # How far the school spreads left and right of this fish, so a leader
        # turns early enough to keep its followers inside the glass
        self.spread = (0, 0)
        small = kind == "tetra"
        self.cruise = (random.uniform(0.22, 0.34) if small else random.uniform(0.12, 0.28)) * (1 - 0.3 * depth)
        self.speed = self.cruise
        self.vy = 0.0
        self.tail = random.uniform(0, 2)  # tail beat phase; the frame is int(tail)
        self.bob = random.uniform(0, 2 * math.pi)
        self.pause = 0
        if leader:
            self.x = leader.x + place[0]
            self.y = min(max(leader.y + place[1], self.top), self.bottom)
            self.facing = leader.facing
        else:
            self.x = random.uniform(0, width - self.w)
            self.y = random.uniform(self.top, self.bottom)
            self.facing = random.choice((1, -1))
            self.pick_turn()
        self.target_y = self.y

    def pick_turn(self) -> None:
        """Choose where to turn around next: usually just short of the glass,
        sometimes partway across, and sometimes out of view for a while (the
        tank is bigger than the panel)."""
        if self.facing > 0:
            glass = self.tank_width - self.w - self.spread[1]
        else:
            glass = self.spread[0]
        roll = random.random()
        if roll < 0.15:
            beyond = random.uniform(self.w + 4, self.w + 30)
        elif roll < 0.35:
            beyond = -random.uniform(8, self.tank_width / 2)
        else:
            beyond = random.uniform(-4, 1)
        self.turn_x = glass + beyond * self.facing
        if (self.turn_x - self.x) * self.facing < 8:
            self.turn_x = glass

    def swim(self) -> None:
        if self.leader:
            self.follow()
        else:
            self.wander()
        self.x += self.speed * self.facing
        self.y += self.vy
        # Tail beats faster when swimming faster
        self.tail = (self.tail + 0.08 + self.speed * 0.5) % 2

    def wander(self) -> None:
        # Now and then glide to a new depth, slowly and smoothly
        if random.random() < 0.006:
            self.target_y = random.uniform(self.top, self.bottom)
        climb = max(-0.07, min(0.07, (self.target_y - self.y) * 0.03))
        self.vy += (climb - self.vy) * 0.1

        # Slow down nearing the turning point, and turn there
        ahead = (self.turn_x - self.x) * self.facing
        want = self.cruise * min(1.0, max(0.15, ahead / 8))
        if self.pause:
            # Hover a moment, then maybe head back the other way
            self.pause -= 1
            want = 0.0
            if not self.pause and random.random() < 0.5:
                self.turn()
        elif random.random() < 0.002:
            self.pause = random.randint(25, 100)
        if ahead <= 0:
            self.turn()
        self.speed += (want - self.speed) * 0.05

    def turn(self) -> None:
        self.facing = -self.facing
        self.pick_turn()

    def follow(self) -> None:
        lead = self.leader
        # Turn a moment after the leader does, each in its own time
        if self.facing != lead.facing and random.random() < 0.12:
            self.facing = lead.facing
        # Drift around within the place, so the school loosens and tightens
        if random.random() < 0.004:
            self.offset = (self.place[0] + random.uniform(-1, 1), self.place[1] + random.uniform(-0.7, 0.7))
        # Speed up or ease off to hold the place (never backing up)
        behind = (lead.x + self.offset[0] - self.x) * self.facing
        want = max(0.0, min(lead.cruise * 1.8, lead.speed + behind * 0.04))
        self.speed += (want - self.speed) * 0.2
        target_y = min(max(lead.y + self.offset[1], self.top), self.bottom)
        self.vy = max(-0.12, min(0.12, (target_y - self.y) * 0.05))

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        x0 = round(self.x)
        y0 = round(self.y + math.sin(t * 1.3 + self.bob) * 0.6)
        for dx, dy, color in self.pixels[self.facing < 0][int(self.tail)]:
            put(fb, width, height, x0 + dx, y0 + dy, color)


class Bubble:
    __slots__ = ("x", "y", "speed", "phase")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.speed = random.uniform(0.12, 0.25)
        self.phase = random.uniform(0, 2 * math.pi)


class Seaweed:
    """A stalk rooted in the sand with leaves on alternate sides, swaying as a
    wave travels up it. haze fades it toward the water color, for the ones at
    the back."""

    def __init__(self, x: int, base: int, length: int, haze: float) -> None:
        self.x = x
        self.base = base
        self.length = length
        self.phase = random.uniform(0, 2 * math.pi)
        color = random.choice([(0, 150, 40), (20, 120, 30), (60, 170, 30)])
        tip = mix(color, (180, 255, 120), 0.4)
        # (color, leaf side or 0) for each segment from the root up
        self.segments = []
        for i in range(length):
            if i >= length - 2:
                c = tip
            else:
                c = mix(color, (0, 0, 0), 0.3 * (i % 2))
            leaf = (1 if i % 6 == 2 else -1) if i % 3 == 2 and i < length - 2 else 0
            self.segments.append((mix(c, HAZE, haze), leaf))

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        phase = t * 0.9 + self.phase
        length = self.length
        for i, (color, leaf) in enumerate(self.segments):
            # Sways more toward the tip
            x = self.x + round(math.sin(phase - i * 0.35) * (i / length) * 2.2)
            y = self.base - i
            put(fb, width, height, x, y, color)
            if leaf:
                put(fb, width, height, x + leaf, y, color)


def render_background(width: int, height: int, sand_top: list[int], props: dict) -> list[bytes]:
    """The water, light rays, and sand for each frame of the loop. props maps
    (x, y) to the color of an ornament pixel resting on the sand; like the
    sand, ornaments are lit by the rays where they reach the floor."""
    floor = min(sand_top)
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    p = (2 * np.pi * np.arange(LOOP_FRAMES) / LOOP_FRAMES).astype(np.float32)[:, None, None]
    depth = np.minimum(y / max(floor - 1, 1), 1)  # 0 at the surface, 1 at the sand
    water = np.array(WATER_TOP, np.float32) * (1 - depth[..., None]) + np.array(WATER_BOTTOM, np.float32) * depth[..., None]

    # Light rays: slanted bands from the surface that sway together, made of
    # sines drifting at different speeds so each ray brightens and fades
    u = x + y * 0.45 + 2.5 * np.sin(p)
    rays = np.sin(u * 0.19 + p * 2) + np.sin(u * 0.31 - p * 3 + 2.0) + 0.5 * np.sin(u * 0.53 + p * 5 + 1.0)
    rays = np.clip((rays - 0.8) / 1.4, 0, 1)
    rays *= rays
    rgb = water + np.array(RAYS, np.float32) * (rays * (1 - 0.7 * depth))[..., None]
    # A ripple shimmering along the surface row
    rgb[:, 0] += np.array(SURFACE, np.float32) * (0.45 + 0.35 * np.sin(x[0] * 0.7 - p[:, 0] * 7))[..., None]

    # Sand: speckled with the odd pebble, darker deeper down, and the ornaments
    solid = np.zeros((height, width, 3), np.float32)
    is_solid = np.zeros((height, width), bool)
    for sx, top in enumerate(sand_top):
        for sy in range(top, height):
            color = random.choice(PEBBLES) if random.random() < 0.05 else random.choice(SAND)
            solid[sy, sx] = [c * (1 - 0.12 * (sy - top)) for c in color]
            is_solid[sy, sx] = True
    for (sx, sy), color in props.items():
        if 0 <= sx < width and 0 <= sy < height:
            solid[sy, sx] = color
            is_solid[sy, sx] = True
    # Brighter where a ray falls
    rgb[:, is_solid] = solid[is_solid] * (0.8 + 0.5 * rays[:, is_solid])[..., None]
    np.minimum(rgb, 255, out=rgb)
    return [frame.tobytes() for frame in rgb.astype(np.uint8)]


class Aquarium:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        a, b = random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)
        self.sand_top = [
            height - SAND_ROWS - round(0.9 * math.sin(x * 0.13 + a) + 0.5 * math.sin(x * 0.29 + b))
            for x in range(width)
        ]
        floor = min(self.sand_top)

        # The chest in one side of the tank and the bubbler in the other
        chest_w = len(CHEST[0])
        self.chest_x = round(width * random.uniform(0.1, 0.3))
        self.bubbler_x = round(width * random.uniform(0.72, 0.88))
        if random.random() < 0.5:
            self.chest_x = width - chest_w - self.chest_x
            self.bubbler_x = width - 1 - self.bubbler_x
        chest_span = range(max(self.chest_x, 0), min(self.chest_x + chest_w, width))
        self.chest_y = max(self.sand_top[x] for x in chest_span) - len(CHEST) + 1
        props = {
            (self.chest_x + dx, self.chest_y + dy): color
            for dx, dy, color in render_sprite(CHEST_PALETTE, CHEST)
        }
        stone_y = self.sand_top[self.bubbler_x] - 1
        props[(self.bubbler_x, stone_y)] = props[(self.bubbler_x + 1, stone_y)] = STONE
        self.background = render_background(width, height, self.sand_top, props)
        self.chest_lid = render_sprite(CHEST_PALETTE, CHEST_OPEN)
        self.chest_timer = round(random.uniform(*CHEST_SECONDS) / FRAME_SECONDS)

        # Seaweed spread along the floor clear of the chest and bubbler; a
        # couple of shorter ones stand in front of the fish
        count = max(3, width // 9)
        spots = [round((i + random.uniform(0.2, 0.8)) * width / count) for i in range(count)]
        spots = [
            x for x in spots
            if x < width and not self.chest_x - 1 <= x <= self.chest_x + chest_w and abs(x - self.bubbler_x) > 1
        ]
        front = set(random.sample(spots, min(2, len(spots) // 3)))
        self.back_weeds = [
            Seaweed(x, self.sand_top[x], random.randint(height // 3, height // 2 + 3), 0.25)
            for x in spots if x not in front
        ]
        self.front_weeds = [
            Seaweed(x, self.sand_top[x], random.randint(height // 4, height // 3 + 1), 0.0)
            for x in front
        ]

        # Each swimmer (the school counts as one) gets its own depth, spread
        # evenly from near to far; fish are drawn far to near
        kinds = [kind for kind, n in TANK for _ in range(n)] + [SCHOOL]
        depths = [i / (len(kinds) - 1) for i in range(len(kinds))]
        random.shuffle(depths)
        top, bottom = 1, floor - 1
        self.fish = []
        for kind, depth in zip(kinds, depths):
            fish = Fish(kind, width, top, bottom, depth)
            self.fish.append(fish)
            if kind == SCHOOL:
                # Keep the leader far enough from the glass, surface and sand
                # for the whole school to fit
                fish.spread = (max(-px for px, py in SCHOOL_PLACES), max(px for px, py in SCHOOL_PLACES))
                fish.top -= min(py for px, py in SCHOOL_PLACES)
                fish.bottom -= max(py for px, py in SCHOOL_PLACES)
                fish.y = fish.target_y = min(max(fish.y, fish.top), fish.bottom)
                fish.pick_turn()
                self.fish += [Fish(kind, width, top, bottom, depth, fish, place) for place in SCHOOL_PLACES]
        self.fish.sort(key=lambda fish: -fish.depth)

        self.bubbles: list[Bubble] = []
        self.frame_count = 0

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.frame_count += 1
        t = self.frame_count * FRAME_SECONDS
        fb = bytearray(self.background[self.frame_count % LOOP_FRAMES])
        bubbles = self.bubbles

        for weed in self.back_weeds:
            weed.draw(fb, width, height, t)

        # Every so often the chest lid lifts and lets out a burst of bubbles
        self.chest_timer -= 1
        if self.chest_timer <= 0:
            self.chest_timer = round(random.uniform(*CHEST_SECONDS) / FRAME_SECONDS)
        elif self.chest_timer <= CHEST_OPEN_FRAMES:
            for dx, dy, color in self.chest_lid:
                put(fb, width, height, self.chest_x + dx, self.chest_y + dy - 1, color)
            if random.random() < 0.5:
                bubbles.append(Bubble(self.chest_x + random.uniform(1, len(CHEST[0]) - 2), self.chest_y - 2))

        # The bubbler sends up a steady stream
        if random.random() < 0.08:
            bubbles.append(Bubble(self.bubbler_x + random.uniform(0, 1), self.sand_top[self.bubbler_x] - 2))

        # Bubbles rise behind the fish
        rising = []
        for b in bubbles:
            b.y -= b.speed
            if b.y < 0.5:
                continue  # popped at the surface
            rising.append(b)
            wobble = math.sin(t * 4 + b.phase) * 0.6
            put(fb, width, height, round(b.x + wobble), round(b.y), BUBBLE)
        self.bubbles = bubbles = rising

        if self.frame_count % 4 == 0:
            self.give_room()
        for fish in self.fish:
            fish.swim()
            fish.draw(fb, width, height, t)
            # Fish blow the occasional bubble
            if random.random() < 0.002:
                mouth = fish.x + (fish.w if fish.facing > 0 else -1)
                bubbles.append(Bubble(mouth, fish.y + fish.h // 2))

        for weed in self.front_weeds:
            weed.draw(fb, width, height, t)

        return fb

    def give_room(self) -> None:
        """Fish that come too close drift apart vertically, so they don't
        swim along stuck together. A school moves aside as one."""
        fish = self.fish
        for i, a in enumerate(fish):
            for b in fish[i + 1:]:
                if (a.leader or a) is (b.leader or b):
                    continue  # the same school
                # Compare centers, doubled to stay in whole sprite sizes
                if (abs(2 * (a.x - b.x) + a.w - b.w) < a.w + b.w + 4
                        and abs(2 * (a.y - b.y) + a.h - b.h) < a.h + b.h + 2):
                    up = -1 if 2 * a.y + a.h < 2 * b.y + b.h else 1  # a is above b
                    for f, away in ((a.leader or a, up), (b.leader or b, -up)):
                        f.target_y = min(max(f.y + 3 * away, f.top), f.bottom)


SCENE = Aquarium


def main() -> None:
    run_scene(Aquarium, "Aquarium LED matrix display")


if __name__ == "__main__":
    main()
