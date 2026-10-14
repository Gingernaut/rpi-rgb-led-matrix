#!/usr/bin/env python
"""Pixel aquarium: fish swimming over a sandy floor among swaying seaweed,
with rising bubbles, a treasure chest, and light rays from the surface.

The water, light rays, sand and ornaments are pre-rendered as a seamless loop
at startup, so each frame copies a background and draws the moving things.
Fish are drawn far to near, the far ones hazed toward the water color, some
seaweed stands in front of them, and the neon tetras swim as a school.

After a few quiet minutes something happens (see EVENTS): a shark cruises
through and the fish scatter, a whale passes in the distance, a diver visits
the chest with fish tagging along, food is sprinkled in, or jellyfish drift
by. Now and then a sea turtle paddles across. To try one out, AQUARIUM_EVENT
shows just that one, a couple of seconds in and then after every
EVENT_TEST_SECONDS of quiet:

    AQUARIUM_EVENT=shark uv run server/displays/preview.py aquarium --seconds 30

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import os
import random

import numpy as np

from matrix_common import REFRESH_HZ, fade_lut, mix, put, put_max, run_scene

FRAME_SECONDS = 3 / REFRESH_HZ  # every 3rd refresh, ~25 fps

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

# Fish behavior around events
FLEE_SPEED = 3.5  # times cruising speed, bolting from a shark
SEEK_SPEED = 2.5  # at most this times cruising speed, going after food
BAIT_BALL = 0.45  # a frightened school closes up to this much of its spacing

# --- Events ------------------------------------------------------------------
#
# After EVENT_SECONDS of quiet (random within) one of EVENTS starts, picked by
# weight and never the same one twice running; the turtle keeps its own
# TURTLE_SECONDS of quiet. One thing happens at a time.
EVENT_SECONDS = (120.0, 600.0)
TURTLE_SECONDS = (180.0, 480.0)
EVENT_TEST_SECONDS = 8.0  # AQUARIUM_EVENT's quiet between showings

SHARK_PALETTE = {
    "G": (105, 120, 138), "D": (72, 85, 102), "W": (195, 200, 205),
    "M": (70, 80, 96), "K": (5, 5, 10),
}
SHARK = [
    [
        "T...........D.......",
        "TT.........DD.......",
        ".TT.....GGGGGGG.....",
        "..TTGGGGGGGGGGGGGG..",
        "..TGGGGGGGGGGMMGGKGG",
        ".TT.WWWWWWWWWWWWWWW.",
        "TT........DDD.......",
        "T...................",
    ],
    [
        "............D.......",
        "T..........DD.......",
        "TTT.....GGGGGGG.....",
        ".TTTGGGGGGGGGGGGGG..",
        "..TGGGGGGGGGGMMGGKGG",
        ".TTTWWWWWWWWWWWWWWW.",
        "..........DDD.......",
        "....................",
    ],
]
SHARK_PALETTE["T"] = SHARK_PALETTE["D"]
SHARK_SPEED = 0.26  # pixels per frame

TURTLE_PALETTE = {
    "S": (140, 112, 50), "H": (80, 58, 24), "R": (200, 180, 110),
    "K": (150, 178, 115), "E": (10, 15, 5),
}
# The domed shell and head, then the flippers for each frame of the stroke
TURTLE_BODY = [
    "......HHHHHHH.......",
    "....HHSSHSSSHSH.....",
    "..HSSSHSSSSHSSSH..KK",
    ".HSSSSHSSSSHSSSSHKKE",
    "RRRRRRRRRRRRRRRRRKK.",
]
TURTLE_FLIPPERS = [
    ["..K.........KKKK....", ".K..............KK..", "...................."],  # reaching forward
    ["..K.........KKK.....", ".K...........KKK....", "...............KK..."],  # pulling down
    [".K.........KKK......", "K.......KKKK........", "......KK............"],  # swept back
    ["..K.........KKK.....", ".K.........KKKK.....", "...................."],  # recovering
]
TURTLE_SPEED = 0.12

DIVER_PALETTE = {
    "Y": (225, 190, 40), "K": (45, 50, 75), "C": (0, 170, 200), "H": (30, 32, 45),
    "S": (225, 165, 125), "M": (150, 215, 255), "F": (240, 200, 30),
}
DIVER = [
    [
        "........YYYYY...",
        "F.....YYYYYYY.HH",
        "FFKKKKKKKKKKKHHM",
        "F.KKKKCCCCCKKHSM",
        "..........KK....",
        "............KS..",
    ],
    [
        "........YYYYY...",
        "......YYYYYYY.HH",
        "F.KKKKKKKKKKKHHM",
        "FFKKKKCCCCCKKHSM",
        "F.........KK....",
        "............KS..",
    ],
]
DIVER_HAND = (13, 5)  # where it holds the coin, in sprite coordinates
DIVER_MOUTH = (15, 1)  # where its breath bubbles out
DIVER_SPEED = 0.15
DIVER_BREATH_SECONDS = 3.5
DIVER_LOOK_SECONDS = 4.0  # at the open chest
# Where curious fish keep their mouths, from the diver's middle
DIVER_FOLLOW_SPOTS = [(-15, -6), (-14, 5), (16, -5), (15, 6), (-21, 0), (22, 1), (2, -9)]

# The whale is drawn from a profile rather than a sprite, being longer than
# the panel: its half-height at each point from the nose (0) to the fluke (1)
WHALE_LENGTH = 96
WHALE_PROFILE = (
    [0.0, 0.02, 0.05, 0.1, 0.2, 0.32, 0.45, 0.6, 0.74, 0.84, 0.9, 0.94, 0.975, 1.0],
    [0.0, 2.2, 3.6, 4.8, 5.8, 6.0, 5.5, 4.3, 2.6, 1.4, 1.0, 2.2, 3.2, 0.5],
)
WHALE_BACK = (30, 42, 62)  # shading down to WHALE_SIDE
WHALE_SIDE = (58, 74, 98)
WHALE_BELLY = (130, 145, 160)
WHALE_PLEAT = (82, 98, 116)
WHALE_LINE = (16, 22, 34)  # mouth and eye
WHALE_KNOB = (100, 116, 136)  # the knobs on its head
WHALE_FIN = (175, 186, 198)
WHALE_OPACITY = 0.92  # a little of the water shows through, for distance
WHALE_SPEED = 0.2
WHALE_SHADOW = 0.25  # how much it dims the tank, passing right over

FOOD = [(230, 120, 40), (200, 60, 40), (220, 180, 90)]
FOOD_PINCHES = 3
FOOD_FLAKES = 12  # in each pinch
FOOD_PINCH_SECONDS = (4.0, 7.0)  # between pinches
FOOD_CHEW_SECONDS = (0.8, 2.0)  # after each bite
FOOD_SECONDS = 50.0  # the fish lose interest in what's left after this

JELLY_COLORS = [(230, 70, 170), (150, 90, 255), (80, 190, 255), (255, 120, 90)]


def frames_for(seconds: float) -> int:
    return max(1, round(seconds / FRAME_SECONDS))


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


class Sprite:
    """Sprite frames as pixel lists for each facing, hazed toward the water by
    depth (0 near to 1 far)."""

    def __init__(self, palette: dict, frames: list[list[str]], depth: float = 0.0) -> None:
        palette = {ch: mix(color, HAZE, depth * 0.22) for ch, color in palette.items()}
        # [facing right, facing left][frame]
        self.pixels = [
            [render_sprite(palette, f) for f in frames],
            [render_sprite(palette, flip(f)) for f in frames],
        ]
        self.w = len(frames[0][0])
        self.h = len(frames[0])

    def draw(self, fb: bytearray, width: int, height: int, x: int, y: int, facing: int, frame: int) -> None:
        for dx, dy, color in self.pixels[facing < 0][frame]:
            put(fb, width, height, x + dx, y + dy, color)

    def at(self, x: float, facing: int, point: tuple[int, int]) -> float:
        """The panel x of a point given in the right-facing sprite's coordinates."""
        return x + (point[0] if facing > 0 else self.w - 1 - point[0])


class Fish:
    """A fish cruising back and forth, now and then pausing, turning around or
    changing depth. Given a leader, it keeps its place near that fish instead.
    Events can send it fleeing out of view (flee/calm), after a point such as
    food (goal), or slow it down (pace).

    depth is 0 (near) to 1 (far): far fish are hazed and swim a little slower."""

    def __init__(self, kind: str, width: int, top: int, bottom: int, depth: float,
                 leader: "Fish | None" = None, place: tuple[int, int] = (0, 0)) -> None:
        palette, frames = SPRITES[kind]
        self.sprite = Sprite(palette, frames, depth)
        self.w = self.sprite.w
        self.h = self.sprite.h
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
        self.fleeing = False
        self.calm_in = 0  # frames until a fleeing fish heads back
        self.goal: tuple[float, float] | None = None  # where to bring its mouth
        self.pace = 1.0  # scales its cruising speed
        self.tight = self.tight_to = 1.0  # a leader's school spacing, eased
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
        elif self.fleeing:
            self.bolt()
        elif self.goal:
            self.seek()
        else:
            self.wander()
        self.x += self.speed * self.facing
        self.y += self.vy
        self.tight += (self.tight_to - self.tight) * 0.05
        # Tail beats faster when swimming faster
        self.tail = (self.tail + 0.08 + self.speed * 0.5) % 2

    def glide(self, rate: float = 0.07) -> None:
        """Ease toward target_y, smoothly and at most rate pixels a frame."""
        climb = max(-rate, min(rate, (self.target_y - self.y) * 0.03))
        self.vy += (climb - self.vy) * 0.1

    def wander(self) -> None:
        # Now and then glide to a new depth, slowly and smoothly
        if random.random() < 0.006:
            self.target_y = random.uniform(self.top, self.bottom)
        self.glide()

        # Slow down nearing the turning point, and turn there
        ahead = (self.turn_x - self.x) * self.facing
        want = self.cruise * self.pace * min(1.0, max(0.15, ahead / 8))
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

    def flee(self, from_x: float) -> None:
        """Bolt away from from_x to somewhere out of view, and wait there
        until calm(). A school balls up tight as it goes."""
        self.fleeing = True
        self.calm_in = 0
        self.pause = 0
        self.facing = 1 if self.x + self.w / 2 >= from_x else -1
        # Far enough past the glass for the whole school to be hidden
        if self.facing > 0:
            self.turn_x = self.tank_width + self.spread[0] + random.uniform(3, 15)
        else:
            self.turn_x = -self.w - self.spread[1] - random.uniform(3, 15)
        self.target_y = random.uniform(self.top, self.bottom)
        self.tight_to = BAIT_BALL

    def calm(self, delay: int) -> None:
        """Head back in after delay frames."""
        if self.fleeing:
            self.calm_in = max(1, delay)

    def bolt(self) -> None:
        self.glide(0.12)
        ahead = (self.turn_x - self.x) * self.facing
        want = self.cruise * FLEE_SPEED if ahead > 0 else 0.0
        self.speed += (want - self.speed) * 0.15
        if self.calm_in:
            self.calm_in -= 1
            if not self.calm_in:
                self.fleeing = False
                self.tight_to = 1.0
                # Back toward the middle of the tank from wherever it hid
                self.facing = 1 if self.x + self.w / 2 < self.tank_width / 2 else -1
                self.pick_turn()

    def seek(self) -> None:
        """Swim to bring the mouth to self.goal, hovering once there."""
        gx, gy = self.goal
        if (gx - (self.x + self.w / 2)) * self.facing < -1:
            self.facing = -self.facing  # it's behind: turn around
        nose = self.x + self.w if self.facing > 0 else self.x
        ahead = (gx - nose) * self.facing
        want = self.cruise * max(0.0, min(SEEK_SPEED, ahead / 4))
        self.speed += (want - self.speed) * 0.1
        self.target_y = min(max(gy - self.h / 2, self.top), self.bottom)
        climb = max(-0.15, min(0.15, (self.target_y - self.y) * 0.06))
        self.vy += (climb - self.vy) * 0.2
        self.pause = 0

    def release(self) -> None:
        """Stop going after the goal and go back to wandering."""
        self.goal = None
        self.target_y = self.y
        self.pick_turn()

    def reaches(self, px: float, py: float) -> bool:
        """Whether (px, py) is at this fish's mouth, in the front of its head."""
        ahead = (px - (self.x + self.w / 2)) * self.facing
        return -1 <= ahead <= self.w / 2 + 1.5 and abs(py - (self.y + self.h / 2)) <= self.h / 2 + 0.5

    def follow(self) -> None:
        lead = self.leader
        # Turn a moment after the leader does, each in its own time
        if self.facing != lead.facing and random.random() < 0.12:
            self.facing = lead.facing
        # Drift around within the place, so the school loosens and tightens
        if random.random() < 0.004:
            self.offset = (self.place[0] + random.uniform(-1, 1), self.place[1] + random.uniform(-0.7, 0.7))
        # Speed up or ease off to hold the place (never backing up)
        behind = (lead.x + self.offset[0] * lead.tight - self.x) * self.facing
        top_speed = max(lead.cruise * 1.8, lead.speed * 1.3)
        want = max(0.0, min(top_speed, lead.speed + behind * 0.04))
        self.speed += (want - self.speed) * 0.2
        target_y = min(max(lead.y + self.offset[1] * lead.tight, self.top), self.bottom)
        self.vy = max(-0.15, min(0.15, (target_y - self.y) * 0.05))

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        y = round(self.y + math.sin(t * 1.3 + self.bob) * 0.6)
        self.sprite.draw(fb, width, height, round(self.x), y, self.facing, int(self.tail))


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


# --- Events ------------------------------------------------------------------

BACK = "back"  # drawn behind the seaweed and fish
AMONG = "among"  # drawn among the fish, at its depth


class Event:
    """Something passing through the tank for a while. Each frame the tank
    calls step(), then draw() at the event's layer; among the fish, it's drawn
    in front of those deeper than its depth (0 near to 1 far)."""

    layer = AMONG
    depth = 0.5
    shade = 1.0  # how much light reaches the tank, when something blocks it

    def __init__(self, tank: "Aquarium") -> None:
        self.tank = tank

    def step(self) -> bool:
        """Advance a frame; False once it's over."""
        return False

    def draw(self, fb: bytearray, t: float) -> None:
        pass

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        """(left, top, right, bottom) boxes the fish steer clear of."""
        return []

    def free_fish(self) -> list[Fish]:
        """The fish that go their own way (school leaders, not followers)."""
        return [fish for fish in self.tank.fish if not fish.leader]


class Shark(Event):
    """A shark cruises slowly through near the glass. The fish bolt out of view
    ahead of it, the tetras balling up tight, and once it's gone they come
    back one at a time."""

    depth = 0.05

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(SHARK_PALETTE, SHARK, self.depth)
        self.facing = random.choice((1, -1))
        self.x = -self.sprite.w - 2.0 if self.facing > 0 else tank.width + 2.0
        self.y0 = random.uniform(5, tank.floor - self.sprite.h - 4)
        self.tail = 0.0
        self.frame_count = 0
        self.scattered = False

    def step(self) -> bool:
        self.frame_count += 1
        self.x += SHARK_SPEED * self.facing
        self.tail = (self.tail + 0.05) % 2  # a slow, heavy beat
        nose = self.x + self.sprite.w if self.facing > 0 else self.x
        if not self.scattered and -8 < nose < self.tank.width + 8:
            # It's coming into view: everyone runs
            self.scattered = True
            for fish in self.free_fish():
                fish.flee(nose - 12 * self.facing)
        if (self.x + self.sprite.w < 0) if self.facing < 0 else (self.x > self.tank.width):
            for fish in self.free_fish():
                fish.calm(frames_for(random.uniform(1.5, 8.0)))
            return False
        return True

    def draw(self, fb: bytearray, t: float) -> None:
        y = round(self.y0 + 1.2 * math.sin(self.frame_count * FRAME_SECONDS * 0.4))
        self.sprite.draw(fb, self.tank.width, self.tank.height, round(self.x), y, self.facing, int(self.tail))


class Whale(Event):
    """A humpback, longer than the tank is wide, glides past in the distance,
    shading the water as it goes. The fish slow down to watch."""

    layer = BACK

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        # The nose starts just out of view
        self.nose = -1.0 if self.facing > 0 else tank.width + 1.0
        self.y = random.uniform(9, 12)
        self.beat = 0.0
        self.beat_step = 2 * math.pi / frames_for(5.0)
        for fish in tank.fish:
            fish.pace = 0.5

    def step(self) -> bool:
        self.nose += WHALE_SPEED * self.facing
        self.beat += self.beat_step
        width = self.tank.width
        tail = self.nose - WHALE_LENGTH * self.facing
        # The share of the panel it covers sets how much it shades the tank
        left, right = sorted((self.nose, tail))
        covered = max(0.0, min(right, width) - max(left, 0)) / width
        self.shade = 1 - WHALE_SHADOW * covered
        if (tail > width) if self.facing > 0 else (tail < 0):
            for fish in self.tank.fish:
                fish.pace = 1.0
            return False
        return True

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        s = (self.nose - (np.arange(width) + 0.5)) * self.facing / WHALE_LENGTH  # 0 nose, 1 fluke tip
        cols = np.flatnonzero((s >= 0) & (s <= 1))
        if not len(cols):
            return
        s = s[cols]
        half = np.interp(s, *WHALE_PROFILE)
        # The tail sweeps up and down, more toward the fluke
        sway = 2.5 * np.clip((s - 0.55) / 0.45, 0, 1) ** 2 * math.sin(self.beat)
        center = self.y + sway
        hump = np.sin(np.pi * np.clip((s - 0.05) / 0.7, 0, 1))  # the arched back
        dorsal = 1.5 * np.clip(1 - np.abs(s - 0.66) / 0.04, 0, 1)
        top = center - half * 0.85 - hump - dorsal
        bottom = center + half * 1.1
        y = np.arange(height, dtype=np.float32)[:, None] + 0.5
        cover = np.clip(np.minimum(y - top, bottom - y) + 0.5, 0, 1) * WHALE_OPACITY
        # Dark on top shading lighter down the side; the front of the belly
        # pale, with pleats down the throat; a long mouth line
        v = (y - top) / np.maximum(bottom - top, 0.1)
        side = np.clip(v * 1.6, 0, 1)[..., None]
        color = np.array(WHALE_BACK, np.float32) * (1 - side) + np.array(WHALE_SIDE, np.float32) * side
        pale = (v > 0.62) & (s < 0.5)
        color[pale] = WHALE_BELLY
        color[pale & (np.arange(height)[:, None] % 2 == 0)] = WHALE_PLEAT
        mouth = 0.45 + s * 1.1  # curving down toward the eye
        color[(np.abs(v - mouth) * (bottom - top) < 0.5) & (s > 0.015) & (s < 0.14)] = WHALE_LINE

        view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
        water = view[:, cols].astype(np.float32)
        water += (color - water) * cover[..., None]
        view[:, cols] = water.astype(np.uint8)

        def blend(px: float, py: float, color: tuple) -> None:
            px, py = round(px), round(py)
            if 0 <= px < width and 0 <= py < height:
                p = (py * width + px) * 3
                for i in range(3):
                    fb[p + i] = int(fb[p + i] + (color[i] - fb[p + i]) * WHALE_OPACITY)

        def body_y(at: float, v: float) -> float:
            """The y a fraction v of the way down the body, at at from the nose."""
            h = float(np.interp(at, *WHALE_PROFILE))
            top = self.y - h * 0.85 - math.sin(math.pi * min(max((at - 0.05) / 0.7, 0), 1))
            return top + v * (self.y + h * 1.1 - top)

        def body_x(at: float) -> float:
            return self.nose - at * WHALE_LENGTH * self.facing

        # Knobs along the top of the head, and the eye just past the mouth
        for at in (0.04, 0.07, 0.10):
            blend(body_x(at), body_y(at, 0.12), WHALE_KNOB)
        blend(body_x(0.155), body_y(0.155, 0.42), WHALE_LINE)

        # The long pale flipper on the near side, beating slowly: broad at the
        # root, tapering to the tip
        root = 0.24
        rx, ry = body_x(root), body_y(root, 0.8)
        angle = 0.5 + 0.15 * math.sin(self.beat * 0.8)
        dx, dy = -math.cos(angle) * self.facing, math.sin(angle)
        length = 16
        for k in range(length):
            px, py = rx + dx * k, ry + dy * k
            for i in range(3 if k < 5 else 2 if k < 11 else 1):
                blend(px, py - i, WHALE_FIN)


class Feeding(Event):
    """Food is sprinkled on the surface a pinch at a time; the flakes float a
    moment, then sink. The fish rush up and snap them up, pausing to chew
    after each bite."""

    layer = BACK

    class Flake:
        __slots__ = ("x", "y", "speed", "color", "wait", "float", "phase")

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        spot = random.uniform(16, tank.width - 16)
        self.flakes = []
        sprinkle = 0
        for _ in range(FOOD_PINCHES):
            drop = spot + random.uniform(-8, 8)
            for _ in range(FOOD_FLAKES):
                f = Feeding.Flake()
                f.x = drop + random.gauss(0, 3)
                f.y = 1.0
                f.speed = random.uniform(0.035, 0.07)
                f.color = random.choice(FOOD)
                f.wait = sprinkle + random.randint(0, frames_for(1.5))  # not in yet
                f.float = random.randint(0, frames_for(2.0))  # on the surface
                f.phase = random.uniform(0, 2 * math.pi)
                self.flakes.append(f)
            sprinkle += frames_for(random.uniform(*FOOD_PINCH_SECONDS))
        self.chewing: dict[Fish, int] = {}  # frames until each fish bites again
        self.frames_left = frames_for(FOOD_SECONDS)
        self.frame_count = 0

    def step(self) -> bool:
        tank = self.tank
        self.frame_count += 1
        self.frames_left -= 1
        chewing = self.chewing
        for fish in list(chewing):
            chewing[fish] -= 1
            if not chewing[fish]:
                del chewing[fish]
        hungry = [fish for fish in tank.fish if fish not in chewing]
        kept = []
        live = []  # in the water
        for f in self.flakes:
            if f.wait:
                f.wait -= 1
                kept.append(f)
                continue
            if f.float:
                f.float -= 1
            else:
                f.y += f.speed
                f.x += math.sin(self.frame_count * 0.05 + f.phase) * 0.03
            # Lost in the sand, or eaten if a fish gets its mouth to it
            if f.y >= tank.sand_top[min(max(round(f.x), 0), tank.width - 1)]:
                continue
            eater = next((fish for fish in hungry if fish.reaches(f.x, f.y)), None)
            if eater:
                chewing[eater] = frames_for(random.uniform(*FOOD_CHEW_SECONDS))
                hungry.remove(eater)
                continue
            kept.append(f)
            live.append(f)
        self.flakes = kept
        if not self.flakes or not self.frames_left:
            for fish in self.free_fish():
                fish.release()
            return False
        if self.frame_count % 5 == 1:
            # Each fish goes for the flake nearest its mouth; a chewing one
            # hangs about where it is
            for fish in self.free_fish():
                if fish in chewing:
                    continue
                mx = fish.x + fish.w / 2 + fish.w / 2 * fish.facing
                my = fish.y + fish.h / 2
                f = min(live, key=lambda f: (f.x - mx) ** 2 + 2 * (f.y - my) ** 2, default=None)
                fish.goal = (f.x, f.y) if f else None
        return True

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        for f in self.flakes:
            if not f.wait:
                put(fb, width, height, round(f.x), round(f.y), f.color)


class Jellyfish(Event):
    """A few glowing jellyfish drift through on the current, pulsing to stay
    up. The fish keep out of their way."""

    layer = BACK

    class Jelly:
        __slots__ = ("x", "y", "vy", "drift", "color", "rim", "period", "clock", "wave")

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        colors = random.sample(JELLY_COLORS, 3)
        self.jellies = []
        for i in range(random.randint(2, 3)):
            j = Jellyfish.Jelly()
            # Staggered in from the side they drift from
            j.x = (-6 - i * random.uniform(14, 22)) if self.facing > 0 else tank.width + 6 + i * random.uniform(14, 22)
            j.y = random.uniform(4, tank.floor - 14)
            j.vy = 0.0
            j.drift = random.uniform(0.07, 0.1) * self.facing
            j.color = colors[i]
            j.rim = mix(colors[i], (255, 255, 255), 0.45)
            j.period = frames_for(random.uniform(2.2, 3.2))
            j.clock = random.randrange(j.period)
            j.wave = random.uniform(0, 2 * math.pi)
            self.jellies.append(j)
        self.frame_count = 0

    def squeeze(self, j: "Jellyfish.Jelly") -> float:
        """1 as the bell contracts for a pulse, easing back to 0."""
        return max(0.0, 1 - j.clock / (j.period * 0.3))

    def step(self) -> bool:
        self.frame_count += 1
        low = self.tank.floor - 13
        for j in self.jellies:
            j.clock += 1
            if j.clock >= j.period:
                j.clock = 0
                # Pulse harder when sinking too low, gently near the surface
                j.vy -= 0.08 if j.y < 4 else 0.17 if j.y > low else 0.12
            j.vy = (j.vy + 0.0016) * 0.96
            j.y += j.vy
            j.x += j.drift
        return any(-8 < j.x < self.tank.width + 8 or (j.x < 0) == (self.facing > 0) for j in self.jellies)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        for j in self.jellies:
            x0, y0 = round(j.x), round(j.y)
            squeezed = self.squeeze(j) > 0.4
            # The bell: a dome, narrower and taller while it pulses
            rows = ((1, 1), (2, 2), (2, 2), (1, 1)) if squeezed else ((1, 1), (2, 2), (3, 3))
            for dy, (a, b) in enumerate(rows):
                for dx in range(-a, b + 1):
                    color = j.rim if dy == 0 or (dy == len(rows) - 1 and dx % 2) else j.color
                    put_max(fb, width, height, x0 + dx, y0 + dy, color)
            # Tentacles trailing below, waving and fading
            base = y0 + len(rows)
            length = 4 if squeezed else 6
            for k, tx in enumerate((-2, 0, 2) if not squeezed else (-1, 0, 1)):
                for i in range(length):
                    sway = math.sin(self.frame_count * 0.08 + j.wave + k * 1.7 - i * 0.6) * (i / length) * 1.2
                    fade = 0.75 * (1 - i / (length + 1))
                    put_max(fb, width, height, x0 + tx + round(sway), base + i,
                            (int(j.color[0] * fade), int(j.color[1] * fade), int(j.color[2] * fade)))

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return [(j.x - 4, j.y - 1, j.x + 4, j.y + 10) for j in self.jellies]


class Diver(Event):
    """A diver swims down to the chest, opens it for a look and leaves with a
    coin, breathing out bubbles as it goes. Some of the fish tag along."""

    depth = 0.3

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(DIVER_PALETTE, DIVER, self.depth)
        w, h = self.sprite.w, self.sprite.h
        # It comes in on the side away from the chest, so it crosses the tank
        chest_mid = tank.chest_x + len(CHEST[0]) / 2
        self.facing = 1 if chest_mid > tank.width / 2 else -1
        self.x = -w - 2.0 if self.facing > 0 else tank.width + 2.0
        self.y = random.uniform(3, 7)
        hand_x = DIVER_HAND[0] if self.facing > 0 else w - 1 - DIVER_HAND[0]
        self.stops = [
            # Hand over the chest, a little above it
            (chest_mid - hand_x, tank.chest_y - DIVER_HAND[1] - 2, frames_for(DIVER_LOOK_SECONDS)),
            # Out the far side, rising
            (tank.width + 2.0 if self.facing > 0 else -w - 2.0, 2.0, 0),
        ]
        self.wait = 0
        self.coin = False
        self.kick = 0.0
        self.breath = frames_for(1.0)
        # The curious fish each keep their own spot around the diver
        curious = [fish for fish in self.free_fish() if random.random() < 0.7]
        spots = random.sample(DIVER_FOLLOW_SPOTS, len(curious))
        self.followers = {fish: (w / 2 + dx, h / 2 + dy) for fish, (dx, dy) in zip(curious, spots)}

    def step(self) -> bool:
        tank = self.tank
        moving = 0.0
        if self.wait:
            self.wait -= 1
            tank.chest_hold = max(tank.chest_hold, 2)
            if not self.wait:
                self.coin = True
                self.stops.pop(0)
        else:
            tx, ty, look = self.stops[0]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            moving = min(DIVER_SPEED, dist * 0.05)
            if dist < 0.3:
                if not look:
                    for fish in self.followers:
                        fish.release()
                    return False
                self.wait = look
            else:
                self.x += dx / dist * moving
                self.y += dy / dist * moving
        # Fins kick while swimming, idle while it looks
        self.kick = (self.kick + 0.03 + moving * 0.35) % 2
        # A burst of breath every few seconds
        self.breath -= 1
        if self.breath <= 0:
            self.breath = frames_for(DIVER_BREATH_SECONDS * random.uniform(0.85, 1.15))
        elif self.breath < 6 and self.breath % 2 == 0:
            tank.bubbles.append(Bubble(self.sprite.at(self.x, self.facing, DIVER_MOUTH), self.y + DIVER_MOUTH[1] - 1))
        if tank.frame_count % 3 == 0:
            for fish, (ox, oy) in self.followers.items():
                fish.goal = (self.x + ox, self.y + oy)
        return True

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        x, y = round(self.x), round(self.y)
        self.sprite.draw(fb, width, height, x, y, self.facing, int(self.kick))
        if self.coin:
            hand = round(self.sprite.at(self.x, self.facing, DIVER_HAND))
            put(fb, width, height, hand, y + DIVER_HAND[1] + 1, CHEST_PALETTE["G"])


class Turtle(Event):
    """A sea turtle paddles slowly across at mid-depth."""

    depth = 0.5

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(TURTLE_PALETTE, [TURTLE_BODY + f for f in TURTLE_FLIPPERS], self.depth)
        self.facing = random.choice((1, -1))
        self.x = -self.sprite.w - 2.0 if self.facing > 0 else tank.width + 2.0
        self.y0 = random.uniform(7, tank.floor - self.sprite.h - 5)
        self.stroke = 0.0
        self.stroke_step = len(TURTLE_FLIPPERS) / frames_for(2.4)
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        self.stroke = (self.stroke + self.stroke_step) % len(TURTLE_FLIPPERS)
        # A little surge on each pull of the flippers
        self.x += TURTLE_SPEED * (1.4 if int(self.stroke) in (1, 2) else 0.7) * self.facing
        return -self.sprite.w - 3 < self.x < self.tank.width + 3

    def draw(self, fb: bytearray, t: float) -> None:
        y = round(self.y0 + 1.5 * math.sin(self.frame_count * FRAME_SECONDS * 0.25))
        self.sprite.draw(fb, self.tank.width, self.tank.height, round(self.x), y, self.facing, int(self.stroke))


# name: (event, weight when picking one at random)
EVENTS = {
    "shark": (Shark, 3),
    "whale": (Whale, 2),
    "diver": (Diver, 2),
    "feeding": (Feeding, 3),
    "jellyfish": (Jellyfish, 2),
}
VISITORS = {"turtle": Turtle}  # each on its own schedule


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
        self.floor = floor = min(self.sand_top)

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
        self.chest_timer = frames_for(random.uniform(*CHEST_SECONDS))
        self.chest_hold = 0  # frames an event keeps the chest open for

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

        # Events: the one under way (if any), and the quiet frames left before
        # the next event and the next visitor
        self.event: Event | None = None
        self.last_event = None
        self.only = os.environ.get("AQUARIUM_EVENT")
        if self.only and self.only not in EVENTS and self.only not in VISITORS:
            raise SystemExit(f"AQUARIUM_EVENT must be one of {', '.join([*EVENTS, *VISITORS])}")
        self.event_wait = frames_for(2.0 if self.only else random.uniform(*EVENT_SECONDS))
        self.visitor_wait = frames_for(random.uniform(*TURTLE_SECONDS))
        self.shade_luts: dict[int, bytes] = {}

    def start(self, name: str) -> None:
        """Begin the named event (from EVENTS or VISITORS) now."""
        event_class = EVENTS[name][0] if name in EVENTS else VISITORS[name]
        self.event = event_class(self)

    def direct(self) -> None:
        """Run the event under way, or count down the quiet to the next."""
        if self.event:
            if not self.event.step():
                self.event = None
            return
        if self.only:
            self.event_wait -= 1
            if self.event_wait <= 0:
                self.event_wait = frames_for(EVENT_TEST_SECONDS)
                self.start(self.only)
            return
        self.event_wait -= 1
        self.visitor_wait -= 1
        if self.event_wait <= 0:
            self.event_wait = frames_for(random.uniform(*EVENT_SECONDS))
            names = [name for name in EVENTS if name != self.last_event]
            self.last_event = random.choices(names, [EVENTS[name][1] for name in names])[0]
            self.start(self.last_event)
        elif self.visitor_wait <= 0:
            self.visitor_wait = frames_for(random.uniform(*TURTLE_SECONDS))
            self.start(random.choice(list(VISITORS)))

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.frame_count += 1
        t = self.frame_count * FRAME_SECONDS
        self.direct()
        event = self.event

        background = self.background[self.frame_count % LOOP_FRAMES]
        if event and event.shade < 1:
            level = round(event.shade * 20)
            if level not in self.shade_luts:
                self.shade_luts[level] = fade_lut(level / 20)
            background = background.translate(self.shade_luts[level])
        fb = bytearray(background)
        bubbles = self.bubbles

        if event and event.layer == BACK:
            event.draw(fb, t)

        for weed in self.back_weeds:
            weed.draw(fb, width, height, t)

        # Every so often the chest lid lifts and lets out a burst of bubbles
        self.chest_timer -= 1
        if self.chest_hold:
            self.chest_hold -= 1
        if self.chest_timer <= 0:
            self.chest_timer = frames_for(random.uniform(*CHEST_SECONDS))
        elif self.chest_timer <= CHEST_OPEN_FRAMES or self.chest_hold:
            for dx, dy, color in self.chest_lid:
                put(fb, width, height, self.chest_x + dx, self.chest_y + dy - 1, color)
            if random.random() < (0.15 if self.chest_hold else 0.5):
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
        among = event if event and event.layer == AMONG else None
        for fish in self.fish:
            if among and among.depth > fish.depth:
                among.draw(fb, t)
                among = None
            fish.swim()
            fish.draw(fb, width, height, t)
            # Fish blow the occasional bubble
            if random.random() < 0.002:
                mouth = fish.x + (fish.w if fish.facing > 0 else -1)
                bubbles.append(Bubble(mouth, fish.y + fish.h // 2))
        if among:
            among.draw(fb, t)

        for weed in self.front_weeds:
            weed.draw(fb, width, height, t)

        return fb

    def give_room(self) -> None:
        """Fish that come too close drift apart vertically, so they don't
        swim along stuck together. A school moves aside as one. They also
        steer above or below anything an event asks them to avoid."""
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
        if not self.event:
            return
        for left, top, right, bottom in self.event.obstacles():
            for f in fish:
                if f.leader or f.goal or not (f.x < right + 3 and f.x + f.w > left - 3):
                    continue
                if f.y < bottom + 1 and f.y + f.h > top - 1:
                    # Whichever way round is shorter, if there's room
                    above, below = top - f.h - 2, bottom + 2
                    go = above if f.y + f.h / 2 < (top + bottom) / 2 and above >= f.top else below
                    f.target_y = min(max(go, f.top), f.bottom)


SCENE = Aquarium


def main() -> None:
    run_scene(Aquarium, "Aquarium LED matrix display")


if __name__ == "__main__":
    main()
