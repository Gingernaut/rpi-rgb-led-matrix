#!/usr/bin/env python
"""Pixel aquarium: fish swimming over a sandy floor among swaying seaweed,
with rising bubbles, a treasure chest, and light rays from the surface.

The water, light rays, sand and ornaments are pre-rendered as a seamless loop
at startup, so each frame copies a background and draws the moving things.
Fish are drawn far to near, the far ones hazed toward the water color, some
seaweed stands in front of them, and the neon tetras swim as a school.

Every minute or so something happens (see EVENTS): a shark cruises through
and the fish scatter, a whale or an old submarine passes in the distance,
jellyfish drift by, a sea turtle paddles across, or, now and then, a moray
eel slides along the bottom. To try one out,
AQUARIUM_EVENT shows just that one, a couple of seconds in and then every
few seconds after (see aquarium_common):

    AQUARIUM_EVENT=shark uv run server/displays/preview.py aquarium --seconds 30

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from aquarium_common import (
    AMONG, BACK, FRAME_SECONDS, JELLY_FRAMES, Event, Fish, Sprite, Tank, flip, frames_for, render_sprite, school,
)
from matrix_common import fade_lut, mix, put, put_max, run_scene, scale

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

# Sprites (see aquarium_common.Sprite), two frames per fish: tail swept up/down
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

# The events' looks (see EVENTS; aquarium_common schedules them)

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
    "S": (80, 170, 70), "H": (40, 112, 45), "R": (215, 205, 125),
    "K": (150, 215, 130), "E": (10, 10, 10),
}
# The domed shell and a big round head, then the flippers for each frame of
# the stroke
TURTLE_BODY = [
    "....HHHHH.....",
    "..HSSHSSSH.KK.",
    ".HSSSHSSSHKKEK",
    "RRRRRRRRRRRKKK",
]
TURTLE_FLIPPERS = [
    ["..K......KKK..", ".K..........K."],  # reaching forward
    ["..K......KK...", ".K........KK.."],  # pulling down
    [".K......KK....", "K......KK....."],  # swept back
    ["..K......KK...", ".K......KKK..."],  # recovering
]
TURTLE_SPEED = 0.12

# A green moray: its head a sprite (facing right; jaws shut, then open), its
# long body drawn behind it as a wave runs down it
MORAY_PALETTE = {
    "G": (95, 150, 55), "D": (60, 105, 38), "L": (165, 195, 95), "E": (250, 235, 120),
    "K": (10, 10, 10), "M": (120, 30, 35), "W": (240, 240, 230),
}
MORAY_HEAD = [
    [
        ".DDDD..",
        "DGGGKG.",
        "GGGGGGG",
        "GLLLLLL",
        ".LLLL..",
    ],
    [
        ".DDDD..",
        "DGGGKGG",
        "GGGGMMW",
        "GLLMMW.",
        ".LLLLL.",
    ],
]
MORAY_EYE = (4, 1)  # drawn with a ring of yellow around the pupil "K"
MORAY_LENGTH = 26  # the body behind the head, in pixels
MORAY_BACK = (60, 105, 38)
MORAY_SIDE = (95, 150, 55)
MORAY_BELLY = (165, 195, 95)
MORAY_FIN = (150, 175, 70)  # the fin along its back
MORAY_SPEED = 0.18
MORAY_PAUSE = (4.0, 7.0)  # seconds it stops partway, gaping

# The whale is drawn from a profile rather than a sprite, for a smooth round
# body: its half-height at each point from the nose (0) to the fluke tips (1).
# The fluke is a whale's tail as in photos of one raised above the water: two
# broad lobes flaring from a narrow stock, with a notch between them.
WHALE_LENGTH = 50
WHALE_PROFILE = (
    [0.0, 0.018, 0.053, 0.106, 0.194, 0.308, 0.44, 0.546, 0.634, 0.704, 0.76,
     0.8, 0.84, 0.87, 0.9, 0.93, 0.96, 1.0],
    [0.0, 2.6, 4.0, 4.8, 5.1, 4.9, 4.0, 3.0, 2.0, 1.3, 1.0,
     1.05, 1.4, 3.0, 4.4, 5.4, 6.1, 6.5],
)
WHALE_NOTCH = 0.09  # how deep the notch cuts into the fluke's trailing edge
WHALE_HUMP = 0.6  # how much its back arches
WHALE_SWAY = 2.0  # how far the fluke sweeps up and down
WHALE_BACK = (40, 72, 125)  # shading down to WHALE_SIDE
WHALE_SIDE = (75, 110, 165)
WHALE_BELLY = (175, 200, 220)
WHALE_PLEAT = (135, 160, 190)
WHALE_LINE = (15, 22, 40)  # its smile and eye
WHALE_GLINT = (240, 245, 255)
WHALE_FIN = (160, 190, 225)
WHALE_OPACITY = 0.95  # a little of the water shows through, for distance
WHALE_SPEED = 0.16
WHALE_SHADOW = 0.12  # how much it dims the tank, passing right over

# An old riveted submarine, facing right; frames turn the propeller
SUB_PALETTE = {
    "Y": (245, 195, 35), "O": (235, 120, 25), "D": (185, 130, 25), "R": (200, 145, 20),
    "P": (255, 240, 170), "S": (160, 160, 150), "L": (255, 255, 220),
}
SUB_HULL = [
    "....................SSS.......",
    "....................S.........",
    "................OOOOSOO.......",
    "................YYYYYYY.......",
    "................YYPYYPY.......",
    "........YYYYYYYYYYYYYYYYYY....",
    ".....YYYYYYYYYYYYYYYYYYYYYYY..",
    "....YYYYPYYYYPYYYYPYYYYPYYYYL.",
    "...OOOOOOOOOOOOOOOOOOOOOOOOOOO",
    "....YRYYRYYRYYRYYRYYRYYRYYRYY.",
    ".....DDDDDDDDDDDDDDDDDDDDDDD..",
    "........DDDDDDDDDDDDDDDDDD....",
]
# The tail, drawn over the hull's first three columns: the rudder, and the
# propeller on its shaft, turning from blades up and down to edge on and back
SUB_PROPS = [
    {6: "S.O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "S.O"},
    {6: "..O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "..O"},
    {6: "..O", 7: "..O", 8: "SSO", 9: "..O", 10: "..O"},
    {6: "..O", 7: "S.O", 8: "SSO", 9: "S.O", 10: "..O"},
]
SUB_LAMP = (28, 7)  # the searchlight, in sprite coordinates
SUB_SPEED = 0.15
SUB_BEAM = (170, 150, 25)  # added to the water by the searchlight at full strength
SUB_BEAM_LENGTH = 26

JELLY_COLORS = [(230, 70, 170), (150, 90, 255), (80, 190, 255), (255, 120, 90)]


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


class Shark(Event):
    """A shark cruises slowly through near the glass. The fish bolt out of view
    ahead of it, the tetras balling up tight, and once it's gone they come
    back one at a time."""

    depth = 0.05

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(SHARK_PALETTE, SHARK, self.depth, HAZE)
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
        # Until it's gone out the other side
        return (self.x + self.sprite.w >= 0) if self.facing < 0 else (self.x <= self.tank.width)

    def end(self) -> None:
        for fish in self.free_fish():
            fish.calm(frames_for(random.uniform(1.5, 8.0)))

    def draw(self, fb: bytearray, t: float) -> None:
        y = round(self.y0 + 1.2 * math.sin(self.frame_count * FRAME_SECONDS * 0.4))
        self.sprite.draw(fb, self.tank.width, self.tank.height, round(self.x), y, self.facing, int(self.tail))


class Whale(Event):
    """A little blue whale glides past in the distance, smiling, shading the
    water as it goes and now and then blowing a bubble. The fish slow down
    to watch."""

    layer = BACK

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        # The nose starts just out of view
        self.nose = -1.0 if self.facing > 0 else tank.width + 1.0
        self.y = tank.floor * random.uniform(0.3, 0.45)  # rows 8-12 on a 32-row panel
        self.beat = 0.0
        self.beat_step = 2 * math.pi / frames_for(3.5)
        self.blow = frames_for(random.uniform(2, 4))
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
        # A bubble from the blowhole every few seconds
        self.blow -= 1
        if not self.blow:
            self.blow = frames_for(random.uniform(2, 4))
            self.tank.bubbles.append(Bubble(self.body_x(0.14), float(self.edges(0.14)[0]) - 1))
        # Until the fluke is out of view
        return (tail <= width) if self.facing > 0 else (tail >= 0)

    def end(self) -> None:
        for fish in self.tank.fish:
            fish.pace = 1.0

    def edges(self, s):
        """The top and bottom of the body at s (a number or array) from the
        nose (0) to the fluke tip (1)."""
        half = np.interp(s, *WHALE_PROFILE)
        # The tail sweeps up and down, more toward the fluke
        sway = WHALE_SWAY * np.clip((s - 0.48) / 0.52, 0, 1) ** 2 * math.sin(self.beat)
        hump = WHALE_HUMP * np.sin(np.pi * np.clip((s - 0.044) / 0.616, 0, 1))
        center = self.y + sway
        return center - half * 0.9 - hump, center + half * 1.1

    def body_x(self, s: float) -> float:
        return self.nose - s * WHALE_LENGTH * self.facing

    def body_y(self, s: float, v: float) -> float:
        """The y a fraction v of the way down the body, at s from the nose."""
        top, bottom = self.edges(s)
        return float(top + v * (bottom - top))

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        s = (self.nose - (np.arange(width) + 0.5)) * self.facing / WHALE_LENGTH
        cols = np.flatnonzero((s >= 0) & (s <= 1))
        if not len(cols):
            return
        s = s[cols]
        top, bottom = self.edges(s)
        y = np.arange(height, dtype=np.float32)[:, None] + 0.5
        cover = np.clip(np.minimum(y - top, bottom - y) + 0.5, 0, 1) * WHALE_OPACITY
        # The notch between the fluke's lobes: the trailing edge runs nearly
        # straight across the lobes and dips in to a point between them
        notch = (bottom - top) / 2 * (1 - np.sqrt(np.clip((1 - s) / WHALE_NOTCH, 0, 1)))
        cover *= np.clip(np.abs(y - (top + bottom) / 2) - notch + 0.5, 0, 1)
        # Darker on top shading lighter down the side, a pale striped belly,
        # and a smile curving up toward the eye
        v = (y - top) / np.maximum(bottom - top, 0.1)
        side = np.clip(v * 1.6, 0, 1)[..., None]
        color = np.array(WHALE_BACK, np.float32) * (1 - side) + np.array(WHALE_SIDE, np.float32) * side
        pale = (v > 0.62) & (s < 0.53)
        color[pale] = WHALE_BELLY
        color[pale & (np.arange(height)[:, None] % 2 == 0)] = WHALE_PLEAT
        smile = 0.64 - 0.24 * (s / 0.15) ** 2
        color[(np.abs(v - smile) * (bottom - top) < 0.5) & (s > 0.044) & (s < 0.15)] = WHALE_LINE

        view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
        water = view[:, cols].astype(np.float32)
        water += (color - water) * cover[..., None]
        view[:, cols] = water.astype(np.uint8)

        def blend(px: float, py: float, color: tuple) -> None:
            # The pixel the point falls in, as for the body (so it mirrors
            # exactly when the whale faces the other way)
            px, py = math.floor(px), math.floor(py)
            if 0 <= px < width and 0 <= py < height:
                p = (py * width + px) * 3
                for i in range(3):
                    fb[p + i] = int(fb[p + i] + (color[i] - fb[p + i]) * WHALE_OPACITY)

        # A big eye with a glint
        ex, ey = self.body_x(0.21), self.body_y(0.21, 0.36)
        blend(ex, ey, WHALE_LINE)
        blend(ex, ey + 1, WHALE_LINE)
        blend(ex + self.facing, ey, WHALE_GLINT)

        # A short flipper, paddling slowly
        rx, ry = self.body_x(0.264), self.body_y(0.264, 0.85)
        angle = 0.9 + 0.3 * math.sin(self.beat * 0.8)
        dx, dy = -math.cos(angle) * self.facing, math.sin(angle)
        for k in range(5):
            # Two pixels wide, toward the tail, down to a rounded tip
            for i in range(2 if k < 4 else 1):
                blend(rx + dx * k - i * self.facing, ry + dy * k, WHALE_FIN)


class Submarine(Event):
    """An old riveted submarine putters past in the distance, portholes
    glowing and propeller turning, sweeping its searchlight through the
    water ahead."""

    layer = BACK
    depth = 0.5  # (for its haze)

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        frames = []
        for prop in SUB_PROPS:
            rows = list(SUB_HULL)
            for row, tail in prop.items():
                rows[row] = "".join(t if t != "." else h for t, h in zip(tail, rows[row])) + rows[row][len(tail):]
            frames.append(rows)
        self.sprite = Sprite(SUB_PALETTE, frames, self.depth, HAZE)
        self.facing = random.choice((1, -1))
        # Far enough out that the searchlight slides into view ahead of it,
        # rather than switching on all at once
        lamp = self.sprite.at(0, self.facing, SUB_LAMP)
        self.x = (-SUB_BEAM_LENGTH if self.facing > 0 else tank.width + SUB_BEAM_LENGTH) - lamp
        self.y0 = tank.floor * random.uniform(0.2, 0.42) - self.sprite.h / 2 + 3
        self.prop = 0.0
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        self.x += SUB_SPEED * self.facing
        self.prop = (self.prop + 0.3) % len(SUB_PROPS)
        if random.random() < 0.06:
            # A bubble churned up by the propeller
            stern = self.sprite.at(self.x, self.facing, (0, 8))
            self.tank.bubbles.append(Bubble(stern, self.y + 8 + random.uniform(-1, 1)))
        # Until it's gone out the other side
        return (self.x + self.sprite.w >= 0) if self.facing < 0 else (self.x <= self.tank.width)

    @property
    def y(self) -> float:
        return self.y0 + 0.8 * math.sin(self.frame_count * FRAME_SECONDS * 0.5)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        y = round(self.y)
        # The searchlight: a cone of light from the lamp, sweeping slowly up
        # and down, fading with distance
        lx = self.sprite.at(self.x, self.facing, SUB_LAMP) + 0.5
        ly = y + SUB_LAMP[1] + 0.5
        ahead = (np.arange(width) + 0.5 - lx) * self.facing
        cols = np.flatnonzero((ahead > 0) & (ahead < SUB_BEAM_LENGTH))
        if len(cols):
            ahead = ahead[cols]
            tilt = 0.12 + 0.12 * math.sin(self.frame_count * FRAME_SECONDS * 0.6)
            off = np.arange(height, dtype=np.float32)[:, None] + 0.5 - ly - ahead * tilt
            spread = ahead * 0.3 + 0.6
            glow = np.clip(1 - np.abs(off) / spread, 0, 1) * (1 - ahead / SUB_BEAM_LENGTH) ** 1.5
            view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
            lit = view[:, cols] + glow[..., None] * np.array(SUB_BEAM, np.float32)
            view[:, cols] = np.minimum(lit, 255).astype(np.uint8)
        self.sprite.draw(fb, width, height, round(self.x), y, self.facing, int(self.prop))


class Jellyfish(Event):
    """A few glowing jellyfish drift sideways through on the current, bells
    pulsing and tentacles trailing behind. The fish keep out of their way."""

    layer = BACK

    class Jelly:
        __slots__ = ("x", "y", "drift", "pixels", "pulse", "pulse_step", "bob")

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        frames = JELLY_FRAMES if self.facing > 0 else [flip(f) for f in JELLY_FRAMES]
        self.w, self.h = len(JELLY_FRAMES[0][0]), len(JELLY_FRAMES[0])
        colors = random.sample(JELLY_COLORS, 3)
        self.jellies = []
        # The first in drifts fastest, so none catches up with another
        drifts = sorted((random.uniform(0.07, 0.1) for _ in range(random.randint(2, 3))), reverse=True)
        for i, drift in enumerate(drifts):
            j = Jellyfish.Jelly()
            # Staggered in from the side they drift from
            gap = i * random.uniform(16, 24)
            j.x = (-self.w - gap) if self.facing > 0 else tank.width + gap
            j.y = random.uniform(2, max(2, tank.floor - self.h - 3))
            j.drift = drift * self.facing
            color = colors[i]
            palette = {
                "R": mix(color, (255, 255, 255), 0.45), "G": scale(color, 0.55),
                "H": mix(color, (255, 255, 255), 0.7), "t": scale(color, 0.5),
            }
            j.pixels = [render_sprite(palette, f) for f in frames]
            j.pulse = random.uniform(0, len(frames))
            j.pulse_step = len(frames) / frames_for(random.uniform(2.2, 3.2))
            j.bob = random.uniform(0, 2 * math.pi)
            self.jellies.append(j)
        self.frame_count = 0

    def y(self, j: "Jellyfish.Jelly") -> float:
        return j.y + 1.2 * math.sin(self.frame_count * FRAME_SECONDS * 0.4 + j.bob)

    def step(self) -> bool:
        self.frame_count += 1
        for j in self.jellies:
            j.pulse = (j.pulse + j.pulse_step) % len(j.pixels)
            j.x += j.drift
        # Until the last is gone out the other side
        return any(-self.w - 2 < j.x < self.tank.width + 2 or (j.x < 0) == (self.facing > 0) for j in self.jellies)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        for j in self.jellies:
            x0, y0 = round(j.x), round(self.y(j))
            for dx, dy, color in j.pixels[int(j.pulse)]:
                put_max(fb, width, height, x0 + dx, y0 + dy, color)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return [(j.x - 1, self.y(j) - 1, j.x + self.w + 1, self.y(j) + self.h + 1) for j in self.jellies]


class Turtle(Event):
    """A sea turtle paddles slowly across at mid-depth."""

    depth = 0.5

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(TURTLE_PALETTE, [TURTLE_BODY + f for f in TURTLE_FLIPPERS], self.depth, HAZE)
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


class Moray(Event):
    """Now and then a moray eel slides along low among the seaweed, a wave
    running down its long body. Partway across it stops, head raised, slowly
    opening and shutting its jaws, then glides on. The fish keep clear."""

    depth = 0.3

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.head = Sprite(MORAY_PALETTE, MORAY_HEAD, self.depth, HAZE)
        self.facing = random.choice((1, -1))
        # The snout starts just out of view
        self.x = -1.0 if self.facing > 0 else tank.width + 1.0
        self.y0 = random.uniform(tank.floor - 9, tank.floor - 5)
        self.stop_at = tank.width * random.uniform(0.45, 0.7)
        self.pause = frames_for(random.uniform(*MORAY_PAUSE))
        self.speed = MORAY_SPEED
        self.phase = 0.0
        self.lift = 0.0  # how far it's raised its head
        self.gape = 0  # frames into the current opening of its jaws
        self.frame_count = 0
        self.body = [mix(c, HAZE, self.depth * 0.22) for c in (MORAY_BACK, MORAY_SIDE, MORAY_BELLY, MORAY_FIN)]

    def step(self) -> bool:
        self.frame_count += 1
        ahead = (self.x - self.stop_at) * self.facing
        stopped = self.pause and ahead >= 0
        if stopped:
            self.pause -= 1
            self.speed *= 0.93
            self.lift += (1 - self.lift) * 0.05
            self.gape = (self.gape + 1) % frames_for(1.6)
        else:
            self.speed += (MORAY_SPEED - self.speed) * 0.03
            self.lift *= 0.95
            self.gape = 0
        self.x += self.speed * self.facing
        # The wave runs down its body as fast as it swims, and slowly at rest
        self.phase += 0.08 + self.speed * 0.6
        tail = self.x - (MORAY_LENGTH + self.head.w) * self.facing
        return (tail <= self.tank.width) if self.facing > 0 else (tail >= 0)

    def at(self, i: float) -> float:
        """The y of the body's middle i pixels back from the head."""
        sway = (0.3 + 1.5 * i / MORAY_LENGTH) * math.sin(self.phase - i * 0.35)
        return self.y0 + sway - self.lift * 1.8 * max(0.0, 1 - i / 8)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        tail = self.x - (MORAY_LENGTH + self.head.w) * self.facing
        left, right = sorted((self.x, tail))
        return [(left, self.y0 - 4, right, self.y0 + 3)]

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        back, side, belly, fin = self.body
        head_w = self.head.w
        # The body, from the tail up to the back of the head, thinning toward
        # the tail, with a fin along its back
        for i in range(MORAY_LENGTH, 0, -1):
            x = math.floor(self.x - (head_w + i - 1) * self.facing)
            y = self.at(i)
            half = 1.7 - 1.4 * i / MORAY_LENGTH
            top, bottom = math.floor(y - half + 0.5), math.floor(y + half + 0.5)
            for py in range(top, bottom + 1):
                put(fb, width, height, x, py, back if py == top else belly if py == bottom and bottom - top >= 2 else side)
            if i > 2:
                put(fb, width, height, x, top - 1, fin)
        # The head, raised a little when it stops, jaws working
        open_ = self.gape > frames_for(0.5) and self.gape < frames_for(1.2)
        hx = math.floor(self.x) - (head_w - 1 if self.facing > 0 else 0)
        hy = math.floor(self.at(0) + 0.5) - 2
        self.head.draw(fb, width, height, hx, hy, self.facing, 1 if open_ else 0)
        ex = math.floor(self.head.at(hx, self.facing, MORAY_EYE))
        put(fb, width, height, ex - self.facing, hy + MORAY_EYE[1], mix(MORAY_PALETTE["E"], HAZE, self.depth * 0.22))


# name: (event, weight when picking one at random)
EVENTS = {
    "shark": (Shark, 3),
    "whale": (Whale, 2),
    "submarine": (Submarine, 2),
    "jellyfish": (Jellyfish, 2),
    "turtle": (Turtle, 2),
    "moray": (Moray, 1),  # only now and then
}


class Aquarium(Tank):
    EVENTS = EVENTS

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
            sprite = Sprite(*SPRITES[kind], depth, HAZE)
            cruise = (0.22, 0.34) if kind == SCHOOL else (0.12, 0.28)

            def make_fish(leader=None, place=(0, 0)):
                return Fish(sprite, width, top, bottom, depth, cruise, leader, place)

            self.fish += school(make_fish, SCHOOL_PLACES) if kind == SCHOOL else [make_fish()]
        self.fish.sort(key=lambda fish: -fish.depth)

        self.bubbles: list[Bubble] = []
        self.frame_count = 0
        self.init_events()
        self.shade_luts: dict[int, bytes] = {}

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
        if self.chest_timer <= 0:
            self.chest_timer = frames_for(random.uniform(*CHEST_SECONDS))
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


SCENE = Aquarium


def main() -> None:
    run_scene(Aquarium, "Aquarium LED matrix display")


if __name__ == "__main__":
    main()
