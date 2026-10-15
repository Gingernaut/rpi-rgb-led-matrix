#!/usr/bin/env python
"""Pixel aquarium: fish swimming over a sandy floor among swaying seaweed,
with rising bubbles, a treasure chest, and light rays from the surface.

The water, light rays, sand and ornaments are pre-rendered as a seamless loop
at startup, so each frame copies a background and draws the moving things.
Fish are drawn far to near, the far ones hazed toward the water color, some
seaweed stands in front of them, and the neon tetras swim as a school.

Every half minute or so something happens (see EVENTS): a shark cruises through
and the fish scatter, a whale or an old submarine passes in the distance,
jellyfish drift by, a sea turtle paddles across, or, now and then, a moray
eel slides along the bottom or a crab scuttles across the sand. To try one
out,
AQUARIUM_EVENT shows just that one, a couple of seconds in and then every
few seconds after (see aquarium_common):

    AQUARIUM_EVENT=shark uv run server/displays/preview.py aquarium --seconds 30

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from aquarium_common import (
    AMONG, BACK, FLEE_SPEED, FRAME_SECONDS, JELLY_FRAMES, SUB_LAMP, SUB_PALETTE, SUB_PROPS, Event, Fish, Sprite,
    Tank, flip, frames_for, render_sprite, school, submarine_frames,
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

# (sprite, how many) swimming on their own; the clownfish keep to their
# anemone (see Clownfish)
TANK = [("clownfish", 2), ("blue_tang", 1), ("yellow_tang", 1)]
CLOWNFISH = "clownfish"
# The tetra school: a leader, and each follower's place relative to it. The
# places are fixed in the tank, not relative to the leader's heading, so when
# the school turns it wheels around in place and the back becomes the front.
SCHOOL = "tetra"
SCHOOL_PLACES = [(-6, -3), (-6, 3), (6, 2), (-12, 0)]
# Their blue stripes catch the light: brightening toward TETRA_SHINE where a
# ray falls on them, and flashing as each turns, edge on for a moment
TETRA_SHINE = (200, 255, 255)
TETRA_SHINES = 5  # brightness steps, sprites made for each
TETRA_EDGE = [".CC.", ".RC."]  # turning, seen end on
TETRA_TURN_FRAMES = 3
TETRA_FLASH_FADE = 0.88  # a frame

# The clownfish's anemone: a bubble-tip anemone on the sand, its tentacles
# fanning up from a short column and swaying, bulbs at their tips
ANEMONE_COLUMN = (150, 62, 92)
ANEMONE_DISC = (190, 85, 115)
ANEMONE_TENTACLE = (225, 120, 150)
ANEMONE_TIP = (255, 200, 215)
ANEMONE_TENTACLES = 11
ANEMONE_ROAM = 15  # how far, either side, the clownfish stray from it
NESTLE_EVERY = 25.0  # seconds, on average, before a clownfish goes home
NESTLE_SECONDS = (5.0, 12.0)  # how long it stays

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

# An orange crab, claws up and eyes on stalks; frames step its legs
CRAB_PALETTE = {
    "B": (245, 100, 30), "D": (190, 65, 20), "C": (255, 135, 50), "E": (15, 10, 10), "L": (220, 85, 28),
}
CRAB = [
    [
        "C.......C",
        "CC.E.E.CC",
        ".CDBBBDC.",
        "LBBBBBBBL",
        "L.L...L.L",
    ],
    [
        "C.......C",
        "CC.E.E.CC",
        ".CDBBBDC.",
        "LBBBBBBBL",
        ".L.L.L.L.",
    ],
]
CRAB_SPEED = 0.15
CRAB_STOPS = (1, 3)  # how many times it stops on the way across

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

SUB_SPEED = 0.15
SUB_BEAM = (170, 150, 25)  # added to the water by the searchlight at full strength
SUB_BEAM_LENGTH = 26

JELLY_COLORS = [(230, 70, 170), (150, 90, 255), (80, 190, 255), (255, 120, 90)]

# A pufferfish: drawn as a round body that blows up into a spiky ball, its
# spots stretching with it
PUFFER_BACK = (215, 180, 85)
PUFFER_SPOT = (105, 75, 35)
PUFFER_BELLY = (240, 236, 214)
PUFFER_FIN = (225, 190, 100)
PUFFER_SPINE = (165, 135, 70)
PUFFER_EYE = (15, 15, 20)
PUFFER_RING = (250, 215, 110)
PUFFER_SIZE = ((4.5, 2.6), (5.5, 5.5))  # half width and height: swimming, blown up
PUFFER_SPEED = 0.1
PUFFER_SECONDS = (4.0, 7.0)  # blown up for this long

# A cuttlefish: its skin flickers through displays while it hovers
CUTTLE_HEAD = 3  # from where the arms join back to the mantle
CUTTLE_PROFILE = (  # its half-height, by pixels back from where the arms join
    [0, 1, 2, 3, 5, 9, 12, 14],
    [1.6, 2.0, 2.2, 2.0, 2.8, 2.8, 2.2, 1.0],
)
CUTTLE_LENGTH = 14
CUTTLE_ARMS = 4
CUTTLE_FIN = (230, 215, 190)  # the fin skirt rippling along its edges
CUTTLE_EYE = (20, 15, 15)
CUTTLE_LID = (235, 200, 120)
CUTTLE_SPEED = 0.08
CUTTLE_DISPLAY_SECONDS = 2.8  # each display it shows while hovering
CUTTLE_DISPLAYS = ["cloud", "zebra", "cloud", "flush", "mottle"]

# A blue-spotted stingray, its disc seen from a little above
RAY_LENGTH = 14
RAY_HALF = 2.6  # the disc's half-height
RAY_TAIL = 11
RAY_SKIN = (190, 150, 65)
RAY_RIM = (120, 92, 40)  # its underside edge, in shadow
RAY_SPOT = (40, 150, 255)
RAY_EYE = (25, 20, 10)
RAY_SPEED = 0.16
RAY_BURIED_SECONDS = (6.0, 10.0)

# A manta ray passing high overhead, seen from a little below: its outline's
# half-height (wingtip to wingtip) by fraction of the way from head to tail
MANTA_LENGTH = 22
MANTA_SPAN = 8.5
MANTA_PROFILE = ([0.0, 0.08, 0.2, 0.32, 0.42, 0.5, 0.6, 0.75, 1.0], [0.18, 0.32, 0.62, 0.88, 1.0, 0.72, 0.42, 0.24, 0.12])
MANTA_TAIL = 10
MANTA_TOP = (32, 42, 58)
MANTA_EDGE = (55, 70, 92)
MANTA_BELLY = (215, 222, 228)
MANTA_MARK = (60, 66, 80)
MANTA_SPEED = 0.17
MANTA_SHADOW = 0.1

# The whale shark (a rare visitor, see RARE_EVENTS): drawn from a profile
# like the whale, the top and bottom of its body by fraction of the way from
# its blunt snout to the root of its tail
WS_LENGTH = 56
WS_TOP = ([0.0, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.78, 0.82], [2.4, 3.6, 4.6, 5.2, 5.3, 4.8, 3.6, 2.6, 1.6, 1.2])
WS_BOTTOM = ([0.0, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 0.7, 0.78, 0.82], [2.0, 2.8, 3.8, 4.4, 4.4, 3.8, 2.6, 1.8, 1.2, 1.0])
WS_TAIL = 0.82  # where its tail fin starts
WS_NOTCH = 0.07  # how deep the fork of its tail cuts in
WS_BACK = (48, 64, 92)
WS_SIDE = (80, 96, 124)
WS_BELLY = (200, 205, 210)
WS_SPOT = (205, 215, 225)
WS_LINE = (15, 20, 30)  # its mouth, eye and gills
WS_FIN = (62, 78, 106)
WS_SPEED = 0.13
WS_SHADOW = 0.15
WS_MINUTES = (8.0, 12.0)  # between visits
REMORA = [(110, 112, 118), (40, 42, 48)]  # body, stripe
PILOT_FISH = [(245, 205, 40), (25, 25, 25)]  # body, bars


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


class Anemone:
    """A bubble-tip anemone on the sand: a short column, and tentacles fanning
    up and out from its disc, swaying in the current. Every other tentacle
    stands in front of a clownfish nestled in it."""

    def __init__(self, cx: int, sand_top: list[int]) -> None:
        self.cx = cx
        self.sand_top = sand_top
        self.disc = min(sand_top[max(cx - 3, 0):cx + 4]) - 2  # the row the tentacles grow from
        n = ANEMONE_TENTACLES
        # (base x, lean, length, phase), from the left
        self.tentacles = []
        for i in range(n):
            k = (i - (n - 1) / 2) / ((n - 1) / 2)  # -1 the leftmost to 1 the rightmost
            length = random.choice((6, 7, 7)) - round(abs(k) * 1.5)
            self.tentacles.append((cx + 0.5 + k * 3.5, k * 0.95, length, random.uniform(0, 2 * math.pi)))

    def draw(self, fb: bytearray, width: int, height: int, t: float, front: bool) -> None:
        if not front:
            # The column, down to the sand, and the disc on top
            for x in range(self.cx - 2, self.cx + 3):
                if 0 <= x < width:
                    for y in range(self.disc + 1, self.sand_top[x]):
                        put(fb, width, height, x, y, ANEMONE_COLUMN)
            for x in range(self.cx - 3, self.cx + 4):
                put(fb, width, height, x, self.disc, ANEMONE_DISC)
        # The ones at the back a shade darker, so the two rows read apart
        stalk = ANEMONE_TENTACLE if front else mix(ANEMONE_TENTACLE, ANEMONE_COLUMN, 0.45)
        for i, (bx, lean, length, phase) in enumerate(self.tentacles):
            if (i % 2 == 1) != front:
                continue
            for j in range(1, length + 1):
                # Swaying more toward the tip, as a wave runs up it
                angle = lean + 0.28 * math.sin(t * 1.1 + phase - j * 0.35) * j / length
                put(fb, width, height, math.floor(bx + math.sin(angle) * j), round(self.disc - math.cos(angle) * j),
                    ANEMONE_TIP if j == length else stalk)


class Clownfish(Fish):
    """A clownfish that keeps near its anemone: it pokes about within a short
    way of it, now and then goes home to nestle among the tentacles a while,
    turning about in them, and when the shark comes it dives in there rather
    than fleeing, and stays until it's gone."""

    def __init__(self, sprite, width: int, depth: float, cruise: tuple[float, float],
                 home: Anemone, slot: tuple[float, float]) -> None:
        self.home = home
        self.slot = slot  # where it sits in the anemone
        self.nestled = 0  # frames left at home
        self.hiding = False
        super().__init__(sprite, width, max(1, home.disc - 17), home.disc + 1, depth, cruise)
        self.x = min(max(slot[0] + random.uniform(-ANEMONE_ROAM, ANEMONE_ROAM), 0), width - self.w)
        self.homebody = True
        self.pick_turn()

    def pick_turn(self) -> None:
        # Back and forth within a short way of home
        lo = max(0.0, self.home.cx - ANEMONE_ROAM - self.w / 2)
        hi = min(self.tank_width - self.w, self.home.cx + ANEMONE_ROAM - self.w / 2)
        short = random.uniform(0, 5)
        self.turn_x = hi - short if self.facing > 0 else lo + short

    def wander(self) -> None:
        if self.hiding:
            self.go_home(FLEE_SPEED)
            if self.calm_in:
                self.calm_in -= 1
                if not self.calm_in:
                    # Lingering a little before venturing out again
                    self.hiding = False
                    self.nestled = frames_for(random.uniform(1.5, 5.0))
            return
        if self.nestled:
            self.go_home(1.0)
            self.nestled -= 1
            if not self.nestled:
                self.pick_turn()
            return
        if random.random() < 1 / frames_for(NESTLE_EVERY):
            self.nestled = frames_for(random.uniform(*NESTLE_SECONDS))
        super().wander()

    def go_home(self, hurry: float) -> None:
        """Swim to its place in the anemone, at hurry times cruising speed,
        and turn about there now and then."""
        sx, sy = self.slot
        dx = sx - self.x
        if abs(dx) > 1.5:
            self.facing = 1 if dx > 0 else -1
        elif random.random() < 0.006:
            self.facing = -self.facing
        want = min(self.cruise * hurry, max(0.0, dx * self.facing) * 0.06)
        self.speed += (want - self.speed) * 0.15
        self.target_y = sy
        climb = max(-0.08 * hurry, min(0.08 * hurry, (sy - self.y) * 0.06))
        self.vy += (climb - self.vy) * 0.2

    def flee(self, from_x: float) -> None:
        self.hiding = True
        self.calm_in = 0
        self.pause = 0

    def calm(self, delay: int) -> None:
        if self.hiding:
            self.calm_in = max(1, delay)


class Tetra(Fish):
    """A neon tetra whose blue stripe catches the light: brighter where a ray
    falls on it, and flashing as it turns, edge on for a moment."""

    def __init__(self, shines: list[Sprite], edge: Sprite, ray_at, *args) -> None:
        super().__init__(shines[0], *args)
        self.shines = shines
        self.edge = edge
        self.ray_at = ray_at  # how strongly a light ray falls on a pixel, 0 to 1
        self.flash = 0.0
        self.turning = 0  # frames left edge on

    def swim(self) -> None:
        facing = self.facing
        super().swim()
        self.flash *= TETRA_FLASH_FADE
        if self.turning:
            self.turning -= 1
        if self.facing != facing:
            self.flash = 1.0
            self.turning = TETRA_TURN_FRAMES

    def draw(self, fb: bytearray, width: int, height: int, t: float) -> None:
        x, y = self.pose(t)
        if self.turning:
            self.edge.draw(fb, width, height, x, y, self.facing, 0)
            return
        shine = max(self.flash, min(1.0, 1.6 * self.ray_at(x + self.w // 2, y)))
        sprite = self.shines[min(len(self.shines) - 1, int(shine * len(self.shines)))]
        sprite.draw(fb, width, height, x, y, self.facing, int(self.tail))


def render_background(width: int, height: int, sand_top: list[int], props: dict) -> tuple[list[bytes], np.ndarray]:
    """The water, light rays, and sand for each frame of the loop, and how
    strong the rays are at each pixel (0 to 255) in each. props maps (x, y)
    to the color of an ornament pixel resting on the sand; like the sand,
    ornaments are lit by the rays where they reach the floor."""
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
    lit = (rays * (1 - 0.7 * depth) * 255).astype(np.uint8)
    return [frame.tobytes() for frame in rgb.astype(np.uint8)], lit


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
        self.sprite = Sprite(SUB_PALETTE, submarine_frames(), self.depth, HAZE)
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


class Crab(Event):
    """Now and then a crab scuttles sideways across the sand in front of
    everything, stopping a few times on the way, claws up."""

    depth = 0.05

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.sprite = Sprite(CRAB_PALETTE, CRAB, self.depth, HAZE)
        self.facing = random.choice((1, -1))
        self.x = -self.sprite.w - 1.0 if self.facing > 0 else tank.width + 1.0
        # Where it stops, in the order it reaches them
        stops = sorted(random.uniform(0.15, 0.85) * tank.width for _ in range(random.randint(*CRAB_STOPS)))
        self.stops = stops if self.facing > 0 else stops[::-1]
        self.rest = 0
        self.speed = CRAB_SPEED
        self.step_phase = 0.0

    def step(self) -> bool:
        if self.rest:
            self.rest -= 1
        elif self.stops and (self.x + self.sprite.w / 2 - self.stops[0]) * self.facing >= 0:
            self.stops.pop(0)
            self.rest = frames_for(random.uniform(1.0, 3.0))
        else:
            self.x += self.speed * self.facing
            # Its legs step only while it walks
            self.step_phase = (self.step_phase + 0.12) % 2
        return -self.sprite.w - 2 < self.x < self.tank.width + 2

    def draw(self, fb: bytearray, t: float) -> None:
        tank = self.tank
        x = round(self.x)
        cols = [tank.sand_top[c] for c in range(max(x, 0), min(x + self.sprite.w, tank.width))]
        # Standing on the sand beneath it
        y = (min(cols) if cols else tank.floor) - self.sprite.h + 1
        self.sprite.draw(fb, tank.width, tank.height, x, y, self.facing, int(self.step_phase))


def hazed(color: tuple, depth: float) -> tuple:
    """A color faded toward the water as a sprite at depth would be."""
    return mix(color, HAZE, depth * 0.22)


class Pufferfish(Event):
    """A pufferfish putters through, its little fins whirring. Partway across
    something startles it and it blows itself up into a spiky ball, the fish
    nearby turning away, and hangs there a few seconds, drifting up a little,
    before it shrinks back down and carries on."""

    depth = 0.3

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        rx = PUFFER_SIZE[0][0]
        self.x = -rx - 3.0 if self.facing > 0 else tank.width + rx + 3.0  # its middle
        self.y = random.uniform(9, tank.floor - 9)
        self.puff_at = tank.width * random.uniform(0.35, 0.65)
        self.puffed = False
        self.blown = 0.0  # from 0 swimming to 1 blown up
        self.hold = 0  # frames left blown up
        self.speed = PUFFER_SPEED
        self.frame_count = 0
        # Spots on its back, in its own coordinates (+u toward its head), so
        # they stretch as it blows up
        self.spots = [(random.uniform(-0.75, 0.55), random.uniform(-0.85, -0.05)) for _ in range(7)]
        self.colors = {
            name: hazed(color, self.depth) for name, color in (
                ("back", PUFFER_BACK), ("spot", PUFFER_SPOT), ("belly", PUFFER_BELLY), ("fin", PUFFER_FIN),
                ("spine", PUFFER_SPINE), ("eye", PUFFER_EYE), ("ring", PUFFER_RING),
            )
        }

    def size(self) -> tuple[float, float]:
        e = self.blown * self.blown * (3 - 2 * self.blown)
        (rx0, ry0), (rx1, ry1) = PUFFER_SIZE
        return rx0 + (rx1 - rx0) * e, ry0 + (ry1 - ry0) * e

    @property
    def cy(self) -> float:
        return self.y + 0.8 * math.sin(self.frame_count * FRAME_SECONDS * 0.9) * (1 - self.blown)

    def step(self) -> bool:
        self.frame_count += 1
        if not self.puffed and (self.x - self.puff_at) * self.facing >= 0:
            self.puffed = True
            self.hold = frames_for(random.uniform(*PUFFER_SECONDS))
            # The fish close by and heading its way turn away
            for fish in self.free_fish():
                middle = fish.x + fish.w / 2
                if abs(middle - self.x) < 16 and (self.x - middle) * fish.facing > 0:
                    fish.turn()
        if self.hold:
            self.hold -= 1
            self.blown = min(1.0, self.blown + 0.07)
            self.speed *= 0.9
            self.y -= 0.012
        else:
            self.blown = max(0.0, self.blown - 0.02)
            if not self.blown:
                self.speed += (PUFFER_SPEED - self.speed) * 0.03
        self.x += self.speed * self.facing
        return -10 < self.x < self.tank.width + 10

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        rx, ry = self.size()
        return [(self.x - rx - 2, self.cy - ry - 2, self.x + rx + 2, self.cy + ry + 2)]

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        c = self.colors
        f = self.facing
        rx, ry = self.size()
        cx, cy = self.x, self.cy
        belly_line = 0.3 - 0.25 * self.blown
        for py in range(math.floor(cy - ry), math.ceil(cy + ry) + 1):
            v = (py + 0.5 - cy) / ry
            for px in range(math.floor(cx - rx), math.ceil(cx + rx) + 1):
                u = (px + 0.5 - cx) / rx * f
                if u * u + v * v > 1.0:
                    continue
                if v > belly_line:
                    color = c["belly"]
                elif any((u - su) ** 2 + (v - sv) ** 2 < 0.045 for su, sv in self.spots):
                    color = c["spot"]
                else:
                    color = c["back"]
                put(fb, width, height, px, py, color)
        # Its tail fan, beating
        beat = int(self.frame_count * 0.25) % 2
        tx = math.floor(cx - (rx + 0.5) * f)
        my = math.floor(cy)
        for dy in (-1, 0, 1):
            put(fb, width, height, tx, my + dy, c["fin"])
        for dy in ((-2, -1, 1, 2) if beat else (-1, 0, 1)):
            put(fb, width, height, tx - f, my + dy, c["fin"])
        # Spines standing out as it blows up
        if self.blown > 0.3:
            spine = scale(c["spine"], min(1.0, (self.blown - 0.3) / 0.5))
            for k in range(12):
                a = k * math.pi / 6 + 0.26
                put(fb, width, height, math.floor(cx + math.cos(a) * (rx + 1)), math.floor(cy + math.sin(a) * (ry + 1)), spine)
        # A big ringed eye, and a little fin whirring behind it
        ex, ey = math.floor(cx + 0.45 * rx * f), math.floor(cy - 0.3 * ry)
        put(fb, width, height, ex, ey, c["eye"])
        put(fb, width, height, ex, ey - 1, c["ring"])
        put(fb, width, height, math.floor(cx - 0.05 * rx * f) - f * beat, my + 1, c["fin"])


class Cuttlefish(Event):
    """A cuttlefish glides in on the fin skirt rippling along its sides, arms
    held together ahead of it, and hovers a while partway across, its skin
    flickering through its displays: dark clouds rolling forward over it,
    bold zebra stripes, a red-purple flush, then back to a sandy mottle
    before it moves on."""

    depth = 0.35

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        # x is where its arms join its head; the rest of it starts out of view
        self.x = -CUTTLE_ARMS - 1.0 if self.facing > 0 else tank.width + CUTTLE_ARMS + 1.0
        self.y = random.uniform(8, tank.floor - 8)
        self.stop_at = tank.width * random.uniform(0.42, 0.58) + CUTTLE_LENGTH / 2 * self.facing
        self.display = frames_for(CUTTLE_DISPLAY_SECONDS)
        self.hover = self.display * len(CUTTLE_DISPLAYS)  # frames left hovering
        self.held = 0  # frames it's hovered
        self.speed = CUTTLE_SPEED
        self.frame_count = 0
        self.noise = {(u, r): random.random() for u in range(-CUTTLE_ARMS, CUTTLE_LENGTH + 1) for r in range(-4, 5)}

    @property
    def cy(self) -> float:
        return self.y + 0.7 * math.sin(self.frame_count * FRAME_SECONDS * 0.5)

    def step(self) -> bool:
        self.frame_count += 1
        if self.hover and (self.x - self.stop_at) * self.facing >= 0:
            self.hover -= 1
            self.held += 1
            self.speed *= 0.95
        else:
            self.speed += (CUTTLE_SPEED - self.speed) * 0.02
        self.x += self.speed * self.facing
        tail = self.x - CUTTLE_LENGTH * self.facing
        return (tail <= self.tank.width + 1) if self.facing > 0 else (tail >= -1)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        left, right = sorted((self.x + CUTTLE_ARMS * self.facing, self.x - CUTTLE_LENGTH * self.facing))
        return [(left, self.cy - 4, right, self.cy + 4)]

    def modes(self) -> tuple[str, str, float]:
        """The display showing, the one before it, and how far it's taken
        over from that one (0 to 1)."""
        if not self.held or not self.hover:
            return "mottle", "mottle", 1.0
        i = min(self.held // self.display, len(CUTTLE_DISPLAYS) - 1)
        before = CUTTLE_DISPLAYS[i - 1] if i else "mottle"
        return CUTTLE_DISPLAYS[i], before, min(1.0, (self.held - i * self.display) / frames_for(0.8))

    def skin(self, mode: str, u: int, r: int, t: float) -> tuple:
        n = self.noise.get((u, r), 0.5)
        if mode == "mottle":
            return (215, 185, 135) if n > 0.72 else (100, 70, 48) if n < 0.25 else (160, 120, 82)
        if mode == "cloud":
            band = math.sin(u * 0.9 + t * 6.5)
            return mix((185, 145, 105), (55, 38, 32), max(0.0, (band - 0.3) / 0.7))
        if mode == "zebra":
            return (235, 228, 215) if math.sin(u * 1.7 + r * 0.9) > 0.1 else (40, 30, 30)
        return (245, 205, 215) if n > 0.85 else (175, 55, 95)  # flush

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        f = self.facing
        t = self.frame_count * FRAME_SECONDS
        cy = self.cy
        my = round(cy)
        mode, before, k = self.modes()

        def color_at(u: int, r: int) -> tuple:
            color = self.skin(mode, u, r, t)
            if k < 1:
                color = mix(self.skin(before, u, r, t), color, k)
            return hazed(color, self.depth)

        fin = hazed(mix(CUTTLE_FIN, HAZE, 0.4), self.depth)
        x0 = math.floor(self.x)
        for u in range(CUTTLE_LENGTH + 1):
            px = x0 - u * f
            half = float(np.interp(u, *CUTTLE_PROFILE))
            top, bottom = round(cy - half), round(cy + half)
            for py in range(top, bottom + 1):
                put(fb, width, height, px, py, color_at(u, py - my))
            # The fin skirt along the mantle, a wave running down it
            if u >= CUTTLE_HEAD:
                wave = math.sin(u * 1.3 - t * 9)
                if wave > -0.3:
                    put(fb, width, height, px, top - 1, fin)
                if wave < 0.3:
                    put(fb, width, height, px, bottom + 1, fin)
        # Its arms, held together and curling a little at the tips
        for a in range(1, CUTTLE_ARMS + 1):
            rows = (-1, 0, 1) if a <= 2 else (-1, 0) if a == 3 else (0,)
            curl = round(0.4 * math.sin(t * 2 + a)) if a >= 3 else 0
            for r in rows:
                put(fb, width, height, x0 + a * f, my + r + curl, scale(color_at(0, r), 0.85))
        # An eye with a pale lid
        put(fb, width, height, x0 - 2 * f, my - 1, hazed(CUTTLE_EYE, self.depth))
        put(fb, width, height, x0 - 2 * f, my - 2, hazed(CUTTLE_LID, self.depth))


class Stingray(Event):
    """A blue-spotted stingray glides in low, the edges of its disc rippling,
    and settles on the sand partway across. It flutters itself under,
    kicking up sand, and lies there a while with only its eyes showing,
    then lifts off in a puff of sand and glides away."""

    depth = 0.15

    class Grain:
        __slots__ = ("x", "y", "vx", "vy", "age", "life", "color")

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        self.x = -2.0 if self.facing > 0 else tank.width + 1.0  # its snout
        # Where it settles: on open sand, clear of the anemone and the chest
        # (land_at is where its snout stops)
        a = tank.anemone.cx
        chest = tank.chest_x + len(CHEST[0]) / 2
        spots = [x for x in range(round(tank.width * 0.25), round(tank.width * 0.75))
                 if abs(x - RAY_LENGTH / 2 * self.facing - a) > RAY_LENGTH / 2 + 8
                 and abs(x - RAY_LENGTH / 2 * self.facing - chest) > RAY_LENGTH / 2 + 4]
        self.land_at = random.choice(spots) if spots else tank.width * random.uniform(0.35, 0.65)
        # Gliding low, but clear over the anemone
        self.cruise_y = tank.anemone.disc - 8 - random.uniform(0, 3)
        self.y = self.cruise_y
        self.state = "glide"
        self.timer = 0
        self.speed = RAY_SPEED
        self.cover = 0.0  # how buried it is
        self.flap = 1.0  # how hard its edges ripple
        self.phase = 0.0
        self.frame_count = 0
        self.grains: list[Stingray.Grain] = []
        self.spots = {(u, r) for u in range(2, RAY_LENGTH - 1) for r in range(-2, 2) if random.random() < 0.16}
        self.sand = {(u, r): random.choice(SAND) for u in range(-1, RAY_LENGTH + RAY_TAIL + 2) for r in range(-4, 4)}
        self.colors = {name: hazed(color, self.depth) for name, color in (
            ("skin", RAY_SKIN), ("rim", RAY_RIM), ("spot", RAY_SPOT), ("eye", RAY_EYE))}

    def rest_y(self) -> float:
        """Its middle when lying on the sand under it."""
        tank = self.tank
        left, right = sorted((self.x, self.x - RAY_LENGTH * self.facing))
        cols = range(max(math.floor(left), 0), min(math.ceil(right) + 1, tank.width))
        ground = min(tank.sand_top[c] for c in cols) if cols else tank.floor
        return ground - 0.5 - RAY_HALF * 0.8

    def kick(self, n: int, strength: float) -> None:
        """Throw up n grains of sand from under its disc."""
        tank = self.tank
        for _ in range(n):
            g = Stingray.Grain()
            g.x = self.x - random.uniform(0, RAY_LENGTH) * self.facing
            col = min(max(int(g.x), 0), tank.width - 1)
            g.y = tank.sand_top[col] - 1.0
            g.vx = random.uniform(-0.25, 0.25)
            g.vy = -random.uniform(0.1, 0.35) * strength
            g.age, g.life = 0, frames_for(random.uniform(0.8, 2.0))
            g.color = random.choice(SAND)
            self.grains.append(g)

    def step(self) -> bool:
        self.frame_count += 1
        state = self.state
        if state in ("glide", "leave"):
            self.speed += (RAY_SPEED - self.speed) * 0.03
            self.y += (self.cruise_y - self.y) * 0.02
            self.flap += (1 - self.flap) * 0.05
            if state == "glide" and (self.x - self.land_at) * self.facing >= 0:
                self.state = "land"
        elif state == "land":
            self.speed *= 0.97
            self.y += (self.rest_y() - self.y) * 0.04
            if abs(self.rest_y() - self.y) < 0.2 and self.speed < 0.02:
                self.state, self.timer = "bury", frames_for(2.0)
        elif state == "bury":
            self.speed *= 0.9
            self.flap = 2.0
            self.cover = min(0.85, self.cover + 0.85 / frames_for(2.0))
            if random.random() < 0.6:
                self.kick(1, 0.6)
            self.timer -= 1
            if self.timer <= 0:
                self.state, self.timer = "rest", frames_for(random.uniform(*RAY_BURIED_SECONDS))
        elif state == "rest":
            self.flap *= 0.9
            self.timer -= 1
            if self.timer <= 0:
                self.state, self.timer = "lift", frames_for(1.0)
                self.kick(25, 1.0)
        elif state == "lift":
            self.cover = max(0.0, self.cover - 0.06)
            self.flap = 1.6
            self.y -= 0.05
            self.speed += (RAY_SPEED - self.speed) * 0.03
            self.timer -= 1
            if self.timer <= 0:
                self.state = "leave"
        self.phase += 0.03 + 0.12 * self.flap
        self.x += self.speed * self.facing

        tank = self.tank
        grains = []
        for g in self.grains:
            g.age += 1
            g.vy += 0.015
            g.x += g.vx
            g.y += g.vy
            col = int(g.x)
            if g.age < g.life and 0 <= col < tank.width and not (g.vy > 0 and g.y >= tank.sand_top[col]):
                grains.append(g)
        self.grains = grains
        tail = self.x - (RAY_LENGTH + RAY_TAIL) * self.facing
        return (tail <= tank.width) if self.facing > 0 else (tail >= 0)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        left, right = sorted((self.x, self.x - RAY_LENGTH * self.facing))
        return [(left, self.y - RAY_HALF - 1, right, self.y + RAY_HALF + 1)]

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        c = self.colors
        f = self.facing
        cy = self.y
        my = round(cy)
        cover = self.cover
        x0 = math.floor(self.x)

        def buried(color: tuple, u: int, r: int, amount: float = cover) -> tuple:
            return mix(color, self.sand[(u, max(-4, min(3, r)))], amount) if amount > 0 else color

        for u in range(RAY_LENGTH + 1):
            s = u / RAY_LENGTH
            half = RAY_HALF * math.sqrt(max(0.0, math.sin(math.pi * min(1.0, 0.06 + s))))
            ripple = 0.7 * self.flap * math.sin(s * 10 - self.phase) * min(1.0, 2 * s)
            top, bottom = round(cy - half - ripple), round(cy + half * 0.8 - ripple * 0.5)
            px = x0 - u * f
            for py in range(top, bottom + 1):
                r = py - my
                v = (py - cy) / max(half, 0.5)
                color = c["rim"] if v > 0.45 else c["spot"] if (u, r) in self.spots else c["skin"]
                put(fb, width, height, px, py, buried(color, u, r))
        # Its eyes, bumps on top near the front, still showing when buried
        for u in (3, 5):
            put(fb, width, height, x0 - u * f, my - 2, c["eye"])
        # The tail, blue stripes along it and a barb near its root
        for k in range(RAY_TAIL):
            u = RAY_LENGTH + k
            ty = round(cy + 0.4 + math.sin(self.phase * 0.5 - k * 0.5) * k / RAY_TAIL)
            color = c["spot"] if k % 3 == 1 else c["skin"]
            put(fb, width, height, x0 - u * f, ty, buried(color, u, ty - my, cover * 0.7))
            if k == 3:
                put(fb, width, height, x0 - u * f, ty - 1, buried(c["rim"], u, ty - my - 1, cover * 0.7))
        for g in self.grains:
            put(fb, width, height, math.floor(g.x), math.floor(g.y), g.color)


class Manta(Event):
    """A manta ray glides past high overhead, seen from a little below: broad
    wings beating slowly in a wave from root to tip, a pale belly between
    dark wing edges, horn-like head fins curled ahead and a thin tail
    behind. Its shadow dims the light as it goes."""

    layer = BACK
    depth = 0.6  # (for its haze)

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        self.nose = -3.0 if self.facing > 0 else tank.width + 3.0
        self.y = random.uniform(8.5, 11.0)
        self.beat = random.uniform(0, 2 * math.pi)
        self.beat_step = 2 * math.pi / frames_for(3.8)
        self.colors = {name: np.array(hazed(color, self.depth), np.float32) for name, color in (
            ("top", MANTA_TOP), ("edge", MANTA_EDGE), ("belly", MANTA_BELLY), ("mark", MANTA_MARK))}

    def step(self) -> bool:
        self.nose += MANTA_SPEED * self.facing
        self.beat += self.beat_step
        width = self.tank.width
        tail = self.nose - (MANTA_LENGTH + MANTA_TAIL) * self.facing
        left, right = sorted((self.nose, self.nose - MANTA_LENGTH * self.facing))
        self.shade = 1 - MANTA_SHADOW * max(0.0, min(right, width) - max(left, 0)) / width
        return (tail <= width) if self.facing > 0 else (tail >= 0)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        f = self.facing
        c = self.colors
        view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
        s = (self.nose - (np.arange(width) + 0.5)) * f / MANTA_LENGTH
        cols = np.flatnonzero((s >= 0) & (s <= 1))
        lift = 0.0
        if len(cols):
            s = s[cols]
            span = np.interp(s, *MANTA_PROFILE) * MANTA_SPAN
            # The wingbeat runs from the root out, raising and lowering both
            # wings: the upper outline reaches higher as the lower one draws in
            flap = np.sin(self.beat - s * 2.2)
            center = self.y - 0.6 * flap
            top = center - span * (0.8 + 0.25 * flap)
            bottom = center + span * (0.8 - 0.25 * flap)
            y = np.arange(height, dtype=np.float32)[:, None] + 0.5
            cover = np.clip(np.minimum(y - top, bottom - y) + 0.5, 0, 1)
            w = (y - center) / np.maximum(span, 0.5)
            color = np.broadcast_to(c["top"], (height, len(cols), 3)).copy()
            belly = (np.abs(w) < 0.5) & (s > 0.08) & (s < 0.66)
            color[belly] = c["belly"]
            body_col = np.floor(s * MANTA_LENGTH).astype(int)[None, :]
            row = np.floor(y - center).astype(int)
            color[belly & (((body_col == 9) & (row == 1)) | ((body_col == 11) & (row == -1))
                           | ((body_col == 12) & (row == 2)))] = c["mark"]
            # Gill slits
            color[belly & np.isin(body_col, (4, 5, 6)) & (np.abs(row) == 2)] = c["mark"]
            color[(np.abs(w) > 0.85) & (cover > 0)] = c["edge"]
            water = view[:, cols].astype(np.float32)
            water += (color - water) * cover[..., None]
            view[:, cols] = water.astype(np.uint8)
            lift = 0.6 * math.sin(self.beat)
        # Its head fins, curled ahead, and the mouth between them
        nx, ny = math.floor(self.nose), math.floor(self.y - lift)
        top = tuple(int(v) for v in c["top"])
        for dx, dy in ((1, -2), (2, -2), (2, -1), (1, 1), (2, 1), (2, 0)):
            put(fb, width, height, nx + dx * f, ny + dy, top)
        # The thin tail
        for k in range(MANTA_TAIL):
            put(fb, width, height, math.floor(self.nose - (MANTA_LENGTH + k) * f),
                math.floor(self.y + 0.5 * math.sin(self.beat - 2.2 - k * 0.3)), top)


class WhaleShark(Event):
    """Once in a while (see RARE_EVENTS) a whale shark cruises slowly through,
    nearly as long as the tank is wide: a broad flat head with a wide mouth,
    its dark back starred with pale spots and checkered with pale lines, its
    tall tail sweeping. Remoras ride under its belly and little golden pilot
    fish swim along ahead of its mouth. The fish slow down to watch."""

    depth = 0.55

    def __init__(self, tank: "Aquarium") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        # Its pilot fish come into view first
        self.nose = -10.0 if self.facing > 0 else tank.width + 10.0
        self.y = tank.floor * random.uniform(0.42, 0.52)
        self.beat = 0.0
        self.beat_step = 2 * math.pi / frames_for(4.5)
        self.frame_count = 0
        self.pilots = [(random.uniform(2, 9), random.uniform(-3, 3), random.uniform(0, 2 * math.pi))
                       for _ in range(3)]
        self.colors = {name: hazed(color, self.depth) for name, color in (
            ("back", WS_BACK), ("side", WS_SIDE), ("belly", WS_BELLY), ("spot", WS_SPOT), ("line", WS_LINE),
            ("fin", WS_FIN), ("remora", REMORA[0]), ("stripe", REMORA[1]), ("pilot", PILOT_FISH[0]),
            ("bar", PILOT_FISH[1]))}
        for fish in tank.fish:
            fish.pace = 0.6

    def end(self) -> None:
        for fish in self.tank.fish:
            fish.pace = 1.0

    def step(self) -> bool:
        self.frame_count += 1
        self.nose += WS_SPEED * self.facing
        self.beat += self.beat_step
        width = self.tank.width
        tail = self.nose - WS_LENGTH * self.facing
        left, right = sorted((self.nose, tail))
        self.shade = 1 - WS_SHADOW * max(0.0, min(right, width) - max(left, 0)) / width
        return (tail <= width) if self.facing > 0 else (tail >= 0)

    def edges(self, s):
        """The top, bottom and middle of its body at s (a number or array)
        from the snout (0) to the tip of its tail (1)."""
        top = np.interp(s, *WS_TOP)
        bottom = np.interp(s, *WS_BOTTOM)
        tail = np.clip((s - WS_TAIL) / (1 - WS_TAIL), 0, 1) ** 1.3 * (0.9 + 0.1 * math.sin(self.beat))
        top = top + 6.8 * tail + np.interp(s, [0.42, 0.5, 0.54], [0, 4.5, 0]) + np.interp(s, [0.71, 0.74, 0.76], [0, 1.6, 0])
        bottom = bottom + 4.2 * tail
        center = self.y + 0.8 * np.clip((s - 0.5) / 0.5, 0, 1) ** 2 * math.sin(self.beat)
        return center - top, center + bottom, center

    def body_x(self, s: float) -> float:
        return self.nose - s * WS_LENGTH * self.facing

    def body_y(self, s: float, v: float) -> float:
        top, bottom, _ = self.edges(s)
        return float(top + v * (bottom - top))

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        c = self.colors
        f = self.facing
        s = (self.nose - (np.arange(width) + 0.5)) * f / WS_LENGTH
        cols = np.flatnonzero((s >= 0) & (s <= 1))
        if len(cols):
            s = s[cols]
            top, bottom, center = self.edges(s)
            y = np.arange(height, dtype=np.float32)[:, None] + 0.5
            cover = np.clip(np.minimum(y - top, bottom - y) + 0.5, 0, 1)
            # The fork of its tail
            k = 1 - np.sqrt(np.clip((1 - s) / WS_NOTCH, 0, 1))
            gap = np.maximum(center - k * (center - top) - y, y - center - k * (bottom - center))
            cover *= np.where(k > 0, np.clip(gap + 0.5, 0, 1), 1)

            v = (y - top) / np.maximum(bottom - top, 0.1)
            side = np.clip(v * 1.5, 0, 1)[..., None]
            color = np.array(c["back"], np.float32) * (1 - side) + np.array(c["side"], np.float32) * side
            body_col = np.floor(s * WS_LENGTH).astype(int)[None, :]
            row = np.floor(y - center).astype(int)
            upper = (v < 0.62) & (s < WS_TAIL)
            # Pale lines across its flank, spots between them, and smaller,
            # closer spots on its head
            lines = upper & (body_col % 5 == 2) & (s > 0.14) & (s < 0.72)
            color[lines] = color[lines] * 0.65 + np.array(c["spot"], np.float32) * 0.35
            spots = upper & ~lines & (
                ((s >= 0.14) & (((body_col % 5 == 0) & (row % 3 == 0)) | ((body_col % 5 == 4) & (row % 3 == 2))))
                | ((s < 0.14) & ((body_col + row) % 3 == 0) & (row % 2 == 0)))
            color[spots] = c["spot"]
            color[(v >= 0.62) & (s < WS_TAIL - 0.02)] = c["belly"]
            # Its wide mouth across the front, and gill slits
            color[(s < 0.05) & (np.abs(v - 0.6) * (bottom - top) < 0.5)] = c["line"]
            gills = np.isin(body_col, (8, 10, 12)) & (v > 0.3) & (v < 0.75)
            color[gills] = color[gills] * 0.55

            view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
            water = view[:, cols].astype(np.float32)
            water += (color - water) * cover[..., None]
            view[:, cols] = water.astype(np.uint8)

        # A small eye just behind the corner of its mouth
        put(fb, width, height, math.floor(self.body_x(0.075)), math.floor(self.body_y(0.075, 0.45)), c["line"])
        # A long pectoral fin, angled down and back
        rx, ry = self.body_x(0.2), self.body_y(0.2, 0.9)
        angle = 0.75 + 0.1 * math.sin(self.beat * 0.5)
        dx, dy = -math.cos(angle) * f, math.sin(angle)
        for k in range(7):
            for i in range(2 if k < 5 else 1):
                put(fb, width, height, math.floor(rx + dx * k - i * f), math.floor(ry + dy * k), c["fin"])
        # Remoras riding under its belly
        for rs in (0.3, 0.42):
            x, y = math.floor(self.body_x(rs)), math.floor(self.edges(rs)[1] + 0.5)
            for i in range(5):
                put(fb, width, height, x - i * f, y, c["remora"])
                put(fb, width, height, x - i * f, y + 1, c["stripe"] if 0 < i < 4 else c["remora"])
        # Its pilot fish ahead of its mouth
        clock = self.frame_count * FRAME_SECONDS
        mouth_y = self.body_y(0.0, 0.5)
        for ahead, dy, phase in self.pilots:
            px = math.floor(self.nose + (ahead + 0.8 * math.sin(clock * 0.7 + phase)) * f)
            py = math.floor(mouth_y + dy + 0.6 * math.sin(clock * 2 + phase))
            for i in range(4):  # from its head back
                color = c["bar"] if i == 2 else c["pilot"]
                put(fb, width, height, px - i * f, py, color)
                if i < 3:
                    put(fb, width, height, px - i * f, py + 1, color)


# name: (event, weight when picking one at random)
EVENTS = {
    "shark": (Shark, 3),
    "whale": (Whale, 2),
    "submarine": (Submarine, 2),
    "jellyfish": (Jellyfish, 2),
    "turtle": (Turtle, 2),
    "moray": (Moray, 1),  # only now and then
    "crab": (Crab, 1),  # only now and then
    "cuttlefish": (Cuttlefish, 2),
    "pufferfish": (Pufferfish, 2),
    "stingray": (Stingray, 2),
    "manta": (Manta, 2),
}
# name: (event, (fewest, most) minutes between its visits)
RARE_EVENTS = {
    "whale-shark": (WhaleShark, WS_MINUTES),
}


class Aquarium(Tank):
    EVENTS = EVENTS
    RARE_EVENTS = RARE_EVENTS

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
        self.background, self.rays = render_background(width, height, self.sand_top, props)

        # The anemone in the open sand between them, as near the middle of it
        # as it fits
        left, right = sorted((self.chest_x + chest_w // 2, self.bubbler_x))
        lo, hi = left + chest_w // 2 + 9, right - 8
        self.anemone = a = Anemone(round(random.uniform(lo, hi)) if lo <= hi else (left + right) // 2, self.sand_top)
        # The other fish keep out of it
        self.scenery = [(a.cx - 8, a.disc - 7, a.cx + 9, height)]
        self.chest_lid = render_sprite(CHEST_PALETTE, CHEST_OPEN)
        self.chest_timer = frames_for(random.uniform(*CHEST_SECONDS))

        # Seaweed spread along the floor clear of the chest and bubbler; a
        # couple of shorter ones stand in front of the fish
        count = max(3, width // 9)
        spots = [round((i + random.uniform(0.2, 0.8)) * width / count) for i in range(count)]
        spots = [
            x for x in spots
            if x < width and not self.chest_x - 1 <= x <= self.chest_x + chest_w and abs(x - self.bubbler_x) > 1
            and abs(x - self.anemone.cx) > 8
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
        # The clownfish's places in the anemone: side by side, one a little
        # higher, among the tentacles
        a = self.anemone
        slots = [(a.cx - 8.0, a.disc - 7.0), (a.cx + 1.0, a.disc - 5.0)]
        random.shuffle(slots)
        for kind, depth in zip(kinds, depths):
            sprite = Sprite(*SPRITES[kind], depth, HAZE)
            cruise = (0.22, 0.34) if kind == SCHOOL else (0.12, 0.28)
            if kind == CLOWNFISH:
                self.fish.append(Clownfish(sprite, width, depth, cruise, a, slots.pop()))
            elif kind == SCHOOL:
                palette, frames = SPRITES[kind]
                shines = [
                    Sprite({**palette, "C": mix(palette["C"], TETRA_SHINE, k / (TETRA_SHINES - 1))}, frames, depth, HAZE)
                    for k in range(TETRA_SHINES)
                ]
                edge = Sprite({**palette, "C": TETRA_SHINE}, [TETRA_EDGE], depth, HAZE)

                def make_tetra(leader=None, place=(0, 0)):
                    return Tetra(shines, edge, self.ray_at, width, top, bottom, depth, cruise, leader, place)

                self.fish += school(make_tetra, SCHOOL_PLACES)
            else:
                self.fish.append(Fish(sprite, width, top, bottom, depth, cruise))
        self.fish.sort(key=lambda fish: -fish.depth)

        self.bubbles: list[Bubble] = []
        self.frame_count = 0
        self.init_events()
        self.shade_luts: dict[int, bytes] = {}

    def ray_at(self, x: int, y: int) -> float:
        """How strongly a light ray falls on pixel (x, y) now, 0 to 1."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.rays[self.frame_count % LOOP_FRAMES, y, x] / 255
        return 0.0

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
        self.anemone.draw(fb, width, height, t, front=False)

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

        self.anemone.draw(fb, width, height, t, front=True)
        for weed in self.front_weeds:
            weed.draw(fb, width, height, t)

        return fb


SCENE = Aquarium


def main() -> None:
    run_scene(Aquarium, "Aquarium LED matrix display")


if __name__ == "__main__":
    main()
