#!/usr/bin/env python
"""The deep sea at night: black water with marine snow drifting down, a
hydrothermal vent smoldering on the sea floor among red-plumed tube worms,
and a few deep-sea creatures, seen mostly by their own light: an anglerfish
with its glowing lure, lanternfish, a viperfish and a glowing jellyfish,
and now and then a vampire squid or a spider crab picking its way along
the floor. They take turns: only one to three are in view at
a time, each staying a while before swimming off.

Light decides what's seen. Each creature has its true colors, which show only
where something lights it (the vent's glow, a lure, a submersible's lamps), and
a dim silhouette everywhere else; its photophores glow regardless. The
background likewise has a lit version, revealed wherever light falls. And
the water itself sparks where anything stirs it, so swimmers leave a faint
wake of light, brighter the faster they go.

Every half minute or so something happens (see EVENTS): a giant squid jets
past and the creatures put out their lights, a submersible's lamps sweep the
dark, a siphonophore or a comb jelly drifts by, something unseen startles
the water into a wave of sparks, the vent erupts, the anglerfish lures in a
meal, a black dragonfish noses through, or a gulper eel opens
its enormous mouth. AQUARIUM_EVENT=<name> shows just that one (see
aquarium_common):

    AQUARIUM_EVENT=squid uv run server/displays/preview.py aquarium-night --seconds 30

Requires the rgbmatrix Python bindings (make build-python) and numpy.
"""

import math
import random

import numpy as np

from aquarium_common import (
    AMONG, BACK, FRAME_SECONDS, JELLY_FRAMES, SUB_LAMP, SUB_PALETTE, SUB_PROPS, Event, Fish, Sprite, Tank, flip,
    frames_for, submarine_frames,
)
from matrix_common import hsv, mix, put, put_max, run_scene, scale

FRONT = "front"  # drawn over everything

WATER_TOP = (0, 6, 20)
WATER_BOTTOM = (0, 1, 6)
WATER_LIT = (30, 66, 92)  # the water where light falls: a blue haze
DIM = 0.22  # how much of their color creatures and the floor show unlit
FLOOR_ROWS = 3  # on average; the floor rolls a row up and down
SILT = [(95, 86, 76), (82, 74, 66), (108, 98, 86), (88, 80, 74)]
ROCKS = [(70, 66, 64), (60, 58, 60), (84, 76, 70)]

# The vent: a tall chimney of rock and mineral crust with a smaller one
# beside it, smoking from both mouths ("M"), veins ("O") glowing from inside
VENT_PALETTE = {
    "K": (88, 72, 60), "D": (64, 54, 48), "Y": (150, 115, 45), "R": (125, 62, 32),
    "O": (130, 70, 30), "M": (130, 70, 30),
}
VENT = [
    "...DMD.......",
    "...KOK.......",
    "..DKOKD......",
    "..KYKOK......",
    "..KKOKD......",
    ".DKOKKK......",
    ".KKKYOK......",
    ".KRKOKKD.DMD.",
    "DKKOKKYK.KOK.",
    "KKKKOKKKDKOKD",
    "KYKOKKRKKOKKK",
    "KKOKKKKKOKKYK",
    "DKKKKOKKKKKKD",
    "KKRKKKKOKKRKK",
]
VENT_DEEP = (110, 20, 0)  # a vein between pulses of heat, which climb it
VENT_HOT = (255, 140, 30)
VENT_MOUTH = (255, 210, 110)
VENT_GLOW = (80, 28, 4)  # added to the water by the vent's glow at full strength
VENT_TINT = (1.0, 0.75, 0.5)  # its light is warm
VENT_REACH = 10.0  # the glow's radius (one sigma), in pixels
VENT_REST = 0.55  # how bright its glow is while it smolders (erupting, it's 1)
SMOKE = [  # by age: leaving the mouth glowing, lit by the vent, cooling grey
    (0.0, (230, 110, 25)), (0.12, (110, 55, 25)), (0.4, (40, 32, 30)), (1.0, (0, 0, 0)),
]
SMOKE_SECONDS = (5.0, 8.0)

ERUPTION_RUMBLE = 4.0  # seconds of rising heat before it blows
ERUPTION_SECONDS = (8.0, 12.0)  # how long it erupts before dying back down
EMBER_HOT = (255, 210, 90)
EMBER_COOL = (190, 40, 0)
EMBER_GRAVITY = 0.012

TUBE = (140, 132, 118)
PLUME = (215, 25, 35)

# The glow around a lure: (dx, dy, strength) out to a few pixels
LURE_HALO = [
    (dx, dy, 0.55 * math.exp(-(dx * dx + dy * dy) / (2 * 1.5 ** 2)))
    for dy in range(-3, 4) for dx in range(-3, 4)
    if 0 < dx * dx + dy * dy <= 10
]

VAMPIRE_CLOUD = (50, 130, 255)  # the glowing specks a startled vampire squid puffs out

SNOW_FLAKES = 28
SPARK = (40, 210, 170)
# Swimmers stir up sparks behind them: WAKE_RATE a frame for each pixel a
# frame they swim faster than WAKE_FROM, each lasting WAKE_LIFE frames
WAKE_RATE = 1.5
WAKE_FROM = 0.05
WAKE_LIFE = (10, 24)

# --- Creatures -----------------------------------------------------------------
#
# Each creature is drawn from sprite rows (facing right; "." transparent) with
# a palette of lit colors (or (lit, unlit) pairs; unlit defaults to the
# creature's dim, or DIM, of lit), and glows: characters that shine whatever
# the light. A glow marked as a lamp also lights its surroundings, like the
# anglerfish's lure, and glows its halo color around it.

CREATURES = {
    "anglerfish": dict(
        # Blue-grey, rimmed paler, all gaping jaw and needle teeth, with a
        # spiny dorsal fin, a fan tail and a long stalk arching high over its
        # head to dangle a lure out in front of its mouth
        palette={"B": (62, 82, 118), "D": (40, 54, 84), "H": (110, 145, 190), "F": (78, 100, 138),
                 "W": ((225, 235, 245), (45, 48, 55)), "M": ((14, 14, 30), (3, 3, 6)),
                 "E": ((200, 230, 250), (50, 60, 75)), "S": (120, 140, 170)},
        glows={"L": (210, 245, 255)},
        halo=(60, 160, 255),
        lamps="L",
        frames=[
            [
                ".........SSS...",
                "........S...S..",
                ".......S.....S.",
                "....F.S......S.",
                "...FF.S......L.",
                "F.DDDDS........",
                "FFDBBBBEBW.....",
                "FFBBBBBBMWMW...",
                "FFBBBBBBMMM....",
                "F.BBBBBBWMWM...",
                "..HBBBBBBBBW...",
                "...HHHHHH......",
            ],
            [
                ".........SSS...",
                "........S...S..",
                ".......S.....S.",
                "....F.S......S.",
                "F..FF.S......L.",
                "FFDDDDS........",
                "FFDBBBBEBW.....",
                ".FBBBBBBMWMW...",
                ".FBBBBBBMMM....",
                "FFBBBBBBWMWM...",
                "F.HBBBBBBBBW...",
                "...HHHHHH......",
            ],
            # Making a catch (see Angler): jaws flung wide, teeth bristling...
            [
                ".........SSS...",
                "........S...S..",
                ".......S.....S.",
                "....F.S..W...S.",
                "...FF.S.BW...L.",
                "F.DDDDSBBMW....",
                "FFDBBBBEMMM....",
                "FFBBBBBMMMMW...",
                "FFBBBBBMMMM....",
                "F.BBBBBBMMMW...",
                "..HBBBBBBMMW...",
                "...HHHHHHHW....",
            ],
            # ...then clamped shut on it, its belly full
            [
                ".........SSS...",
                "........S...S..",
                ".......S.....S.",
                "....F.S......S.",
                "...FF.S......L.",
                "F.DDDDS........",
                "FFDBBBBEBB.....",
                "FFBBBBBBBBW....",
                "FFBBBBBBWBWB...",
                "F.BBBBBBBBB....",
                "..HBBBBBBBBH...",
                "...HHHHHHHH....",
            ],
        ],
    ),
    "lanternfish": dict(
        palette={"B": (95, 110, 140), "D": (60, 72, 100), "E": ((200, 220, 230), (40, 45, 50)),
                 "T": (80, 95, 125)},
        glows={"G": (60, 210, 255)},
        frames=[
            ["..DDDD.", "TBBGBBE", "..GBGB."],
            ["T.DDDD.", ".BBGBBE", "T.GBGB."],
        ],
    ),
    "viperfish": dict(
        palette={"B": (60, 70, 90), "D": (40, 46, 62), "W": ((230, 230, 215), (90, 90, 85)),
                 "E": ((210, 220, 230), (50, 55, 60)), "R": (70, 80, 100), "T": (55, 64, 84)},
        glows={"L": (90, 170, 255), "G": (50, 120, 255)},
        frames=[
            [
                ".......L......",
                "......R.......",
                "T..DDDDDDDDD..",
                "TTBBBBBBBBBEW.",
                "T..BBBBBBBBB.W",
                "....G.G.G.GBW.",
            ],
            [
                ".......L......",
                "T.....R.......",
                ".T.DDDDDDDDD..",
                "..BBBBBBBBBEW.",
                ".T.BBBBBBBBB.W",
                "T...G.G.G.GBW.",
            ],
        ],
    ),
    "jellyfish": dict(
        # Lit only by itself: a bell rimmed in light, pulsing as it drifts
        # sideways, tentacles trailing behind and below
        palette={},
        glows={"R": (150, 120, 255), "G": (60, 50, 160), "H": (230, 120, 230), "t": (45, 36, 110)},
        frames=JELLY_FRAMES,
    ),
    "vampire-squid": dict(
        # Mantle first, flapping the two ear-like fins near its tip, a big
        # blue eye, and behind it a dark cape of webbed arms fringed with
        # pale spines, blue lights at the arm tips and fin bases. The last
        # frame is the cape pulled up over itself, spines out.
        palette={"M": (150, 35, 50), "N": (110, 24, 38), "C": (80, 16, 30), "W": (125, 30, 45),
                 "c": (205, 160, 165), "F": (175, 60, 75)},
        glows={"E": (90, 170, 255), "g": (60, 140, 255), "b": (40, 90, 200)},
        frames=[
            [
                "........FF...",
                "g......FNNN..",
                "cWc...bMMMMM.",
                "WCCCWMMMMMMMM",
                "WCCCCEMMMMMMM",
                "cWc...bMMMMM.",
                "g......FNNN..",
                "........FF...",
            ],
            [
                ".............",
                "........NNN..",
                "gWc...bMMMMMF",
                "cCCCWMMMMMMMF",
                "cCCCCEMMMMMMF",
                "gWc...bMMMMMF",
                "........NNN..",
                ".............",
            ],
            [
                "....c.c.c....",
                "...cWCCCWc...",
                "..cWCCCCCWc..",
                "..cCCCCCCCc..",
                "..cWCCCCCWc..",
                "...cWCCCWc...",
                "....c.c.c....",
                ".............",
            ],
        ],
    ),
    "spider-crab": dict(
        # A Japanese spider crab: a small spotted orange body slung between
        # very long spindly legs, knees high, picking its way sideways
        palette={"B": (215, 95, 45), "S": (245, 195, 145), "L": (195, 85, 42), "K": (225, 110, 55),
                 "E": ((20, 15, 15), (4, 3, 3))},
        frames=[
            [
                "...KK.......KK...",
                "..L..L.E.E.L..L..",
                ".L...LBSBBBL...L.",
                ".L..L.BBBSB.L..L.",
                "L...L..BBB..L...L",
                "L..L.........L..L",
            ],
            [
                "....KK.....KK....",
                "...L..LE.EL..L...",
                "..L..LBSBBBL..L..",
                ".L..L.BBBSB.L..L.",
                ".L..L..BBB..L..L.",
                "L...L.......L...L",
            ],
        ],
    ),
}

# (creature, how many, cruising speeds, how high it keeps from 0 the surface
# to 1 the floor, how likely it is to be the next to visit) swimming on their
# own. The vampire squid and the spider crab only come by now and then.
TANK = [
    ("anglerfish", 1, (0.03, 0.06), (0.3, 1.0), 1.0),
    ("lanternfish", 2, (0.18, 0.28), (0.0, 0.8), 1.0),
    ("viperfish", 1, (0.08, 0.14), (0.1, 0.8), 1.0),
    ("jellyfish", 1, (0.04, 0.07), (0.0, 0.7), 1.0),
    ("vampire-squid", 1, (0.05, 0.08), (0.15, 0.8), 0.25),
    ("spider-crab", 1, (0.05, 0.08), (1.0, 1.0), 0.25),
]

# The creatures take turns: ON_SCREEN of them (a school counts each fish) are
# in view at once, the number changing every CROWD_SECONDS. Each stays for
# VISIT_SECONDS, then swims off to make room for another.
ON_SCREEN = (1, 3)
CROWD_SECONDS = (20.0, 45.0)
VISIT_SECONDS = (25.0, 60.0)
ENTRY_SECONDS = (3.0, 8.0)  # at least this long between arrivals

# --- Events --------------------------------------------------------------------

SQUID_HEAD = 23  # mantle tip to the base of the arms
SQUID_PROFILE = (  # the body's half-height, by pixels back from the mantle tip
    [0, 1, 2, 4, 6, 7, 9, 13, 17, 19, 20, 22, 23],
    [0.6, 2.0, 3.6, 3.8, 2.6, 2.4, 2.7, 3.0, 3.0, 2.6, 2.8, 2.8, 2.2],
)
SQUID_FINS = 6  # the fins, at the mantle tip, end this far back
SQUID_ARMS = 14
SQUID_TENTACLES = 25
SQUID_BACK = (120, 30, 26)
SQUID_BELLY = (175, 70, 55)
SQUID_FIN = (150, 50, 42)
SQUID_ARM = (130, 38, 32)
SQUID_EYE = (12, 8, 14)
SQUID_GLINT = (235, 235, 215)
SQUID_KICK = 0.22  # speed gained by each jet of the mantle
SQUID_DRAG = 0.985

# The day tank's old submarine, darker down here with so little light to
# catch, but for its glowing portholes and searchlight
SUB_NIGHT_PALETTE = {ch: c if ch in "PL" else scale(c, 0.38) for ch, c in SUB_PALETTE.items()}
SUB_SPEED = 0.12
SUB_BEAM_LENGTH = 40

SIPHON_SEGMENTS = 40
SIPHON_GAP = 2.2  # pixels between segments
SIPHON_SPEED = 0.13
SIPHON_FLOAT = (255, 120, 60)
SIPHON_BELL = (70, 100, 150)
SIPHON_STEM = (22, 28, 50)

COMB_SPEED = (0.06, 0.09)

STARTLE_SPEED = 0.45  # how fast the wave of sparks spreads, pixels a frame

# The anglerfish's catch: a bristlemouth, a dark little shape behind a row
# of blue lights until the lure's glow turns it silver, edging in to look
PREY = dict(
    palette={"B": ((185, 200, 215), (24, 28, 38)), "D": ((125, 140, 160), (18, 21, 30)),
             "E": ((245, 250, 255), (40, 44, 52)), "T": ((135, 150, 170), (18, 21, 30))},
    glows={"g": (35, 105, 230)},
    frames=[
        ["..DDD.", "TBBBBE", "..ggg."],
        ["T.DDD.", ".BBBBE", "T.ggg."],
    ],
)
PREY_SPEED = 0.1
CATCH_SECONDS = 60.0  # the most it waits for its meal
CATCH_ROOM = 18  # the open water it wants ahead of its lure
CATCH_MOUTH = (10, 7)  # where it swallows, in the anglerfish's sprite coordinates
CATCH_GAPE = 0.24  # seconds its jaws hang open before it lunges
CATCH_LUNGE = 0.55  # how fast it lunges, pixels a frame
CATCH_SUCK = 0.3  # how much of the way to the mouth the prey is drawn each frame
CATCH_BURST = (60, 140, 255)  # the prey's lights, scattered as it's snapped up

# Something crossing the panel (the dragonfish, the gulper) asks the
# creatures in view to keep clear of it and this much of the water ahead
CLEAR_AHEAD = 10

# A black dragonfish: long, slender and black, fangs bared, a barbel dangling
# from its chin to a glowing tip, and a red photophore under its eye
DRAGON = dict(
    # A sheen along its back ("D") to show its outline, its dorsal and anal
    # fins far back by the tail, a row of lights along its belly, and its
    # jaws open on long fangs
    palette={"B": ((80, 66, 96), (36, 30, 46)), "D": ((110, 94, 128), (54, 46, 68)),
             "F": ((96, 78, 110), (46, 38, 58)), "E": ((190, 200, 210), (70, 72, 80)),
             "W": ((240, 235, 220), (110, 106, 98)), "M": ((14, 6, 12), (6, 2, 6)),
             "b": ((120, 104, 130), (52, 44, 62))},
    glows={"R": (255, 35, 25), "P": (60, 130, 255), "g": (75, 55, 185), "L": (210, 235, 255)},
    frames=[
        [
            "...F..............DDD....",
            "T.FFDDDDDDDDDDDDDDBPBEDD.",
            "TTBBBBBBBBBBBBBBBBBBRBMMW",
            "T.FF..g..g..g..g..bBBWMW.",
            "...F...............b.....",
            "...................b.....",
            "..................b......",
            "..................L......",
        ],
        [
            "..F...............DDD....",
            "TFF.DDDDDDDDDDDDDDBPBEDD.",
            ".TBBBBBBBBBBBBBBBBBBRBMMW",
            "TF.F..g..g..g..g..bBBWMW.",
            "..F................b.....",
            "....................b....",
            "...................b.....",
            "...................L.....",
        ],
    ],
)
DRAGON_SPEED = 0.1

# A gulper eel: a long whip of a body behind a head that's nearly all mouth,
# a light glowing at its tail tip
GULPER_LENGTH = 40  # the body behind the head, in pixels
GULPER_JAW = 8  # the head, snout to the hinge of its jaws
GULPER_POUCH = 9.0  # how far the lower jaw drops, fully open
GULPER_BODY = (105, 78, 112)  # lit; unlit it shows at a little under half this
GULPER_RIM = (190, 160, 180)  # its jaws' edges
GULPER_MOUTH = (60, 24, 54)  # inside the pouch
GULPER_STOP = (3, 8)  # where it stops to feed: its tail tip this far inside the glass
GULPER_TAIL = (255, 70, 140)
GULPER_SPEED = 0.16
GULPER_SPECK = (110, 200, 255)  # the copepods it gulps


class Glower:
    """A creature's sprite frames, each facing, as (dx, dy, unlit, lit) body
    pixels plus (dx, dy, color) glows and (dx, dy) lamps."""

    def __init__(self, palette: dict, frames: list[list[str]], glows: dict | None = None, lamps: str = "",
                 dim: float = DIM) -> None:
        glows = glows or {}
        colors = {
            ch: (c[1], c[0]) if isinstance(c[0], tuple) else (scale(c, dim), c)
            for ch, c in palette.items()
        }
        # Tails ("T") and fins share a color unless the palette says otherwise
        if colors:
            colors.setdefault("T", colors.get("D") or next(iter(colors.values())))
        self.body, self.glows, self.lamps = [], [], []
        for rows_of in (lambda f: f, flip):
            body, glow, lamp = [], [], []
            for f in frames:
                rows = rows_of(f)
                body.append([(dx, dy, *colors[ch]) for dy, row in enumerate(rows)
                             for dx, ch in enumerate(row) if ch in colors])
                glow.append([(dx, dy, glows[ch]) for dy, row in enumerate(rows)
                             for dx, ch in enumerate(row) if ch in glows])
                lamp.append([(dx, dy) for dy, row in enumerate(rows)
                             for dx, ch in enumerate(row) if ch in lamps])
            self.body.append(body)
            self.glows.append(glow)
            self.lamps.append(lamp)
        self.w = len(frames[0][0])
        self.h = len(frames[0])
        self.frames = len(frames)

    at = Sprite.at

    def draw(self, fb: bytearray, width: int, height: int, x: int, y: int, facing: int, frame: int,
             light: list | None = None, glow: float = 1.0) -> None:
        side = facing < 0
        for dx, dy, unlit, lit in self.body[side][frame]:
            px, py = x + dx, y + dy
            if 0 <= px < width and 0 <= py < height:
                level = light[py][px] if light else 0.0
                p = (py * width + px) * 3
                if level <= 0.02:
                    fb[p:p + 3] = bytes(unlit)
                else:
                    level = min(level, 1.0)
                    fb[p] = int(unlit[0] + (lit[0] - unlit[0]) * level)
                    fb[p + 1] = int(unlit[1] + (lit[1] - unlit[1]) * level)
                    fb[p + 2] = int(unlit[2] + (lit[2] - unlit[2]) * level)
        if glow > 0.02:
            for dx, dy, color in self.glows[side][frame]:
                put_max(fb, width, height, x + dx, y + dy,
                        (min(255, int(color[0] * glow)), min(255, int(color[1] * glow)),
                         min(255, int(color[2] * glow))))


class Critter(Fish):
    """A deep-sea fish: a Fish whose lights can be turned down, flashed, and
    (for a lure) pulse, and which visits the panel for a while at a time
    (see NightTank.cast)."""

    def __init__(self, kind: str, *args, **kwargs) -> None:
        spec = CREATURES[kind]
        sprite = Glower(spec["palette"], spec["frames"], spec.get("glows"), spec.get("lamps", ""),
                        spec.get("dim", DIM))
        super().__init__(sprite, *args, **kwargs)
        self.kind = kind
        self.glow = self.glow_to = 1.0
        self.flash = 0.0  # extra brightness from a startle, fading
        self.pulse = 0.35 if spec.get("lamps") else 0.0
        self.pulse_phase = random.uniform(0, 2 * math.pi)
        self.away = False  # out of view, waiting its turn
        self.leaving = False  # swimming off the panel
        self.visit = 0  # frames left before it leaves

    def pick_turn(self) -> None:
        # Its visits decide when it leaves the panel, so it turns back at the
        # glass rather than wandering out of view
        super().pick_turn()
        glass = self.glass(self.facing)
        if (self.turn_x - glass) * self.facing > 0:
            self.turn_x = glass

    def wander(self) -> None:
        if not self.leaving:
            super().wander()
            return
        # Off the panel at an easy pace, then out of the picture
        self.glide()
        self.speed += (self.cruise * self.pace - self.speed) * 0.05
        if (self.turn_x - self.x) * self.facing <= 0:
            self.leaving = False
            self.away = True

    def swim(self) -> None:
        super().swim()
        self.glow += (self.glow_to - self.glow) * 0.04
        self.flash *= 0.94

    def level(self, t: float) -> float:
        """How bright its lights are now."""
        pulse = 1 - self.pulse * (0.5 + 0.5 * math.sin(t * 2.1 + self.pulse_phase))
        return self.glow * pulse * (1 + 1.5 * self.flash)

    def frame(self) -> int:
        """Which sprite frame to draw."""
        return int(self.tail)

    def draw_extra(self, fb: bytearray, width: int, height: int) -> None:
        """Anything it draws besides its sprite."""

    def lamps(self, t: float) -> list[tuple[int, int]]:
        x, y = self.pose(t)
        return [(x + dx, y + dy) for dx, dy in self.sprite.lamps[self.facing < 0][self.frame()]]


class VampireSquid(Critter):
    """A vampire squid flaps slowly along. Now and then, and whenever the
    giant squid comes by, it pulls its cape up over itself into a spiky ball
    and hangs there a while before swimming on. Startled, it first puffs out
    a cloud of glowing blue specks to slip away behind."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beats = 2  # swimming; the last frame is the cloak
        self.cloak = 0  # frames left cloaked
        self.hid = False  # already cloaked for the dimming the giant squid brings
        self.cloud: list[Spark] = []

    def swim(self) -> None:
        startled = self.flash > 0.8
        dimmed = self.glow_to < 0.5
        if not dimmed:
            self.hid = False
        if not self.cloak and not self.fleeing and (
                startled or (dimmed and not self.hid) or random.random() < 1 / frames_for(25)):
            self.cloak = frames_for(random.uniform(3, 6))
            self.hid = dimmed
            if startled:
                self.puff()
        for s in self.cloud:
            s.age += 1
            s.x += s.vx
            s.y += s.vy
            s.vx *= 0.96
            s.vy = s.vy * 0.96 - 0.002
        self.cloud = [s for s in self.cloud if s.age < s.life]
        if self.cloak:
            # Drifting to a stop
            self.cloak -= 1
            self.speed *= 0.95
            self.vy *= 0.9
            self.x += self.speed * self.facing
            self.y += self.vy
            self.glow += (self.glow_to - self.glow) * 0.04
            self.flash *= 0.94
            return
        super().swim()

    def puff(self) -> None:
        """A cloud of glowing specks from its arm tips."""
        cx = self.x + (2 if self.facing > 0 else self.w - 3)
        cy = self.y + self.h / 2
        for _ in range(30):
            s = Spark()
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.06, 0.3)
            s.x, s.y = cx, cy
            s.vx, s.vy = math.cos(angle) * speed - 0.08 * self.facing, math.sin(angle) * speed * 0.7
            s.age, s.life = 0, frames_for(random.uniform(2.0, 4.5))
            s.color = mix(VAMPIRE_CLOUD, (200, 240, 255), random.random() * 0.4)
            self.cloud.append(s)

    def frame(self) -> int:
        return 2 if self.cloak else int(self.tail)

    def draw_extra(self, fb: bytearray, width: int, height: int) -> None:
        for s in self.cloud:
            fade = 1 - s.age / s.life
            if random.random() < 0.85:  # twinkling
                put_max(fb, width, height, round(s.x), round(s.y), scale(s.color, fade * min(self.glow, 1.0)))


class Crab(Critter):
    """A crab walking sideways along the floor, its legs moving only while
    it walks, stopping now and then."""

    ground: list[int]  # the floor's top row in each column, set by the tank

    def swim(self) -> None:
        tail = self.tail
        super().swim()
        self.tail = (tail + self.speed * 1.2) % self.beats
        # Standing on the floor beneath it
        width = len(self.ground)
        cols = [min(max(c, 0), width - 1) for c in range(int(self.x), int(self.x) + self.w)]
        self.y = self.target_y = min(self.ground[c] for c in cols) - self.h
        self.vy = 0.0

    def pose(self, t: float) -> tuple[int, int]:
        return round(self.x), round(self.y)


class Angler(Critter):
    """The anglerfish. Its tail beats through the first two frames; the
    other two are for the catch event, which sets jaw to one of them while
    it strikes."""

    GAPE, SHUT = 2, 3

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.beats = 2
        self.tail %= self.beats
        self.jaw: int | None = None

    def frame(self) -> int:
        return int(self.tail) if self.jaw is None else self.jaw


class Flake:
    __slots__ = ("x", "y", "speed", "drift", "bright")


class Puff:
    __slots__ = ("x", "y", "vx", "vy", "age", "life", "heat")


class Spark:
    __slots__ = ("x", "y", "vx", "vy", "age", "life", "color")


class Worm:
    __slots__ = ("x", "base", "length", "phase", "out", "hide")


def smoke_color(age: float) -> tuple:
    """The smoke's color at age (0 leaving the vent, 1 gone), from SMOKE."""
    for (a0, c0), (a1, c1) in zip(SMOKE, SMOKE[1:]):
        if age <= a1:
            return mix(c0, c1, (age - a0) / (a1 - a0))
    return SMOKE[-1][1]


def glow_kernel(radius: float) -> np.ndarray:
    r = int(math.ceil(radius * 2.5))
    d = np.arange(-r, r + 1, dtype=np.float32)
    return np.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * radius * radius))


def add_light(light: np.ndarray, x: float, y: float, kernel: np.ndarray, strength: float) -> None:
    """Add a soft round light centered on pixel (x, y)."""
    h, w = light.shape
    r = kernel.shape[0] // 2
    x, y = int(x), int(y)
    x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, w), max(y - r, 0), min(y + r + 1, h)
    if x0 < x1 and y0 < y1:
        light[y0:y1, x0:x1] += kernel[y0 - y + r:y1 - y + r, x0 - x + r:x1 - x + r] * strength


# --- Events --------------------------------------------------------------------


class Squid(Event):
    """A giant squid jets past, mantle first, its arms trailing: each pulse of
    the mantle drives it on, and it glides, arms spreading, until the next.
    The creatures put out their lights and keep still until it's gone."""

    depth = 0.4

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        self.tip = -1.0 if self.facing > 0 else tank.width + 1.0
        self.y0 = random.uniform(9, tank.floor - 9)
        self.speed = 0.15
        self.period = frames_for(2.6)
        self.clock = 0
        self.frame_count = 0
        self.wave = random.uniform(0, 2 * math.pi)
        for fish in tank.fish:
            fish.glow_to = 0.08
            fish.pace = 0.5

    def end(self) -> None:
        for fish in self.tank.fish:
            fish.glow_to = 1.0
            fish.pace = 1.0

    @property
    def squeeze(self) -> float:
        """1 as the mantle contracts for a jet, easing back to 0."""
        return max(0.0, 1 - self.clock / (self.period * 0.35))

    @property
    def y(self) -> float:
        return self.y0 + 1.5 * math.sin(self.frame_count * 0.012 + self.wave)

    def step(self) -> bool:
        self.frame_count += 1
        self.clock += 1
        if self.clock >= self.period:
            self.clock = 0
            self.speed += SQUID_KICK
            # The jet blasts out of the funnel, under its head, in a spray of
            # sparks streaming back among its arms, and churns up more all
            # down its arms and tentacles
            funnel = self.tip - (SQUID_HEAD + 1) * self.facing
            self.tank.burst(funnel, self.y + 1, 16, 0.6, heading=0.0 if self.facing < 0 else math.pi, spread=0.3)
            for _ in range(8):
                back = random.uniform(SQUID_HEAD, SQUID_HEAD + SQUID_TENTACLES + 3)
                self.tank.stir(self.tip - back * self.facing, self.y + random.uniform(-5, 5), 1)
        self.speed *= SQUID_DRAG
        self.tip += self.speed * self.facing
        # And as it goes it stirs up more around its body, the faster the more
        back = random.uniform(2, SQUID_HEAD + SQUID_TENTACLES)
        edge = random.uniform(4, 6) if back < SQUID_HEAD else random.uniform(0, 5)
        self.tank.stir(self.tip - back * self.facing, self.y + random.choice((-1, 1)) * edge, self.speed * 4)
        tail = self.tip - (SQUID_HEAD + SQUID_TENTACLES) * self.facing
        return (tail <= self.tank.width) if self.facing > 0 else (tail >= 0)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        back = self.tip - (SQUID_HEAD + SQUID_ARMS) * self.facing
        left, right = sorted((self.tip, back))
        return [(left, self.y - 5, right, self.y + 5)]

    def draw(self, fb: bytearray, t: float) -> None:
        tank = self.tank
        width, height = tank.width, tank.height
        light = tank.light
        cy = self.y
        squeeze = self.squeeze

        # The mantle, fins and head, column by column
        u = (self.tip - (np.arange(width) + 0.5)) * self.facing
        cols = np.flatnonzero((u >= 0) & (u <= SQUID_HEAD))
        if len(cols):
            uc = u[cols]
            half = np.interp(uc, *SQUID_PROFILE)
            half = half * np.where((uc > SQUID_FINS) & (uc < 18), 1 - 0.2 * squeeze, 1.0)
            top, bottom = cy - half, cy + half
            y = np.arange(height, dtype=np.float32)[:, None] + 0.5
            cover = np.clip(np.minimum(y - top, bottom - y) + 0.5, 0, 1)
            v = np.clip((y - top) / np.maximum(bottom - top, 0.1), 0, 1)[..., None]
            color = np.array(SQUID_BACK, np.float32) * (1 - v) + np.array(SQUID_BELLY, np.float32) * v
            fin = (uc < SQUID_FINS)[None, :, None]
            color = np.where(fin, np.array(SQUID_FIN, np.float32), color)
            # Unlit it's a dark red shape; light brings out its color
            lit = light[:, cols, None] if light is not None else 0.0
            color = color * (0.45 + 0.55 * np.minimum(lit, 1.0))
            view = np.frombuffer(fb, np.uint8).reshape(height, width, 3)
            water = view[:, cols].astype(np.float32)
            water += (color - water) * cover[..., None]
            view[:, cols] = water.astype(np.uint8)

        def body_x(back: float) -> int:
            return math.floor(self.tip - back * self.facing)

        # The arms trail behind, drawn together while it jets, spreading as
        # it glides; the two long tentacles wave further back
        spread = 0.35 + 0.65 * (1 - squeeze)
        phase = self.frame_count * 0.09 + self.wave
        for k, base in enumerate((-1.6, -0.8, 0.0, 0.8, 1.6, -0.4, 0.4)):
            for i in range(SQUID_ARMS):
                along = i / SQUID_ARMS
                ay = cy + base * (1 + along * 1.6 * spread) + math.sin(phase - i * 0.5 + k) * along * 1.2
                put(fb, width, height, body_x(SQUID_HEAD + i), math.floor(ay),
                    scale(SQUID_ARM, 0.5 * (1 - 0.5 * along)))
        for k, base in enumerate((-0.6, 0.6)):
            for i in range(SQUID_TENTACLES):
                along = i / SQUID_TENTACLES
                ay = cy + base * (1 + along * 3 * spread) + math.sin(phase * 0.8 - i * 0.35 + k * 2) * along * 2.2
                color = scale(SQUID_ARM, 0.55 * (1 - 0.4 * along))
                put(fb, width, height, body_x(SQUID_HEAD + i), math.floor(ay), color)
                if i >= SQUID_TENTACLES - 4:  # the clubs at the ends
                    put(fb, width, height, body_x(SQUID_HEAD + i), math.floor(ay) + (1 if base > 0 else -1), color)

        # Its huge eye, catching what light there is
        ex, ey = body_x(20), math.floor(cy - 1)
        for dx in (0, -self.facing):
            put(fb, width, height, ex + dx, ey, SQUID_EYE)
            put(fb, width, height, ex + dx, ey + 1, SQUID_EYE)
        put(fb, width, height, ex, ey, SQUID_GLINT)


class Submersible(Event):
    """The old submarine from the day tank putters past, its searchlight
    sweeping a cone of light through the dark: inside it the water turns
    blue, the snow sparkles, the floor and the creatures show their colors,
    and fish turn away."""

    layer = BACK

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.sprite = Sprite(SUB_NIGHT_PALETTE, submarine_frames())
        self.prop = 0.0
        self.facing = random.choice((1, -1))
        lamp = self.sprite.at(0, self.facing, SUB_LAMP)
        self.x = (-SUB_BEAM_LENGTH if self.facing > 0 else tank.width + SUB_BEAM_LENGTH) - lamp
        self.y0 = random.uniform(2, tank.floor * 0.3)
        self.frame_count = 0
        self.beam = np.zeros((tank.height, tank.width), np.float32)

    @property
    def y(self) -> float:
        return self.y0 + 0.7 * math.sin(self.frame_count * FRAME_SECONDS * 0.5)

    def step(self) -> bool:
        self.frame_count += 1
        self.x += SUB_SPEED * self.facing
        self.prop = (self.prop + 0.3) % len(SUB_PROPS)
        tank = self.tank
        width, height = tank.width, tank.height
        # The cone of light from the lamp, angled down toward the floor and
        # sweeping slowly, fading with distance
        lx = self.sprite.at(self.x, self.facing, SUB_LAMP) + 0.5
        ly = round(self.y) + SUB_LAMP[1] + 0.5
        ahead = (np.arange(width, dtype=np.float32) + 0.5 - lx) * self.facing
        tilt = 0.42 + 0.15 * math.sin(self.frame_count * FRAME_SECONDS * 0.45)
        off = np.arange(height, dtype=np.float32)[:, None] + 0.5 - ly - ahead * tilt
        spread = ahead * 0.32 + 0.8
        self.beam = (np.clip(1.15 - np.abs(off) / spread, 0, 1)
                     * np.clip(1 - ahead / SUB_BEAM_LENGTH, 0, 1) ** 1.2 * (ahead > 0)).astype(np.float32)
        # Fish caught in the light, facing into it, turn away
        if self.frame_count % 4 == 0:
            for fish in self.free_fish():
                cx, cy = int(fish.x + fish.w / 2), int(fish.y + fish.h / 2)
                if (0 <= cx < width and 0 <= cy < height and self.beam[cy, cx] > 0.3
                        and fish.facing != self.facing):
                    fish.turn()
        return (self.x + self.sprite.w >= 0) if self.facing < 0 else (self.x <= width)

    def light(self) -> np.ndarray:
        return self.beam

    def draw(self, fb: bytearray, t: float) -> None:
        self.sprite.draw(fb, self.tank.width, self.tank.height, round(self.x), round(self.y),
                         self.facing, int(self.prop))


class Siphonophore(Event):
    """A siphonophore drifts through: a colony strung out longer than the
    tank, a little float leading swimming bells and a long chain of glowing
    beads trailing tentacles, with flashes rippling down its length."""

    layer = BACK

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        self.head = -1.0 if self.facing > 0 else tank.width + 1.0
        self.y0 = random.uniform(7, tank.floor - 10)
        self.phase = random.uniform(0, 2 * math.pi)
        self.hue = random.uniform(0.55, 0.62)
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        self.head += SIPHON_SPEED * self.facing
        tail = self.head - SIPHON_SEGMENTS * SIPHON_GAP * self.facing
        return (tail <= self.tank.width + 2) if self.facing > 0 else (tail >= -2)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        clock = self.frame_count * FRAME_SECONDS
        for i in range(SIPHON_SEGMENTS):
            x = self.head - i * SIPHON_GAP * self.facing
            if not -3 < x < width + 3:
                continue
            y = self.y0 + 3 * math.sin(x * 0.07 + self.phase) + 0.8 * math.sin(clock * 0.5 - i * 0.25)
            px, py = math.floor(x), math.floor(y)
            if i:
                # The stem back to the segment ahead
                ahead = self.head - (i - 0.5) * SIPHON_GAP * self.facing
                ay = self.y0 + 3 * math.sin(ahead * 0.07 + self.phase) + 0.8 * math.sin(clock * 0.5 - i * 0.25 + 0.125)
                put_max(fb, width, height, math.floor(ahead), math.floor(ay), SIPHON_STEM)
            if i == 0:
                put_max(fb, width, height, px, py, SIPHON_FLOAT)
                put_max(fb, width, height, px - self.facing, py, scale(SIPHON_FLOAT, 0.6))
            elif i < 6:
                # The swimming bells, clear, rimmed with light
                put_max(fb, width, height, px, py, SIPHON_BELL)
                put_max(fb, width, height, px, py + 1, scale(SIPHON_BELL, 0.6))
            else:
                flash = (0.5 + 0.5 * math.sin(clock * 3 - i * 0.4)) ** 3
                color = hsv(self.hue + 0.1 * (i / SIPHON_SEGMENTS), 0.75, 0.4 + 0.6 * flash)
                put_max(fb, width, height, px, py, color)
                if i % 3 == 0:
                    # A tentacle hanging down, its tip catching the flash
                    length = 2 + (i * 7) % 3
                    sway = math.sin(clock * 0.8 + i)
                    for k in range(1, length + 1):
                        put_max(fb, width, height, px + round(sway * k / length), py + k,
                                scale(color, 0.25 if k < length else 0.6))


class CombJelly(Event):
    """A comb jelly drifts by, rows of beating combs rippling with rainbow
    light down its clear body, two fine tentacles trailing."""

    layer = BACK

    class Jelly:
        __slots__ = ("x", "y", "drift", "phase", "size")

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.facing = random.choice((1, -1))
        self.jellies = []
        drifts = [random.uniform(*COMB_SPEED)]
        for i, drift in enumerate(drifts):
            j = CombJelly.Jelly()
            j.x = (-5 - i * random.uniform(18, 26)) if self.facing > 0 else tank.width + 5 + i * random.uniform(18, 26)
            j.y = random.uniform(7, tank.floor - 10)
            j.drift = drift * self.facing
            j.phase = random.uniform(0, 2 * math.pi)
            j.size = random.choice((4.5, 5.5))
            self.jellies.append(j)
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        for j in self.jellies:
            j.x += j.drift
        return any(-8 < j.x < self.tank.width + 8 or (j.x < 0) == (self.facing > 0) for j in self.jellies)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        return [(j.x - 4, j.y - j.size - 1, j.x + 4, j.y + j.size + 1) for j in self.jellies]

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        clock = self.frame_count * FRAME_SECONDS
        for j in self.jellies:
            cy = j.y + 1.2 * math.sin(clock * 0.4 + j.phase)
            b = j.size
            a = b * 0.55
            rows = range(math.floor(-b), math.ceil(b) + 1)
            # A faint clear body
            for dy in rows:
                edge = a * math.sqrt(max(0.0, 1 - (dy / b) ** 2))
                for side in (-1, 1):
                    put_max(fb, width, height, round(j.x + side * edge), math.floor(cy + dy), (14, 22, 32))
            # Four comb rows, light rippling down them in rainbow colors
            for m, f in enumerate((-0.7, -0.22, 0.22, 0.7)):
                for dy in rows:
                    if abs(dy) > b - 0.5:
                        continue
                    edge = a * math.sqrt(max(0.0, 1 - (dy / b) ** 2))
                    beat = max(0.0, math.sin(clock * 5 - dy * 0.9 + m * 0.8 + j.phase)) ** 2
                    color = hsv(clock * 0.15 + dy * 0.08 + m * 0.12, 0.85, 0.25 + 0.75 * beat)
                    put_max(fb, width, height, round(j.x + f * edge), math.floor(cy + dy), color)
            # Two tentacles trailing behind it
            for side in (-1, 1):
                for k in range(1, 9):
                    tx = j.x - self.facing * k * 0.7 + side * 1.5
                    ty = cy + b + k * 0.5 + math.sin(clock * 1.3 - k * 0.6 + side) * k * 0.15
                    put_max(fb, width, height, round(tx), math.floor(ty), (40, 22, 34) if k % 3 else (90, 50, 80))


class Startle(Event):
    """Something unseen brushes past: a ring of blue-green sparks spreads
    through the water from where it touched, the creatures flash and dart,
    and the tube worms snatch in their plumes."""

    layer = FRONT

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.cx = random.uniform(tank.width * 0.2, tank.width * 0.8)
        self.cy = random.uniform(6, tank.floor - 6)
        self.radius = 0.0
        self.sparks: list[Spark] = []
        self.reached: set[int] = set()
        self.done = math.hypot(tank.width, tank.height)

    def step(self) -> bool:
        tank = self.tank
        self.radius += STARTLE_SPEED
        r = self.radius
        if r < self.done:
            # New sparks along the front of the ring, as many as it's long
            for _ in range(min(8, 1 + int(r * 0.25))):
                a = random.uniform(0, 2 * math.pi)
                s = Spark()
                s.x = self.cx + math.cos(a) * (r + random.uniform(-1, 1))
                s.y = self.cy + math.sin(a) * (r + random.uniform(-1, 1)) * 0.8
                s.age, s.life = 0, random.randint(10, 22)
                s.color = mix(SPARK, (150, 255, 255), random.random() * 0.5)
                self.sparks.append(s)
        # The ring reaches each creature and worm in turn
        for fish in tank.fish:
            if id(fish) not in self.reached and math.hypot(fish.x + fish.w / 2 - self.cx, (fish.y - self.cy) / 0.8) < r:
                self.reached.add(id(fish))
                fish.flash = 1.0
                if not fish.leader:
                    fish.speed = fish.cruise * 3
        for w in tank.worms:
            if id(w) not in self.reached and math.hypot(w.x - self.cx, (w.base - self.cy) / 0.8) < r:
                self.reached.add(id(w))
                w.hide = frames_for(random.uniform(2.5, 5))
        for s in self.sparks:
            s.age += 1
        self.sparks = [s for s in self.sparks if s.age < s.life]
        return r < self.done or bool(self.sparks)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        for s in self.sparks:
            put_max(fb, width, height, round(s.x), round(s.y), scale(s.color, 1 - s.age / s.life))


class Eruption(Event):
    """The vent wakes: its veins brighten and its glow grows as the heat
    rises, then it blows, pouring out smoke, flinging glowing embers that
    arc up and fall back, and lighting up the water around it, until it
    dies back down to a smolder. The tube worms snatch in their plumes, and
    the creatures bolt out of view, coming back one at a time as it cools."""

    layer = FRONT

    class Ember:
        __slots__ = ("x", "y", "vx", "vy", "age", "life")

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.rumble = frames_for(ERUPTION_RUMBLE)
        self.erupting = frames_for(random.uniform(*ERUPTION_SECONDS))
        self.embers: list[Eruption.Ember] = []
        tank.activity_to = 0.35

    def end(self) -> None:
        self.tank.activity_to = 0.0
        self.calm()

    def calm(self) -> None:
        for fish in self.free_fish():
            fish.calm(frames_for(random.uniform(3.0, 9.0)))

    def step(self) -> bool:
        tank = self.tank
        if self.rumble:
            self.rumble -= 1
            if not self.rumble:
                # It blows
                tank.activity_to = 1.0
                for w in tank.worms:
                    w.hide = self.erupting + frames_for(random.uniform(1, 4))
                for fish in self.free_fish():
                    if tank.in_view(fish) and not fish.leaving:
                        fish.flee(tank.mouths[0][0])
        elif self.erupting:
            self.erupting -= 1
            if not self.erupting:
                tank.activity_to = 0.0
                self.calm()
            # Embers flung from the mouths, the main one most
            for k, (mx, my) in enumerate(tank.mouths):
                rate = tank.activity * (1.2 if k == 0 else 0.3)
                while rate > 0:
                    if random.random() < rate:
                        e = Eruption.Ember()
                        e.x, e.y = mx + 0.5, my - 0.5
                        e.vx, e.vy = random.uniform(-0.22, 0.22), -random.uniform(0.3, 0.7)
                        e.age, e.life = 0, frames_for(random.uniform(1.2, 2.8))
                        self.embers.append(e)
                    rate -= 1
        embers = []
        for e in self.embers:
            e.age += 1
            e.vy += EMBER_GRAVITY
            e.x += e.vx
            e.y += e.vy
            col = int(e.x)
            if e.age < e.life and 0 <= col < tank.width and e.y < tank.floor_top[col]:
                embers.append(e)
        self.embers = embers
        # Until it's died back down
        return bool(self.rumble or self.erupting or self.embers or tank.activity > 0.08)

    def draw(self, fb: bytearray, t: float) -> None:
        width, height = self.tank.width, self.tank.height
        for e in self.embers:
            age = e.age / e.life
            color = mix(EMBER_HOT, EMBER_COOL, min(1.0, age * 1.5))
            put_max(fb, width, height, math.floor(e.x), math.floor(e.y), scale(color, 1 - age * age))


class Catch(Event):
    """The anglerfish hangs still, its lure twitching, and something small
    comes to look: a bristlemouth, a row of blue lights in the dark until
    the lure's glow turns it silver, edging closer, stopping, edging closer.
    When it's near enough the jaws fly open, the anglerfish lunges, the rush
    of water sweeps the bristlemouth in, the jaws snap shut on it in a spray
    of its scattered light, and the lure goes dim a moment before it glows
    again. The other creatures keep out of the way."""

    depth = 0.5  # just behind the anglerfish, set once it's found

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.angler = next(f for f in tank.fish if isinstance(f, Angler))
        self.kept = self.angler  # it stays in view till it's done (see NightTank.cast)
        self.held: list[Critter] = []  # once it settles, it keeps its place and the others keep clear
        self.depth = self.angler.depth + 0.001
        self.state = "come"  # come, settle, lure, gape, strike, gulp, rest
        self.clock = 0  # frames in this state
        self.frame_count = 0
        self.prey = None
        self.prey_sprite = Glower(PREY["palette"], PREY["frames"], PREY["glows"])

    def lure(self) -> tuple[float, float]:
        a = self.angler
        x, y = a.pose(self.frame_count * FRAME_SECONDS)
        (lx, ly), = a.sprite.lamps[a.facing < 0][0]
        return x + lx, y + ly

    def room(self, facing: int) -> float:
        """The open water ahead of the lure, facing that way: up to the
        glass, or the vent if that's nearer."""
        tank, a = self.tank, self.angler
        lx, _ = self.lure()
        if facing != a.facing:
            lx = 2 * (a.x + a.w / 2) - lx  # where the lure would be, turned
        vent = (tank.vent_x, tank.vent_x + len(tank.vent_rows[0]))
        if facing > 0:
            return max(0.0, vent[0] - lx) if vent[1] >= lx else tank.width - lx
        return max(0.0, lx - vent[1]) if vent[0] <= lx else lx

    def hold(self) -> None:
        """Keep the anglerfish where it is, facing the way it faces."""
        a = self.angler
        a.pace = 0.0
        a.pause = 0
        a.turn_x = a.x + 40 * a.facing
        a.target_y = a.y

    def step(self) -> bool:
        tank, a = self.tank, self.angler
        self.frame_count += 1
        self.clock += 1
        a.visit = max(a.visit, frames_for(15))
        a.leaving = False
        if self.frame_count > frames_for(CATCH_SECONDS):
            return False
        width = tank.width

        if self.state == "come":
            if a.away:
                # Make room for it if the panel's full, then bring it in
                present = [u for u in tank.units() if not u.away]
                if sum(tank.size[id(u)] for u in present) >= ON_SCREEN[1]:
                    staying = [u for u in present if not u.leaving]
                    if staying:
                        tank.leave(min(staying, key=lambda u: u.visit))
                    return True
                tank.arrive(a)
            # Hurrying, it swims on into view, until there's open
            # water ahead of it (or behind it, and it turns) for its meal to
            # come through; there it hangs still
            a.pace = 4.0
            a.pause = 0
            if (a.x + a.w > width - 1) if a.facing > 0 else (a.x < 1):
                a.turn()  # it was swimming out of view: back across
            a.turn_x = a.x + 40 * a.facing
            if 1 <= a.x and a.x + a.w <= width - 1:
                if self.room(a.facing) >= CATCH_ROOM:
                    self.state, self.clock = "settle", 0
                elif self.room(-a.facing) >= CATCH_ROOM:
                    a.turn()
                    self.state, self.clock = "settle", 0
                if self.state == "settle":
                    self.held = [a]
        elif self.state == "settle":
            self.hold()
            if a.speed < 0.01 and self.clock > frames_for(1.5):
                lx, ly = self.lure()
                p = self.prey = Fish(self.prey_sprite, width, 1, tank.floor - 1, 0.0, (PREY_SPEED, PREY_SPEED))
                p.facing = -a.facing
                # Out in the dark ahead, where only its faint lights show
                ahead = lx + random.uniform(CATCH_ROOM - 4, CATCH_ROOM + 6) * a.facing
                p.x = min(ahead, width + 1.0) if a.facing > 0 else max(ahead - p.w, -p.w - 1.0)
                p.y = min(max(ly - 1 + random.uniform(-4, 4), 1.0), tank.floor - 4.0)
                p.speed = PREY_SPEED
                p.wait = 0  # frames it hangs back, looking
                self.state, self.clock = "lure", 0
        elif self.state == "lure":
            self.hold()
            p = self.prey
            lx, ly = self.lure()
            head = p.x + (p.w if p.facing > 0 else -1)
            dx, dy = lx - head, ly - (p.y + 1)
            # The lure twitches now and then, more eagerly as its meal nears
            if random.random() < (0.03 if abs(dx) > 8 else 0.06):
                a.flash = max(a.flash, random.uniform(0.3, 0.6))
            if p.wait:
                p.wait -= 1
                want = 0.0
            else:
                want = PREY_SPEED * min(1.0, abs(dx) / 6 + 0.3)
                # It stops to look, more often the closer it gets
                if random.random() < (0.01 if abs(dx) > 12 else 0.025):
                    p.wait = frames_for(random.uniform(0.6, 1.8))
            p.speed += (want - p.speed) * 0.1
            p.x += p.speed * p.facing
            p.y += max(-0.05, min(0.05, dy * 0.03))
            p.tail = (p.tail + 0.08 + p.speed * 0.8) % 2
            if math.hypot(dx, dy) < 2.2 or dx * p.facing < 0:
                # Close enough: the jaws fly open, the lure flaring, and the
                # bristlemouth freezes
                a.jaw = Angler.GAPE
                a.flash = 1.0
                p.speed = 0.0
                self.state, self.clock = "gape", 0
        elif self.state == "gape":
            self.hold()
            if self.clock >= frames_for(CATCH_GAPE):
                self.state, self.clock = "strike", 0
        elif self.state == "strike":
            # It lunges, and the water rushing into its mouth sweeps the
            # bristlemouth in with it, tumbling
            a.x += CATCH_LUNGE * a.facing
            p = self.prey
            x, y = a.pose(self.frame_count * FRAME_SECONDS)
            mx, my = a.sprite.at(x, a.facing, CATCH_MOUTH), y + CATCH_MOUTH[1]
            px, py = p.x + p.w / 2, p.y + 1.5
            p.x += (mx - px) * CATCH_SUCK
            p.y += (my - py) * CATCH_SUCK
            p.tail = (p.tail + 0.5) % 2
            if math.hypot(mx - px, my - py) < 1.5 or self.clock >= frames_for(0.4):
                # Snapped up: its lights scatter out of the jaws as they
                # clamp shut, and the lure goes dim
                tank.burst(mx + a.facing, my - 1, 16, 0.4, CATCH_BURST, 0.0 if a.facing > 0 else math.pi, 1.5)
                tank.stir(mx + a.facing, my - 1, 4)
                self.prey = None
                a.jaw = Angler.SHUT
                a.flash = 0.0
                a.glow_to = 0.2
                self.state, self.clock = "gulp", 0
        elif self.state == "gulp":
            self.hold()
            if self.clock == frames_for(1.2):
                a.jaw = None
            if self.clock > frames_for(1.8):
                a.glow_to = 1.0
                self.state, self.clock = "rest", 0
        elif self.state == "rest":
            self.hold()
            if self.clock > frames_for(2.5):
                return False
        return True

    def end(self) -> None:
        a = self.angler
        a.pace = 1.0
        a.jaw = None
        a.glow_to = 1.0
        a.flash = 0.0
        if not a.away:
            a.visit = frames_for(random.uniform(8, 20))
            a.pick_turn()

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        # Once it's settled, the others keep out of the way of the
        # anglerfish and the water ahead of its lure, where its meal comes
        if not self.held:
            return []
        a = self.angler
        lx, ly = self.lure()
        far = lx + (CATCH_ROOM + 4) * a.facing
        left, right = sorted((lx, far))
        return [(a.x, a.y, a.x + a.w, a.y + a.h), (left, ly - 4, right, ly + 4)]

    def draw(self, fb: bytearray, t: float) -> None:
        p = self.prey
        if p is not None:
            light = self.tank.light
            # Its lights twinkle as it comes
            glow = 0.75 + 0.25 * math.sin(t * 5 + p.bob)
            self.prey_sprite.draw(fb, self.tank.width, self.tank.height, round(p.x), round(p.y), p.facing,
                                  int(p.tail), light.tolist() if light is not None else None, glow)


class Dragonfish(Event):
    """A black dragonfish noses through, slow and dark against the dark, the
    barbel at its chin dangling a glowing lure and a red photophore under
    its eye. Partway across it stops a while, as if looking. It keeps to the
    clearest water it can, and the fish keep clear of it."""

    depth = 0.3

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        self.sprite = Glower(DRAGON["palette"], DRAGON["frames"], DRAGON["glows"], lamps="L", dim=1.0)
        self.facing = random.choice((1, -1))
        self.x = -self.sprite.w - 1.0 if self.facing > 0 else tank.width + 1.0
        # At a depth clear of whoever's about, barbel and all
        self.y0 = tank.lane(1, tank.floor - 11, 1, self.sprite.h + 1)
        self.stop_at = tank.width * random.uniform(0.35, 0.6)
        self.pause = frames_for(random.uniform(3.0, 6.0))
        self.speed = DRAGON_SPEED
        self.tail = 0.0
        self.frame_count = 0

    @property
    def y(self) -> float:
        return self.y0 + 0.8 * math.sin(self.frame_count * FRAME_SECONDS * 0.35)

    def step(self) -> bool:
        tank = self.tank
        width = tank.width
        self.frame_count += 1
        middle = self.x + self.sprite.w / 2
        if self.pause and (middle - self.stop_at) * self.facing >= 0:
            self.pause -= 1
            self.speed *= 0.95
        else:
            self.speed += (DRAGON_SPEED - self.speed) * 0.03
        self.x += self.speed * self.facing
        self.tail = (self.tail + 0.04 + self.speed * 0.4) % 2
        tank.stir(self.sprite.at(self.x, self.facing, (0, 2)), self.y + 2, max(0.0, self.speed - WAKE_FROM) * WAKE_RATE)
        # Until it's gone out the other side
        return (self.x + self.sprite.w >= 0) if self.facing < 0 else (self.x <= width)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        # The whole fish, barbel and all, and the water it's heading into
        left, right = self.x - 1, self.x + self.sprite.w + 1
        if self.facing > 0:
            right += CLEAR_AHEAD
        else:
            left -= CLEAR_AHEAD
        return [(left, self.y - 1, right, self.y + self.sprite.h)]

    def draw(self, fb: bytearray, t: float) -> None:
        tank = self.tank
        x, y = round(self.x), round(self.y)
        light = tank.light.tolist() if tank.light is not None else None
        self.sprite.draw(fb, tank.width, tank.height, x, y, self.facing, int(self.tail), light)
        # A soft glow around the lure
        for lx, ly in self.sprite.lamps[self.facing < 0][int(self.tail)]:
            for hx, hy, strength in LURE_HALO:
                put_max(fb, tank.width, tank.height, x + lx + hx, y + ly + hy, scale((60, 120, 200), strength * 0.6))


class Gulper(Event):
    """A gulper eel wriggles past: a long whip of a body, a light flickering
    pink at its tail tip, and a head that's nearly all mouth. Once all of it
    is in view it stops among a cloud of glowing copepods and opens its
    enormous mouth, the lower jaw dropping into a great pouch, and the
    copepods are swept in before it shuts and swims on. It keeps to the
    clearest water it can, and the fish keep clear of it."""

    depth = 0.35

    class Speck:
        __slots__ = ("x", "y", "vx", "vy", "phase")

    def __init__(self, tank: "NightTank") -> None:
        super().__init__(tank)
        # It stops with the whole of it in view, the light at its tail tip
        # too. It comes at a depth, and from a side, that keep its body and
        # the pouch it'll open there as clear as can be of whoever's about.
        tail = GULPER_JAW + GULPER_LENGTH + random.uniform(*GULPER_STOP)
        ends = {1: tail, -1: tank.width - 1 - tail}

        def blocking(facing: int, y: float) -> float:
            stop = ends[facing]
            left, right = sorted((stop - (GULPER_JAW + 2) * facing, stop + 6 * facing))
            return tank.blocking(y - 4, y + 4) + tank.crowding(left, y - 4, right, y + 3 + GULPER_POUCH)

        self.facing, self.y0 = min(((f, float(y)) for f in (1, -1) for y in np.arange(5, tank.floor - 12)),
                                   key=lambda fy: blocking(*fy))
        self.x = -1.0 if self.facing > 0 else tank.width + 1.0  # the snout
        self.stop_at = ends[self.facing]
        self.fade = 0.0  # the copepods' glow: coming on as they come into view, going as it leaves them
        self.state = "swim"  # swim, open, hold, shut, swim on
        self.clock = 0
        self.speed = GULPER_SPEED
        self.gape = 0.0
        self.phase = random.uniform(0, 2 * math.pi)
        self.frame_count = 0
        self.specks: list[Gulper.Speck] = []

    def at(self, i: float) -> float:
        """The y of the body's middle i pixels back from the hinge."""
        sway = (0.2 + 2.6 * i / GULPER_LENGTH) * math.sin(self.phase - i * 0.28)
        return self.y0 + sway

    def mouth(self) -> tuple[float, float]:
        """Where things go in: the middle of the open pouch."""
        return self.x - GULPER_JAW * 0.45 * self.facing, self.y0 + 1 + GULPER_POUCH * self.gape * 0.4

    def step(self) -> bool:
        tank = self.tank
        self.frame_count += 1
        self.clock += 1
        if self.state == "swim":
            if (self.x - self.stop_at) * self.facing >= -24 and not self.specks:
                # A cloud of copepods twinkling ahead of it
                for _ in range(18):
                    s = Gulper.Speck()
                    a, r = random.uniform(0, 2 * math.pi), random.uniform(0.5, 4.0)
                    s.x = self.stop_at + 1.5 * self.facing + math.cos(a) * r
                    s.y = self.y0 + 3 + math.sin(a) * r * 0.8
                    s.vx = s.vy = 0.0
                    s.phase = random.uniform(0, 2 * math.pi)
                    self.specks.append(s)
            if (self.x - self.stop_at) * self.facing >= 0:
                self.state, self.clock = "open", 0
        elif self.state == "open":
            self.gape = min(1.0, self.gape + 1 / frames_for(1.3))
            if self.gape >= 1:
                self.state, self.clock = "hold", 0
        elif self.state == "hold":
            if self.clock > frames_for(3) or (not self.specks and self.clock > frames_for(1.8)):
                self.state, self.clock = "shut", 0
        elif self.state == "shut":
            self.gape = max(0.0, self.gape - 1 / frames_for(0.5))
            if not self.gape and self.clock > frames_for(1.5):
                self.state, self.clock = "swim on", 0
        moving = self.state in ("swim", "swim on")
        want = GULPER_SPEED if moving else 0.0
        if self.state == "swim" and self.specks:
            want *= max(0.15, min(1.0, (self.stop_at - self.x) * self.facing / 10))
        self.speed += (want - self.speed) * 0.05
        self.x += self.speed * self.facing
        self.phase += 0.04 + self.speed * 0.7

        # The copepods drift, and an open mouth draws them in
        mx, my = self.mouth()
        kept = []
        for s in self.specks:
            if self.gape > 0.5:
                dx, dy = mx - s.x, my - s.y
                d = math.hypot(dx, dy)
                if d < 2:
                    continue  # swallowed
                # Swept in, faster the wider it gapes
                pull = 0.05 * self.gape * random.uniform(0.5, 1.0)
                s.vx = s.vx * 0.82 + dx / d * pull
                s.vy = s.vy * 0.82 + dy / d * pull
            else:
                s.vx = s.vx * 0.94 + random.uniform(-0.004, 0.004)
                s.vy = s.vy * 0.94 + random.uniform(-0.004, 0.004)
            s.x += s.vx
            s.y += s.vy
            kept.append(s)
        if self.state == "swim on":
            self.fade = max(0.0, self.fade - 1 / frames_for(1.5))
            if not self.fade:
                kept = []
        elif kept:
            self.fade = min(1.0, self.fade + 1 / frames_for(1.5))
        self.specks = kept

        # Its tail lashes up a few sparks
        if moving:
            tip = self.x - (GULPER_JAW + GULPER_LENGTH) * self.facing
            tank.stir(tip, self.at(GULPER_LENGTH), self.speed * 1.5)
        tail = self.x - (GULPER_JAW + GULPER_LENGTH + 2) * self.facing
        return (tail <= tank.width) if self.facing > 0 else (tail >= 0)

    def obstacles(self) -> list[tuple[float, float, float, float]]:
        # Its body; its head, with room for the pouch it'll open and the
        # copepods; and the water it's heading into
        f = self.facing
        back = self.x - (GULPER_JAW + GULPER_LENGTH + 1) * f
        left, right = sorted((self.x + CLEAR_AHEAD * f, back))
        head_left, head_right = sorted((self.x + CLEAR_AHEAD * f, self.x - (GULPER_JAW + 2) * f))
        return [(left, self.y0 - 4, right, self.y0 + 4),
                (head_left, self.y0 - 4, head_right, self.y0 + 3 + GULPER_POUCH)]

    def draw(self, fb: bytearray, t: float) -> None:
        tank = self.tank
        width, height = tank.width, tank.height
        light = tank.light
        f = self.facing

        def lit(color: tuple, px: int, py: int) -> tuple:
            # Unlit it's a dim shape; light brings out its color
            level = min(1.0, float(light[py, px])) if light is not None and 0 <= px < width and 0 <= py < height else 0.0
            return scale(color, 0.42 + 0.58 * level)

        def dot(px: int, py: int, color: tuple) -> None:
            put(fb, width, height, px, py, lit(color, px, py))

        # The body, tail first, thinning to a whip
        for i in range(GULPER_LENGTH, 0, -1):
            px = math.floor(self.x - (GULPER_JAW + i - 1) * f)
            py = math.floor(self.at(i) + 0.5)
            dot(px, py, GULPER_BODY)
            if i < GULPER_LENGTH * 0.35:
                dot(px, py + 1, scale(GULPER_BODY, 0.75))
        # The light at its tail tip, flickering
        tip = math.floor(self.x - (GULPER_JAW + GULPER_LENGTH) * f)
        ty = math.floor(self.at(GULPER_LENGTH) + 0.5)
        bright = 0.55 + 0.45 * abs(math.sin(t * 7.3 + self.phase)) * random.uniform(0.7, 1.0)
        put_max(fb, width, height, tip, ty, scale(GULPER_TAIL, bright))
        for hx, hy, strength in LURE_HALO:
            if abs(hx) + abs(hy) <= 2:
                put_max(fb, width, height, tip + hx, ty + hy, scale(GULPER_TAIL, strength * bright * 0.7))

        # The head: the upper jaw a straight edge from snout to hinge, the
        # lower one dropping away from it as the mouth opens, the pouch
        # between them bellying out
        drop = GULPER_POUCH * self.gape
        hy = self.y0 - 1
        for c in range(GULPER_JAW + 1):
            px = math.floor(self.x - c * f)
            k = 1 - c / GULPER_JAW  # 1 at the snout, 0 at the hinge
            upper = math.floor(hy - 0.6 * self.gape * k + 0.5)
            lower = math.floor(self.y0 + 1 + drop * k ** 0.8 + 0.5)
            belly = math.floor(self.y0 + 1 + drop * (math.sin(math.pi * (0.25 + 0.75 * k)) * 0.75 + 0.25 * k) + 0.5) \
                if self.gape else lower
            bottom = max(lower, belly)
            for py in range(upper, bottom + 1):
                edge = py == upper or py == bottom
                color = GULPER_RIM if edge and self.gape > 0.05 else GULPER_BODY if edge or py <= upper + 1 else GULPER_MOUTH
                if not self.gape:
                    color = GULPER_BODY if py == upper else scale(GULPER_BODY, 0.8)
                dot(px, py, color)
        # A tiny eye at the snout
        put(fb, width, height, math.floor(self.x - 1 * f), math.floor(hy - 0.6 * self.gape + 0.5) + 1, (150, 150, 160))

        # The copepods, twinkling
        for s in self.specks:
            b = (0.45 + 0.55 * max(0.0, math.sin(t * 4 + s.phase))) * self.fade
            put_max(fb, width, height, round(s.x), round(s.y), scale(GULPER_SPECK, b))


# name: (event, weight when picking one at random)
EVENTS = {
    "squid": (Squid, 3),
    "submersible": (Submersible, 2),
    "siphonophore": (Siphonophore, 2),
    "comb-jelly": (CombJelly, 2),
    "startle": (Startle, 2),
    "eruption": (Eruption, 2),
    "catch": (Catch, 2),
    "dragonfish": (Dragonfish, 2),
    "gulper": (Gulper, 2),
}


class NightTank(Tank):
    EVENTS = EVENTS

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        a, b = random.uniform(0, 2 * math.pi), random.uniform(0, 2 * math.pi)
        self.floor_top = [
            height - FLOOR_ROWS - round(0.9 * math.sin(x * 0.11 + a) + 0.6 * math.sin(x * 0.27 + b))
            for x in range(width)
        ]
        self.floor = floor = min(self.floor_top)

        # The vent toward one side (mirrored at random), its base sunk into
        # the floor; its veins and mouths glow, in panel coordinates
        rows = VENT if random.random() < 0.5 else flip(VENT)
        vent_w, vent_h = len(rows[0]), len(rows)
        self.vent_x = round(width * random.uniform(0.12, 0.25))
        if random.random() < 0.5:
            self.vent_x = width - vent_w - self.vent_x
        self.vent_y = max(self.floor_top[max(self.vent_x, 0):self.vent_x + vent_w]) - vent_h + 1
        self.vent_rows = rows
        self.veins = [(self.vent_x + dx, self.vent_y + dy) for dy, row in enumerate(rows)
                      for dx, ch in enumerate(row) if ch == "O"]
        # The main mouth first: it smokes the most
        self.mouths = sorted(((self.vent_x + dx, self.vent_y + dy) for dy, row in enumerate(rows)
                              for dx, ch in enumerate(row) if ch == "M"), key=lambda m: m[1])
        # How active the vent is, from 0 smoldering to 1 erupting, easing
        # toward activity_to (which an eruption raises)
        self.activity = self.activity_to = 0.0

        # The vent's glow, centered on the main chimney a little below its
        # mouth
        yy = np.arange(height, dtype=np.float32)[:, None]
        xx = np.arange(width, dtype=np.float32)[None, :]
        glow_x, glow_y = self.mouths[0][0], self.vent_y + vent_h * 0.45
        glow = np.exp(-((xx - glow_x) ** 2 + (yy - glow_y) ** 2) / (2 * VENT_REACH ** 2)).astype(np.float32)
        self.vent_light = glow * VENT_REST
        self.dark_bg, self.lit_bg = self.render_background()
        self.background = self.dark_bg.astype(np.uint8).tobytes()
        # Erupting, it brightens the water around it this much more
        self.eruption_glow = (np.array(VENT_GLOW, np.float32) * 1.5 * glow[..., None] ** 2).astype(np.float32)

        # A bed of tube worms around the vent, taller close to it
        self.worms = []
        spots = [x for x in range(self.vent_x - 9, self.vent_x + vent_w + 9)
                 if 0 <= x < width and not self.vent_x - 1 <= x <= self.vent_x + vent_w]
        for x in random.sample(spots, min(len(spots), 12)):
            w = Worm()
            w.x, w.base = x, self.floor_top[x] - 1
            near = min(abs(x - self.vent_x), abs(x - self.vent_x - vent_w))
            w.length = random.randint(2, 3) + (2 if near < 4 else 1 if near < 7 else 0)
            w.phase = random.uniform(0, 2 * math.pi)
            w.out, w.hide = 1.0, 0
            self.worms.append(w)

        # The creatures, each at its own depth, drawn far to near
        kinds = [(kind, cruise, band, odds) for kind, n, cruise, band, odds in TANK for _ in range(n)]
        depths = [i / (len(kinds) - 1) for i in range(len(kinds))]
        random.shuffle(depths)
        self.fish = []
        self.odds = {}  # how likely each is to be the next to visit
        for (kind, cruise, band, odds), depth in zip(kinds, depths):
            top = 1 + round(band[0] * (floor - 2))
            bottom = 1 + round(band[1] * (floor - 2))
            critter = {"anglerfish": Angler, "vampire-squid": VampireSquid, "spider-crab": Crab}.get(kind, Critter)(
                kind, width, top, bottom, depth, cruise)
            if isinstance(critter, Crab):
                critter.ground = self.floor_top
            self.odds[id(critter)] = odds
            self.fish.append(critter)
        self.fish.sort(key=lambda fish: -fish.depth)

        # A few start in view, the rest waiting their turns
        self.size = {id(f): 1 + sum(g.leader is f for g in self.fish) for f in self.fish if not f.leader}
        self.crowd = random.randint(*ON_SCREEN)
        self.crowd_wait = frames_for(random.uniform(*CROWD_SECONDS))
        self.entry_wait = frames_for(random.uniform(*ENTRY_SECONDS))
        self.last_gone = None
        room = self.crowd
        for f in random.sample(self.units(), len(self.size)):
            if self.size[id(f)] <= room and self.odds[id(f)] >= 1:
                room -= self.size[id(f)]
                f.visit = frames_for(random.uniform(*VISIT_SECONDS) * random.uniform(0.3, 1))
            else:
                f.away = True

        self.flakes = []
        for _ in range(SNOW_FLAKES):
            f = Flake()
            self.respawn(f)
            f.y = random.uniform(0, floor)
            self.flakes.append(f)
        self.puffs: list[Puff] = []
        self.sparks: list[Spark] = []
        self.wake: list[Spark] = []  # stirred up by swimmers (see stir)
        self.lure_kernel = glow_kernel(3.0)
        self.light = None
        self.frame_count = 0
        self.init_events()

    def stir(self, x: float, y: float, amount: float, vx: float = 0.0) -> None:
        """Stir the water at (x, y) into about amount sparks (a fraction is
        the chance of one), drifting off at vx."""
        while amount > 0:
            if amount >= 1 or random.random() < amount:
                s = Spark()
                s.x, s.y = x + random.uniform(-0.8, 0.8), y + random.uniform(-0.8, 0.8)
                s.vx, s.vy = vx + random.uniform(-0.02, 0.02), random.uniform(-0.015, 0.015)
                s.age, s.life = 0, random.randint(*WAKE_LIFE)
                s.color = scale(mix(SPARK, (150, 255, 255), random.random() * 0.6), random.uniform(0.45, 1.0))
                self.wake.append(s)
            amount -= 1

    def burst(self, x: float, y: float, n: int, speed: float, color: tuple = SPARK,
              heading: float | None = None, spread: float = math.pi) -> None:
        """Throw n sparks of about color out from (x, y) at up to speed
        pixels a frame: every way, or within spread of heading (radians, 0
        to the right), slowing as they go."""
        for _ in range(n):
            a = random.uniform(-math.pi, math.pi) if heading is None else heading + random.uniform(-spread, spread)
            v = speed * random.uniform(0.3, 1.0)
            s = Spark()
            s.x, s.y = x + random.uniform(-0.5, 0.5), y + random.uniform(-0.5, 0.5)
            s.vx, s.vy = math.cos(a) * v, math.sin(a) * v
            s.age, s.life = 0, random.randint(WAKE_LIFE[0] + 6, WAKE_LIFE[1] + 12)
            s.color = scale(mix(color, (190, 240, 255), random.random() * 0.5), random.uniform(0.7, 1.0))
            self.wake.append(s)

    def crowding(self, left: float, top: float, right: float, bottom: float) -> float:
        """How much of the box the creatures in view fill, where they are
        and where they're headed (in square pixels, give or take)."""
        total = 0.0
        for f in self.fish:
            if self.in_view(f):
                w = min(right, f.x + f.w) - max(left, f.x)
                h = min(bottom, max(f.y, f.target_y) + f.h + 1) - max(top, min(f.y, f.target_y) - 1)
                if w > 0 and h > 0:
                    total += w * h
        return total

    def blocking(self, top: float, bottom: float) -> float:
        """How much a band across the panel from top to bottom would be in
        the way: of the creatures in view, of any it would leave no room to
        get out of its way, above or below, in the water they keep to, and
        (a little) of the vent and its worms. With a little chance in it,
        for an even choice among bands equally clear."""
        blocking = self.crowding(0, top, self.width, bottom) + random.uniform(0, 2)
        for f in self.fish:
            if (self.in_view(f) and not f.leader
                    and f.top + f.h + f.stack[1] + 2 > top and f.bottom - f.stack[0] - 2 < bottom):
                blocking += 0.5 * f.w * f.h
        vent_w = len(self.vent_rows[0]) + 4
        return blocking + 0.5 * max(0.0, bottom - self.vent_y + 2) * vent_w

    def lane(self, low: float, high: float, up: float, down: float) -> float:
        """A y between low and high for something about to cross the panel
        whose body reaches up above y and down below it, as clear as can be
        (see blocking)."""
        return min((float(y) for y in np.arange(low, high + 0.01, 1.0)),
                   key=lambda y: self.blocking(y - up, y + down))

    def units(self) -> list[Critter]:
        """The creatures that go their own way (with their schools)."""
        return [f for f in self.fish if not f.leader]

    def in_view(self, fish: Critter) -> bool:
        return not (fish.leader or fish).away

    def cast(self) -> None:
        """Keep a few creatures in view: send one off when its visit is up or
        the panel's too crowded, and bring another in when there's room."""
        self.crowd_wait -= 1
        if self.crowd_wait <= 0:
            self.crowd = random.randint(*ON_SCREEN)
            self.crowd_wait = frames_for(random.uniform(*CROWD_SECONDS))
        units = self.units()
        kept = getattr(self.event, "kept", None)  # the one an event needs in view stays till it's over
        present = 0  # counting those still on their way out
        pinned = 0
        staying = []
        for u in units:
            if u.away:
                continue
            present += self.size[id(u)]
            if u is kept:
                pinned += self.size[id(u)]
            elif not u.leaving and not u.fleeing:
                u.visit -= 1
                if u.visit <= 0:
                    self.leave(u)
                else:
                    staying.append(u)
        if staying and pinned + sum(self.size[id(u)] for u in staying) > self.crowd:
            self.leave(min(staying, key=lambda u: u.visit))
        self.entry_wait -= 1
        if self.entry_wait <= 0 and self.activity < 0.2:
            waiting = [u for u in units if u.away and self.size[id(u)] <= self.crowd - present]
            # Someone other than the last to leave, if there's a choice
            if len(waiting) > 1 and self.last_gone in waiting:
                waiting.remove(self.last_gone)
            if waiting:
                self.arrive(random.choices(waiting, [self.odds[id(u)] for u in waiting])[0])
                self.entry_wait = frames_for(random.uniform(*ENTRY_SECONDS))

    def leave(self, fish: Critter) -> None:
        """Send a creature (and its school) off the nearer side of the panel."""
        fish.leaving = True
        fish.pause = 0
        fish.facing = 1 if fish.x + fish.w / 2 >= self.width / 2 else -1
        if fish.facing > 0:
            fish.turn_x = self.width + fish.spread[0] + 3
        else:
            fish.turn_x = -fish.w - fish.spread[1] - 3
        self.last_gone = fish

    def arrive(self, fish: Critter) -> None:
        """Bring a creature (and its school) in from one side for a visit."""
        fish.away = False
        fish.visit = frames_for(random.uniform(*VISIT_SECONDS))
        fish.facing = random.choice((1, -1))
        fish.x = -fish.w - fish.spread[1] - 2.0 if fish.facing > 0 else self.width + fish.spread[0] + 2.0
        fish.y = fish.target_y = random.uniform(fish.top, fish.bottom)
        fish.vy = 0.0
        fish.speed = fish.cruise
        fish.pick_turn()
        # Its lights as they should be now (they didn't change while it was
        # away), and nothing left over from its last visit
        fish.glow, fish.flash = fish.glow_to, 0.0
        if isinstance(fish, VampireSquid):
            fish.cloud.clear()
            fish.cloak = 0
        for f in self.fish:
            if f.leader is fish:
                f.x, f.facing, f.speed, f.vy = fish.x + f.place[0], fish.facing, fish.cruise, 0.0
                f.y = min(max(fish.y + f.place[1], f.top), f.bottom)

    def give_room(self) -> None:
        # Only among those in view, and not one an event is holding in place
        # (the anglerfish making its catch)
        everyone = self.fish
        held = getattr(self.event, "held", ())
        self.fish = [f for f in everyone if self.in_view(f) and f not in held]
        try:
            super().give_room()
        finally:
            self.fish = everyone

    def render_background(self) -> tuple[np.ndarray, np.ndarray]:
        """The water, floor and vent, unlit (but for the vent's own glow) and
        fully lit, as (height, width, 3) float32."""
        width, height = self.width, self.height
        y = np.arange(height, dtype=np.float32)[:, None]
        depth = np.minimum(y / max(self.floor - 1, 1), 1)[..., None]
        water = np.array(WATER_TOP, np.float32) * (1 - depth) + np.array(WATER_BOTTOM, np.float32) * depth
        water = np.broadcast_to(water, (height, width, 3))
        lit = np.broadcast_to(np.array(WATER_LIT, np.float32), (height, width, 3)).copy()
        solid = np.zeros((height, width), bool)
        chimney = np.zeros((height, width), bool)
        for x, top in enumerate(self.floor_top):
            for sy in range(top, height):
                color = random.choice(ROCKS) if random.random() < 0.08 else random.choice(SILT)
                lit[sy, x] = [c * (1 - 0.1 * (sy - top)) for c in color]
                solid[sy, x] = True
        for dy, row in enumerate(self.vent_rows):
            for dx, ch in enumerate(row):
                x, y = self.vent_x + dx, self.vent_y + dy
                if ch != "." and 0 <= x < width and 0 <= y < height:
                    lit[y, x] = VENT_PALETTE[ch]
                    solid[y, x] = chimney[y, x] = True
        dark = np.where(solid[..., None], lit * DIM, water)
        # The vent's warm glow on the floor and in the water around it
        glow = self.vent_light[..., None]
        warm = lit * np.array(VENT_TINT, np.float32)
        dark = np.where(solid[..., None], dark + (warm - dark) * glow,
                        dark + np.array(VENT_GLOW, np.float32) * glow ** 2)
        # The chimney stands dark against its own glow, lit only at the edges
        dark = np.where(chimney[..., None], lit * (DIM + 0.25 * self.vent_light[..., None] ** 3), dark)
        return np.minimum(dark, 255), np.minimum(lit, 255)

    def respawn(self, f: Flake) -> None:
        f.x = random.uniform(0, self.width)
        f.y = -1.0
        f.speed = random.uniform(0.015, 0.05)
        f.drift = random.uniform(0, 2 * math.pi)
        f.bright = random.randint(22, 55)

    def frame(self) -> bytearray:
        width, height = self.width, self.height
        self.frame_count += 1
        t = self.frame_count * FRAME_SECONDS
        self.direct()
        event = self.event

        self.cast()
        shown = [fish for fish in self.fish if self.in_view(fish)]
        for fish in shown:
            fish.swim()
            # Swimming faster than a drift, it stirs up sparks behind it
            excess = fish.speed - WAKE_FROM
            if excess > 0 and not isinstance(fish, Crab):
                x, y = fish.pose(t)
                tail = x - 1 if fish.facing > 0 else x + fish.w
                fish_y = y + fish.h * 0.6 if fish.kind != "anglerfish" else y + 8
                self.stir(tail, fish_y, excess * WAKE_RATE, -0.01 * fish.facing)
        if self.frame_count % 4 == 0:
            self.give_room()

        # The vent heats up quickly and cools slowly
        rate = 0.04 if self.activity_to > self.activity else 0.012
        self.activity += (self.activity_to - self.activity) * rate
        active = self.activity

        # What light there is: the vent's flickering glow, the lures, and any
        # an event brings
        flicker = 0.85 + 0.1 * math.sin(t * 1.7) + 0.05 * math.sin(t * 5.3)
        if active > 0.3:
            flicker += random.uniform(-0.1, 0.1) * active
        light = self.vent_light * (flicker * (1 + 1.0 / VENT_REST * active))
        for fish in shown:
            if fish.sprite.lamps[0][0]:
                for lx, ly in fish.lamps(t):
                    add_light(light, lx, ly, self.lure_kernel, 0.6 * min(fish.level(t), 1.2))
        extra = event.light() if event is not None and hasattr(event, "light") else None
        bg = None
        if extra is not None:
            light = light + extra
            bg = self.dark_bg + (self.lit_bg - self.dark_bg) * np.minimum(extra, 1)[..., None]
        if active > 0.02:
            bg = (self.dark_bg if bg is None else bg) + self.eruption_glow * active
        fb = bytearray(self.background if bg is None else np.minimum(bg, 255).astype(np.uint8).tobytes())
        self.light = light
        light_rows = light.tolist()

        if event and event.layer == BACK:
            event.draw(fb, t)

        # Marine snow sinking, lit up where light falls on it
        floor_top = self.floor_top
        for f in self.flakes:
            f.y += f.speed
            x = f.x + 0.6 * math.sin(t * 0.3 + f.drift)
            px, py = int(x) % width, int(f.y)
            if py >= floor_top[px]:
                self.respawn(f)
                continue
            if py >= 0:
                level = min(1.0, light_rows[py][px])
                b = f.bright + int(200 * level * level)
                put_max(fb, width, height, px, py, (b, b, min(255, b + 8)))

        # Smoke from the mouths, a thin wisp while it smolders and billowing
        # when it erupts: glowing as it leaves, lit by the vent as it rises,
        # cooling grey and spreading as it climbs
        for k, (mx, my) in enumerate(self.mouths):
            rate = (0.2 + 3.0 * active) if k == 0 else (0.07 + 1.0 * active)
            while rate > 0:
                if random.random() < rate:
                    p = Puff()
                    p.x, p.y = mx + random.uniform(-0.5, 0.5), my - 0.5
                    p.vx, p.vy = random.uniform(-0.03, 0.03), -random.uniform(0.07, 0.11) * (1 + 1.2 * active)
                    p.age, p.life = 0, frames_for(random.uniform(*SMOKE_SECONDS))
                    p.heat = 0.5 + 0.5 * active
                    self.puffs.append(p)
                rate -= 1
        current = 0.006 * math.sin(t * 0.04)  # the plume leans with a slow current
        puffs = []
        for p in self.puffs:
            p.age += 1
            if p.age >= p.life or p.y < -1:
                continue
            puffs.append(p)
            p.vx += random.uniform(-0.008, 0.008) + current * 0.05
            p.x += p.vx
            p.y += p.vy
            p.vy *= 0.997
            color = scale(smoke_color(p.age / p.life), p.heat)
            px, py = round(p.x), round(p.y)
            put_max(fb, width, height, px, py, color)
            if p.age > p.life * 0.3:
                put_max(fb, width, height, px + 1, py, scale(color, 0.6))
        self.puffs = puffs

        # The mouths and veins glowing from within, slow pulses of heat
        # climbing the veins, and the whole vent glowing hot as it erupts
        for mx, my in self.mouths:
            put(fb, width, height, mx, my, scale(VENT_MOUTH, min(1.0, (0.5 + 0.5 * active) * flicker)))
        for vx, vy in self.veins:
            pulse = max(0.0, math.sin(t * 1.3 + (vy - self.vent_y) * 0.7 + vx * 0.3)) ** 3
            put(fb, width, height, vx, vy, mix(VENT_DEEP, VENT_HOT, min(1.0, 0.05 + 0.55 * pulse + 0.9 * active)))

        # Tube worms, their plumes swaying, snatched in when startled
        for w in self.worms:
            if w.hide:
                w.hide -= 1
                w.out = max(0.0, w.out - 0.25)
            else:
                w.out = min(1.0, w.out + 0.02)
            # Lit by the vent, or a lamp; dim away from it
            level = min(1.0, light_rows[min(max(w.base, 0), height - 1)][w.x])
            tube_color = scale(TUBE, 0.15 + 0.85 * level)
            tube = w.length - (1 if w.out < 0.5 else 0)
            for k in range(tube):
                put(fb, width, height, w.x, w.base - k, tube_color if k == tube - 1 else scale(tube_color, 0.6))
            if w.out > 0.3:
                sway = round(0.6 * math.sin(t * 0.8 + w.phase))
                plume = scale(PLUME, w.out * (0.3 + 0.7 * level))
                put(fb, width, height, w.x + sway, w.base - tube, plume)
                if w.out > 0.8:
                    put(fb, width, height, w.x + sway - 1, w.base - tube, scale(plume, 0.6))
                    put(fb, width, height, w.x + sway + 1, w.base - tube, scale(plume, 0.6))

        # Now and then a stray spark of light in the dark
        if random.random() < 0.025:
            s = Spark()
            s.x, s.y = random.uniform(0, width), random.uniform(0, self.floor - 1)
            s.age, s.life, s.color = 0, random.randint(8, 20), mix(SPARK, (120, 160, 255), random.random())
            self.sparks.append(s)
        sparks = []
        for s in self.sparks:
            s.age += 1
            if s.age < s.life:
                sparks.append(s)
                fade = 1 - abs(2 * s.age / s.life - 1)
                put_max(fb, width, height, round(s.x), round(s.y), scale(s.color, fade))
        self.sparks = sparks

        among = event if event and event.layer == AMONG else None
        for fish in shown:
            if among and among.depth > fish.depth:
                among.draw(fb, t)
                among = None
            x, y = fish.pose(t)
            level = fish.level(t)
            fish.sprite.draw(fb, width, height, x, y, fish.facing, fish.frame(), light_rows, level)
            fish.draw_extra(fb, width, height)
            if fish.sprite.lamps[0][0] and level > 0.05:
                # A soft glow diffusing out around a lure
                color = CREATURES[fish.kind]["halo"]
                for lx, ly in fish.lamps(t):
                    for hx, hy, strength in LURE_HALO:
                        put_max(fb, width, height, lx + hx, ly + hy, scale(color, strength * min(level, 1.0)))
        if among:
            among.draw(fb, t)

        # The wake swimmers leave, flaring and fading in the water around them
        wake = []
        for s in self.wake:
            s.age += 1
            if s.age < s.life:
                wake.append(s)
                s.x += s.vx
                s.y += s.vy
                s.vx *= 0.96
                s.vy *= 0.96
                put_max(fb, width, height, round(s.x), round(s.y), scale(s.color, (1 - s.age / s.life) ** 1.5))
        self.wake = wake

        if event and event.layer == FRONT:
            event.draw(fb, t)
        return fb


SCENE = NightTank


def main() -> None:
    run_scene(NightTank, "Deep-sea night aquarium LED matrix display")


if __name__ == "__main__":
    main()
