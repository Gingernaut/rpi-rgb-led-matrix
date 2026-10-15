#!/usr/bin/env python
"""Tide pool: a cutaway of a rocky Pacific Northwest tide pool, its water
rising and falling with the real tide in Seattle.

The water level follows NOAA's tide predictions for Seattle (station
9447130, fetched daily): when the tide is low the pool is cut off from the
sea, still and clear, with barnacles, mussels and rockweed out in the air;
as the tide comes in the sea washes over the rim, the water deepens and the
surge sways everything in it. In the pool live green anemones, ochre sea
stars, purple urchins, a mottled sculpin that rests and darts, hermit crabs
picking along the bottom, and an opalescent nudibranch.

Every minute or so something happens (see EVENTS): a big wave crashes over
the rim, a gull drops by to poke around (or bobs on the water when the tide
is in), a red octopus slips out of its den to explore, two hermit crabs swap
shells, or, at high tide, a school of shiner perch swims through.
AQUARIUM_EVENT=<name> shows just that one (see aquarium_common), and
TIDE_FEET=<feet> holds the tide at that height (about -2 to 13 here):

    TIDE_FEET=2 AQUARIUM_EVENT=wave uv run server/displays/preview.py tide_pool --seconds 30

Requires the rgbmatrix Python bindings (make build-python), numpy and requests.
"""

import bisect
import json
import math
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests

from aquarium_common import FRAME_SECONDS, Event, Tank, flip, frames_for
from matrix_common import run_scene

WIDTH, HEIGHT = 64, 32
TAU = math.tau

# --- The tide ----------------------------------------------------------------

TIDE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
TIDE_STATION = "9447130"  # Seattle
TIDE_MAX_AGE = 12 * 3600
TIDE_DAYS = 4  # of predictions fetched, from yesterday
TIDE_DAY = 24.84 * 3600  # a lunar day: the tides come round again about this often
FETCH_RETRY = 300
FETCH_RETRY_MAX = 3600
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "tides"
CACHE_FILE = CACHE_DIR / "hilo.json"

# The water's surface (rows down from the top) by tide height in feet: at
# LOW_FEET and below the pool is cut off and its water rests at the rim;
# by HIGH_FEET the sea has risen to HIGH_LEVEL
LOW_FEET, HIGH_FEET = 6.0, 12.0
RIM_LEVEL, HIGH_LEVEL = 15.4, 9.2


def log(message: str) -> None:
    print(f"tide_pool: {message}", file=sys.stderr, flush=True)


class Tide:
    """The tide's height now, from NOAA's predicted highs and lows (kept
    fresh by a background thread), with cosine curves between them. Past
    the end of them (NOAA unreachable for days), the same tide a whole number
    of lunar days earlier; before any have arrived, a stand-in tide of the
    right sort."""

    def __init__(self) -> None:
        fixed = os.environ.get("TIDE_FEET")
        try:
            self.fixed = None if fixed is None else float(fixed)
        except ValueError:
            raise SystemExit(f"TIDE_FEET must be a number of feet, not {fixed!r}") from None
        # (times, heights), swapped in whole so the fetch thread can't be
        # caught halfway through replacing them
        self.points: tuple[list[float], list[float]] = ([], [])
        self.fetched = 0.0  # when the predictions in hand were fetched
        if self.fixed is None:
            self.load()
            threading.Thread(target=self.loop, name="tide-feed", daemon=True).start()

    def feet(self, now: float) -> float:
        if self.fixed is not None:
            return self.fixed
        times, heights = self.points
        if len(times) >= 2 and now >= times[-1]:
            now -= (math.floor((now - times[-1]) / TIDE_DAY) + 1) * TIDE_DAY
        i = bisect.bisect_right(times, now)
        if 0 < i < len(times):
            k = (now - times[i - 1]) / (times[i] - times[i - 1])
            return heights[i - 1] + (heights[i] - heights[i - 1]) * (1 - math.cos(math.pi * k)) / 2
        # A semidiurnal stand-in: highs and lows about 12.4 hours apart
        return 6.0 + 5.0 * math.sin(now / (12.42 * 3600) * TAU)

    def load(self) -> None:
        try:
            self.parse(json.loads(CACHE_FILE.read_text()))
            self.fetched = CACHE_FILE.stat().st_mtime
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def parse(self, data: dict) -> None:
        points = []
        for p in data["predictions"]:
            when = datetime.strptime(p["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            points.append((when.timestamp(), float(p["v"])))
        points.sort()
        if len(points) < 2:
            raise ValueError("no predictions")
        self.points = ([p[0] for p in points], [p[1] for p in points])

    def loop(self) -> None:
        retry = FETCH_RETRY
        while True:
            if time.time() - self.fetched < TIDE_MAX_AGE:
                time.sleep(600)
                continue
            try:
                start = datetime.now(timezone.utc) - timedelta(days=1)
                r = requests.get(TIDE_URL, timeout=15, params={
                    "product": "predictions", "application": "pixel_display",
                    "begin_date": start.strftime("%Y%m%d"), "range": TIDE_DAYS * 24,
                    "datum": "MLLW", "station": TIDE_STATION, "time_zone": "gmt",
                    "units": "english", "interval": "hilo", "format": "json"})
                r.raise_for_status()
                data = r.json()
                self.parse(data)
                self.fetched = time.time()
                retry = FETCH_RETRY
            except Exception as e:  # keep what we have; try again later
                log(f"tide fetch failed, retrying: {type(e).__name__}: {e}")
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)
                continue
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                CACHE_FILE.write_text(json.dumps(data))
            except OSError as e:  # the predictions are in hand; only the next start misses them
                log(f"couldn't cache the tides: {e}")


def level_for(feet: float) -> float:
    k = min(1.0, max(0.0, (feet - LOW_FEET) / (HIGH_FEET - LOW_FEET)))
    return RIM_LEVEL + (HIGH_LEVEL - RIM_LEVEL) * k


# --- The setting -------------------------------------------------------------

# The rock's top row in each column: the high ledge on the left, the pool
# floor (a boulder, and open sand beyond it), and the rim on the right the
# sea spills over
ROCK_TOP = (
    [9, 8, 8, 9, 9, 10, 10, 11, 12, 13, 15, 18, 21, 24, 26]
    + [27, 27, 28, 28, 28, 27, 28, 28]
    + [26, 24, 23, 23, 24, 26]
    + [27, 28, 28, 28, 27, 27, 28, 28, 28, 28, 28, 27, 28, 28, 28, 27, 27, 27, 27, 26]
    + [25, 23, 21, 19, 17, 16, 15, 15, 15, 16, 16, 15, 16, 17, 18]
)
assert len(ROCK_TOP) == WIDTH
FLOOR_LEFT, FLOOR_RIGHT = 15, 48  # the pool's floor, between the walls
BOULDER = (23, 28)  # on the floor: its first and last columns
DEN = (7, 20, 13, 25)  # the octopus's den in the left wall: left, top, right, bottom
HORIZON = 7
BACK_TOP = 10  # the top of the pool's far wall, behind everything (rows)
BACK_ROCK = [(36, 34, 34), (31, 30, 31), (41, 38, 36)]  # in shadow, so the near rock stands out
BACK_WEED = (34, 52, 28)

SKY_TOP = (56, 104, 166)
SKY_LOW = (132, 170, 204)
CLOUD = (196, 204, 216)
FAR_SEA = (38, 92, 128)
ROCK = [(112, 98, 84), (100, 88, 77), (126, 112, 94), (88, 78, 69)]
ROCK_EDGE = (168, 152, 128)
ROCK_SHADE = 0.9  # rock darkens this much a row deeper into the stone, down to
ROCK_SHADE_ROWS = 4  # this many rows in
WET_ROCK = 0.84  # how much darker rock is below the high-tide line
LICHEN = (200, 140, 40)
CORALLINE = (186, 98, 120)
SAND = [(146, 130, 98), (130, 116, 88), (160, 144, 108)]  # wet
DEN_DARK = (14, 11, 10)

WATER_TOP = (26, 134, 146)
WATER_DEEP = (12, 76, 104)
WATER_ALPHA = (0.22, 0.42)  # turquoise at the surface, and deep down, over the far wall
WATER_FILTER = (0.66, 0.92, 0.96)  # what the water lets through (red goes first)
# How much of the water's color reaches what's in it: all of it over the far
# wall, but the near rock is cut away in front of the water, and the creatures
# are close to the glass, so they keep their colors
ROCK_WATER, SAND_WATER, CREATURE_WATER = 0.3, 0.45, 0.35
SURFACE_LIGHT = (70, 120, 120)
FOAM = (215, 230, 235)
BUBBLE = (150, 210, 220)

# Kelp and rockweed: (x, y anchored (on the rock), length, color, bladder color)
WEEDS = [(14, 26, 7, (110, 100, 34), (150, 140, 60)), (29, 27, 6, (96, 120, 40), (130, 160, 60)),
         (36, 28, 8, (120, 96, 30), (160, 130, 50)), (45, 27, 7, (100, 110, 36), (140, 150, 60)),
         (51, 22, 6, (116, 104, 32), (156, 140, 56)), (57, 15, 5, (112, 100, 34), (150, 136, 56)),
         (3, 9, 4, (110, 98, 32), (150, 134, 54))]


def mix(a, b, t: float) -> tuple:
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class Art:
    """Sprite frames for numpy frames: each character a color from the
    palette, "." clear, facing right and (flipped) left."""

    def __init__(self, palette: dict, frames: list[list[str]]) -> None:
        self.w, self.h = len(frames[0][0]), len(frames[0])
        self.frames = []
        for f in frames:
            faces = []
            for rows in (f, flip(f)):
                pts = [(c, r, palette[ch]) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch != "."]
                faces.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
                              np.array([p[2] for p in pts], np.float32)))
            self.frames.append(faces)

    def draw(self, frame: np.ndarray, x: float, y: float, index: int = 0, facing: int = 1) -> None:
        dx, dy, colors = self.frames[index % len(self.frames)][facing < 0]
        xs, ys = dx + round(x), dy + round(y)
        ok = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        frame[ys[ok], xs[ok]] = colors[ok]


# --- Creatures ---------------------------------------------------------------

ANEMONE_OPEN = [
    ["t.t.t.t", ".TtTtT.", "T.TTT.T", ".TTTTT.", "..CCC..", "..CCC.."],
    [".t.t.t.", "tTtTtT.", ".TTTTTT", ".TTTTT.", "..CCC..", "..CCC.."],
    ["t.t.t.t", ".TtTtTt", "TTTTTT.", ".TTTTT.", "..CCC..", "..CCC.."],
]
ANEMONE_SHUT = [["......."] * 3 + ["..CCC..", ".CCCCC.", "..CCC.."]]  # a squat, dark knob
GREEN_ANEMONE = {"T": (90, 210, 130), "t": (190, 255, 210), "C": (46, 120, 76)}
PINK_ANEMONE = {"T": (110, 190, 140), "t": (250, 150, 200), "C": (90, 130, 100)}

STAR = [["..O..", "OoOOO", ".OOo.", ".O.O.", "O...O"],
        ["..O..", "OOOoO", ".oOO.", ".O.O.", ".O..O"]]
PURPLE_STAR = {"O": (140, 46, 150), "o": (196, 110, 210)}
ORANGE_STAR = {"O": (226, 108, 34), "o": (255, 170, 90)}
BLOOD_STAR = {"O": (204, 44, 60), "o": (250, 120, 120)}

URCHIN = [["s.s.s", ".sUs.", "sUUUs", ".UUU."], [".s.s.", "s.U.s", "sUUUs", ".UUU."]]
URCHIN_COLORS = {"U": (104, 34, 132), "s": (164, 84, 204)}

MUSSELS = ["MmM.Mm", "MMMMMM", ".MM.M."]
MUSSEL_COLORS = {"M": (40, 44, 80), "m": (104, 118, 164)}
BARNACLE = {"b": (204, 198, 182), "d": (110, 104, 92)}

# A hermit crab: its shell (a spiral, "s") behind, and in front its eye
# stalk, big claw ("c") and walking legs
CRAB_W = 8
CRAB_FRAMES = [
    ["..sS....", ".SsSS.e.", "SSSsSCC.", "SsSSSCcc", ".SSS.L.L"],
    ["..sS....", ".SsSS.e.", "SSSsSCC.", "SsSSSCcc", ".SSSL.L."],
]
CRAB_HIDDEN = [["..sS....", ".SsSS...", "SSSsSc..", "SsSSSc..", ".SSS...."]]  # the claw over the door
CRAB_BARE = [["........", "......e.", "..PP.CC.", ".PPPPCcc", "..PP.L.L"]]  # out of its shell
SHELL_EMPTY = [["..sS....", ".SsSS...", "SSSsa...", "SsSSa...", ".SSS...."]]  # left on the sand, its door dark
SWAP_SPOT = 30  # where crabs trade shells: the open sand, right of the boulder (and room for two)
# A tan moon snail's shell and a white turban's
SHELLS = [{"S": (186, 122, 62), "s": (238, 192, 124)}, {"S": (214, 210, 200), "s": (118, 110, 124)}]
CRAB_BODY = {"C": (236, 84, 40), "c": (255, 146, 76), "L": (214, 76, 40), "e": (255, 226, 160),
             "P": (246, 168, 160), "a": (44, 30, 28)}

# A sculpin, resting on its fins: a big head with the eye on top, a mottled
# body tapering to the tail
SCULPIN = [["...dd.e.", "tbBbBBBB", ".tBBbBBm", "..f.ff.."],
           ["..ddd.e.", "tBbBbBBB", "t.BBbBBm", ".ff..f.."]]
SCULPIN_COLORS = {"B": (126, 136, 66), "b": (58, 70, 40), "t": (110, 118, 60), "d": (160, 168, 90),
                  "e": (255, 214, 60), "m": (196, 200, 140), "f": (150, 160, 84)}

NUDIBRANCH = [["o.o.o", "WWWWw"], [".o.o.", "WWWWw"]]
NUDIBRANCH_COLORS = {"o": (255, 140, 40), "W": (220, 228, 240), "w": (110, 160, 255)}


def ground(x: float, w: int = 1) -> int:
    """The top of the rock under a creature w wide at x."""
    c0 = max(0, min(WIDTH - 1, round(x)))
    c1 = max(0, min(WIDTH - 1, round(x) + w - 1))
    return min(ROCK_TOP[c0:c1 + 1])


class Anemone:
    """Opens to wave its tentacles in the water; shuts when out of the water
    or touched."""

    def __init__(self, x: int, palette: dict) -> None:
        self.open = Art(palette, ANEMONE_OPEN)
        self.shut = Art(palette, ANEMONE_SHUT)
        self.x = x
        self.y = ground(x + 2, 3) - 6
        self.closed_for = 0
        self.phase = random.uniform(0, 3)

    def draw(self, frame: np.ndarray, pool: "TidePool") -> None:
        self.phase += 0.04 + 0.08 * pool.surge_strength
        if self.closed_for:
            self.closed_for -= 1
        if self.closed_for or self.y + 2 < pool.level:
            self.shut.draw(frame, self.x, self.y)
        else:
            self.open.draw(frame, self.x, self.y, int(self.phase) % 3)


def crab_spot(x: float) -> float:
    """x kept to the pool's floor, for a crab."""
    return min(max(x, FLOOR_LEFT), FLOOR_RIGHT - CRAB_W)


def crab_y(x: float) -> int:
    """The top of a crab at x, standing on the rock under its middle."""
    return ground(x + 2, CRAB_W - 4) - 5


class Crab:
    """A hermit crab picking its way along the bottom, now and then
    stopping, and pulling into its shell when startled."""

    def __init__(self, x: float, shell: dict) -> None:
        self.shell = shell
        self.make_art()
        self.x = x
        self.facing = random.choice((1, -1))
        self.speed = 0.0
        self.step = 0.0
        self.pause = random.randint(0, 80)
        self.hidden = 0
        self.goal: float | None = None  # where an event wants it
        self.bare = False  # out of its shell

    def make_art(self) -> None:
        palette = {**CRAB_BODY, **self.shell}
        self.walk = Art(palette, CRAB_FRAMES)
        self.tucked = Art(palette, CRAB_HIDDEN)
        self.naked = Art(palette, CRAB_BARE)

    def move(self) -> None:
        if self.hidden:
            self.hidden -= 1
            return
        if self.goal is not None:
            ahead = self.goal - self.x
            if abs(ahead) < 0.5:
                self.speed = 0.0
                return
            self.facing = 1 if ahead > 0 else -1
            self.speed = min(abs(ahead), 0.2 if self.bare else 0.12)  # out of its shell, it hurries
        elif self.bare:
            return
        elif self.pause:
            self.pause -= 1
            self.speed = 0.0
            return
        else:
            self.speed = 0.08
            if random.random() < 0.01:
                self.pause = random.randint(40, 200)
            if random.random() < 0.004:
                self.facing = -self.facing
        self.x += self.speed * self.facing
        if self.x != crab_spot(self.x):
            self.x = crab_spot(self.x)
            self.facing = 1 if self.x <= FLOOR_LEFT else -1
        self.step += self.speed * 0.8

    def draw(self, frame: np.ndarray) -> None:
        y = crab_y(self.x)
        if self.bare:
            self.naked.draw(frame, self.x, y, 0, self.facing)
        elif self.hidden:
            self.tucked.draw(frame, self.x, y, 0, self.facing)
        else:
            self.walk.draw(frame, self.x, y, int(self.step) % 2, self.facing)


def sculpin_spot(x: float) -> float:
    """The nearest place to x for the sculpin to rest: the open sand right
    of the boulder."""
    return min(max(x, BOULDER[1] + 1), FLOOR_RIGHT - 8)


class Sculpin:
    """Rests on the bottom, perfectly still, then darts to a new spot."""

    def __init__(self) -> None:
        self.art = Art(SCULPIN_COLORS, SCULPIN)
        self.x = sculpin_spot(random.uniform(FLOOR_LEFT, FLOOR_RIGHT))
        self.y = ground(self.x, 8) - 4.0
        self.facing = random.choice((1, -1))
        self.rest = frames_for(random.uniform(3, 10))
        self.dart: tuple | None = None  # (from x, from y, to x, to y, frames, frame)

    def flee(self, from_x: float) -> None:
        self.go(self.x + (10 if self.x > from_x else -10))

    def go(self, x: float, anywhere: bool = False) -> None:
        """Dart to the place nearest x it likes to rest, or (anywhere) to
        anywhere on the floor."""
        x = min(max(x, FLOOR_LEFT), FLOOR_RIGHT - 8) if anywhere else sculpin_spot(x)
        frames = frames_for(0.35 + abs(x - self.x) * 0.03)
        self.dart = (self.x, self.y, x, ground(x, 8) - 4.0, frames, 0)
        self.facing = 1 if x > self.x else -1

    def move(self) -> None:
        if self.dart:
            x0, y0, x1, y1, frames, f = self.dart
            f += 1
            k = f / frames
            ease = 1 - (1 - k) ** 2
            self.x = x0 + (x1 - x0) * ease
            self.y = y0 + (y1 - y0) * ease - math.sin(math.pi * k) * 2.5
            self.dart = None if f >= frames else (x0, y0, x1, y1, frames, f)
            if self.dart is None:
                self.rest = frames_for(random.uniform(4, 14))
            return
        self.rest -= 1
        if self.rest <= 0:
            self.go(self.x + random.choice((-1, 1)) * random.uniform(6, 16))

    def draw(self, frame: np.ndarray) -> None:
        self.art.draw(frame, self.x, self.y, 1 if self.dart else 0, self.facing)


class Nudibranch:
    """Glides along the bottom, very slowly."""

    def __init__(self) -> None:
        self.art = Art(NUDIBRANCH_COLORS, NUDIBRANCH)
        self.x = random.uniform(FLOOR_LEFT + 2, FLOOR_RIGHT - 6)
        self.facing = random.choice((1, -1))

    def move(self) -> None:
        self.x += 0.01 * self.facing
        if not FLOOR_LEFT < self.x < FLOOR_RIGHT - 6:
            self.facing = -self.facing

    def draw(self, frame: np.ndarray, t: float) -> None:
        self.art.draw(frame, self.x, ground(self.x, 5) - 2, int(t * 1.5) % 2, self.facing)


# --- The pool ----------------------------------------------------------------


class TidePool(Tank):
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the tide pool is drawn for a {WIDTH}x{HEIGHT} panel")
        self.width, self.height = width, height
        self.floor = 28
        self.fish = []  # (the Tank's fish: this pool's creatures are below)
        self.frame_count = 0
        self.tide = Tide()
        self.level = level_for(self.tide.feet(time.time()))
        self.extra = 0.0  # water a wave has thrown in, above the level
        self.surge_strength = 0.0
        self.surge = 0.0  # the sway of the water now, -1 to 1 at full strength
        self.bubbles: list[list[float]] = []  # [x, y, speed]
        self.foam: list[list[float]] = []  # [x, y, vx, vy, life]
        self.make_setting()
        # Two anemones in the pool, and one up on the rim, open only when the
        # tide is in
        self.anemones = [Anemone(16, GREEN_ANEMONE), Anemone(22, PINK_ANEMONE), Anemone(57, GREEN_ANEMONE)]
        self.stars = [(Art(PURPLE_STAR, STAR), 10, 17), (Art(ORANGE_STAR, STAR), 50, 19),
                      (Art(BLOOD_STAR, STAR), 46, 22)]
        self.urchins = [(Art(URCHIN_COLORS, URCHIN), 11, 23), (Art(URCHIN_COLORS, URCHIN), 13, 24)]
        self.crabs = [Crab(22.0, SHELLS[0]), Crab(40.0, SHELLS[1])]
        self.sculpin = Sculpin()
        self.nudibranch = Nudibranch()
        self.weeds = [{"x": x, "y": y, "n": n, "color": c, "bladder": b, "phase": random.uniform(0, TAU)}
                      for x, y, n, c, b in WEEDS]
        self.motes = [[random.uniform(0, WIDTH), random.uniform(10, HEIGHT), random.uniform(0, TAU)]
                      for _ in range(14)]
        self.init_events()

    # Setting ----------------------------------------------------------

    def make_setting(self) -> None:
        """The sky, the far sea, the rock and sand, drawn once."""
        rng = random.Random(9)
        bg = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        for r in range(HEIGHT):
            k = min(1.0, r / HORIZON)
            bg[r] = mix(SKY_TOP, SKY_LOW, k)
        bg[HORIZON:] = FAR_SEA
        for r in range(HORIZON, HEIGHT):
            bg[r] *= 1 - 0.04 * (r - HORIZON)
        # The pool's far wall, in shadow, with a little weed on it
        for r in range(BACK_TOP, HEIGHT):
            for c in range(WIDTH):
                color = rng.choice(BACK_ROCK)
                if r == BACK_TOP:
                    color = mix(color, ROCK_EDGE, 0.25)
                elif rng.random() < 0.06:
                    color = BACK_WEED
                bg[r, c] = color
        self.clouds = [[rng.uniform(0, WIDTH), rng.uniform(1, 4), rng.uniform(5, 9)] for _ in range(3)]
        high_line = level_for(HIGH_FEET + 1)
        rock = np.zeros((HEIGHT, WIDTH), bool)
        water = np.ones((HEIGHT, WIDTH), np.float32)  # see ROCK_WATER
        for c in range(WIDTH):
            top = ROCK_TOP[c]
            floor = FLOOR_LEFT <= c <= FLOOR_RIGHT
            for r in range(top, HEIGHT):
                rock[r, c] = True
                water[r, c] = ROCK_WATER
                if floor and r >= top and ROCK_TOP[c] >= 26:
                    color = rng.choice(SAND)
                    water[r, c] = SAND_WATER
                else:
                    # Lit along its top and the faces toward the pool
                    face = (c > 0 and ROCK_TOP[c - 1] > r) or (c < WIDTH - 1 and ROCK_TOP[c + 1] > r)
                    color = mix(rng.choice(ROCK), ROCK_EDGE, 0.6 if r == top else 0.35 if face else 0.0)
                    # Darker into the stone, away from its lit top and sides
                    inside = min(r - top, *(abs(c - k) for k in range(max(0, c - 4), min(WIDTH, c + 5))
                                             if ROCK_TOP[k] > r), ROCK_SHADE_ROWS)
                    color = tuple(v * ROCK_SHADE ** (inside * 0.5) for v in color)
                    if r > high_line:
                        color = tuple(v * WET_ROCK for v in color)
                    if r < high_line and rng.random() < 0.08:
                        color = LICHEN
                    elif r > RIM_LEVEL + 3 and rng.random() < 0.025:
                        color = CORALLINE
                bg[r, c] = color
        # The octopus's den
        x0, y0, x1, y1 = DEN
        bg[y0 + 1:y1, x0:x1] = DEN_DARK
        bg[y0, x0 + 1:x1 - 1] = DEN_DARK
        self.den = (bg == DEN_DARK).all(axis=2)
        # Mussels on the rim, barnacles and limpets up on the ledge
        art = Art(MUSSEL_COLORS, [MUSSELS])
        art.draw(bg, 51, 15)
        for c in (1, 3, 4, 6, 8):
            top = ROCK_TOP[c]
            bg[top, c] = BARNACLE["b"]
            bg[top + 1, c] = BARNACLE["d"]
        self.background = bg
        self.rock = rock
        self.water = water

    # Helpers for events -----------------------------------------------

    def surface(self) -> float:
        return self.level - self.extra

    def startle(self, x: float, radius: float = 14.0) -> None:
        """Everything near x pulls in or darts off."""
        for crab in self.crabs:
            if abs(crab.x + CRAB_W / 2 - x) < radius and not crab.bare:
                crab.hidden = frames_for(random.uniform(3, 7))
        for anemone in self.anemones:
            if abs(anemone.x + 3 - x) < radius:
                anemone.closed_for = frames_for(random.uniform(4, 9))
        if abs(self.sculpin.x + 4 - x) < radius and not self.sculpin.dart:
            self.sculpin.flee(x)

    def bubble(self, x: float, y: float) -> None:
        if y > self.surface() + 1:
            self.bubbles.append([x, y, random.uniform(0.15, 0.35)])

    # Drawing ----------------------------------------------------------

    def draw_clouds(self, frame: np.ndarray) -> None:
        """A few soft clouds drifting slowly over the sky (behind the rock)."""
        for cloud in self.clouds:
            cloud[0] += 0.004
            if cloud[0] - cloud[2] > WIDTH:
                cloud[0] = -cloud[2]
            x, y, size = cloud
            x0, x1 = max(0, int(x - size)), min(WIDTH, int(x + size) + 1)
            if x0 >= x1:
                continue
            xs = np.arange(x0, x1, dtype=np.float32)
            for r in range(max(0, int(y) - 1), min(HORIZON, int(y) + 3)):
                k = np.clip(1 - ((xs - x) / size) ** 2 - ((r - y - 0.5) / 1.6) ** 2, 0, 1) * 0.7
                sky = ~self.rock[r, x0:x1]
                frame[r, x0:x1][sky] += (np.array(CLOUD, np.float32) - frame[r, x0:x1][sky]) * k[sky, None]

    def draw_weeds(self, frame: np.ndarray, t: float) -> None:
        """Kelp and rockweed: floating up and swaying in the water, lying
        limp over the rock in the air."""
        surface = self.surface()
        xs, ys, colors = [], [], []
        for weed in self.weeds:
            x, y = float(weed["x"]), float(weed["y"])
            sway = math.sin(t * 0.7 + weed["phase"]) * 0.25 + self.surge * 0.6
            for i in range(weed["n"]):
                if y > surface + 0.5:
                    angle = -math.pi / 2 + sway * (i + 1) / weed["n"]
                else:
                    angle = math.atan2(1.0, 0.6 if x < WIDTH / 2 else -0.6)  # draped over the rock
                x += math.cos(angle) * 0.9
                y += math.sin(angle) * 0.9
                xs.append(round(x))
                ys.append(round(y))
                colors.append(weed["bladder"] if i % 3 == 2 else weed["color"])
        xs, ys = np.array(xs), np.array(ys)
        ok = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        frame[ys[ok], xs[ok]] = np.array(colors, np.float32)[ok]

    def draw_water(self, frame: np.ndarray, t: float, ink: np.ndarray) -> np.ndarray:
        """Tint everything under the surface (the creatures, ink, less), with
        a bright surface line and waves when the sea is in. Returns the
        surface row per column."""
        surface = self.surface()
        cols = np.arange(WIDTH, dtype=np.float32)
        swell = self.surge_strength
        line = (surface + swell * (0.7 * np.sin(cols * 0.28 - (t * 1.6) % TAU)
                                   + 0.4 * np.sin(cols * 0.11 + (t * 0.7) % TAU))
                + 0.12 * np.sin(cols * 0.9 + (t * 2.3) % TAU))
        rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
        depth = rows - line[None, :]
        strength = np.where(ink, np.float32(CREATURE_WATER), self.water)
        cover = np.clip(depth + 0.5, 0, 1)
        tint = cover * strength
        deep = np.clip(depth / 16, 0, 1)
        color = np.array(WATER_TOP, np.float32) + (np.array(WATER_DEEP, np.float32) - WATER_TOP) * deep[..., None]
        alpha = (WATER_ALPHA[0] + (WATER_ALPHA[1] - WATER_ALPHA[0]) * deep) * tint
        # Light rippling down through the water
        light = 1 + 0.12 * np.sin(cols * 0.45 + rows * 0.2 - (t * 0.9) % TAU) * cover
        filtered = 1 + (np.array(WATER_FILTER, np.float32) - 1) * (tint * (0.6 + 0.4 * deep))[..., None]
        frame *= filtered * light[..., None]
        frame += (color - frame) * alpha[..., None]
        # The surface catches the light (not on the cut face of the rock)
        edge = np.clip(1 - np.abs(depth - 0.3) * 1.6, 0, 1) * np.where(self.rock, 0.3, 1.0)
        frame += edge[..., None] * np.array(SURFACE_LIGHT, np.float32)
        return line

    def draw_bits(self, frame: np.ndarray, line: np.ndarray, t: float) -> None:
        """Bubbles, foam, and motes drifting in the surge."""
        live = []
        for b in self.bubbles:
            b[1] -= b[2]
            b[0] += self.surge * 0.1
            c, r = round(b[0]), round(b[1])
            if 0 <= c < WIDTH and r > line[c] and r >= 0 and not self.rock[r, c]:
                frame[r, c] = np.maximum(frame[r, c], BUBBLE)
                live.append(b)
        self.bubbles = live
        live = []
        foam = np.array(FOAM, np.float32)
        for f in self.foam:
            # Spray flies, and where it falls on the water it floats a while
            f[0] += f[2]
            f[4] -= 1
            c = round(f[0])
            if f[4] <= 0 or not 0 <= c < WIDTH:
                continue
            if f[1] < line[c] - 0.6:
                f[1] += f[3]
                f[3] += 0.04
            else:
                f[1] = line[c] - 0.3
                f[2] = f[2] * 0.96 + self.surge * 0.004
            r = round(f[1])
            if not 0 <= r < HEIGHT or self.rock[r, c]:
                continue
            frame[r, c] += (foam - frame[r, c]) * min(1.0, f[4] / 12)
            live.append(f)
        self.foam = live
        for m in self.motes:
            m[0] = (m[0] + self.surge * 0.08 + 0.01) % WIDTH
            m[1] += math.sin(t * 0.5 + m[2]) * 0.01
            c, r = int(m[0]), round(m[1])
            if 0 <= r < HEIGHT and r > line[c] + 1 and not self.rock[r, c]:
                frame[r, c] += 14

    def frame(self) -> np.ndarray:
        t = self.frame_count * FRAME_SECONDS
        now = time.time()
        # The tide moves slowly; ease toward it (a wave's extra water drains)
        target = level_for(self.tide.feet(now))
        self.level += (target - self.level) * 0.01
        self.extra *= 0.985
        submerged = min(1.0, max(0.0, (RIM_LEVEL - self.level) / (RIM_LEVEL - HIGH_LEVEL)))
        self.surge_strength = submerged + self.extra * 0.25
        self.surge = math.sin((t * 0.55) % TAU) * self.surge_strength
        # Fish only visit when the sea is in
        self.EVENTS = SEA_EVENTS if submerged > 0.3 or self.only else POOL_EVENTS
        self.direct()

        frame = self.background.copy()
        self.draw_clouds(frame)
        self.draw_weeds(frame, t)
        scenery = frame.copy()  # to tell the creatures (drawn next) from it
        for anemone in self.anemones:
            anemone.draw(frame, self)
        for art, x, y in self.stars:
            art.draw(frame, x, y, int(t / 7) % 2)
        for art, x, y in self.urchins:
            art.draw(frame, x, y, int(t * 2 + x) % 2)
        self.nudibranch.move()
        self.nudibranch.draw(frame, t)
        if self.event:
            self.event.draw_behind(frame, t)
        for crab in self.crabs:
            crab.move()
            crab.draw(frame)
        self.sculpin.move()
        self.sculpin.draw(frame)
        if self.event:
            self.event.draw(frame, t)
        line = self.draw_water(frame, t, (frame != scenery).any(axis=2))
        self.draw_bits(frame, line, t)
        if self.event:
            self.event.draw_above(frame, t, line)
        # Whitecaps on the swell when the sea is in, and now and then a
        # bubble from an anemone or the rocks
        if random.random() < 0.08 * (self.surge_strength - 0.5):
            x = random.uniform(0, WIDTH - 1)
            self.foam.append([x, line[round(x)], random.uniform(-0.15, 0.05), 0.0, random.randint(15, 35)])
        if random.random() < 0.02:
            x = random.uniform(FLOOR_LEFT, FLOOR_RIGHT)
            self.bubble(x, ground(x) - 1)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


# --- Events ------------------------------------------------------------------


class PoolEvent(Event):
    """An event in the pool, drawn behind the crabs (draw_behind), among them
    (draw) or over the water (draw_above)."""

    def draw_behind(self, frame: np.ndarray, t: float) -> None:
        pass

    def draw_above(self, frame: np.ndarray, t: float, line: np.ndarray) -> None:
        pass


SWELL = (34, 112, 134)  # a wave's body, before it breaks
SWELL_TOP = (96, 182, 196)


class Wave(PoolEvent):
    """A big wave rises up beyond the rim and breaks over it: spray flies,
    white water sweeps in across the pool, the water surges up and churns
    with bubbles, and everything in the pool ducks."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.frame_count = 0
        self.crash = frames_for(1.3)
        self.sweep = frames_for(2.0)  # the white water crossing the pool
        self.rise = 0.0  # the swell, 0 to 1 at its height
        self.front = float(WIDTH)  # its face

    def step(self) -> bool:
        self.frame_count += 1
        f = self.frame_count
        pool = self.tank
        if f < self.crash:
            # The swell rising, and a little spindrift off its crest
            self.rise = (f / self.crash) ** 1.5
            self.front = WIDTH - 1 - 14 * math.sqrt(f / self.crash)
            if random.random() < self.rise:
                pool.foam.append([self.front + random.uniform(0, 4), self.crest() - 0.5, random.uniform(-0.4, 0.1),
                                  random.uniform(-0.5, -0.1), random.randint(8, 16)])
        elif f == self.crash:
            # It breaks: a burst of spray off the crest
            pool.startle(40, 40)
            top = self.crest()
            for _ in range(36):
                pool.foam.append([self.front + random.uniform(-1, 8), top + random.uniform(0, 3),
                                  random.uniform(-1.0, -0.15), random.uniform(-1.0, -0.2), random.randint(20, 40)])
        if self.crash <= f < self.crash + self.sweep:
            # The swell collapses as the white water pours over the rim and
            # sweeps across, raising the water and stirring up bubbles
            k = (f - self.crash) / self.sweep
            self.rise = max(0.0, 1 - k * 3)
            pool.extra += (3.0 if pool.level > RIM_LEVEL - 2 else 1.8) / self.sweep
            edge = self.front - k * 30
            surface = pool.surface()
            for _ in range(2):
                pool.foam.append([edge + random.uniform(0, 4), surface, random.uniform(-0.25, -0.05), 0.0,
                                  random.randint(20, 50)])
            if random.random() < 0.6:
                pool.bubble(edge + random.uniform(0, 8), random.uniform(surface + 2, surface + 9))
        return f < self.crash + frames_for(7)

    def crest(self) -> float:
        """The top of the swell, at its face."""
        return min(self.tank.surface(), ROCK_TOP[min(WIDTH - 1, round(self.front))]) - 7 * self.rise

    def draw_above(self, frame: np.ndarray, t: float, line: np.ndarray) -> None:
        """The swell beyond the rim: steep in front, sloping off behind."""
        if self.rise <= 0.05:
            return
        for c in range(max(0, math.ceil(self.front)), WIDTH):
            base = min(line[c], ROCK_TOP[c])
            height = 7 * self.rise * max(0.3, 1 - (c - self.front) / 20)
            top = round(base - height)
            for r in range(max(0, top), round(base)):
                k = (r - top) / max(1.0, height)
                frame[r, c] = FOAM if r == top else mix(SWELL_TOP, SWELL, min(1.0, k * 1.5))


GULL = {"W": (232, 234, 238), "G": (150, 156, 170), "D": (60, 62, 70), "Y": (240, 200, 40),
        "L": (232, 170, 90), "e": (20, 20, 20)}
GULL_STAND = [
    ["...WWe.", "...WWWY", "GGGWWW.", "GGGWW..", ".DGWW..", "...L.L."],
    ["...WWe.", "...WWWY", "GGGWWW.", "GGGWW..", ".DGWW..", "..L..L."],
]
GULL_PECK = [[".......", "...WW..", "GGGWWWe", "GGGWWWY", ".DGWW..", "...L.L."]]
GULL_FLY = [
    ["GG.....", ".GGWWe.", "..WWWWY", "..GGW..", ".GG....", "......."],
    [".......", "...WWe.", "DGWWWWY", "GGGG...", "......."],
]
GULL_SWIM = [["...WWe.", "...WWWY", "GGGWWW.", "DGWWWW."]]


class Gull(PoolEvent):
    """A gull glides in, lands on the ledge, has a look about and pokes at
    the water (or, when the tide's in, bobs on it a while), then flies off."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.stand = Art(GULL, GULL_STAND)
        self.peck_art = Art(GULL, GULL_PECK)
        self.fly = Art(GULL, [GULL_FLY[0], GULL_FLY[1] + ["......."]])
        self.swim = Art(GULL, GULL_SWIM)
        self.afloat = pool.level < ROCK_TOP[4] + 2  # the ledge is under water
        self.x, self.y = -8.0, 0.0
        self.facing = 1
        if self.afloat:
            self.spot = (random.uniform(26, 44), 0.0)
        else:
            # On the ledge, standing on the rock under its middle foot
            # (column 3 of it, facing either way)
            x = float(random.randint(0, 3))
            self.spot = (x, ROCK_TOP[round(x) + 3] - 6.0)
        self.phase = "in"
        self.wait = 0
        self.frame_count = 0
        self.pecks = 0
        self.pecking = 0

    def step(self) -> bool:
        self.frame_count += 1
        pool = self.tank
        if self.phase == "in":
            tx, ty = self.spot
            if self.afloat:
                ty = pool.surface() - 3.5
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)
            if d < 0.6:
                self.phase = "here"
                self.wait = frames_for(random.uniform(8, 14))
                if self.afloat:
                    pool.foam += [[self.x + 3 + random.uniform(-2, 2), pool.surface() - 1, random.uniform(-0.3, 0.3),
                                   -0.3, 14] for _ in range(4)]  # a splash
                return True
            speed = min(0.7, 0.12 + d * 0.05)
            self.x += dx / d * speed
            self.y += dy / d * speed
        elif self.phase == "here":
            self.wait -= 1
            if self.afloat:
                self.y = pool.surface() - 3.5 + math.sin(self.frame_count * 0.08) * 0.4
                self.x += pool.surge * 0.03
            elif self.pecks < 3 and random.random() < 0.01:
                # A peck at the edge of the water, if it can reach
                self.pecks += 1
                self.pecking = frames_for(0.6)
                pool.startle(12, 12)
            if self.pecking:
                self.pecking -= 1
            if random.random() < 0.01:
                self.facing = -self.facing
            if not self.afloat and pool.surface() < self.y + 6:
                self.wait = 0  # the tide's come up over its feet
            if self.wait <= 0:
                self.phase = "off"
                self.facing = 1
        else:
            self.x += 0.6
            self.y -= 0.35
            return self.x < WIDTH + 8 and self.y > -8
        return True

    def draw_above(self, frame: np.ndarray, t: float, line: np.ndarray) -> None:
        if self.phase == "here":
            if self.afloat:
                self.swim.draw(frame, self.x, self.y + 2, 0, self.facing)
            elif self.pecking:
                self.peck_art.draw(frame, self.x, self.y, 0, self.facing)
            else:
                self.stand.draw(frame, self.x, self.y, int(t * 0.7) % 2, self.facing)
        else:
            self.fly.draw(frame, self.x, self.y, int(t * 5) % 2, self.facing)


OCTOPUS = {"R": (206, 58, 44), "r": (244, 124, 96), "e": (250, 226, 130), "a": (226, 96, 72), "t": (252, 176, 150)}
CAMO = {"R": (120, 92, 58), "r": (164, 132, 88), "e": (250, 226, 130), "a": (140, 108, 70), "t": (190, 160, 120)}
OCTOPUS_BODY = [".RRR.", "RRrRR", "eRRRe", ".RRR."]  # its mantle, eyes at the sides
OCTOPUS_HOME = (7.0, 21.0)  # curled up in the den
OCTOPUS_ARMS = (-1.35, -0.8, -0.3, 0.3, 0.8, 1.35)  # eight arms (we see six), fanned out from straight down
CAMO_STEPS = 12  # of its blush from red to brown
DEN_SHADE = 0.35  # how much of the octopus shows in the dark of its den


class Octopus(PoolEvent):
    """A red octopus pours itself out of its den, explores the bottom with
    its arms rippling, blushing brown to blend in now and then, and goes
    home again."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.x, self.y = OCTOPUS_HOME
        self.stops = [random.uniform(18, 38) for _ in range(random.randint(1, 3))] + [None]
        self.goal = (self.stops[0], ground(self.stops[0], 5) - 6.0)
        self.camo = 0.0
        self.camo_to = 0.0
        self.wait = 0
        self.frame_count = 0
        self.arm_phase = 0.0
        self.arts: dict[int, tuple[dict, Art]] = {}  # (palette, body) by how brown, in CAMO_STEPS

    def step(self) -> bool:
        self.frame_count += 1
        pool = self.tank
        # Now and then it blushes brown, for a few seconds
        self.camo += (self.camo_to - self.camo) * 0.03
        if random.random() < (0.01 if self.camo_to else 0.002):
            self.camo_to = 1.0 - self.camo_to
        self.arm_phase %= TAU
        if self.wait:
            self.wait -= 1
            self.arm_phase += 0.02
            return True
        gx, gy = self.goal
        dx, dy = gx - self.x, gy - self.y
        d = math.hypot(dx, dy)
        if d > 0.4:
            speed = 0.09
            self.x += dx / d * speed
            self.y += dy / d * speed
            self.arm_phase += 0.12
            pool.startle(self.x + 2, 8)
            return True
        self.stops.pop(0)
        if not self.stops:
            return False
        nxt = self.stops[0]
        if nxt is None:
            self.goal = OCTOPUS_HOME
        else:
            self.goal = (nxt, ground(nxt, 5) - 6.0)
        self.wait = frames_for(random.uniform(2, 5))
        return True

    def draw(self, frame: np.ndarray, t: float) -> None:
        brown = round(self.camo * CAMO_STEPS)
        if brown not in self.arts:
            palette = {ch: mix(OCTOPUS[ch], CAMO[ch], brown / CAMO_STEPS) for ch in OCTOPUS}
            self.arts[brown] = (palette, Art(palette, [OCTOPUS_BODY]))
        palette, body = self.arts[brown]
        in_den = frame[self.tank.den].copy()
        # The arms, curling and reaching, spreading out along the bottom
        cx, cy = self.x + 2, self.y + 3
        x0, y0, x1, y1 = DEN
        for k, spread in enumerate(OCTOPUS_ARMS):
            side = 1 if spread > 0 else -1
            x, y = cx + spread * 1.2, cy
            for s in range(7):
                bend = math.sin(self.arm_phase * 2 + k * 1.3 + s * 0.6) * 0.5
                angle = math.pi / 2 + spread + bend - side * s * 0.06
                x += math.cos(angle) * 0.85
                y += math.sin(angle) * 0.85
                c = round(x)
                if not 0 <= c < WIDTH:
                    break
                floor = y1 if x0 <= c < x1 and y >= y0 else ROCK_TOP[c]
                if y > floor - 0.5:
                    if y > floor + 2.5:
                        break  # up against a wall of rock: the rest is behind it
                    y = floor - 0.5
                    x += side * 0.3
                    c = min(max(round(x), 0), WIDTH - 1)
                r = round(y)
                if 0 <= r < HEIGHT:
                    frame[r, c] = mix(palette["a"], palette["t"], s / 7)
        body.draw(frame, self.x, self.y)
        # Whatever of it is still in the den, in the dark there
        frame[self.tank.den] = in_den + (frame[self.tank.den] - in_den) * DEN_SHADE


class ShellSwap(PoolEvent):
    """Two hermit crabs meet claw to claw out on the open sand and tap at
    each other's shells, then both climb out, scuttle past each other
    (quickly!), back into the other's shell, and go on their way."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.a, self.b = sorted(pool.crabs, key=lambda c: c.x)
        self.spots = (SWAP_SPOT, SWAP_SPOT + CRAB_W)
        self.a.goal, self.b.goal = self.spots
        for crab in (self.a, self.b):
            crab.hidden = crab.pause = 0
        self.shells = [Art({**CRAB_BODY, **crab.shell}, SHELL_EMPTY) for crab in (self.a, self.b)]
        self.phase = "meet"
        self.wait = 0
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        a, b = self.a, self.b
        sculpin = self.tank.sculpin  # (shooed out of the way)
        if not sculpin.dart and SWAP_SPOT - 8 < sculpin.x < SWAP_SPOT + 2 * CRAB_W:
            sculpin.go(FLOOR_LEFT, anywhere=True)
        if self.phase == "meet":
            if abs(a.x - a.goal) < 0.6 and abs(b.x - b.goal) < 0.6 or self.frame_count > frames_for(40):
                a.x, b.x = self.spots
                a.facing, b.facing = 1, -1
                self.phase = "look"
                self.wait = frames_for(3)
        elif self.phase == "look":
            # Tapping, one and then the other
            self.wait -= 1
            tap = self.wait // frames_for(0.4) % 4
            a.goal = self.spots[0] - (tap == 1)
            b.goal = self.spots[1] + (tap == 3)
            if self.wait <= 0:
                self.phase = "out"
                a.x, b.x = self.spots
                a.goal = b.goal = None
                a.bare = b.bare = True
                self.wait = frames_for(1.0)
        elif self.phase == "out":
            # A moment out in the open, then each to the other's shell
            self.wait -= 1
            if self.wait == 0:
                a.goal, b.goal = self.spots[1], self.spots[0]
            arrived = a.goal is not None and abs(a.x - a.goal) < 0.5 and abs(b.x - b.goal) < 0.5
            if arrived or self.wait < -frames_for(5):
                # Backed in, so facing the way the shell lay
                a.x, b.x = self.spots[1], self.spots[0]
                a.facing, b.facing = -1, 1
                a.shell, b.shell = b.shell, a.shell
                a.make_art()
                b.make_art()
                a.bare = b.bare = False
                a.goal, b.goal = a.x, b.x
                self.phase = "settle"
                self.wait = frames_for(1.5)
        elif self.phase == "settle":
            self.wait -= 1
            if self.wait <= 0:
                self.phase = "apart"
                a.goal, b.goal = crab_spot(a.x + 6), crab_spot(b.x - 10)
                self.wait = frames_for(4)
        else:
            self.wait -= 1
            if self.wait <= 0:
                return False
        return True

    def draw_behind(self, frame: np.ndarray, t: float) -> None:
        if self.phase == "out":
            # The two empty shells, where they were left
            for shell, x, facing in zip(self.shells, self.spots, (1, -1)):
                shell.draw(frame, x, crab_y(x), 0, facing)

    def end(self) -> None:
        for crab in (self.a, self.b):
            crab.goal = None
            crab.bare = False


PERCH = {"S": (196, 206, 214), "s": (140, 156, 170), "y": (230, 200, 90), "e": (20, 20, 30)}
PERCH_FRAMES = [[".sSSS.", "sSyyeS", ".sSSS."], ["s.SSS.", ".SyyeS", "s.SSS."]]


class Perch(PoolEvent):
    """With the sea in, a school of shiner perch flickers through."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.art = Art(PERCH, PERCH_FRAMES)
        self.facing = random.choice((1, -1))
        top = pool.level + 2
        self.fish = []
        start = -8.0 if self.facing > 0 else WIDTH + 2.0
        # In loose ranks, so they don't bunch into one blob
        rows = max(1, min(3, int((22 - top) // 3.5)))
        speed = random.uniform(0.35, 0.45)
        for i in range(random.randint(5, 8)):
            self.fish.append({"x": start - self.facing * (i // rows * 8 + random.uniform(0, 3)),
                              "y": top + (i % rows) * 3.5 + random.uniform(0, 1),
                              "speed": speed * random.uniform(0.97, 1.03), "phase": random.uniform(0, TAU)})
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        for f in self.fish:
            f["x"] += f["speed"] * self.facing
            f["y"] += math.sin(self.frame_count * 0.08 + f["phase"]) * 0.05
            f["y"] = max(f["y"], self.tank.surface() + 1.5)
        return any((-8 < f["x"] < WIDTH + 2) or (f["x"] < 0) == (self.facing > 0) for f in self.fish)

    def draw(self, frame: np.ndarray, t: float) -> None:
        for f in self.fish:
            self.art.draw(frame, f["x"], f["y"], int(t * 6 + f["phase"]) % 2, self.facing)


SEA_EVENTS = {
    "wave": (Wave, 3),
    "gull": (Gull, 2),
    "octopus": (Octopus, 2),
    "shell-swap": (ShellSwap, 1),
    "perch": (Perch, 2),
}
POOL_EVENTS = {name: event for name, event in SEA_EVENTS.items() if name != "perch"}
TidePool.EVENTS = SEA_EVENTS


SCENE = TidePool


def main() -> None:
    run_scene(TidePool, "Tide pool LED matrix display")


if __name__ == "__main__":
    main()
