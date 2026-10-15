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
FETCH_RETRY = 300
FETCH_RETRY_MAX = 3600
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "tides"

# The water's surface (rows down from the top) by tide height in feet: at
# LOW_FEET and below the pool is cut off and its water rests at the rim;
# by HIGH_FEET the sea has risen to HIGH_LEVEL
LOW_FEET, HIGH_FEET = 6.0, 12.0
RIM_LEVEL, HIGH_LEVEL = 15.4, 9.2


def log(message: str) -> None:
    print(f"tide_pool: {message}", file=sys.stderr, flush=True)


class Tide:
    """The tide's height now, from NOAA's predicted highs and lows (kept
    fresh by a background thread), with cosine curves between them. Before
    any have arrived, a stand-in tide of the right sort."""

    def __init__(self) -> None:
        self.fixed = os.environ.get("TIDE_FEET")
        self.times: list[float] = []
        self.heights: list[float] = []
        if self.fixed is None:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self.load()
            threading.Thread(target=self.loop, name="tide-feed", daemon=True).start()

    def feet(self, now: float) -> float:
        if self.fixed is not None:
            return float(self.fixed)
        times, heights = self.times, self.heights
        i = bisect.bisect_right(times, now)
        if 0 < i < len(times):
            k = (now - times[i - 1]) / (times[i] - times[i - 1])
            return heights[i - 1] + (heights[i] - heights[i - 1]) * (1 - math.cos(math.pi * k)) / 2
        # A semidiurnal stand-in: highs and lows about 12.4 hours apart
        return 6.0 + 5.0 * math.sin(now / (12.42 * 3600) * TAU)

    def load(self) -> bool:
        try:
            self.parse(json.loads((CACHE_DIR / "hilo.json").read_text()))
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def parse(self, data: dict) -> None:
        times, heights = [], []
        for p in data["predictions"]:
            when = datetime.strptime(p["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            times.append(when.timestamp())
            heights.append(float(p["v"]))
        if len(times) < 2:
            raise ValueError("no predictions")
        self.times, self.heights = times, heights

    def loop(self) -> None:
        retry = FETCH_RETRY
        while True:
            path = CACHE_DIR / "hilo.json"
            try:
                fresh = time.time() - path.stat().st_mtime < TIDE_MAX_AGE
            except OSError:
                fresh = False
            if fresh and self.times:
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
                path.write_text(json.dumps(data))
                retry = FETCH_RETRY
            except Exception as e:  # keep what we have; try again later
                log(f"tide fetch failed, retrying: {type(e).__name__}: {e}")
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)


def level_for(feet: float) -> float:
    k = min(1.0, max(0.0, (feet - LOW_FEET) / (HIGH_FEET - LOW_FEET)))
    return RIM_LEVEL + (HIGH_LEVEL - RIM_LEVEL) * k


# --- The setting -------------------------------------------------------------

# The rock's top row in each column: the high ledge on the left, the pool
# floor (with a boulder), and the rim on the right the sea spills over
ROCK_TOP = (
    [9, 8, 8, 9, 9, 10, 10, 11, 12, 13, 15, 18, 21, 24, 26]
    + [27, 27, 28, 28, 28, 27, 27, 28, 28, 28, 28, 27, 26, 25, 24, 24, 24, 25, 26, 27]
    + [28, 28, 28, 27, 27, 28, 28, 28, 28, 28, 27, 27, 27, 26]
    + [25, 23, 21, 19, 17, 16, 15, 15, 15, 16, 16, 15, 16, 17, 18]
)
assert len(ROCK_TOP) == WIDTH
FLOOR_LEFT, FLOOR_RIGHT = 15, 48  # the pool's floor, between the walls
DEN = (8, 21, 12, 25)  # the octopus's den in the left wall: left, top, right, bottom
HORIZON = 7
BACK_TOP = 10  # the top of the pool's far wall, behind everything (rows)
BACK_ROCK = [(36, 34, 34), (31, 30, 31), (41, 38, 36)]  # in shadow, so the near rock stands out
BACK_WEED = (34, 52, 28)

SKY_TOP = (56, 104, 166)
SKY_LOW = (132, 170, 204)
CLOUD = (196, 204, 216)
FAR_SEA = (38, 92, 128)
ROCK = [(122, 106, 90), (108, 94, 82), (138, 122, 102), (96, 84, 74)]
ROCK_EDGE = (176, 160, 134)
ROCK_SHADE = 0.9  # rock darkens this much a row deeper into the stone, down to
ROCK_SHADE_ROWS = 4  # this many rows in
WET_ROCK = 0.9  # how much darker rock is below the high-tide line
LICHEN = (200, 140, 40)
CORALLINE = (186, 98, 120)
SAND = [(176, 158, 118), (158, 140, 104), (190, 172, 130)]
DEN_DARK = (14, 11, 10)

WATER_TOP = (40, 176, 180)
WATER_DEEP = (14, 80, 110)
WATER_ALPHA = (0.06, 0.24)  # haze at the surface, and deep down: a tide pool is clear
WATER_FILTER = (0.66, 0.92, 0.96)  # what the water lets through (red goes first)
SURFACE_LIGHT = (70, 120, 120)
FOAM = (215, 230, 235)

# Kelp and rockweed: (x, y anchored, length, color, bladder color)
WEEDS = [(14, 25, 7, (110, 100, 34), (150, 140, 60)), (27, 27, 6, (96, 120, 40), (130, 160, 60)),
         (36, 24, 8, (120, 96, 30), (160, 130, 50)), (46, 27, 7, (100, 110, 36), (140, 150, 60)),
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

    def draw(self, frame: np.ndarray, x: float, y: float, index: int = 0, facing: int = 1, tint=None) -> None:
        dx, dy, colors = self.frames[index % len(self.frames)][facing < 0]
        xs, ys = dx + round(x), dy + round(y)
        ok = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        frame[ys[ok], xs[ok]] = colors[ok] if tint is None else tint(colors[ok])


# --- Creatures ---------------------------------------------------------------

ANEMONE_OPEN = [
    ["t.t.t.t", ".TtTtT.", "T.TTT.T", ".TTTTT.", "..CCC..", "..CCC.."],
    [".t.t.t.", "tTtTtT.", ".TTTTTT", ".TTTTT.", "..CCC..", "..CCC.."],
    ["t.t.t.t", ".TtTtTt", "TTTTTT.", ".TTTTT.", "..CCC..", "..CCC.."],
]
ANEMONE_SHUT = [["......."] * 3 + ["..TTT..", ".CCCCC.", "..CCC.."]]
GREEN_ANEMONE = {"T": (90, 210, 130), "t": (190, 255, 210), "C": (46, 120, 76)}
PINK_ANEMONE = {"T": (110, 190, 140), "t": (250, 150, 200), "C": (90, 130, 100)}

STAR = [["..O..", "OoOOO", ".OOo.", ".O.O.", "O...O"],
        ["..O..", "OOOoO", ".oOO.", ".O.O.", ".O..O"]]
PURPLE_STAR = {"O": (140, 46, 150), "o": (196, 110, 210)}
ORANGE_STAR = {"O": (226, 108, 34), "o": (255, 170, 90)}

URCHIN = [["s.s.s", ".sUs.", "sUUUs", ".UUU."], [".s.s.", "s.U.s", "sUUUs", ".UUU."]]
URCHIN_COLORS = {"U": (104, 34, 132), "s": (164, 84, 204)}

MUSSELS = ["MmM.Mm", "MMMMMM", ".MM.M."]
MUSSEL_COLORS = {"M": (28, 30, 58), "m": (84, 96, 134)}
BARNACLE = {"b": (204, 198, 182), "d": (110, 104, 92)}

CRAB_FRAMES = [
    [".SSS...", "SsSSe..", "SSSSCC.", ".L.L.L."],
    [".SSS...", "SsSSe..", "SSSSCC.", "L.L.L.."],
]
CRAB_HIDDEN = [[".SSS...", "SsSS...", "SSSS...", "......."]]
CRAB_BARE = [["..PP...", ".PPPe..", "PPPPCC.", ".L.L.L."]]  # out of its shell
SHELLS = [{"S": (150, 108, 74), "s": (200, 160, 120)}, {"S": (196, 186, 170), "s": (140, 128, 116)},
          {"S": (74, 70, 80), "s": (130, 124, 136)}]
CRAB_BODY = {"C": (226, 92, 40), "L": (200, 80, 36), "e": (20, 20, 20), "P": (236, 150, 140)}

SCULPIN = [[".....BB.", "tBbBbBBe", "..ff.ff."], [".....BB.", "tbBbBbBe", ".ff..ff."]]
SCULPIN_COLORS = {"B": (112, 100, 60), "b": (78, 110, 66), "t": (92, 80, 50), "e": (236, 200, 60),
                  "f": (134, 112, 70)}

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
        self.bare = 0  # frames out of its shell

    def make_art(self) -> None:
        palette = {**CRAB_BODY, **self.shell}
        self.walk = Art(palette, CRAB_FRAMES)
        self.tucked = Art(palette, CRAB_HIDDEN)
        self.naked = Art(palette, CRAB_BARE)

    def move(self) -> None:
        if self.hidden:
            self.hidden -= 1
            return
        if self.bare:
            return
        if self.goal is not None:
            ahead = self.goal - self.x
            if abs(ahead) < 0.5:
                self.speed = 0.0
                return
            self.facing = 1 if ahead > 0 else -1
            self.speed = 0.12
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
        if self.x < FLOOR_LEFT or self.x > FLOOR_RIGHT - 7:
            self.x = min(max(self.x, FLOOR_LEFT), FLOOR_RIGHT - 7)
            self.facing = 1 if self.x <= FLOOR_LEFT else -1
        self.step += self.speed * 0.8

    def draw(self, frame: np.ndarray) -> None:
        y = ground(self.x, 7) - 4
        if self.bare:
            self.naked.draw(frame, self.x, y, 0, self.facing)
        elif self.hidden:
            self.tucked.draw(frame, self.x, y, 0, self.facing)
        else:
            self.walk.draw(frame, self.x, y, int(self.step) % 2, self.facing)


class Sculpin:
    """Rests on the bottom, perfectly still, then darts to a new spot."""

    def __init__(self) -> None:
        self.art = Art(SCULPIN_COLORS, SCULPIN)
        self.x = random.uniform(FLOOR_LEFT + 2, FLOOR_RIGHT - 10)
        self.y = ground(self.x, 8) - 3.0
        self.facing = random.choice((1, -1))
        self.rest = frames_for(random.uniform(3, 10))
        self.dart: tuple | None = None  # (from x, from y, to x, to y, frames, frame)

    def flee(self, from_x: float) -> None:
        self.go(self.x + (10 if self.x > from_x else -10))

    def go(self, x: float) -> None:
        x = min(max(x, FLOOR_LEFT + 1), FLOOR_RIGHT - 9)
        frames = frames_for(0.35 + abs(x - self.x) * 0.03)
        self.dart = (self.x, self.y, x, ground(x, 8) - 3.0, frames, 0)
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
        self.anemones = [Anemone(18, GREEN_ANEMONE), Anemone(39, GREEN_ANEMONE), Anemone(30, PINK_ANEMONE)]
        self.stars = [(Art(PURPLE_STAR, STAR), 10, 17), (Art(ORANGE_STAR, STAR), 51, 20),
                      (Art(PURPLE_STAR, STAR), 44, 23)]
        self.urchins = [(Art(URCHIN_COLORS, URCHIN), 13, 23), (Art(URCHIN_COLORS, URCHIN), 15, 24)]
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
        self.sky = bg.copy()
        self.clouds = [[rng.uniform(0, WIDTH), rng.uniform(1, 4), rng.uniform(5, 9)] for _ in range(3)]
        high_line = level_for(HIGH_FEET + 1)
        rock = np.zeros((HEIGHT, WIDTH), bool)
        for c in range(WIDTH):
            top = ROCK_TOP[c]
            floor = FLOOR_LEFT <= c <= FLOOR_RIGHT
            for r in range(top, HEIGHT):
                rock[r, c] = True
                if floor and r >= top and ROCK_TOP[c] >= 26:
                    color = rng.choice(SAND)
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
        # Mussels on the rim, barnacles and limpets up on the ledge
        art = Art(MUSSEL_COLORS, [MUSSELS])
        art.draw(bg, 53, 14)
        for c in (1, 3, 4, 6, 8):
            top = ROCK_TOP[c]
            bg[top, c] = BARNACLE["b"]
            bg[top + 1, c] = BARNACLE["d"]
        self.background = bg
        self.rock = rock

    # Helpers for events -----------------------------------------------

    def surface(self) -> float:
        return self.level - self.extra

    def startle(self, x: float, radius: float = 14.0) -> None:
        """Everything near x pulls in or darts off."""
        for crab in self.crabs:
            if abs(crab.x + 3 - x) < radius and not crab.bare:
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

    def draw_water(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Tint everything under the surface, with a bright surface line and
        waves when the sea is in. Returns the surface row per column."""
        surface = self.surface()
        cols = np.arange(WIDTH, dtype=np.float32)
        swell = self.surge_strength
        line = (surface + swell * (0.7 * np.sin(cols * 0.28 - (t * 1.6) % TAU)
                                   + 0.4 * np.sin(cols * 0.11 + (t * 0.7) % TAU))
                + 0.12 * np.sin(cols * 0.9 + (t * 2.3) % TAU))
        rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
        depth = rows - line[None, :]
        cover = np.clip(depth + 0.5, 0, 1)
        deep = np.clip(depth / 16, 0, 1)
        color = np.array(WATER_TOP, np.float32) + (np.array(WATER_DEEP, np.float32) - WATER_TOP) * deep[..., None]
        alpha = (WATER_ALPHA[0] + (WATER_ALPHA[1] - WATER_ALPHA[0]) * deep) * cover
        # Light rippling down through the water
        light = 1 + 0.12 * np.sin(cols * 0.45 + rows * 0.2 - (t * 0.9) % TAU) * cover
        filtered = 1 + (np.array(WATER_FILTER, np.float32) - 1) * (cover * (0.6 + 0.4 * deep))[..., None]
        frame *= filtered * light[..., None]
        frame += (color - frame) * alpha[..., None]
        # The surface catches the light
        edge = np.clip(1 - np.abs(depth - 0.3) * 1.6, 0, 1)
        frame += edge[..., None] * np.array(SURFACE_LIGHT, np.float32)
        return line

    def draw_bits(self, frame: np.ndarray, line: np.ndarray, t: float) -> None:
        """Bubbles, foam, and motes drifting in the surge."""
        live = []
        for b in self.bubbles:
            b[1] -= b[2]
            b[0] += self.surge * 0.1
            c, r = round(b[0]), round(b[1])
            if 0 <= c < WIDTH and r > line[c] and r >= 0:
                frame[r, c] = np.maximum(frame[r, c], (150, 210, 220))
                live.append(b)
        self.bubbles = live
        live = []
        for f in self.foam:
            f[0] += f[2]
            f[1] += f[3]
            f[3] += 0.03
            f[4] -= 1
            c, r = round(f[0]), round(f[1])
            if f[4] > 0 and 0 <= c < WIDTH and 0 <= r < HEIGHT:
                frame[r, c] = frame[r, c] + (np.array(FOAM, np.float32) - frame[r, c]) * min(1.0, f[4] / 10)
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
        line = self.draw_water(frame, t)
        self.draw_bits(frame, line, t)
        if self.event:
            self.event.draw_above(frame, t, line)
        # Now and then a bubble from an anemone or the rocks
        if random.random() < 0.02:
            self.bubble(random.uniform(FLOOR_LEFT, FLOOR_RIGHT), 26)
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


class Wave(PoolEvent):
    """A big wave crashes over the rim: spray flies, the water surges up and
    churns full of bubbles, and everything in the pool ducks."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.frame_count = 0
        self.crash = frames_for(0.8)

    def step(self) -> bool:
        self.frame_count += 1
        f = self.frame_count
        pool = self.tank
        if f < self.crash:
            # Spray thrown up over the rim
            for _ in range(5):
                pool.foam.append([random.uniform(50, 63), random.uniform(13, 16), random.uniform(-0.7, -0.1),
                                  random.uniform(-0.9, -0.3), random.randint(12, 28)])
        elif f == self.crash:
            pool.extra += 3.5 if pool.level > RIM_LEVEL - 2 else 2.0
            pool.startle(40, 40)
            for _ in range(30):
                pool.bubble(random.uniform(30, 62), random.uniform(pool.surface() + 2, 27))
        elif f < self.crash + frames_for(2.5):
            for _ in range(2):
                pool.bubble(random.uniform(20, 62), random.uniform(pool.surface() + 2, 27))
            if random.random() < 0.5:
                pool.foam.append([random.uniform(20, 63), pool.surface() - 0.5, random.uniform(-0.3, 0.1),
                                  random.uniform(-0.2, 0), random.randint(10, 30)])
        return f < self.crash + frames_for(7)


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
        self.spot = (random.uniform(26, 44), 0.0) if self.afloat else (random.uniform(1, 5), ROCK_TOP[3] - 6.0)
        self.phase = "in"
        self.wait = 0
        self.frame_count = 0
        self.pecks = 0

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
                    pool.foam += [[self.x + 3 + random.uniform(-2, 2), pool.surface(), random.uniform(-0.3, 0.3),
                                   -0.2, 12] for _ in range(4)]
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
            if getattr(self, "pecking", 0):
                self.pecking -= 1
            if random.random() < 0.01:
                self.facing = -self.facing
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
            elif getattr(self, "pecking", 0):
                self.peck_art.draw(frame, self.x, self.y, 0, self.facing)
            else:
                self.stand.draw(frame, self.x, self.y, int(t * 0.7) % 2, self.facing)
        else:
            self.fly.draw(frame, self.x, self.y, int(t * 5) % 2, self.facing)


OCTOPUS = {"R": (196, 52, 40), "r": (240, 110, 80), "e": (240, 220, 120), "d": (120, 32, 30)}
OCTOPUS_BODY = [[".RRR.", "RRrRR", "RRRRR", ".eRe."]]
CAMO = {"R": (110, 84, 56), "r": (150, 120, 80), "e": (240, 220, 120), "d": (70, 56, 40)}


class Octopus(PoolEvent):
    """A red octopus pours itself out of its den, explores the bottom with
    its arms rippling, blushing brown to blend in now and then, and goes
    home again."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.red = Art(OCTOPUS, OCTOPUS_BODY)
        self.brown = Art(CAMO, OCTOPUS_BODY)
        self.x, self.y = 9.0, 22.0
        self.home = (9.0, 22.0)
        self.stops = [random.uniform(20, 40) for _ in range(random.randint(1, 3))] + [None]
        self.goal = (self.stops[0], ground(self.stops[0], 5) - 6.0)
        self.camo = 0.0
        self.camo_to = 0.0
        self.wait = 0
        self.frame_count = 0
        self.arm_phase = 0.0
        self.alpha = 0.0  # how far out of the den

    def step(self) -> bool:
        self.frame_count += 1
        pool = self.tank
        self.camo += (self.camo_to - self.camo) * 0.03
        if random.random() < 0.006:
            self.camo_to = 1.0 - self.camo_to
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
            self.goal = self.home
        else:
            self.goal = (nxt, ground(nxt, 5) - 6.0)
        self.wait = frames_for(random.uniform(2, 5))
        return True

    def draw(self, frame: np.ndarray, t: float) -> None:
        palette = OCTOPUS if self.camo < 0.5 else CAMO
        arm = mix(OCTOPUS["R"], CAMO["R"], self.camo)
        tip = mix(OCTOPUS["d"], CAMO["d"], self.camo)
        # Eight arms (we see five or six), curling and reaching
        cx, cy = self.x + 2, self.y + 3.5
        for k in range(6):
            base = -0.6 + k * 0.45
            x, y = cx + (k - 2.5) * 0.7, cy
            for s in range(5):
                bend = math.sin(self.arm_phase * 2 + k * 1.3 + s * 0.6) * 0.5
                angle = math.pi / 2 + base * 1.3 + bend
                x += math.cos(angle) * 0.9
                y += math.sin(angle) * 0.8
                floor = ground(x)
                if y > floor - 0.5:
                    y = floor - 0.5
                    x += (k - 2.5) * 0.25
                c, r = round(x), round(y)
                if 0 <= c < WIDTH and 0 <= r < HEIGHT and not (DEN[0] <= c < DEN[2] and DEN[1] <= r < DEN[3] and
                                                                 self.x < DEN[2] - 3):
                    frame[r, c] = mix(arm, tip, s / 5)
        (self.red if palette is OCTOPUS else self.brown).draw(frame, self.x, self.y)


class ShellSwap(PoolEvent):
    """Two hermit crabs meet, look each other's shells over, climb out
    (quickly!) and trade."""

    def __init__(self, pool: TidePool) -> None:
        super().__init__(pool)
        self.a, self.b = sorted(pool.crabs, key=lambda c: c.x)
        middle = (self.a.x + self.b.x) / 2
        middle = min(max(middle, FLOOR_LEFT + 8), FLOOR_RIGHT - 14)
        self.a.goal, self.b.goal = middle - 4, middle + 4
        self.a.hidden = self.b.hidden = 0
        self.phase = "meet"
        self.wait = 0
        self.frame_count = 0

    def step(self) -> bool:
        self.frame_count += 1
        a, b = self.a, self.b
        if self.phase == "meet":
            if abs(a.x - a.goal) < 0.6 and abs(b.x - b.goal) < 0.6 or self.frame_count > frames_for(40):
                a.facing, b.facing = 1, -1
                self.phase = "look"
                self.wait = frames_for(3)
        elif self.phase == "look":
            self.wait -= 1
            if self.wait <= 0:
                self.phase = "out"
                a.bare = b.bare = frames_for(1.2)
                self.wait = frames_for(1.2)
        elif self.phase == "out":
            self.wait -= 1
            if self.wait <= 0:
                a.shell, b.shell = b.shell, a.shell
                a.make_art()
                b.make_art()
                a.bare = b.bare = 0
                self.phase = "apart"
                a.goal, b.goal = a.x - 8, b.x + 8
                self.wait = frames_for(5)
        else:
            self.wait -= 1
            if self.wait <= 0:
                return False
        return True

    def end(self) -> None:
        for crab in (self.a, self.b):
            crab.goal = None
            crab.bare = 0


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
