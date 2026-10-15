#!/usr/bin/env python
"""Tides: the next 48 hours of tide at Redondo, for planning a dive.

    REDONDO        ^ 6.2ft     the station, and the tide now, rising or falling
    ~~~~/\\~~~~/\\~~~~~~~~      the tide curve, filled like water, from just
    ~~~/  \\~~/  \\~~~~~~~       before now across the next 48 hours, darker at
                               night, with midnights dotted, the 0 ft line,
                               now marked, and each high and low tide dotted
    v Thu 9:11a     -0.7ft     each high and low in turn, its dot blinking

The predictions are NOAA's for Des Moines, East Passage (station 9446248,
about a mile from the Redondo dive site), fetched twice a day (see
noaa_tides). Heights are feet above mean lower low water, so minus tides
(the big exchanges, with the strongest currents between) dip below the
0 ft line and their lows are dotted orange. Daylight comes from the Sun's
position at the station.

Requires the rgbmatrix Python bindings (make build-python), numpy and requests.
"""

import math
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from matrix_common import REFRESH_HZ, BitmapFont, run_scene
from noaa_tides import Predictions

FRAME_SECONDS = 3 / REFRESH_HZ  # every 3rd refresh, ~25 fps

STATION = "9446248"  # Des Moines, East Passage
NAME = "REDONDO"
LAT, LON = 47.35, -122.32
TIMEZONE = ZoneInfo("America/Los_Angeles")

WIDTH, HEIGHT = 64, 32
# (A cell is left clear all round the edge, two at the sides)
MARGIN = 2
HEADER_TOP = 1
GRAPH_TOP, GRAPH_BOTTOM = 8, 24  # rows: the graph fills 8 to 23
TEXT_TOP = 25

# The chart: HOURS across the graph's width, starting PAST_HOURS before now,
# and LOW_FEET to HIGH_FEET up its height (fixed, so one day's exchange can
# be compared with another's)
HOURS = 48.0
PAST_HOURS = 1.0
LOW_FEET, HIGH_FEET = -3.0, 13.0
GRAPH_LEFT, GRAPH_RIGHT = MARGIN, WIDTH - MARGIN  # columns
HOURS_PER_COLUMN = HOURS / (GRAPH_RIGHT - GRAPH_LEFT)
REDRAW_SECONDS = 30  # the chart is redrawn this often (now moves a column in 48 minutes)

EVENT_SECONDS = 3.0  # each high and low on the bottom line in turn
BLINK_SECONDS = 1.0

NAME_COLOR = (90, 170, 205)
WHITE = (225, 225, 225)
DIM = (120, 120, 132)
DAY = (14, 22, 40)
NIGHT = (2, 3, 8)
WATER = (22, 96, 176)
WATER_NIGHT = 0.55  # how bright the water is at night
SURFACE = (90, 180, 240)
MIDNIGHT = (40, 40, 52)
ZERO_LINE = (46, 46, 56)
NOW_LINE = (60, 60, 70)
HIGH_COLOR = (235, 235, 235)
LOW_COLOR = (70, 215, 255)
MINUS_COLOR = (255, 150, 40)  # a low below 0 ft
RISING_COLOR = (60, 210, 90)
FALLING_COLOR = (255, 150, 40)

UP_ARROW = [".#.",
            "###"]
DOWN_ARROW = ["###",
              ".#."]


def sun_elevation(t: float) -> float:
    """The Sun's elevation (degrees) at the station at time t, from the
    Astronomical Almanac's low-precision formulas (good to a fraction of a
    degree, plenty for telling day from night)."""
    n = t / 86400.0 + 2440587.5 - 2451545.0
    g = math.radians(357.528 + 0.9856003 * n)
    longitude = math.radians(280.460 + 0.9856474 * n + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    tilt = math.radians(23.439 - 4e-7 * n)
    ra = math.atan2(math.cos(tilt) * math.sin(longitude), math.cos(longitude))
    dec = math.asin(math.sin(tilt) * math.sin(longitude))
    hour = math.radians((280.46061837 + 360.98564736629 * n + LON) % 360) - ra
    lat = math.radians(LAT)
    return math.degrees(math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour)))


def rgb(color) -> np.ndarray:
    return np.array(color, np.float32)


class Tides:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the chart is laid out for a {WIDTH}x{HEIGHT} panel")
        self.frame_count = 0
        self.font = BitmapFont("tom-thumb")
        self.cache: dict = {}
        self.icons = {name: np.array([[c == "#" for c in row] for row in rows], np.float32)
                      for name, rows in (("up", UP_ARROW), ("down", DOWN_ARROW))}
        self.chart = None
        self.chart_key = None
        self.events: list[tuple[float, float, str, int, int]] = []  # (time, feet, kind, column, row)
        self.predictions = Predictions(STATION)

    # Layout helpers -------------------------------------------------------

    def text(self, text: str) -> np.ndarray:
        found = self.cache.get(text)
        if found is None:
            if len(self.cache) > 200:
                self.cache.clear()
            found = self.cache[text] = self.font.render(text, proportional=True)
        return found

    def put(self, frame: np.ndarray, part: str, x: int, top: int, color) -> int:
        """Paint text (or an icon, by name) at x on a line; returns its width."""
        art = self.icons.get(part)
        if art is None:
            art = self.text(part)
        else:  # on the text's baseline
            top += self.font.ascent - art.shape[0]
        w = min(art.shape[1], WIDTH - x)
        if w > 0 and x >= 0:
            frame[top:top + art.shape[0], x:x + w] += art[:, :w, None] * rgb(color)
        return art.shape[1]

    @staticmethod
    def column_time(c: float, start: float) -> float:
        """The time at the middle of column c."""
        return start + (c - GRAPH_LEFT + 0.5) * HOURS_PER_COLUMN * 3600

    @staticmethod
    def surface_y(feet: float) -> float:
        """Where the water's surface is drawn for a height: an edge between
        rows, GRAPH_BOTTOM at LOW_FEET up to GRAPH_TOP at HIGH_FEET."""
        k = (feet - LOW_FEET) / (HIGH_FEET - LOW_FEET)
        return GRAPH_BOTTOM - min(1.0, max(0.0, k)) * (GRAPH_BOTTOM - GRAPH_TOP)

    # The chart ------------------------------------------------------------

    def draw_chart(self, now: float) -> np.ndarray:
        """The station's name and the graph, without anything that blinks
        or changes by the minute."""
        frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        start = now - PAST_HOURS * 3600
        rows = np.arange(GRAPH_TOP, GRAPH_BOTTOM, dtype=np.float32)
        zero = self.surface_y(0.0)
        midnights = set()
        local = datetime.fromtimestamp(start, TIMEZONE)
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        while midnight.timestamp() < start + HOURS * 3600:
            midnights.add(GRAPH_LEFT + int((midnight.timestamp() - start) / (HOURS_PER_COLUMN * 3600)))
            midnight += timedelta(days=1)
        for c in range(GRAPH_LEFT, GRAPH_RIGHT):
            t = self.column_time(c, start)
            # Day or night (blending through twilight, the Sun 0 to 6 degrees down)
            light = min(1.0, max(0.0, (sun_elevation(t) + 6) / 6))
            back = rgb(NIGHT) + (rgb(DAY) - rgb(NIGHT)) * light
            column = np.tile(back, (len(rows), 1))
            if c in midnights:
                column[::2] = MIDNIGHT
            column[int(zero) - GRAPH_TOP] = np.maximum(column[int(zero) - GRAPH_TOP], ZERO_LINE)
            feet = self.predictions.feet(t)
            if feet is not None:
                y = self.surface_y(feet)
                cover = np.clip(rows + 1 - y, 0, 1)[:, None]
                water = rgb(WATER) * (WATER_NIGHT + (1 - WATER_NIGHT) * light)
                column = column * (1 - cover) + water * cover
                # The surface, brighter
                top = int(math.floor(y))
                if GRAPH_TOP <= top < GRAPH_BOTTOM:
                    column[top - GRAPH_TOP] = np.maximum(column[top - GRAPH_TOP],
                                                         rgb(SURFACE) * (1 - (y - top)) * (0.6 + 0.4 * light))
            frame[GRAPH_TOP:GRAPH_BOTTOM, c] = column
        # The highs and lows along the chart
        self.events = []
        for t, feet, kind in self.predictions.events(start, start + HOURS * 3600):
            c = GRAPH_LEFT + int((t - start) / (HOURS_PER_COLUMN * 3600))
            r = min(GRAPH_BOTTOM - 1, max(GRAPH_TOP, int(math.floor(self.surface_y(feet))) - (1 if kind == "H" else 0)))
            if GRAPH_LEFT <= c < GRAPH_RIGHT:
                self.events.append((t, feet, kind, c, r))
                frame[r, c] = self.event_color(kind, feet)
        self.put(frame, NAME, MARGIN, HEADER_TOP, NAME_COLOR)
        return frame

    @staticmethod
    def event_color(kind: str, feet: float) -> tuple:
        if kind == "H":
            return HIGH_COLOR
        return MINUS_COLOR if feet < 0 else LOW_COLOR

    # The frame ------------------------------------------------------------

    def frame(self) -> np.ndarray:
        now = time.time()
        key = (int(now // REDRAW_SECONDS), self.predictions.points[0][:1], len(self.predictions.points[0]))
        if key != self.chart_key:
            self.chart, self.chart_key = self.draw_chart(now), key
        frame = self.chart.copy()
        blink = (now % BLINK_SECONDS) < BLINK_SECONDS * 0.6

        feet = self.predictions.feet(now)
        if feet is None:
            message = "no tide data"
            self.put(frame, message, (WIDTH - self.text(message).shape[1]) // 2, TEXT_TOP, DIM)
        else:
            # Now: a line down the chart, and a dot where the water is
            start = now - PAST_HOURS * 3600
            c = GRAPH_LEFT + int(PAST_HOURS / HOURS_PER_COLUMN)
            frame[GRAPH_TOP:GRAPH_BOTTOM, c] = np.maximum(frame[GRAPH_TOP:GRAPH_BOTTOM, c], NOW_LINE)
            r = min(GRAPH_BOTTOM - 1, max(GRAPH_TOP, int(math.floor(self.surface_y(feet)))))
            frame[r, c] = WHITE
            # The tide now, rising or falling, at the top right
            upcoming = self.predictions.events(now, now + 86400)
            rising = bool(upcoming) and upcoming[0][2] == "H"
            label = f"{feet:.1f}ft"
            x = WIDTH - MARGIN - self.text(label).shape[1]
            self.put(frame, label, x, HEADER_TOP, WHITE)
            self.put(frame, "up" if rising else "down", x - 4, HEADER_TOP, RISING_COLOR if rising else FALLING_COLOR)
            # Each coming high and low in turn, its dot blinking
            ahead = [e for e in self.events if e[0] >= now]
            if ahead:
                t, height, kind, ec, er = ahead[int(now / EVENT_SECONDS) % len(ahead)]
                color = self.event_color(kind, height)
                if not blink:
                    frame[er, ec] = rgb(color) * 0.25
                when = datetime.fromtimestamp(t, TIMEZONE)
                label = f"{when:%a} {when.hour % 12 or 12}:{when.minute:02d}{'a' if when.hour < 12 else 'p'}"
                x = MARGIN + self.put(frame, "up" if kind == "H" else "down", MARGIN, TEXT_TOP, color) + 2
                self.put(frame, label, x, TEXT_TOP, WHITE)
                level = f"{height:.1f}ft"
                self.put(frame, level, WIDTH - MARGIN - self.text(level).shape[1], TEXT_TOP, color)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


SCENE = Tides


def main() -> None:
    run_scene(Tides, "Tide chart LED matrix display")


if __name__ == "__main__":
    main()
