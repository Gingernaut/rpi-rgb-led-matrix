#!/usr/bin/env python
"""Flights: the sky out a west-facing window in Seward Park, Seattle, with the
planes in it where they really are right now.

The panel looks west from the window: south at the left edge, north at the
right, VIEW_LEFT to VIEW_RIGHT degrees around, and from the horizon (above
the strip of text) up about 40 degrees. Boeing Field and Sea-Tac lie off to
the southwest, so most of what flies by is landing or taking off there, or
crossing high overhead.

Positions come from adsb.fi's open ADS-B feed (no key), polled every few
seconds; between polls each plane flies on along its reported track and
speed, and when a new report puts it somewhere a little different it eases
there instead of jumping. Each plane is drawn at its real compass direction
and height above the horizon, bigger the nearer it is: a silhouette by day,
with navigation lights, strobes and (low and landing) landing lights, and
only its lights by night. High jets leave contrails in daylight.

The sky follows the real Sun (and Moon): blue by day, sunset colors, stars at
night. Along the horizon are the trees of Beacon Hill, downtown's towers to
the northwest, the Olympics beyond, and the two airports' beacons after dark.

The strip below names one plane in view, marked in the sky, taking turns
when there are several: its airline and flight number (or registration),
then its route (when adsbdb's guess fits where it is) and its type, and its
altitude, climbing or descending. With nothing in view it shows the time.

For previews, FLIGHTS_FRAME_CLOCK=1 keeps time by frames (as the other
scenes do) rather than by the wall clock, so a GIF plays at real speed.

Requires the rgbmatrix Python bindings (make build-python), numpy and requests.
"""

import json
import math
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

from matrix_common import REFRESH_HZ, BitmapFont, run_scene

FRAME_SECONDS = 3 / REFRESH_HZ  # every 3rd refresh, ~25 fps

# The window
HOME_LAT, HOME_LON = 47.552, -122.265
HOME_HEIGHT = 40.0  # meters above sea level
TIMEZONE = ZoneInfo("America/Los_Angeles")

WIDTH, HEIGHT = 64, 32
# Compass degrees at the panel's left and right edges, and degrees of
# elevation per row. Pixel centers are at whole coordinates; the horizon is
# the bottom edge of row HORIZON_ROW, and the strip of text is below it.
VIEW_LEFT, VIEW_RIGHT = 190.0, 350.0
AZ_PER_PX = (VIEW_RIGHT - VIEW_LEFT) / WIDTH
EL_PER_PX = 2.0
HORIZON_ROW = 19
HORIZON_Y = HORIZON_ROW + 0.5
STRIP_TOP = HORIZON_ROW + 1
MAX_RANGE = 70_000  # meters; farther planes are too small and low to see

# Planes in view (and, below, how big to draw them) by distance
NEAR = 6_000
MID = 16_000

FEED_URL = "https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{dist}"
FEED_RADIUS_NM = 40
POLL_SECONDS = 5
STALE_SECONDS = 30  # forget a plane not heard from in this long
NO_DATA_SECONDS = 90  # say so in the strip after this long without a poll
ROUTE_URL = "https://api.adsbdb.com/v0/callsign/{callsign}"
ROUTE_GAP = 1.5  # seconds between route lookups
ROUTE_KEEP = 7 * 86400  # how long a looked-up route is trusted
ROUTE_KEEP_MISSING = 86400
# Seconds to wait after a failed fetch, doubling while it keeps failing
FETCH_RETRY = 30
FETCH_RETRY_MAX = 600
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "flights"
USER_AGENT = "pixel-display (home LED matrix; github.com/Gingernaut/rpi-rgb-led-matrix)"

# A reported position that doesn't line up with where the plane was drawn
# is eased into over SETTLE_SECONDS; one this far off is jumped to
SETTLE_SECONDS = 1.2
JUMP_METERS = 3_000

# A route is shown only if the plane is within ROUTE_SLACK of the great
# circle between its endpoints (adsbdb's routes are often out of date)
ROUTE_SLACK = 250_000

# The strip names each plane in view for FEATURE_SECONDS in turn (a plane
# within CLOSE stays named while it's that close), its second line
# alternating between route and type every LINE_SECONDS
FEATURE_SECONDS = 10.0
CLOSE = 4_500
LINE_SECONDS = 4.0
CARD_FADE_SECONDS = 0.4

# High jets leave contrails by day
CONTRAIL_FEET = 26_000
CONTRAIL_SAMPLES = 36
CONTRAIL_EVERY = 0.5  # seconds between trail points
# Landing lights: below LANDING_FEET and descending, or below TAKEOFF_FEET
LANDING_FEET = 10_000
TAKEOFF_FEET = 2_500

EARTH_RADIUS = 6_371_000.0
REFRACTION = 0.13  # light bends over the horizon a little: the Earth looks bigger
METERS_PER_FOOT = 0.3048
KNOTS = 0.514444  # m/s
FEET_PER_MINUTE = 0.00508  # m/s
# Meters per degree of latitude and longitude at home
M_PER_LAT = 111_132.92 - 559.82 * math.cos(math.radians(2 * HOME_LAT)) + 1.175 * math.cos(math.radians(4 * HOME_LAT))
M_PER_LON = 111_412.84 * math.cos(math.radians(HOME_LAT)) - 93.5 * math.cos(math.radians(3 * HOME_LAT))

# Landmarks on the horizon, (latitude, longitude)
BOEING_FIELD = (47.5300, -122.3019)
SEA_TAC = (47.4502, -122.3088)
DOWNTOWN = (47.6045, -122.3306)
SPACE_NEEDLE = (47.6205, -122.3493)

# --- Look --------------------------------------------------------------------

# The sky's colors by the Sun's elevation: (elevation, zenith, horizon, glow
# around the Sun), blended between. Kept moderate: a whole panel of bright
# sky draws a lot of current.
SKY_KEYS = [
    (-18.0, (1, 1, 6), (5, 5, 12), (0, 0, 0)),
    (-12.0, (2, 3, 12), (12, 9, 20), (4, 2, 2)),
    (-7.0, (6, 9, 34), (48, 26, 42), (40, 14, 8)),
    (-3.0, (14, 24, 66), (130, 60, 40), (110, 45, 10)),
    (1.0, (20, 42, 104), (170, 96, 52), (140, 70, 16)),
    (6.0, (20, 56, 132), (130, 118, 120), (90, 60, 20)),
    (14.0, (16, 58, 140), (86, 126, 172), (40, 34, 14)),
    (40.0, (14, 56, 140), (78, 120, 170), (24, 22, 10)),
]
HORIZON_FALLOFF = 9.0  # degrees: how quickly the horizon color gives way to the zenith's
GLOW_WIDTH = 22.0  # degrees around the Sun

SUN_COLOR = (255, 236, 180)
SUNSET_COLOR = (255, 130, 50)
MOON_COLOR = (215, 215, 195)
STAR_COLOR = (190, 190, 210)
STARS = 32

TREES_DAY = (10, 24, 14)
TREES_NIGHT = (2, 3, 4)
TOWERS_DAY = (34, 40, 54)
TOWERS_NIGHT = (5, 5, 8)
WINDOW_LIGHT = (90, 70, 30)
HOUSE_LIGHT = (110, 80, 30)
TOWER_BEACON = (200, 20, 10)
MOUNTAIN_DAY = (46, 62, 92)
SNOW = (165, 172, 190)

PLANE_DAY = (40, 44, 58)
PLANE_NIGHT = (12, 12, 16)
RED_LIGHT = (255, 30, 20)
GREEN_LIGHT = (30, 255, 60)
STROBE = (255, 255, 255)
LANDING_LIGHT = (255, 250, 225)
CONTRAIL = (210, 214, 224)

MARGIN = 1  # columns left clear at the strip's ends
TEXT = (220, 220, 220)
DIM_TEXT = (110, 110, 120)
CLIMB_COLOR = (60, 210, 90)
DESCEND_COLOR = (255, 150, 40)
DEFAULT_AIRLINE = (220, 220, 220)

# The Olympics: (compass degrees, height in rows, degrees per row of slope)
OLYMPICS = [(258, 2.0, 5), (266, 3.0, 3), (272, 2.6, 3), (278, 4.2, 2.5), (285, 5.2, 2.5),
            (291, 4.4, 2.5), (298, 3.4, 3), (305, 2.8, 3), (313, 2.2, 4)]

# Downtown and the Space Needle, standing on the horizon: "#" building, "o"
# a light at night, "r" a red beacon
DOWNTOWN_ROWS = [
    "...r.....",
    "...#.....",
    "..##..r..",
    "..##..#..",
    "..##.##..",
    ".###.###.",
    "#########",
    "#########",
]
NEEDLE_ROWS = [
    "..r..",
    "..#..",
    "#o#o#",
    ".###.",
    "..#..",
    "..#..",
    ".#.#.",
    ".#.#.",
]

# Sprites, nose to the right: "#" body, "w" wing (where the nav light goes),
# "t" tail (strobe), "n" nose (landing light)
AIRLINER = {
    "near": ["#........",
             "##.......",
             "t#######n",
             "....ww..."],
    "mid": ["#....",
            "t###n",
            "..w.."],
    "far": ["tw#n"],
}
LIGHT_PLANE = {
    "near": ["..www.",
             "t####n",
             "...#.#"],
    "mid": [".ww.",
            "t##n"],
    "far": ["twn"],
}
HELICOPTER = {
    "near": ["######",
             "...#..",
             "t#####",
             "...wn."],
    "mid": ["####",
            "t#wn"],
    "far": ["twn"],
}

# Short names for common types (ICAO designator: name); others come from
# the feed's description, or the designator itself
TYPE_NAMES = {
    "B712": "717", "B737": "737-700", "B738": "737-800", "B739": "737-900",
    "B37M": "737 MAX 7", "B38M": "737 MAX 8", "B39M": "737 MAX 9", "B3XM": "737 MAX 10",
    "B752": "757-200", "B753": "757-300", "B762": "767-200", "B763": "767-300", "B764": "767-400",
    "B772": "777-200", "B77L": "777-200LR", "B77W": "777-300ER", "B778": "777-8", "B779": "777-9",
    "B788": "787-8", "B789": "787-9", "B78X": "787-10", "B744": "747-400", "B748": "747-8",
    "A319": "A319", "A320": "A320", "A321": "A321", "A20N": "A320neo", "A21N": "A321neo",
    "A332": "A330-200", "A333": "A330-300", "A339": "A330-900", "A359": "A350-900",
    "A35K": "A350-1000", "A388": "A380", "BCS1": "A220-100", "BCS3": "A220-300",
    "E170": "E170", "E75L": "E175", "E75S": "E175", "E190": "E190", "E195": "E195",
    "DH8D": "Q400", "DH8C": "Dash 8", "CRJ2": "CRJ-200", "CRJ7": "CRJ-700", "CRJ9": "CRJ-900",
    "AT72": "ATR 72", "AT76": "ATR 72",
    "C150": "Cessna 150", "C152": "Cessna 152", "C172": "Cessna 172", "C182": "Cessna 182",
    "C208": "Caravan", "PC12": "Pilatus PC-12", "SR20": "Cirrus SR20", "SR22": "Cirrus SR22",
    "DHC2": "Beaver", "DHC3": "Otter", "DHC6": "Twin Otter", "P28A": "Piper", "BE36": "Bonanza",
    "M20P": "Mooney", "R22": "Robinson R22", "R44": "Robinson R44", "R66": "Robinson R66",
    "EC35": "EC135", "EC45": "H145", "AS50": "AStar", "B06": "Bell 206", "B407": "Bell 407",
    "P8": "P-8", "K35R": "KC-135", "KC46": "KC-46", "C17": "C-17", "C130": "C-130", "E6": "E-6",
}

# Airlines by ICAO prefix: (name, color), for when adsbdb has no route
AIRLINES = {
    "ASA": ("Alaska", (70, 190, 235)), "QXE": ("Horizon", (70, 190, 235)),
    "DAL": ("Delta", (235, 60, 70)), "UAL": ("United", (90, 140, 255)),
    "AAL": ("American", (200, 210, 225)), "SWA": ("Southwest", (255, 185, 40)),
    "SKW": ("SkyWest", (200, 200, 210)), "FFT": ("Frontier", (60, 205, 110)),
    "JBU": ("JetBlue", (80, 130, 255)), "HAL": ("Hawaiian", (210, 100, 230)),
    "NKS": ("Spirit", (240, 225, 40)), "SCX": ("Sun Country", (255, 150, 40)),
    "ACA": ("Air Canada", (240, 60, 60)), "JZA": ("Jazz", (240, 60, 60)),
    "WJA": ("WestJet", (60, 200, 200)), "FDX": ("FedEx", (170, 100, 255)),
    "UPS": ("UPS", (215, 165, 60)), "BOE": ("Boeing", (90, 150, 255)),
    "EJA": ("NetJets", (190, 190, 200)), "GTI": ("Atlas", (190, 190, 210)),
    "BAW": ("British", (110, 150, 235)), "DLH": ("Lufthansa", (245, 195, 40)),
    "KAL": ("Korean Air", (110, 175, 240)), "EVA": ("EVA Air", (70, 180, 120)),
    "CPA": ("Cathay", (70, 170, 150)), "JAL": ("JAL", (240, 60, 60)),
    "ANA": ("ANA", (70, 120, 235)), "ICE": ("Icelandair", (240, 200, 60)),
    "AFR": ("Air France", (90, 120, 230)), "KLM": ("KLM", (80, 175, 255)),
    "AMX": ("Aeromexico", (120, 150, 230)), "VOI": ("Volaris", (180, 80, 200)),
}

# Words dropped from adsbdb's airline names to fit the strip
AIRLINE_WORDS = {"airlines", "airline", "air lines", "airways", "inc", "inc.", "ltd", "corporation"}
CALLSIGN = re.compile(r"^([A-Z]{3})0*(\d[0-9A-Z]*)$")

UP_ARROW = [".#.",
            "###"]
DOWN_ARROW = ["###",
              ".#."]
ROUTE_ARROW = ["#..",
               "##.",
               "###",
               "##.",
               "#.."]


# --- Sun and Moon ------------------------------------------------------------


def julian_day(t: float) -> float:
    return t / 86400.0 + 2440587.5


def alt_az(ra: float, dec: float, jd: float) -> tuple[float, float]:
    """Elevation and compass azimuth (degrees) at home of a point at right
    ascension and declination (radians)."""
    d = jd - 2451545.0
    sidereal = math.radians((280.46061837 + 360.98564736629 * d + HOME_LON) % 360)
    hour = sidereal - ra
    lat = math.radians(HOME_LAT)
    el = math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour))
    az = math.atan2(-math.sin(hour), math.tan(dec) * math.cos(lat) - math.sin(lat) * math.cos(hour))
    return math.degrees(el), math.degrees(az) % 360


def equatorial(longitude: float, latitude: float, jd: float) -> tuple[float, float]:
    """Ecliptic to equatorial coordinates (radians)."""
    tilt = math.radians(23.439 - 4e-7 * (jd - 2451545.0))
    ra = math.atan2(math.sin(longitude) * math.cos(tilt) - math.tan(latitude) * math.sin(tilt), math.cos(longitude))
    dec = math.asin(math.sin(latitude) * math.cos(tilt) + math.cos(latitude) * math.sin(tilt) * math.sin(longitude))
    return ra, dec


def sun_longitude(jd: float) -> float:
    """The Sun's ecliptic longitude (radians), from the Astronomical Almanac's
    low-precision formula."""
    n = jd - 2451545.0
    g = math.radians(357.528 + 0.9856003 * n)
    return math.radians(280.460 + 0.9856474 * n + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))


def sun_position(t: float) -> tuple[float, float]:
    jd = julian_day(t)
    return alt_az(*equatorial(sun_longitude(jd), 0.0, jd), jd)


def moon_position(t: float) -> tuple[float, float, float]:
    """The Moon's elevation, azimuth and lit fraction, to a fraction of a
    degree (the Almanac's low-precision series, without parallax)."""
    jd = julian_day(t)
    c = (jd - 2451545.0) / 36525

    def s(a: float, b: float) -> float:
        return math.sin(math.radians(a + b * c))

    longitude = math.radians(218.32 + 481267.881 * c + 6.29 * s(135.0, 477198.87) - 1.27 * s(259.3, -413335.36)
                             + 0.66 * s(235.7, 890534.22) + 0.21 * s(269.9, 954397.74)
                             - 0.19 * s(357.5, 35999.05) - 0.11 * s(186.5, 966404.03))
    latitude = math.radians(5.13 * s(93.3, 483202.02) + 0.28 * s(228.2, 960400.89)
                            - 0.28 * s(318.3, 6003.15) - 0.17 * s(217.6, -407332.21))
    el, az = alt_az(*equatorial(longitude, latitude, jd), jd)
    elongation = math.acos(math.cos(latitude) * math.cos(longitude - sun_longitude(jd)))
    return el, az, (1 - math.cos(elongation)) / 2


# --- The view ----------------------------------------------------------------


def view_x(az: float) -> float:
    """Panel x of a compass direction (off the panel when out of view)."""
    middle = (VIEW_LEFT + VIEW_RIGHT) / 2
    return ((az - middle + 180) % 360 - 180 + (VIEW_RIGHT - VIEW_LEFT) / 2) / AZ_PER_PX - 0.5


def view_y(el: float) -> float:
    return HORIZON_Y - el / EL_PER_PX


def local(lat: float, lon: float) -> tuple[float, float]:
    """East and north of home, in meters."""
    return (lon - HOME_LON) * M_PER_LON, (lat - HOME_LAT) * M_PER_LAT


def look(east: float, north: float, up: float) -> tuple[float, float, float]:
    """Compass azimuth and elevation (degrees) and distance (m) from the
    window to a point, allowing for the Earth's curve."""
    ground = math.hypot(east, north)
    up -= ground * ground / (2 * EARTH_RADIUS) * (1 - REFRACTION)
    return (math.degrees(math.atan2(east, north)) % 360, math.degrees(math.atan2(up, ground)),
            math.hypot(ground, up))


def landmark_x(place: tuple[float, float]) -> float:
    east, north = local(*place)
    return view_x(math.degrees(math.atan2(east, north)) % 360)


def rgb(color) -> np.ndarray:
    return np.array(color, np.float32)


def mask(rows: list[str], chars: str = None) -> np.ndarray:
    """A sprite's opaque pixels (or just those in chars) as a 0/1 array."""
    return np.array([[c != "." if chars is None else c in chars for c in row] for row in rows], np.float32)


def stamp(frame: np.ndarray, alpha: np.ndarray, left: float, top: float, color) -> None:
    """Paint color through alpha (a 2D 0-1 mask) with its top-left corner at
    a sub-pixel position, spread over neighboring pixels so it glides."""
    x0, y0 = math.floor(left), math.floor(top)
    fx, fy = left - x0, top - y0
    h, w = alpha.shape
    a = np.zeros((h + 1, w + 1), np.float32)
    a[:h, :w] += alpha * ((1 - fx) * (1 - fy))
    a[:h, 1:] += alpha * (fx * (1 - fy))
    a[1:, :w] += alpha * ((1 - fx) * fy)
    a[1:, 1:] += alpha * (fx * fy)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w + 1, frame.shape[1]), min(y0 + h + 1, frame.shape[0])
    if xa >= xb or ya >= yb:
        return
    a = a[ya - y0:yb - y0, xa - x0:xb - x0, None]
    region = frame[ya:yb, xa:xb]
    region += (rgb(color) - region) * a


GLOW_DY, GLOW_DX = (a.ravel() for a in np.mgrid[-2:3, -2:3])


def glow(frame: np.ndarray, x: float, y: float, color, level: float = 1.0, spread: float = 0.35) -> None:
    """A light at a sub-pixel position, added to the frame: a bright core and,
    with a bigger spread, a soft halo."""
    if level <= 0:
        return
    xi = GLOW_DX + math.floor(x + 0.5)
    yi = GLOW_DY + math.floor(y + 0.5)
    ok = (yi >= 0) & (yi < frame.shape[0]) & (xi >= 0) & (xi < frame.shape[1])
    if not ok.any():
        return
    xi, yi = xi[ok], yi[ok]
    d2 = ((xi - x) ** 2 + (yi - y) ** 2).astype(np.float32)
    frame[yi, xi] += (np.exp(-d2 / spread) * level)[:, None] * rgb(color)


# --- Planes ------------------------------------------------------------------


def log(message: str) -> None:
    print(f"flights: {message}", file=sys.stderr, flush=True)


class Plane:
    """One aircraft: what it is, and where it's drawn, flown on from its last
    report and eased onto each new one."""

    def __init__(self, report: dict) -> None:
        self.id = report["hex"]
        self.anchor = (0.0, 0.0, 0.0)  # east, north, up at anchor_time
        self.anchor_time = 0.0
        self.velocity = (0.0, 0.0, 0.0)
        self.error = (0.0, 0.0, 0.0)  # drawn minus reported, fading out
        self.error_time = 0.0
        self.heard = 0.0
        self.facing = 1
        self.phase = random.uniform(0, 1)  # its strobes' timing
        self.trail: list[tuple[float, float]] = []  # recent (az, el), for contrails
        self.trail_time = 0.0
        self.named = -math.inf  # when the strip last named it
        self.update(report, None)

    def update(self, report: dict, now: float | None) -> None:
        self.callsign = (report.get("flight") or "").strip()
        self.registration = (report.get("r") or "").strip()
        self.type = (report.get("t") or "").strip()
        self.description = (report.get("desc") or "").strip()
        self.owner = (report.get("ownOp") or "").strip()
        self.category = report.get("category") or ""
        self.feet = report["feet"]
        self.climb = report.get("geom_rate", report.get("baro_rate")) or 0
        speed = (report.get("gs") or 0) * KNOTS
        track = math.radians(report.get("track") or 0)
        position = (*local(report["lat"], report["lon"]), self.feet * METERS_PER_FOOT - HOME_HEIGHT)
        old = self.position(now) if now is not None else None
        self.anchor, self.anchor_time = position, report["at"]
        self.velocity = (speed * math.sin(track), speed * math.cos(track), self.climb * FEET_PER_MINUTE)
        self.heard = report["at"]
        if old is None:
            self.error = (0.0, 0.0, 0.0)
            return
        new = self.position(now, settled=True)
        error = tuple(o - n for o, n in zip(old, new))
        self.error = error if math.dist(old, new) < JUMP_METERS else (0.0, 0.0, 0.0)
        self.error_time = now

    def position(self, now: float, settled: bool = False) -> tuple[float, float, float]:
        dt = now - self.anchor_time
        fade = 0.0 if settled else math.exp(-max(0.0, now - self.error_time) / SETTLE_SECONDS)
        return tuple(a + v * dt + e * fade for a, v, e in zip(self.anchor, self.velocity, self.error))

    def kind(self) -> dict:
        if self.category == "A7":
            return HELICOPTER
        if self.category in ("A1", "A2", "B1", "B4"):
            return LIGHT_PLANE
        return AIRLINER

    def landing_lights(self) -> bool:
        return self.feet < TAKEOFF_FEET or (self.feet < LANDING_FEET and self.climb < -300)


class Feed:
    """The planes nearby, polled from adsb.fi by a background thread, and
    their routes, looked up from adsbdb when asked for. Readers take
    `reports` (replaced whole on each poll) and `routes`."""

    def __init__(self) -> None:
        self.reports: list[dict] = []
        self.polled = 0  # counts polls, so readers see new reports
        self.polled_at = -math.inf
        self.routes: dict[str, dict | None] = {}
        self.route_times: dict[str, float] = {}
        self.wanted: list[str] = []  # callsigns to look up, most wanted first
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.load_routes()
        threading.Thread(target=self.poll_loop, name="flights-feed", daemon=True).start()
        threading.Thread(target=self.route_loop, name="flights-routes", daemon=True).start()

    def poll_loop(self) -> None:
        retry, error = FETCH_RETRY, None
        while True:
            started = time.monotonic()
            try:
                self.poll()
            except Exception as e:  # keep the last reports; try again later
                message = f"{type(e).__name__}: {e}"
                if message != error:
                    log(f"feed failed, retrying: {message}")
                error = message
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)
                continue
            if error:
                log("feed recovered")
            retry, error = FETCH_RETRY, None
            time.sleep(max(0.5, POLL_SECONDS - (time.monotonic() - started)))

    def poll(self) -> None:
        url = FEED_URL.format(lat=HOME_LAT, lon=HOME_LON, dist=FEED_RADIUS_NM)
        r = self.session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # (adsb.fi gives "now" in seconds; readsb's own format, in milliseconds)
        now = data.get("now") or time.time()
        if now > 1e11:
            now /= 1000
        reports = []
        for a in data.get("aircraft") or data.get("ac") or []:
            feet = a.get("alt_geom", a.get("alt_baro"))
            if not isinstance(feet, (int, float)) or feet < -300 or a.get("lat") is None or a.get("hex") is None:
                continue  # on the ground, or no position
            seen = a.get("seen_pos", 0) or 0
            if seen > STALE_SECONDS:
                continue
            a["feet"] = feet
            a["at"] = now - seen
            reports.append(a)
        self.reports = reports
        self.polled_at = time.time()
        self.polled += 1

    # Routes ---------------------------------------------------------------

    def route(self, callsign: str) -> dict | None:
        """The route for callsign if known (or known to be missing), else
        None after asking for it."""
        if callsign in self.routes and time.time() - self.route_times.get(callsign, 0) < (
                ROUTE_KEEP if self.routes[callsign] else ROUTE_KEEP_MISSING):
            return self.routes[callsign]
        if callsign not in self.wanted:
            self.wanted.insert(0, callsign)
            del self.wanted[20:]
        return None

    def route_loop(self) -> None:
        retry = FETCH_RETRY
        while True:
            time.sleep(ROUTE_GAP)
            if not self.wanted:
                continue
            callsign = self.wanted[0]
            try:
                route = self.lookup(callsign)
            except Exception as e:
                log(f"route lookup failed for {callsign}: {type(e).__name__}: {e}")
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)
                continue
            retry = FETCH_RETRY
            self.routes[callsign] = route
            self.route_times[callsign] = time.time()
            if callsign in self.wanted:
                self.wanted.remove(callsign)
            self.save_routes()

    def lookup(self, callsign: str) -> dict | None:
        r = self.session.get(ROUTE_URL.format(callsign=callsign), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        found = r.json().get("response")
        if not isinstance(found, dict) or "flightroute" not in found:
            return None
        route = found["flightroute"]
        origin, destination = route.get("origin") or {}, route.get("destination") or {}
        try:
            return {
                "airline": (route.get("airline") or {}).get("name"),
                "from": origin["iata_code"], "to": destination["iata_code"],
                "from_at": (origin["latitude"], origin["longitude"]),
                "to_at": (destination["latitude"], destination["longitude"]),
            }
        except (KeyError, TypeError):
            return None

    def load_routes(self) -> None:
        try:
            saved = json.loads((CACHE_DIR / "routes.json").read_text())
            self.routes = saved["routes"]
            self.route_times = saved["times"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def save_routes(self) -> None:
        now = time.time()
        keep = [c for c, at in self.route_times.items() if now - at < ROUTE_KEEP]
        path = CACHE_DIR / "routes.json"
        try:
            path.with_suffix(".tmp").write_text(json.dumps({
                "routes": {c: self.routes[c] for c in keep if c in self.routes},
                "times": {c: self.route_times[c] for c in keep},
            }))
            path.with_suffix(".tmp").replace(path)
        except OSError as e:
            log(f"couldn't save routes: {e}")


def off_route(route: dict, lat: float, lon: float) -> float:
    """How far (m) a point is from the great circle between a route's ends
    (or past either end)."""
    p1, p2, p3 = (tuple(map(math.radians, p)) for p in (route["from_at"], route["to_at"], (lat, lon)))

    def distance(a, b):
        h = math.sin((b[0] - a[0]) / 2) ** 2 + math.cos(a[0]) * math.cos(b[0]) * math.sin((b[1] - a[1]) / 2) ** 2
        return 2 * math.asin(min(1.0, math.sqrt(h)))

    def bearing(a, b):
        y = math.sin(b[1] - a[1]) * math.cos(b[0])
        x = math.cos(a[0]) * math.sin(b[0]) - math.sin(a[0]) * math.cos(b[0]) * math.cos(b[1] - a[1])
        return math.atan2(y, x)

    d13, length = distance(p1, p3), distance(p1, p2)
    cross = math.asin(max(-1.0, min(1.0, math.sin(d13) * math.sin(bearing(p1, p3) - bearing(p1, p2)))))
    along = math.acos(max(-1.0, min(1.0, math.cos(d13) / max(math.cos(cross), 1e-9))))
    if math.cos(bearing(p1, p3) - bearing(p1, p2)) < 0:
        along = -along
    beyond = max(0.0, -along, along - length)
    return EARTH_RADIUS * max(abs(cross), beyond)


def short_airline(name: str) -> str:
    words = [w for w in name.split() if w.lower() not in AIRLINE_WORDS]
    text = " ".join(words) or name
    return text.replace(" Air Lines", "").strip()


def type_name(plane: Plane) -> str:
    if plane.type in TYPE_NAMES:
        return TYPE_NAMES[plane.type]
    if plane.description:
        text = re.sub(r"\(.*?\)", "", plane.description).strip()
        return " ".join(w if any(ch.isdigit() for ch in w) else w.title() for w in text.split())
    return plane.type


# --- The scene ---------------------------------------------------------------


class Flights:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the view is drawn for a {WIDTH}x{HEIGHT} panel")
        self.frame_count = 0
        self.frame_clock = os.environ.get("FLIGHTS_FRAME_CLOCK") == "1"
        self.start_time = time.time()
        self.font = BitmapFont("tom-thumb")
        self.text_cache: dict = {}
        self.icons = {name: mask(rows) for name, rows in
                      (("up", UP_ARROW), ("down", DOWN_ARROW), ("route", ROUTE_ARROW))}
        self.sprites = {}
        for kind in (AIRLINER, LIGHT_PLANE, HELICOPTER):
            for size, rows in kind.items():
                right = {"body": mask(rows, "#wtn"), "rows": rows}
                left = {"body": mask([r[::-1] for r in rows], "#wtn"), "rows": [r[::-1] for r in rows]}
                self.sprites[id(kind), size] = (right, left)
        self.make_horizon()
        rng = random.Random(7)
        self.stars = [(rng.uniform(0, WIDTH - 1), rng.uniform(0, HORIZON_ROW - 4), rng.uniform(0.3, 1.0),
                       rng.uniform(0, math.tau)) for _ in range(STARS)]
        self.sky = None
        self.sky_time = -math.inf
        self.light = 1.0  # daylight, 0 (night) to 1
        self.sun = self.moon = None

        self.planes: dict[str, Plane] = {}
        self.polled = 0
        self.featured: Plane | None = None
        self.featured_since = -math.inf
        self.card_time = -math.inf
        self.feed = Feed()

    def now(self) -> float:
        if self.frame_clock:
            return self.start_time + self.frame_count * FRAME_SECONDS
        return time.time()

    # The horizon ----------------------------------------------------------

    def make_horizon(self) -> None:
        """Heights (in rows above the horizon) of the trees, the city and the
        mountains along the horizon, and where the lights are."""
        rng = random.Random(3)
        x = np.arange(WIDTH)
        az = VIEW_LEFT + (x + 0.5) * AZ_PER_PX
        # The Olympics, across the Sound: a ragged ridge with a few peaks
        ridge = np.zeros(WIDTH)
        for peak_az, height, width in OLYMPICS:
            ridge = np.maximum(ridge, height - np.abs(az - peak_az) / width)
        ridge += np.array([rng.uniform(-0.3, 0.3) for _ in x])
        self.mountains = np.clip(np.round(ridge), 0, None).astype(int)
        # Trees along Beacon Hill, in clumps
        trees = []
        level = 1.5
        for _ in x:
            level = min(2.4, max(0.6, level + rng.uniform(-0.7, 0.7)))
            trees.append(round(level))
        self.trees = np.array(trees)
        # Downtown to the northwest, with the Space Needle just right of it
        # (in truth it stands right behind downtown from here)
        city = {}  # (column, row): sprite character
        left = round(landmark_x(DOWNTOWN)) - len(DOWNTOWN_ROWS[0]) // 2
        for rows, x0 in ((DOWNTOWN_ROWS, left), (NEEDLE_ROWS, left + len(DOWNTOWN_ROWS[0]) + 1)):
            for r, row in enumerate(rows):
                for c, ch in enumerate(row):
                    if ch != "." and 0 <= x0 + c < WIDTH:
                        city[x0 + c, HORIZON_ROW + 1 - len(rows) + r] = ch
        self.city = city
        self.windows = [p for p, ch in city.items() if ch == "o" or (ch == "#" and rng.random() < 0.3)]
        self.tops = [p for p, ch in city.items() if ch == "r"]
        self.beacons = [round(landmark_x(SEA_TAC)), round(landmark_x(BOEING_FIELD))]
        self.houses = [(c, HORIZON_ROW) for c in rng.sample(range(WIDTH), 9)]
        # The top row of the scenery in each column (planes below it are hidden)
        skyline = HORIZON_ROW + 1 - self.trees
        for (c, r) in city:
            skyline[c] = min(skyline[c], r)
        self.skyline = skyline

    def draw_horizon(self, frame: np.ndarray, t: float) -> None:
        light = self.light
        dark = 1 - light
        # The Olympics: blue with snowy peaks by day, dark against a sunset
        sun_el = self.sun[0]
        day = min(1.0, max(0.0, (sun_el - 3) / 10))
        dusk = min(1.0, max(0.0, (sun_el + 9) / 6))
        horizon = self.sky[HORIZON_ROW - 1]
        for c in range(WIDTH):
            height = self.mountains[c]
            if height and dusk > 0:
                color = (horizon[c] * 0.55 + (rgb(MOUNTAIN_DAY) - horizon[c] * 0.55) * day)
                color = horizon[c] + (color - horizon[c]) * dusk
                frame[HORIZON_ROW + 1 - height:HORIZON_ROW + 1, c] = color
                # Snow on the peaks
                peak = height >= 4 and height >= max(self.mountains[max(c - 1, 0):c + 2])
                if peak:
                    top = HORIZON_ROW + 1 - height
                    frame[top, c] = color + (rgb(SNOW) - color) * 0.7 * day
        trees = rgb(TREES_NIGHT) + (rgb(TREES_DAY) - rgb(TREES_NIGHT)) * light
        towers = rgb(TOWERS_NIGHT) + (rgb(TOWERS_DAY) - rgb(TOWERS_NIGHT)) * light
        for (c, r) in self.city:
            frame[r, c] = towers
        for c in range(WIDTH):
            frame[HORIZON_ROW + 1 - self.trees[c]:HORIZON_ROW + 1, c] = trees
        if dark > 0.3:
            fade = min(1.0, (dark - 0.3) / 0.4)
            for c, r in self.windows:
                frame[r, c] = np.maximum(frame[r, c], rgb(WINDOW_LIGHT) * fade)
            for c, r in self.houses:
                frame[r, c] = np.maximum(frame[r, c], rgb(HOUSE_LIGHT) * fade * 0.6)
            # Red lights atop the tallest towers and the Needle, slowly blinking
            blink = 0.5 + 0.5 * math.cos(t * math.tau / 2.0)
            for c, r in self.tops:
                frame[r, c] = np.maximum(frame[r, c], rgb(TOWER_BEACON) * blink * fade)
            # Airport beacons: a white flash, then a green one
            for i, c in enumerate(self.beacons):
                phase = (t / 2.0 + i * 0.37) % 1.0
                if 0 <= c < WIDTH and phase < 0.12:
                    frame[HORIZON_ROW - self.trees[c], c] = rgb(STROBE) * 0.8 * fade
                elif 0 <= c < WIDTH and 0.5 <= phase < 0.62:
                    frame[HORIZON_ROW - self.trees[c], c] = rgb(GREEN_LIGHT) * 0.8 * fade

    # The sky --------------------------------------------------------------

    def update_sky(self, now: float) -> None:
        """The sky's colors, the Sun and the Moon, every few seconds."""
        if abs(now - self.sky_time) < 10 and self.sky is not None:
            return
        self.sky_time = now
        sun_el, sun_az = sun_position(now)
        self.sun = (sun_el, sun_az)
        self.moon = moon_position(now)
        keys = SKY_KEYS
        s = min(max(sun_el, keys[0][0]), keys[-1][0])
        i = next(i for i in range(len(keys) - 1) if keys[i + 1][0] >= s)
        f = (s - keys[i][0]) / (keys[i + 1][0] - keys[i][0])
        zenith, horizon, sun_glow = (rgb(a) + (rgb(b) - rgb(a)) * f for a, b in zip(keys[i][1:], keys[i + 1][1:]))
        rows, cols = np.mgrid[0:HORIZON_ROW + 1, 0:WIDTH]
        el = (HORIZON_Y - rows) * EL_PER_PX
        az = VIEW_LEFT + (cols + 0.5) * AZ_PER_PX
        blend = np.exp(-el / HORIZON_FALLOFF)[..., None]
        sky = zenith + (horizon - zenith) * blend
        # Glow around the Sun (angular distance on the sphere)
        se, sa = math.radians(sun_el), math.radians(sun_az)
        e, a = np.radians(el), np.radians(az)
        cos_d = np.sin(e) * math.sin(se) + np.cos(e) * math.cos(se) * np.cos(a - sa)
        distance = np.degrees(np.arccos(np.clip(cos_d, -1, 1)))
        sky += np.exp(-distance / GLOW_WIDTH)[..., None] * sun_glow
        self.sky = sky.astype(np.float32)
        self.light = min(1.0, max(0.0, (sun_el + 8) / 10))

    def draw_sky(self, frame: np.ndarray, t: float) -> None:
        frame[:HORIZON_ROW + 1] = self.sky
        stars = min(1.0, max(0.0, (-self.sun[0] - 7) / 5))
        if stars > 0:
            for x, y, level, phase in self.stars:
                twinkle = 0.75 + 0.25 * math.sin(t * 1.7 + phase)
                frame[round(y), round(x)] += rgb(STAR_COLOR) * (level * twinkle * stars)
        moon_el, moon_az, lit = self.moon
        x = view_x(moon_az)
        if moon_el > -1 and -2 < x < WIDTH + 1:
            level = (0.35 + 0.65 * lit) * (0.45 if self.light > 0.6 else 1.0)
            glow(frame, x, view_y(moon_el + 0.5), MOON_COLOR, level, spread=0.9)
        sun_el, sun_az = self.sun
        x = view_x(sun_az)
        if sun_el > -1.5 and -2 < x < WIDTH + 1:
            low = min(1.0, max(0.0, (8 - sun_el) / 8))
            color = rgb(SUN_COLOR) + (rgb(SUNSET_COLOR) - rgb(SUN_COLOR)) * low
            glow(frame, x, view_y(sun_el + 0.5), color, 1.0, spread=1.1)

    # Planes ---------------------------------------------------------------

    def track(self, now: float) -> None:
        """Bring the planes up to date with the feed's latest reports."""
        if self.feed.polled != self.polled:
            self.polled = self.feed.polled
            for report in self.feed.reports:
                plane = self.planes.get(report["hex"])
                if plane is None:
                    self.planes[report["hex"]] = Plane(report)
                elif report["at"] > plane.heard:
                    plane.update(report, now)
        for key in [k for k, p in self.planes.items() if now - p.heard > STALE_SECONDS]:
            del self.planes[key]

    def in_view(self, now: float) -> list[tuple[float, Plane, float, float]]:
        """(distance, plane, x, y) for each plane in view, nearest first."""
        seen = []
        for plane in self.planes.values():
            east, north, up = plane.position(now)
            if abs(east) > MAX_RANGE or abs(north) > MAX_RANGE:
                continue
            az, el, distance = look(east, north, up)
            x, y = view_x(az), view_y(el)
            if distance > MAX_RANGE or not (-3 < x < WIDTH + 2) or y < -2 or el < -0.5:
                continue
            # The way it's going across the view, for which way it faces
            ahead = look(*(p + v for p, v in zip((east, north, up), plane.velocity)))
            dx = view_x(ahead[0]) - x
            if abs(dx) > 0.02:
                plane.facing = 1 if dx > 0 else -1
            plane.az, plane.el = az, el
            seen.append((distance, plane, x, y))
        seen.sort(key=lambda s: s[0])
        return seen

    def hidden(self, x: float, y: float) -> bool:
        """Whether a point is behind the trees, towers or out of view."""
        c = round(x)
        return not (0 <= c < WIDTH) or y >= self.skyline[c] - 0.5 or y < 0

    def draw_contrail(self, frame: np.ndarray, plane: Plane, now: float) -> None:
        if plane.feet < CONTRAIL_FEET or self.light < 0.4:
            plane.trail.clear()
            return
        if now - plane.trail_time >= CONTRAIL_EVERY:
            plane.trail_time = now
            plane.trail.append((plane.az, plane.el))
            del plane.trail[:-CONTRAIL_SAMPLES]
        n = len(plane.trail)
        for i, (az, el) in enumerate(plane.trail[:-1]):
            x, y = view_x(az), view_y(el)
            if not self.hidden(x, y):
                fade = (i + 1) / n
                stamp(frame, ONE * (0.5 * fade * self.light), x, y, CONTRAIL)

    def draw_plane(self, frame: np.ndarray, plane: Plane, x: float, y: float, distance: float, t: float) -> tuple:
        """Draw a plane centered at (x, y); returns its box (left, top, right, bottom)."""
        size = "near" if distance < NEAR else "mid" if distance < MID else "far"
        kind = plane.kind()
        sprite = self.sprites[id(kind), size][plane.facing < 0]
        body = sprite["body"]
        h, w = body.shape
        left, top = x - (w - 1) / 2, y - (h - 1) / 2
        light = self.light
        color = rgb(PLANE_NIGHT) + (rgb(PLANE_DAY) - rgb(PLANE_NIGHT)) * light
        # Far off, a plane is a faint speck by day and only lights by night
        presence = 1.0 if size != "far" else 0.6
        stamp(frame, body * presence, left, top, color)
        night = 1 - light
        lights = 0.55 + 0.45 * night
        # Navigation light on the near wingtip: green heading right (north,
        # its right side toward the window), red heading left
        nav = GREEN_LIGHT if plane.facing > 0 else RED_LIGHT
        strobe_on = (t + plane.phase) % 1.1 < 0.07 or 0.2 < (t + plane.phase) % 1.1 < 0.27
        beacon_on = (t * 0.9 + plane.phase * 0.5) % 1.0 < 0.15
        for r, row in enumerate(sprite["rows"]):
            for c, ch in enumerate(row):
                px, py = left + c, top + r
                if ch == "w":
                    glow(frame, px, py, nav, 0.5 * lights)
                elif ch == "t":
                    if strobe_on:
                        glow(frame, px, py, STROBE, lights, spread=0.5)
                    elif beacon_on:
                        glow(frame, px, py, RED_LIGHT, 0.7 * lights)
                elif ch == "n" and plane.landing_lights():
                    glow(frame, px, py, LANDING_LIGHT, 0.5 + 0.6 * night, spread=0.6 + 0.6 * night)
        return left - 1, top - 1, left + w, top + h

    def feature(self, seen: list, now: float) -> None:
        """Pick the plane the strip names: a close one while it's close, else
        each in view in turn, the least recently named first."""
        visible = {plane.id: (distance, plane) for distance, plane, x, y in seen
                   if not self.hidden(x, y)}
        current = self.featured
        if current is not None and current.id in visible:
            distance = visible[current.id][0]
            if distance < CLOSE or now - self.featured_since < FEATURE_SECONDS:
                return
        if not visible:
            if current is not None:
                self.featured = None
                self.card_time = now
            return
        close = [p for d, p in visible.values() if d < CLOSE]
        if close:
            choice = close[0]
        else:
            choice = min(visible.values(), key=lambda dp: (dp[1].named, dp[0]))[1]
        if choice is not current:
            self.featured = choice
            self.card_time = now
        choice.named = now
        self.featured_since = now

    def draw_marker(self, frame: np.ndarray, box: tuple, color, t: float) -> None:
        """Corner ticks around the named plane, gently pulsing."""
        left, top, right, bottom = box
        level = 0.55 + 0.25 * math.sin(t * math.tau / 1.6)
        for cx, cy in ((left, top), (right, top), (left, bottom), (right, bottom)):
            c, r = round(cx), round(cy)
            if 0 <= c < WIDTH and 0 <= r < STRIP_TOP and r < self.skyline[c]:
                frame[r, c] = frame[r, c] + (rgb(color) - frame[r, c]) * level

    # The strip ------------------------------------------------------------

    def text(self, text: str) -> np.ndarray:
        found = self.text_cache.get(text)
        if found is None:
            if len(self.text_cache) > 300:
                self.text_cache.clear()
            found = self.text_cache[text] = self.font.render(text, proportional=True)
        return found

    def put(self, frame: np.ndarray, art: np.ndarray, x: int, row: int, color, level: float = 1.0) -> None:
        """Paint a mask (text or icon) at column x of a strip line (0 or 1)."""
        top = STRIP_TOP + row * self.font.height
        if art.shape[0] < self.font.height:  # an icon: on the text's baseline
            top += self.font.ascent - art.shape[0]
        w = min(art.shape[1], WIDTH - x)
        if w <= 0 or x < 0:
            return
        frame[top:top + art.shape[0], x:x + w] += art[:, :w, None] * (rgb(color) * level)

    def card(self, plane: Plane, now: float) -> tuple[list, list]:
        """The strip's two lines for a plane: ([(mask, color, x)], ...)."""
        name, color = None, DEFAULT_AIRLINE
        match = CALLSIGN.match(plane.callsign)
        route = None
        if match:
            prefix, number = match.groups()
            found = self.feed.route(plane.callsign)
            if found:
                lat = HOME_LAT + plane.anchor[1] / M_PER_LAT
                lon = HOME_LON + plane.anchor[0] / M_PER_LON
                if off_route(found, lat, lon) < ROUTE_SLACK:
                    route = found
            known = AIRLINES.get(prefix)
            airline = known[0] if known else short_airline(found["airline"]) if found and found["airline"] else prefix
            if known:
                color = known[1]
            name = f"{airline} {number}"
        if not name:
            name = plane.registration or plane.callsign or plane.id.upper()
            if plane.owner and not plane.registration.startswith("N") and len(plane.owner) < 14:
                name = plane.owner.title()
        if self.text_width(name) > WIDTH - 2 * MARGIN:
            name = plane.callsign or name
        first = [(self.text(name), color, MARGIN)]

        # Altitude, with an arrow climbing or descending, at the right
        altitude = f"{round(plane.feet / 100) * 100}ft"
        right = [self.text(altitude)]
        climbing = self.icons["up"] if plane.climb > 300 else self.icons["down"] if plane.climb < -300 else None
        alt_w = right[0].shape[1]
        second = []
        x = WIDTH - MARGIN - alt_w
        second.append((right[0], TEXT, x))
        if climbing is not None:
            x -= climbing.shape[1] + 1
            second.append((climbing, CLIMB_COLOR if plane.climb > 0 else DESCEND_COLOR, x))
        room = x - 2 - MARGIN
        # On the left, the route and the type, taking turns
        options = []
        if route:
            options.append(("route", route))
        kind = type_name(plane)
        if kind:
            words = kind.split()
            while words and self.text_width(" ".join(words)) > room:
                words.pop()
            if not words and self.text_width(plane.type) <= room:
                words = [plane.type]
            if words:
                options.append(("type", " ".join(words)))
        if options:
            which, value = options[int(now / LINE_SECONDS) % len(options)]
            if which == "route":
                a, b = self.text(value["from"]), self.text(value["to"])
                arrow = self.icons["route"]
                second += [(a, TEXT, MARGIN), (arrow, DIM_TEXT, MARGIN + a.shape[1] + 1),
                           (b, TEXT, MARGIN + a.shape[1] + arrow.shape[1] + 2)]
            else:
                second.append((self.text(value), DIM_TEXT, MARGIN))
        return first, second

    def text_width(self, text: str) -> int:
        return self.text(text).shape[1]

    def draw_strip(self, frame: np.ndarray, now: float) -> None:
        fade = min(1.0, (now - self.card_time) / CARD_FADE_SECONDS)
        if self.featured is not None:
            first, second = self.card(self.featured, now)
        else:
            clock = datetime.fromtimestamp(now, TIMEZONE)
            label = f"{clock.hour % 12 or 12}:{clock.minute:02d} {'AM' if clock.hour < 12 else 'PM'}"
            quiet = "no flight data" if time.time() - self.feed.polled_at > NO_DATA_SECONDS else "no planes in view"
            t1, t2 = self.text(label), self.text(quiet)
            first = [(t1, TEXT, (WIDTH - t1.shape[1]) // 2)]
            second = [(t2, DIM_TEXT, (WIDTH - t2.shape[1]) // 2)]
        for row, parts in enumerate((first, second)):
            for art, color, x in parts:
                self.put(frame, art, x, row, color, fade)

    # The frame ------------------------------------------------------------

    def frame(self) -> np.ndarray:
        now = self.now()
        t = self.frame_count * FRAME_SECONDS
        self.update_sky(now)
        frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        self.draw_sky(frame, t)

        self.track(now)
        seen = self.in_view(now)
        self.feature(seen, now)
        marker = None
        # Farthest first, so nearer planes pass in front
        for distance, plane, x, y in reversed(seen):
            self.draw_contrail(frame, plane, now)
            box = self.draw_plane(frame, plane, x, y, distance, t)
            if plane is self.featured:
                marker = box
        self.draw_horizon(frame, t)
        if marker is not None:
            color = DEFAULT_AIRLINE
            match = CALLSIGN.match(self.featured.callsign)
            if match and match.group(1) in AIRLINES:
                color = AIRLINES[match.group(1)][1]
            self.draw_marker(frame, marker, color, t)
        frame[STRIP_TOP:] = 0
        self.draw_strip(frame, now)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


ONE = np.ones((1, 1), np.float32)

SCENE = Flights


def main() -> None:
    run_scene(Flights, "Flights out the west window LED matrix display")


if __name__ == "__main__":
    main()
