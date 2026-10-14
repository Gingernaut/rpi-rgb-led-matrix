#!/usr/bin/env python
"""ISS tracker: where the International Space Station is right now over a
day/night world map, and when it can next be seen from Seattle.

The ISS's orbit comes from its published elements (a TLE from CelesTrak,
fetched twice a day) and is worked out locally with SGP4, so the station
glides smoothly along its ground track: the past few minutes trail behind it
and dots march along where it's headed. Seattle is a pink dot inside a faint
ring: when the ISS is inside that ring it's at least MIN_ELEVATION above
Seattle's horizon.

The strip along the bottom shows the next pass visible from Seattle (the ISS
high enough, lit by the Sun, against a dark sky) with the forecast cloud
cover from Open-Meteo, alternating with where it appears and disappears, how
high it climbs, and for how long. Within SOON_SECONDS it counts down, and
while a pass is up the whole panel turns into a sky view: the pass's arc
across a compass dome, where to look, how high, and how long is left. A pass
forecast to be clouded over (CLOUDY_PERCENT) is listed, grayed out, but
doesn't take over the panel.

For previews, ISS_START (ISO time, e.g. 2026-10-15T19:40-07:00) runs the
scene from that moment instead of now, at ISS_SPEED times real time, keeping
time by frames like the other scenes.

Requires the rgbmatrix Python bindings (make build-python), numpy and sgp4.
"""

import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from sgp4.api import Satrec

from matrix_common import REFRESH_HZ, BitmapFont, run_scene
from planets import WORLD

FRAME_SECONDS = 2 / REFRESH_HZ  # every 2nd refresh, ~37.5 fps

# Where the passes are seen from
OBSERVER = "Seattle"
OBSERVER_LAT, OBSERVER_LON = 47.6062, -122.3321
OBSERVER_HEIGHT_KM = 0.05
TIMEZONE = ZoneInfo("America/Los_Angeles")

# A pass is visible while the ISS is at least MIN_ELEVATION up, sunlit, and
# the sky is dark (the Sun at least 6 degrees down: civil twilight's end).
# Only passes peaking at MIN_PEAK or higher are listed: lower ones are short,
# faint and usually behind trees and buildings.
MIN_ELEVATION = 10.0
DARK_SUN_ELEVATION = -6.0
MIN_PEAK = 20.0
MIN_VISIBLE_SECONDS = 30
PREDICT_DAYS = 10
PREDICT_STEP = 10  # seconds between samples when searching for passes
PREDICT_EVERY = 3600  # re-predict hourly, as the orbit elements age

# Cloud cover (percent) at or under CLEAR_PERCENT shows green, under
# CLOUDY_PERCENT yellow, and from CLOUDY_PERCENT up gray, as clouded out
CLEAR_PERCENT = 30
CLOUDY_PERCENT = 70

# Count down to a pass this far ahead; switch to the sky view PREVIEW_SECONDS
# before the ISS rises into view, and back AFTER_SECONDS after it's gone
SOON_SECONDS = 30 * 60
PREVIEW_SECONDS = 60
AFTER_SECONDS = 5
SKY_FADE_SECONDS = 1.0

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
TLE_MAX_AGE = 12 * 3600  # CelesTrak updates every few hours and blocks frequent fetches
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_MAX_AGE = 3600
FORECAST_STALE = 24 * 3600  # ignore a cached forecast older than this
FETCH_RETRY = 300  # seconds to wait after a failed fetch
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "iss"

WIDTH, HEIGHT = 64, 32
# The map covers 90N to 56.25S (the ISS never goes south of 51.6S) in rows
# 0-25, 5.625 degrees per pixel; the text strip takes the rows below
MAP_ROWS = 26
STRIP_TOP = MAP_ROWS

# Land coverage per map pixel, 0-9, from Natural Earth's 1:50m land polygons
LAND = [
    "0000000000000000000000000000000000000000000000000000000000000000",
    "0000000000000114667565667887520000232000112100003310000000000000",
    "0000000000454467772346999999500000221000013210135686201022100000",
    "2015654345688666788610699986100000246521254487999999999779765444",
    "4348999999999998644860398301420003878978999999999999999999999998",
    "0026733699999994007750022000001317938999999999999999999985446320",
    "1011000049999999969997100000005737879999999999999999999972034000",
    "0000000003999999999974300000000499999889879999999999999973110000",
    "0000000001999999999410000000004754578448559999999999999414000000",
    "0000000000699999995000000000003756224689869999999999945351000000",
    "0000000000069997660000000000017999766589899999999999960200000000",
    "0000000000015960032000000000069999999969966599999999930000000000",
    "0000100000000584414310000000089999999977993029935984020000000000",
    "0000000000000013710101000001099999999997520007301594041000000000",
    "0000000000000000137984000000039989999999800002301320134000000000",
    "0000000000000000018999810000001105999999200000000543822000000000",
    "0000000000000001049999985200000002999993000000000163552464210000",
    "0000000000000000019999999700000000699991000000000013232125412000",
    "0000000000000000005999999200000000799993500000000000004835000100",
    "1000000000000000000599998000000000799942700000000000269999300101",
    "0000000000000000000599961000000000399811200000000000899999920000",
    "0000000000000000000799810000000000089400000000000000597799920000",
    "0000000000000000000997100000000000010000000000000000110038500013",
    "0000000000000000002960000000000000000000000000000000000001200041",
    "0000000000000000004910000000000000000000000010000000000000000110",
    "0000000000000000002711000100000000000000000000000000000000000000",
]

# Map colors by day; the night side keeps NIGHT of each channel
OCEAN = (0, 26, 80)
LAND_GREEN = (30, 95, 25)
DESERT = (140, 105, 50)
ICE = (120, 130, 140)
NIGHT = (0.16, 0.18, 0.3)
# Ground track: the past TRAIL_MINUTES fading out behind the ISS, and dots
# marching TRACK_MINUTES ahead
TRAIL_MINUTES = 40
TRACK_MINUTES = 50
TRACK_STEP = 20  # seconds between track points (about 0.25 px)
TRAIL_COLOR = (255, 90, 20)
TRACK_COLOR = (110, 110, 130)
ISS_COLOR = (255, 255, 235)
HALO_COLOR = (255, 190, 70)
HOME_COLOR = (255, 40, 140)
RING_COLOR = (70, 12, 40)

# Text colors
WHITE = (230, 230, 230)
GRAY = (90, 90, 90)
LABEL_GRAY = (55, 55, 65)
GOLD = (255, 170, 0)
CYAN = (60, 190, 255)
CLEAR_COLOR = (40, 220, 60)
PARTLY_COLOR = (230, 200, 0)
PAGE_SECONDS = 4.0
PAGE_FADE = 0.4

CLOUD_ICON = [
    "..##....",
    ".####.#.",
    "########",
    ".######.",
]

# WGS84
EARTH_RADIUS = 6378.137
FLATTENING = 1 / 298.257223563
ISS_ALTITUDE = 420.0  # km, for the viewing ring
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# --- Orbit and sky -----------------------------------------------------------


def julian_date(t: np.ndarray) -> np.ndarray:
    return t / 86400.0 + 2440587.5


def sidereal_angle(jd: np.ndarray) -> np.ndarray:
    """Greenwich mean sidereal time in radians (IAU 1982, as SGP4 uses)."""
    c = (jd - 2451545.0) / 36525
    seconds = 67310.54841 + (876600 * 3600 + 8640184.812866) * c + 0.093104 * c * c - 6.2e-6 * c ** 3
    return np.radians((seconds % 86400) / 240)


def sun_direction(jd: np.ndarray) -> np.ndarray:
    """Unit vectors toward the Sun in equatorial (inertial) coordinates, from
    the Astronomical Almanac's low-precision formula (about 0.01 degrees)."""
    n = jd - 2451545.0
    mean_longitude = np.radians(280.460 + 0.9856474 * n)
    anomaly = np.radians(357.528 + 0.9856003 * n)
    longitude = mean_longitude + np.radians(1.915 * np.sin(anomaly) + 0.020 * np.sin(2 * anomaly))
    tilt = np.radians(23.439 - 4e-7 * n)
    return np.stack([np.cos(longitude), np.cos(tilt) * np.sin(longitude), np.sin(tilt) * np.sin(longitude)], axis=-1)


def earth_fixed(v: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate inertial vectors (n, 3) into Earth-fixed ones."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([c * v[:, 0] + s * v[:, 1], c * v[:, 1] - s * v[:, 0], v[:, 2]], axis=-1)


def observer_frame(lat: float, lon: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """The observer's Earth-fixed position (km) and east/north/up unit vectors."""
    phi, lam = math.radians(lat), math.radians(lon)
    e2 = FLATTENING * (2 - FLATTENING)
    n = EARTH_RADIUS / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    position = np.array([
        (n + height) * math.cos(phi) * math.cos(lam),
        (n + height) * math.cos(phi) * math.sin(lam),
        (n * (1 - e2) + height) * math.sin(phi),
    ])
    enu = np.array([
        [-math.sin(lam), math.cos(lam), 0],
        [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)],
        [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)],
    ])
    return position, enu


OBSERVER_POSITION, OBSERVER_ENU = observer_frame(OBSERVER_LAT, OBSERVER_LON, OBSERVER_HEIGHT_KM)


@dataclass
class Sky:
    """Where the ISS is at each of a run of times."""

    lat: np.ndarray  # under the ISS, degrees
    lon: np.ndarray
    elevation: np.ndarray  # seen from the observer, degrees
    azimuth: np.ndarray  # degrees east of north
    sunlit: np.ndarray  # the ISS is in sunlight
    sun_elevation: np.ndarray  # the Sun seen from the observer, degrees

    def visible(self) -> np.ndarray:
        return (self.elevation >= MIN_ELEVATION) & self.sunlit & (self.sun_elevation <= DARK_SUN_ELEVATION)


class Orbit:
    def __init__(self, line1: str, line2: str) -> None:
        self.satellite = Satrec.twoline2rv(line1, line2)
        self.lines = (line1, line2)

    def sky(self, t) -> Sky:
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        jd = julian_date(t)
        whole = np.floor(jd)
        error, position, _ = self.satellite.sgp4_array(whole, jd - whole)
        position[error != 0] = np.nan
        angle = sidereal_angle(jd)
        sun = sun_direction(jd)

        fixed = earth_fixed(position, angle)
        lon = np.degrees(np.arctan2(fixed[:, 1], fixed[:, 0]))
        geocentric = np.arctan2(fixed[:, 2], np.hypot(fixed[:, 0], fixed[:, 1]))
        lat = np.degrees(np.arctan(np.tan(geocentric) / (1 - FLATTENING) ** 2))

        e, n, u = ((fixed - OBSERVER_POSITION) @ OBSERVER_ENU.T).T
        elevation = np.degrees(np.arctan2(u, np.hypot(e, n)))
        azimuth = np.degrees(np.arctan2(e, n)) % 360

        # In sunlight unless behind the Earth, inside its (cylindrical) shadow
        along = np.einsum("ij,ij->i", position, sun)
        across = np.linalg.norm(position - along[:, None] * sun, axis=1)
        sunlit = (along > 0) | (across > EARTH_RADIUS)
        sun_up = earth_fixed(sun, angle) @ OBSERVER_ENU[2]
        sun_elevation = np.degrees(np.arcsin(np.clip(sun_up, -1, 1)))
        return Sky(lat, lon, elevation, azimuth, sunlit, sun_elevation)


@dataclass
class Pass:
    start: float  # the ISS comes into view (unix time)
    end: float  # leaves view: sets below MIN_ELEVATION or enters the Earth's shadow
    peak_elevation: float
    start_azimuth: float
    end_azimuth: float
    # Its whole arc above the horizon, every PATH_STEP seconds from path_start,
    # for the sky view: azimuth and elevation (n, 2), and whether visible (n,)
    path_start: float
    path: np.ndarray
    path_visible: np.ndarray


PATH_STEP = 5


def find_passes(orbit: Orbit, start: float, days: float = PREDICT_DAYS) -> list[Pass]:
    """Visible passes over the observer from start (including one under way)."""
    passes = []
    # A day at a time, so the GIL is free between chunks for the frames
    for day in range(math.ceil(days)):
        t = start + day * 86400 + np.arange(0, 86400, PREDICT_STEP, dtype=np.float64)
        visible = orbit.sky(t).visible()
        if not visible.any():
            continue
        changes = np.flatnonzero(np.diff(visible.astype(np.int8))) + 1
        bounds = np.concatenate([[0], changes, [len(t)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if visible[a]:
                found = refine_pass(orbit, t[a], t[b - 1])
                # (A pass across midnight shows up in both days)
                if found is not None and not (passes and abs(found.start - passes[-1].start) < 60):
                    passes.append(found)
        time.sleep(0)
    return passes


def refine_pass(orbit: Orbit, first: float, last: float) -> Pass | None:
    """A pass found at PREDICT_STEP resolution, timed to the second."""
    t = np.arange(first - 15 * 60, last + 15 * 60, 1.0)
    sky = orbit.sky(t)
    visible = sky.visible()
    # The visible stretch containing the coarse one, and the above-horizon arc around it
    middle = int(np.searchsorted(t, (first + last) / 2))
    if not visible[middle]:
        middle = int(np.searchsorted(t, first))
        if not visible[middle]:
            return None
    a = middle - np.argmin(visible[middle::-1]) + 1 if not visible[: middle + 1].all() else 0
    b = middle + np.argmin(visible[middle:]) if not visible[middle:].all() else len(t)
    up = sky.elevation >= 0
    lo = a - np.argmin(up[a::-1]) + 1 if not up[: a + 1].all() else 0
    hi = a + np.argmin(up[a:]) if not up[a:].all() else len(t)

    peak = float(sky.elevation[a:b].max())
    if peak < MIN_PEAK or t[b - 1] - t[a] < MIN_VISIBLE_SECONDS:
        return None
    arc = slice(lo, hi, PATH_STEP)
    return Pass(
        start=float(t[a]),
        end=float(t[b - 1]),
        peak_elevation=peak,
        start_azimuth=float(sky.azimuth[a]),
        end_azimuth=float(sky.azimuth[b - 1]),
        path_start=float(t[lo]),
        path=np.stack([sky.azimuth[arc], sky.elevation[arc]], axis=-1),
        path_visible=visible[arc],
    )


def compass(azimuth: float) -> str:
    return COMPASS[round(azimuth / 45) % 8]


# --- Data: orbit elements and cloud forecast ---------------------------------


@dataclass
class Forecast:
    times: np.ndarray  # unix times, hourly
    cloud: np.ndarray  # total cloud cover, percent

    def cloud_during(self, start: float, end: float) -> float | None:
        """Average cloud cover over a pass, or None past the forecast."""
        if start < self.times[0] or end > self.times[-1]:
            return None
        t = np.linspace(start, end, 5)
        return float(np.interp(t, self.times, self.cloud).mean())


def log(message: str) -> None:
    print(f"iss: {message}", file=sys.stderr, flush=True)


class Feed:
    """The ISS's orbit, the cloud forecast and the passes they predict, kept
    fresh by a background thread. Readers just take the attributes: each is
    replaced whole."""

    def __init__(self, clock) -> None:
        self.clock = clock
        self.orbit: Orbit | None = None
        self.forecast: Forecast | None = None
        self.passes: list[Pass] = []
        self.predicted_at = -math.inf
        self.next_fetch = {"tle": 0.0, "forecast": 0.0}
        self.errors: dict[str, str] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Start from the cache, so the first frames have something to show;
        # the thread fetches anything missing or stale
        self.load_cached()
        self.predict()
        threading.Thread(target=self.loop, name="iss-feed", daemon=True).start()

    def load_cached(self) -> None:
        try:
            self.orbit = parse_tle((CACHE_DIR / "tle.txt").read_text())
        except (OSError, ValueError):
            pass
        try:
            path = CACHE_DIR / "forecast.json"
            if time.time() - path.stat().st_mtime < FORECAST_STALE:
                self.forecast = parse_forecast(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError):
            pass

    def loop(self) -> None:
        while True:
            self.update()
            time.sleep(60)

    def update(self) -> None:
        changed = self.fetch("tle", CACHE_DIR / "tle.txt", TLE_MAX_AGE, self.fetch_tle)
        self.fetch("forecast", CACHE_DIR / "forecast.json", FORECAST_MAX_AGE, self.fetch_forecast)
        if changed or self.clock() - self.predicted_at > PREDICT_EVERY:
            self.predict()

    def fetch(self, name: str, path: Path, max_age: float, fetch) -> bool:
        """Refresh the cached file at path with fetch() when it's older than
        max_age. Returns whether new data arrived."""
        now = time.time()
        try:
            fresh = now - path.stat().st_mtime < max_age
        except OSError:
            fresh = False
        if fresh or now < self.next_fetch[name]:
            return False
        try:
            fetch(path)
        except Exception as e:  # keep the cached data; retry later
            error = f"{type(e).__name__}: {e}"
            if self.errors.get(name) != error:
                log(f"{name} fetch failed, retrying: {error}")
            self.errors[name] = error
            self.next_fetch[name] = now + FETCH_RETRY
            return False
        if self.errors.pop(name, None):
            log(f"{name} fetch recovered")
        return True

    def fetch_tle(self, path: Path) -> None:
        r = requests.get(TLE_URL, timeout=15)
        r.raise_for_status()
        orbit = parse_tle(r.text)
        path.write_text(r.text)
        self.orbit = orbit

    def fetch_forecast(self, path: Path) -> None:
        r = requests.get(FORECAST_URL, timeout=15, params={
            "latitude": OBSERVER_LAT,
            "longitude": OBSERVER_LON,
            "hourly": "cloud_cover",
            "timeformat": "unixtime",
            "past_days": 1,
            "forecast_days": 8,
        })
        r.raise_for_status()
        forecast = parse_forecast(r.json())
        path.write_text(r.text)
        self.forecast = forecast

    def predict(self) -> None:
        if self.orbit is None:
            return
        now = self.clock()
        self.passes = find_passes(self.orbit, now)
        self.predicted_at = now


def parse_tle(text: str) -> Orbit:
    lines = [line.strip() for line in text.splitlines()]
    line1 = next((line for line in lines if line.startswith("1 ")), None)
    line2 = next((line for line in lines if line.startswith("2 ")), None)
    if line1 is None or line2 is None:
        raise ValueError("no TLE in response")
    return Orbit(line1, line2)


def parse_forecast(data: dict) -> Forecast:
    hourly = data["hourly"]
    cloud = np.array([np.nan if c is None else c for c in hourly["cloud_cover"]], dtype=np.float64)
    times = np.array(hourly["time"], dtype=np.float64)
    keep = ~np.isnan(cloud)
    if keep.sum() < 2:
        raise ValueError("empty forecast")
    return Forecast(times[keep], cloud[keep])


# --- Drawing -----------------------------------------------------------------


class Splat:
    """Fixed sub-pixel points (pixel centers at integers), each spread
    bilinearly over four pixels, drawn with new weights every frame."""

    def __init__(self, x: np.ndarray, y: np.ndarray, height: int, width: int, wrap: bool) -> None:
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        index, share = [], []
        for dx, dy, s in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            if wrap:
                xi = xi % width
            ok = (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
            index.append(np.where(ok, yi * width + xi, 0))
            share.append(np.where(ok, s, 0))
        self.index = np.concatenate(index)
        self.share = np.concatenate(share).astype(np.float32)
        self.shape = (height, width)

    def draw(self, weight) -> np.ndarray:
        """The points with these weights (one each, or one for all), as a layer."""
        weight = np.tile(weight, 4) if np.ndim(weight) else weight
        layer = np.bincount(self.index, self.share * weight, self.shape[0] * self.shape[1])
        return layer.reshape(self.shape).astype(np.float32)


GLOW_DY, GLOW_DX = (a.ravel() for a in np.mgrid[-2:4, -2:4])


def draw_glow(frame: np.ndarray, x: float, y: float, halo: float, core_color: np.ndarray,
              halo_color: np.ndarray, wrap: bool) -> None:
    """A bright point with a soft halo at a sub-pixel position."""
    height, width = frame.shape[:2]
    xi = GLOW_DX + math.floor(x)
    yi = GLOW_DY + math.floor(y)
    d2 = ((xi - x) ** 2 + (yi - y) ** 2).astype(np.float32)
    if wrap:
        xi = xi % width
    ok = (yi >= 0) & (yi < height) & (xi >= 0) & (xi < width)
    xi, yi, d2 = xi[ok], yi[ok], d2[ok]
    frame[yi, xi] += (halo * np.exp(-d2 / 2.5))[:, None] * halo_color
    frame[yi, xi] = np.maximum(frame[yi, xi], np.minimum(np.exp(-d2 / 0.35), 1)[:, None] * core_color)


def map_xy(lat, lon):
    """Map pixel coordinates (pixel centers at integers) of a latitude/longitude."""
    return (np.asarray(lon) + 180) / 360 * WIDTH - 0.5, (90 - np.asarray(lat)) / 180 * HEIGHT - 0.5


def view_ring() -> tuple[np.ndarray, np.ndarray]:
    """The ground the ISS must be over to be at least MIN_ELEVATION up from the
    observer: a circle around the observer, as map points."""
    e = math.radians(MIN_ELEVATION)
    radius = math.acos(EARTH_RADIUS * math.cos(e) / (EARTH_RADIUS + ISS_ALTITUDE)) - e
    phi, lam = math.radians(OBSERVER_LAT), math.radians(OBSERVER_LON)
    bearing = np.radians(np.arange(0, 360, 2.0))
    lat = np.arcsin(math.sin(phi) * math.cos(radius) + math.cos(phi) * math.sin(radius) * np.cos(bearing))
    lon = lam + np.arctan2(np.sin(bearing) * math.sin(radius) * math.cos(phi), math.cos(radius) - math.sin(phi) * np.sin(lat))
    return map_xy(np.degrees(lat), (np.degrees(lon) + 180) % 360 - 180)


def base_map() -> tuple[np.ndarray, np.ndarray]:
    """The map's day colors and each pixel's latitude/longitude (radians)."""
    rows, cols = np.mgrid[0:MAP_ROWS, 0:WIDTH]
    lat = 90 - (rows + 0.5) * 180 / HEIGHT
    lon = -180 + (cols + 0.5) * 360 / WIDTH
    land = np.array([[int(c) / 9 for c in row] for row in LAND], dtype=np.float32)[..., None]
    desert = np.array([[WORLD[int((90 - la) // 10)][int((lo + 180) // 10)] == "d" for lo, la in zip(r_lon, r_lat)]
                       for r_lon, r_lat in zip(lon, lat)])
    ground = np.where(desert[..., None], np.array(DESERT, np.float32), np.array(LAND_GREEN, np.float32))
    ground = np.where((lat > 64)[..., None], np.array(ICE, np.float32), ground)
    colors = np.array(OCEAN, np.float32) * (1 - land) + ground * land
    return colors, np.radians(lat), np.radians(lon)


def cloud_color(cloud: float | None) -> tuple[int, int, int]:
    if cloud is None:
        return GRAY
    if cloud <= CLEAR_PERCENT:
        return CLEAR_COLOR
    return PARTLY_COLOR if cloud < CLOUDY_PERCENT else GRAY


def clouded_out(cloud: float | None) -> bool:
    return cloud is not None and cloud >= CLOUDY_PERCENT


def day_label(start: float, now: float) -> str:
    when = datetime.fromtimestamp(start, TIMEZONE)
    days = (when.date() - datetime.fromtimestamp(now, TIMEZONE).date()).days
    if days == 0:
        return "TODAY"
    if days == 1:
        return "TMRW"
    if days < 7:
        return when.strftime("%a").upper()
    return f"{when.month}/{when.day}"


def clock_label(t: float) -> str:
    when = datetime.fromtimestamp(t, TIMEZONE)
    return f"{when.hour % 12 or 12}:{when.minute:02d}{'AM' if when.hour < 12 else 'PM'}"


def countdown(seconds: float) -> str:
    seconds = max(0, int(math.ceil(seconds)))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def rgb(color) -> np.ndarray:
    return np.array(color, np.float32)


DOME_CENTER = 15.5
DOME_RADIUS = 15.0


def dome_xy(azimuth, elevation):
    """Sky view coordinates: the zenith at the center, the horizon at the rim, north up."""
    r = DOME_RADIUS * (90 - np.asarray(elevation)) / 90
    a = np.radians(azimuth)
    return DOME_CENTER + r * np.sin(a), DOME_CENTER - r * np.cos(a)


class IssTracker:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the ISS map is drawn for a {WIDTH}x{HEIGHT} panel")
        self.frame_count = 0
        start = os.environ.get("ISS_START")
        self.sim_start = datetime.fromisoformat(start).timestamp() if start else None
        self.sim_speed = float(os.environ.get("ISS_SPEED", "1"))

        self.font = BitmapFont("tom-thumb")  # 4x6's zero reads as an 8
        self.big_font = BitmapFont("7x13B")
        self.cloud_icon = np.array([[c == "#" for c in row] for row in CLOUD_ICON], dtype=np.float32)
        self.text_cache: dict = {}

        self.day_colors, self.lat, self.lon = base_map()
        self.map_rgb = None
        self.map_time = -math.inf
        ring_x, ring_y = view_ring()
        ring = Splat(ring_x, ring_y, MAP_ROWS, WIDTH, wrap=True).draw(0.5)
        self.ring = np.clip(ring, 0, 1)[..., None] * rgb(RING_COLOR)
        self.home_x, self.home_y = (round(float(v)) for v in map_xy(OBSERVER_LAT, OBSERVER_LON))
        self.track = None
        self.track_time = -math.inf

        self.dome = self.draw_dome()
        self.arc: tuple[Pass, Splat, np.ndarray, np.ndarray] | None = None
        self.sky_mix = 0.0
        self.sky_pass: Pass | None = None  # the pass the sky view shows (kept while fading out)

        self.feed = Feed(self.now)

    def now(self) -> float:
        if self.sim_start is None:
            return time.time()
        return self.sim_start + self.frame_count * FRAME_SECONDS * self.sim_speed

    # The map --------------------------------------------------------------

    def shaded_map(self, now: float) -> np.ndarray:
        """The map lit by the Sun, recomputed every few seconds."""
        if abs(now - self.map_time) > 10:
            jd = julian_date(np.array([now]))
            sun = earth_fixed(sun_direction(jd), sidereal_angle(jd))[0]
            dec, sun_lon = math.asin(sun[2]), math.atan2(sun[1], sun[0])
            altitude = np.degrees(np.arcsin(
                np.sin(self.lat) * math.sin(dec) + np.cos(self.lat) * math.cos(dec) * np.cos(self.lon - sun_lon)))
            light = np.clip((altitude + 9) / 12, 0, 1)[..., None]
            light = light * light * (3 - 2 * light)
            night = self.day_colors * rgb(NIGHT)
            self.map_rgb = (night + (self.day_colors - night) * light).astype(np.float32)
            self.map_time = now
        return self.map_rgb

    def ground_track(self, orbit: Orbit, now: float):
        """The track around now, recomputed every few seconds: its points
        prepared for drawing, their minutes from now, and map x (unwrapped,
        to interpolate the ISS's position between points) and y."""
        if self.track is None or abs(now - self.track_time) > 5:
            minutes = np.arange(-TRAIL_MINUTES * 60, TRACK_MINUTES * 60 + 1, TRACK_STEP) / 60
            sky = orbit.sky(now + minutes * 60)
            ok = ~np.isnan(sky.lat)
            x, y = map_xy(sky.lat[ok], sky.lon[ok])
            self.track = (Splat(x, y, MAP_ROWS, WIDTH, wrap=True), minutes[ok], np.unwrap(x, period=WIDTH), y)
            self.track_time = now
        splat, minutes, x, y = self.track
        return splat, minutes - (now - self.track_time) / 60, x, y

    def draw_map(self, now: float, next_pass: Pass | None, cloud: float | None) -> np.ndarray:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        world = frame[:MAP_ROWS]
        world[:] = self.shaded_map(now)
        t = self.frame_count * FRAME_SECONDS

        # Seattle's viewing ring, pulsing when a clear pass is coming up
        soon = (next_pass is not None and next_pass.start - now < SOON_SECONDS and not clouded_out(cloud))
        world += self.ring * (1.6 + 1.2 * math.sin(t * 4) if soon else 1.0)

        orbit = self.feed.orbit
        track = self.ground_track(orbit, now) if orbit is not None else None
        if track is not None and len(track[1]) > 1:
            splat, minutes, x, y = track
            past = minutes <= 0
            trail = np.where(past, np.clip(1 + minutes / TRAIL_MINUTES, 0, 1) ** 2 * 0.5, 0)
            # Dots march along the track ahead: a bright point every 3 minutes
            phase = (minutes - t * 0.6) / 3
            dots = np.clip(np.cos(phase * 2 * math.pi) * 3 - 2, 0, 1) * 0.6 + 0.12
            ahead = np.where(past, 0, dots * (1 - minutes / (TRACK_MINUTES + 5)))
            world += np.minimum(splat.draw(trail), 1)[..., None] * rgb(TRAIL_COLOR)
            world += np.minimum(splat.draw(ahead), 1)[..., None] * rgb(TRACK_COLOR)

        # Seattle
        world[self.home_y, self.home_x] = rgb(HOME_COLOR) * ((0.75 + 0.25 * math.sin(t * 4)) if soon else 1.0)

        if track is not None and len(track[1]) > 1 and track[1][0] <= 0 <= track[1][-1]:
            _, minutes, x, y = track
            draw_glow(world, float(np.interp(0, minutes, x)) % WIDTH, float(np.interp(0, minutes, y)),
                      0.45 + 0.15 * math.sin(t * 3), rgb(ISS_COLOR), rgb(HALO_COLOR), wrap=True)

        self.draw_strip(frame, now, next_pass, cloud)
        return frame

    # Text -----------------------------------------------------------------

    def text(self, text: str, font: BitmapFont | None = None) -> np.ndarray:
        font = font or self.font
        key = (text, id(font))
        mask = self.text_cache.get(key)
        if mask is None:
            mask = font.render(text, proportional=True)
            self.cache(key, mask)
        return mask

    def cache(self, key, value) -> None:
        if len(self.text_cache) > 200:
            self.text_cache.clear()
        self.text_cache[key] = value

    def line(self, parts: list, width: int) -> np.ndarray:
        """A line of (text or icon mask, color) parts, centered in width
        columns, as RGB. Cached: a line's text changes at most each second."""
        key = tuple((part if isinstance(part, str) else id(part), color) for part, color in parts) + (width,)
        out = self.text_cache.get(key)
        if out is not None:
            return out
        pieces = []
        for part, color in parts:
            if isinstance(part, str):
                mask = self.text(part)
            else:
                # Sitting on the baseline, like the text
                mask = np.zeros((self.font.height, part.shape[1]), np.float32)
                mask[self.font.ascent - part.shape[0]:self.font.ascent] = part
            pieces += [mask[..., None] * rgb(color), np.zeros((self.font.height, 2, 3), np.float32)]
        row = np.concatenate(pieces[:-1], axis=1)[:, :width]
        out = np.zeros((self.font.height, width, 3), np.float32)
        x = (width - row.shape[1]) // 2
        out[:, x:x + row.shape[1]] = row
        self.cache(key, out)
        return out

    def strip_pages(self, now: float, next_pass: Pass | None, cloud: float | None) -> list[list]:
        if self.feed.orbit is None:
            return [[("NO ISS DATA", GRAY)]]
        if next_pass is None:
            return [[(f"NO PASSES {PREDICT_DAYS}D", GRAY)]]
        p = next_pass
        cloudy = clouded_out(cloud)
        percent = "?" if cloud is None else f"{round(cloud)}%"
        weather = [(self.cloud_icon, WHITE if cloud is not None else GRAY), (percent, cloud_color(cloud))]
        route = f"{compass(p.start_azimuth)}>{compass(p.end_azimuth)}"
        minutes = max(1, round((p.end - p.start) / 60))
        details = [(route, GRAY if cloudy else CYAN), (f"{round(p.peak_elevation)}° {minutes}M", GRAY if cloudy else WHITE)]
        if now >= p.start:
            # Under way but clouded over (a clear one takes over the panel)
            return [[("OVERHEAD", GRAY)] + weather, details]
        if p.start - now < SOON_SECONDS:
            return [[("ISS IN", GRAY if cloudy else GOLD), (countdown(p.start - now), GRAY if cloudy else WHITE)] + weather,
                    details]
        pages = [[(f"{day_label(p.start, now)} {clock_label(p.start)}", GRAY if cloudy else WHITE)] + weather, details]
        if cloudy:
            clear = self.next_clear_pass(now)
            if clear is not None:
                pages.append([("CLEAR", CLEAR_COLOR), (f"{day_label(clear.start, now)} {clock_label(clear.start)}", WHITE)])
        return pages

    def draw_strip(self, frame: np.ndarray, now: float, next_pass: Pass | None, cloud: float | None) -> None:
        pages = self.strip_pages(now, next_pass, cloud)
        t = self.frame_count * FRAME_SECONDS
        index = int(t // PAGE_SECONDS) % len(pages)
        into = t % PAGE_SECONDS
        level = 1.0 if len(pages) == 1 else min(1.0, into / PAGE_FADE, (PAGE_SECONDS - into) / PAGE_FADE)
        frame[STRIP_TOP:STRIP_TOP + self.font.height] += self.line(pages[index], WIDTH) * level

    # The sky view ---------------------------------------------------------

    def draw_dome(self) -> np.ndarray:
        """The sky view's fixed parts: the horizon, a fainter ring at 45
        degrees up, and the compass points just inside the rim."""
        yy, xx = np.mgrid[0:HEIGHT, 0:32]
        dist = np.hypot(xx - DOME_CENTER, yy - DOME_CENTER)
        rings = np.clip(1 - abs(dist - DOME_RADIUS + 0.3), 0, 1) + np.clip(1 - abs(dist - DOME_RADIUS / 2), 0, 1) * 0.2
        dome = rings[..., None] * rgb((0, 45, 100))
        for label, (x, y), color in (("N", (14, 2), (220, 40, 40)), ("E", (27, 14), LABEL_GRAY),
                                     ("S", (14, 26), LABEL_GRAY), ("W", (2, 14), LABEL_GRAY)):
            mask = self.text(label)
            dome[y:y + mask.shape[0], x:x + mask.shape[1]] += mask[..., None] * rgb(color)
        return dome.astype(np.float32)

    def pass_arc(self, p: Pass) -> tuple[Splat, np.ndarray, np.ndarray]:
        """The pass's arc across the dome, prepared for drawing, its points'
        times, and their dome x and y."""
        if self.arc is None or self.arc[0] is not p:
            x, y = dome_xy(p.path[:, 0], p.path[:, 1])
            times = p.path_start + np.arange(len(p.path)) * PATH_STEP
            self.arc = (p, Splat(x, y, HEIGHT, 32, wrap=False), times, np.stack([x, y]))
        return self.arc[1:]

    def draw_sky(self, now: float, p: Pass, cloud: float | None) -> np.ndarray:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        sky = frame[:, :32]
        sky[:] = self.dome
        t = self.frame_count * FRAME_SECONDS

        # The pass's arc: dim where it's out of view, brighter where it's
        # visible, brightest where the ISS has already been (drawn over the
        # compass points)
        splat, times, xy = self.pass_arc(p)
        weight = np.where(p.path_visible, np.where(times <= now, 0.9, 0.45), 0.18) * 0.6
        np.maximum(sky, np.minimum(splat.draw(weight), 1)[..., None] * rgb(HALO_COLOR) * 0.6, out=sky)

        # The ISS, between the arc's points
        i = (now - p.path_start) / PATH_STEP
        up = 0 <= i <= len(times) - 1
        nearest = min(max(round(i), 0), len(times) - 1)
        in_view = up and bool(p.path_visible[nearest])
        elevation = float(np.interp(i, np.arange(len(times)), p.path[:, 1]))
        azimuth = float(p.path[nearest, 0])
        if up:
            x = float(np.interp(i, np.arange(len(times)), xy[0]))
            y = float(np.interp(i, np.arange(len(times)), xy[1]))
            draw_glow(sky, x, y, (0.6 + 0.25 * math.sin(t * 4)) if in_view else 0.0,
                      rgb(ISS_COLOR if in_view else (110, 110, 110)), rgb(HALO_COLOR), wrap=False)

        # Where to look, how high, how long
        right = frame[:, 32:]
        right[0:self.font.height] += self.line([("LOOK UP", GOLD)], 32) * (0.75 + 0.25 * math.sin(t * 4))
        direction = self.text(compass(azimuth if in_view or now >= p.start else p.start_azimuth), self.big_font)
        x = (32 - direction.shape[1]) // 2
        right[7:7 + direction.shape[0], x:x + direction.shape[1]] += direction[..., None] * rgb(WHITE)
        if now < p.start:
            status = [("IN", GRAY), (countdown(p.start - now), WHITE)]
        else:
            status = [(f"{round(max(elevation, 0))}°", CYAN), (countdown(p.end - now), WHITE)]
        right[20:20 + self.font.height] += self.line(status, 32)
        percent = "?" if cloud is None else f"{round(cloud)}%"
        right[26:26 + self.font.height] += self.line([(self.cloud_icon, WHITE), (percent, cloud_color(cloud))], 32)
        return frame

    # ----------------------------------------------------------------------

    def next_clear_pass(self, now: float) -> Pass | None:
        forecast = self.feed.forecast
        for p in self.feed.passes:
            if p.end >= now and forecast is not None:
                cloud = forecast.cloud_during(p.start, p.end)
                if cloud is not None and not clouded_out(cloud):
                    return p
        return None

    def cloud(self, p: Pass | None) -> float | None:
        forecast = self.feed.forecast
        return None if p is None or forecast is None else forecast.cloud_during(p.start, p.end)

    def frame(self) -> np.ndarray:
        now = self.now()
        passes = self.feed.passes
        upcoming = next((p for p in passes if p.end >= now), None)
        # A pass about to rise or up now takes over the panel, unless clouded over
        showing = next((p for p in passes if p.start - PREVIEW_SECONDS <= now <= p.end + AFTER_SECONDS), None)
        if showing is not None and not clouded_out(self.cloud(showing)):
            self.sky_pass = showing
        else:
            showing = None
        step = FRAME_SECONDS / SKY_FADE_SECONDS
        self.sky_mix = min(1.0, self.sky_mix + step) if showing else max(0.0, self.sky_mix - step)

        m = self.sky_mix * self.sky_mix * (3 - 2 * self.sky_mix)
        if m < 1:
            frame = self.draw_map(now, upcoming, self.cloud(upcoming))
        if m > 0:
            sky = self.draw_sky(now, self.sky_pass, self.cloud(self.sky_pass))
            frame = sky if m >= 1 else frame * (1 - m) + sky * m
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


SCENE = IssTracker


def main() -> None:
    run_scene(IssTracker, "ISS tracker LED matrix display")


if __name__ == "__main__":
    main()
