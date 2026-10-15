#!/usr/bin/env python
"""Flights: the planes out a west-facing window in Seward Park, Seattle, as
a card of facts about each in turn.

Planes count as in view when they're west of the window (VIEW_LEFT to
VIEW_RIGHT degrees around), within VIEW_RANGE and above the horizon. Boeing
Field and Sea-Tac lie off to the southwest, so most are landing or taking off
there. The card shows, for one plane at a time (a close one stays up while
it's close; otherwise each takes a turn):

    Alaska 321              the flight (or a private plane's registration)
    SJC > SEA               its route, when known and plausible (else the
                            operator of a non-airline plane)
    737-900         4mi W   its type, and how far away and which way
    ^ 3100ft       210mph   altitude, climbing or descending, and speed
    .   .  o    .           where each plane in view is across the window

Positions come from adsb.fi's open ADS-B feed (no key), polled every few
seconds; between polls each plane flies on along its reported track. Routes
come from adsbdb, shown only when the plane is near the great circle between
the two airports and going the right way along it, since its routes are often
out of date. A plane squawking an emergency code says so in red.

Requires the rgbmatrix Python bindings (make build-python), numpy and requests.
"""

import json
import math
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
# How far sea level lies above the WGS84 ellipsoid here (it's below), the
# feed's GPS altitudes being heights above the ellipsoid
GEOID = -22.0  # meters
TIMEZONE = ZoneInfo("America/Los_Angeles")
WIDTH, HEIGHT = 64, 32

# What counts as in view: compass degrees across the window (south to
# north), how far, and how high above the horizon
VIEW_LEFT, VIEW_RIGHT = 190.0, 350.0
VIEW_RANGE = 30_000  # meters
MIN_ELEVATION = 0.5  # degrees

FEED_URL = "https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{dist}"
FEED_RADIUS_NM = 25
POLL_SECONDS = 5
STALE_SECONDS = 30  # forget a plane not heard from in this long
NO_DATA_SECONDS = 90  # say so after this long without a poll
ROUTE_URL = "https://api.adsbdb.com/v0/callsign/{callsign}"
ROUTE_GAP = 1.5  # seconds between route lookups
ROUTE_KEEP = 7 * 86400  # how long a looked-up route is trusted
ROUTE_KEEP_MISSING = 86400
# Seconds to wait after a failed fetch, doubling while it keeps failing
FEED_RETRY = 10
FETCH_RETRY = 30
FETCH_RETRY_MAX = 600
CLOCK_SLACK = 60  # trust the feed's clock over ours only when they're this close
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "flights"
USER_AGENT = "pixel-display (home LED matrix; github.com/Gingernaut/rpi-rgb-led-matrix)"

# A route is shown only if the plane is within ROUTE_SLACK of the great
# circle between its endpoints, and going the right way: away from the
# origin (not descending) when within NEAR_AIRPORT of it, toward the
# destination (not climbing) when near that, and heading within ROUTE_TURN
# degrees of the destination when near neither
ROUTE_SLACK = 250_000
NEAR_AIRPORT = 80_000
ROUTE_TURN = 100

# Each plane in view gets the card for CARD_SECONDS in turn; one within
# CLOSE keeps it while it's that close
CARD_SECONDS = 8.0
CLOSE = 4_500
FADE_SECONDS = 0.3

EARTH_RADIUS = 6_371_000.0
REFRACTION = 0.13  # light bends over the horizon a little: the Earth looks bigger
METERS_PER_FOOT = 0.3048
METERS_PER_MILE = 1609.344
KNOTS = 0.514444  # m/s
MPH_PER_KNOT = 1.150779
FEET_PER_MINUTE = 0.00508  # m/s
CLIMBING = 300  # feet per minute, up or down, to count as climbing or descending
# Meters per degree of latitude and longitude at home
M_PER_LAT = 111_132.92 - 559.82 * math.cos(math.radians(2 * HOME_LAT)) + 1.175 * math.cos(math.radians(4 * HOME_LAT))
M_PER_LON = 111_412.84 * math.cos(math.radians(HOME_LAT)) - 93.5 * math.cos(math.radians(3 * HOME_LAT))
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
EMERGENCIES = {"7500": "HIJACK", "7600": "NO RADIO", "7700": "EMERGENCY"}

# The card's rows: the title, three lines of small text, and the dots
# (A cell is left clear all round the edge, two at the sides)
TITLE_TOP = 1
LINE_TOPS = (9, 16, 23)
DOTS_ROW = 30
MARGIN = 2  # columns left clear at each side

WHITE = (220, 220, 220)
DIM = (120, 120, 132)
FAINT = (45, 45, 55)
CLIMB_COLOR = (60, 210, 90)
DESCEND_COLOR = (255, 150, 40)
ALERT = (255, 30, 30)
DEFAULT_AIRLINE = (220, 220, 220)
MILITARY = (150, 175, 90)

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
# Words that mark a non-airline operator as a business or agency, so it's
# not a private owner, whose name is never shown. A company suffix alone
# isn't enough ("JOHN SMITH LLC"), unless only one word is left without it
# ("BOEING CO"); nor is a trust, which is usually a person's.
BUSINESS = re.compile(r"\b(AIRLINES?|AIR|AVIATION|AIRCRAFT|AEROSPACE|FLIGHT|FLYING|CHARTER|HELICOPTERS?|"
                      r"CLUB|SCHOOL|ACADEMY|COLLEGE|UNIVERSITY|CITY|COUNTY|STATE|DEPARTMENT|POLICE|SHERIFF|"
                      r"PATROL|FIRE|NETWORK|SERVICES?|BANK)\b")
COMPANY_WORDS = {"inc", "llc", "co", "corp", "corporation", "company", "ltd", "lp"}
SMALL_WORDS = {"of", "and", "the", "for", "at", "in"}


def log(message: str) -> None:
    print(f"flights: {message}", file=sys.stderr, flush=True)


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


class Plane:
    """One aircraft from the feed, flown on from its last report."""

    def __init__(self, report: dict) -> None:
        self.id = report["hex"]
        self.shown = -math.inf  # when it last had the card
        self.update(report)

    def update(self, report: dict) -> None:
        self.callsign = (report.get("flight") or "").strip()
        self.registration = (report.get("r") or "").strip()
        self.type = (report.get("t") or "").strip()
        self.description = (report.get("desc") or "").strip()
        self.owner = (report.get("ownOp") or "").strip()
        self.military = bool((report.get("dbFlags") or 0) & 1)
        self.squawk = report.get("squawk") or ""
        self.feet = report["feet"]
        climb = report.get("geom_rate")
        self.climb = (report.get("baro_rate") if climb is None else climb) or 0
        self.knots = report.get("gs") or 0
        self.heading = report.get("track")  # degrees, or None when not reported
        # (without a track it stays put rather than flying off north)
        speed = 0 if self.heading is None else self.knots * KNOTS
        track = math.radians(self.heading or 0)
        self.anchor = (*local(report["lat"], report["lon"]), self.feet * METERS_PER_FOOT - HOME_HEIGHT)
        self.velocity = (speed * math.sin(track), speed * math.cos(track), self.climb * FEET_PER_MINUTE)
        self.heard = report["at"]

    def position(self, now: float) -> tuple[float, float, float]:
        dt = now - self.heard
        return tuple(a + v * dt for a, v in zip(self.anchor, self.velocity))

    def place(self) -> tuple[float, float]:
        """Latitude and longitude at the last report."""
        return HOME_LAT + self.anchor[1] / M_PER_LAT, HOME_LON + self.anchor[0] / M_PER_LON


class Feed:
    """The planes nearby, polled from adsb.fi by a background thread, and
    their routes, looked up from adsbdb by another when asked for. Readers
    take `reports` (replaced whole on each poll) and route()."""

    def __init__(self) -> None:
        self.reports: list[dict] = []
        self.polled = 0  # counts polls, so readers see new reports
        self.polled_at = -math.inf
        # callsign: (when looked up, route or None if adsbdb has none),
        # replaced whole when pruned, so a reader never sees it change size
        self.routes: dict[str, tuple[float, dict | None]] = {}
        self.wanted: list[str] = []  # callsigns to look up, most wanted first
        self.wanted_lock = threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.load_routes()
        threading.Thread(target=self.poll_loop, name="flights-feed", daemon=True).start()
        threading.Thread(target=self.route_loop, name="flights-routes", daemon=True).start()

    @staticmethod
    def session() -> requests.Session:
        session = requests.Session()  # (one per thread: sessions aren't thread-safe)
        session.headers["User-Agent"] = USER_AGENT
        return session

    def poll_loop(self) -> None:
        session = self.session()
        retry, error = FEED_RETRY, None
        while True:
            started = time.monotonic()
            try:
                self.poll(session)
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
            retry, error = FEED_RETRY, None
            time.sleep(max(0.5, POLL_SECONDS - (time.monotonic() - started)))

    def poll(self, session: requests.Session) -> None:
        url = FEED_URL.format(lat=HOME_LAT, lon=HOME_LON, dist=FEED_RADIUS_NM)
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # (adsb.fi gives "now" in seconds; readsb's own format, in
        # milliseconds.) Report times are kept on our clock: if the two
        # disagree, e.g. before the Pi's clock is set, use ours.
        received = time.time()
        now = data.get("now")
        if isinstance(now, (int, float)) and now > 1e11:
            now /= 1000
        if not isinstance(now, (int, float)) or abs(now - received) > CLOCK_SLACK:
            now = received
        reports = []
        for a in data.get("aircraft") or data.get("ac") or []:
            # Prefer the GPS altitude (true height, from the ellipsoid) to the
            # barometric one (off by hundreds of feet with the weather)
            geom, baro = a.get("alt_geom"), a.get("alt_baro")
            feet = geom - GEOID / METERS_PER_FOOT if isinstance(geom, (int, float)) else baro
            if (baro == "ground" or not isinstance(feet, (int, float)) or feet < -300
                    or a.get("lat") is None or a.get("lon") is None or a.get("hex") is None):
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
        """The route for callsign if known, else None; one not looked up
        lately is asked for (and the old one returned meanwhile)."""
        known = self.routes.get(callsign)
        if known is None or time.time() - known[0] > (ROUTE_KEEP if known[1] else ROUTE_KEEP_MISSING):
            with self.wanted_lock:
                if not self.wanted or self.wanted[0] != callsign:
                    if callsign in self.wanted:
                        self.wanted.remove(callsign)
                    self.wanted.insert(0, callsign)
                    del self.wanted[20:]
        return known[1] if known else None

    def route_loop(self) -> None:
        session = self.session()
        retry = FETCH_RETRY
        while True:
            time.sleep(ROUTE_GAP)
            with self.wanted_lock:
                callsign = self.wanted[0] if self.wanted else None
            if callsign is None:
                continue
            try:
                route = self.lookup(session, callsign)
            except Exception as e:
                log(f"route lookup failed for {callsign}: {type(e).__name__}: {e}")
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)
                continue
            retry = FETCH_RETRY
            self.routes[callsign] = (time.time(), route)
            with self.wanted_lock:
                if callsign in self.wanted:
                    self.wanted.remove(callsign)
            self.save_routes()

    def lookup(self, session: requests.Session, callsign: str) -> dict | None:
        r = session.get(ROUTE_URL.format(callsign=callsign), timeout=10)
        if 400 <= r.status_code < 500 and r.status_code != 429:
            return None  # unknown (or a callsign it won't take): don't keep asking
        r.raise_for_status()
        found = r.json().get("response")
        if not isinstance(found, dict) or not isinstance(found.get("flightroute"), dict):
            return None
        route = found["flightroute"]
        origin, destination = route.get("origin") or {}, route.get("destination") or {}
        try:
            return checked({
                "airline": (route.get("airline") or {}).get("name"),
                "flight": route.get("callsign_iata"),
                "from": origin.get("iata_code") or origin["icao_code"],
                "to": destination.get("iata_code") or destination["icao_code"],
                "from_at": (origin["latitude"], origin["longitude"]),
                "to_at": (destination["latitude"], destination["longitude"]),
            })
        except (KeyError, TypeError, ValueError, AttributeError):
            return None

    def load_routes(self) -> None:
        try:
            saved = json.loads((CACHE_DIR / "routes.json").read_text())
            routes, times = saved["routes"], saved["times"]
        except (OSError, ValueError, KeyError, TypeError):
            return
        for callsign, at in times.items():
            try:
                route = routes.get(callsign)
                self.routes[callsign] = (float(at), None if route is None else checked(route))
            except (KeyError, TypeError, ValueError, AttributeError):
                pass

    def save_routes(self) -> None:
        """Forget routes past keeping, and write the rest to the cache."""
        now = time.time()
        self.routes = {c: (at, route) for c, (at, route) in self.routes.items()
                       if now - at < (ROUTE_KEEP if route else ROUTE_KEEP_MISSING)}
        path = CACHE_DIR / "routes.json"
        try:
            path.with_suffix(".tmp").write_text(json.dumps({
                "routes": {c: route for c, (at, route) in self.routes.items()},
                "times": {c: at for c, (at, route) in self.routes.items()},
            }))
            path.with_suffix(".tmp").replace(path)
        except OSError as e:
            log(f"couldn't save routes: {e}")


def checked(route: dict) -> dict:
    """A route with the fields the card uses, of the right types (raising
    KeyError, TypeError or ValueError if they're missing or wrong)."""
    airline, flight = route.get("airline"), route.get("flight")
    if not (isinstance(route["from"], str) and isinstance(route["to"], str) and route["from"] and route["to"]):
        raise ValueError("no airport codes")
    return {
        "airline": airline if isinstance(airline, str) and airline else None,
        "flight": flight if isinstance(flight, str) and len(flight) > 2 else None,
        "from": route["from"], "to": route["to"],
        "from_at": (float(route["from_at"][0]), float(route["from_at"][1])),
        "to_at": (float(route["to_at"][0]), float(route["to_at"][1])),
    }


def arc(a: tuple, b: tuple) -> float:
    """The angle (radians) between two (latitude, longitude) points in radians."""
    h = math.sin((b[0] - a[0]) / 2) ** 2 + math.cos(a[0]) * math.cos(b[0]) * math.sin((b[1] - a[1]) / 2) ** 2
    return 2 * math.asin(min(1.0, math.sqrt(h)))


def bearing(a: tuple, b: tuple) -> float:
    """The initial compass bearing (radians) from a to b, both in radians."""
    y = math.sin(b[1] - a[1]) * math.cos(b[0])
    x = math.cos(a[0]) * math.sin(b[0]) - math.sin(a[0]) * math.cos(b[0]) * math.cos(b[1] - a[1])
    return math.atan2(y, x)


def off_route(route: dict, lat: float, lon: float) -> float:
    """How far (m) a point is from the great circle between a route's ends
    (or past either end)."""
    p1, p2, p3 = (tuple(map(math.radians, p)) for p in (route["from_at"], route["to_at"], (lat, lon)))
    d13, length = arc(p1, p3), arc(p1, p2)
    turn = bearing(p1, p3) - bearing(p1, p2)
    cross = math.asin(max(-1.0, min(1.0, math.sin(d13) * math.sin(turn))))
    along = math.acos(max(-1.0, min(1.0, math.cos(d13) / max(math.cos(cross), 1e-9))))
    if math.cos(turn) < 0:
        along = -along
    beyond = max(0.0, -along, along - length)
    return EARTH_RADIUS * max(abs(cross), beyond)


def plausible(route: dict, plane: Plane) -> bool:
    """Whether a plane could be flying a route (adsbdb's are often out of
    date): near its great circle, and going the right way along it."""
    lat, lon = plane.place()
    if off_route(route, lat, lon) > ROUTE_SLACK:
        return False
    here = (math.radians(lat), math.radians(lon))
    origin, destination = (tuple(map(math.radians, route[end])) for end in ("from_at", "to_at"))
    near_origin = arc(here, origin) * EARTH_RADIUS < NEAR_AIRPORT
    near_destination = arc(here, destination) * EARTH_RADIUS < NEAR_AIRPORT
    if near_origin != near_destination:  # (both for a hop too short to tell)
        return plane.climb > -CLIMBING if near_origin else plane.climb < CLIMBING
    if near_origin or plane.heading is None:
        return True
    turn = math.degrees(bearing(here, destination)) - plane.heading
    return abs((turn + 180) % 360 - 180) < ROUTE_TURN


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


# --- The card ---------------------------------------------------------------


class Flights:
    FRAME_SECONDS = FRAME_SECONDS

    def __init__(self, width: int, height: int) -> None:
        if (width, height) != (WIDTH, HEIGHT):
            raise ValueError(f"the card is laid out for a {WIDTH}x{HEIGHT} panel")
        self.frame_count = 0
        self.title_font = BitmapFont("5x7")
        self.font = BitmapFont("tom-thumb")
        self.cache: dict = {}
        self.icons = {name: np.array([[c == "#" for c in row] for row in rows], np.float32)
                      for name, rows in (("up", UP_ARROW), ("down", DOWN_ARROW), ("route", ROUTE_ARROW))}
        self.planes: dict[str, Plane] = {}
        self.polled = 0
        self.current: Plane | None = None
        self.since = -math.inf  # when the current plane's turn began
        self.changed = -math.inf  # when the card last changed planes (it fades in)
        self.card = None
        self.card_key = None
        self.feed = Feed()

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
                    plane.update(report)
        for key in [k for k, p in self.planes.items() if now - p.heard > STALE_SECONDS]:
            del self.planes[key]

    def in_view(self, now: float) -> list[tuple[Plane, float, float]]:
        """(plane, distance, azimuth) for each plane out the window, nearest first."""
        seen = []
        for plane in self.planes.values():
            east, north, up = plane.position(now)
            if abs(east) > VIEW_RANGE or abs(north) > VIEW_RANGE:
                continue
            az, el, distance = look(east, north, up)
            if distance < VIEW_RANGE and el >= MIN_ELEVATION and VIEW_LEFT <= az <= VIEW_RIGHT:
                seen.append((plane, distance, az))
        seen.sort(key=lambda s: s[1])
        return seen

    def choose(self, seen: list, now: float) -> None:
        """Give the card to a close plane while it's close, else to each in
        view in turn, the one shown longest ago first."""
        distances = {plane.id: distance for plane, distance, az in seen}
        current = self.current
        if current is not None and current.id in distances:
            if distances[current.id] < CLOSE or now - self.since < CARD_SECONDS:
                return
        if not seen:
            if current is not None:
                self.current, self.since = None, now
                self.changed = now
            return
        close = [plane for plane, distance, az in seen if distance < CLOSE]
        choice = close[0] if close else min(seen, key=lambda s: (s[0].shown, s[1]))[0]
        if choice is not current:  # (a plane taking another turn doesn't fade in again)
            self.current, self.changed = choice, now
        choice.shown = self.since = now

    # The card -------------------------------------------------------------

    def text(self, text: str, font: BitmapFont | None = None) -> np.ndarray:
        font = font or self.font
        key = (text, id(font))
        found = self.cache.get(key)
        if found is None:
            if len(self.cache) > 300:
                self.cache.clear()
            found = self.cache[key] = font.render(text, proportional=True)
        return found

    def width(self, text: str, font: BitmapFont | None = None) -> int:
        return self.text(text, font).shape[1]

    def fit(self, options: list, room: int, font: BitmapFont | None = None) -> str:
        """The first of options (all non-empty strings) that fits in room
        columns, else the last cut down word by word."""
        for option in options:
            if option and self.width(option, font) <= room:
                return option
        words = (options[-1] if options else "").split()
        while len(words) > 1 and self.width(" ".join(words), font) > room:
            words.pop()
        return " ".join(words)

    def title(self, plane: Plane, route: dict | None) -> tuple[str, tuple]:
        """The flight (airline and number) or registration, and its color.
        (route here may be out of date: its airline and flight number come
        from the callsign, so they're right even so.)"""
        match = CALLSIGN.match(plane.callsign)
        room = WIDTH - 2 * MARGIN
        if not match:
            # (a TIS-B target's made-up address starts with ~)
            name = plane.registration or plane.callsign or plane.id.lstrip("~").upper()
            return self.fit([name], room, self.title_font), MILITARY if plane.military else WHITE
        prefix, number = match.groups()
        known = AIRLINES.get(prefix)
        color = known[1] if known else MILITARY if plane.military else DEFAULT_AIRLINE
        airline = known[0] if known else short_airline(route["airline"]) if route and route.get("airline") else None
        flight = route.get("flight") if route else None
        options = [f"{airline} {number}" if airline else None,
                   f"{flight[:2]} {flight[2:]}" if flight else None, plane.callsign]
        return self.fit([o for o in options if o], room, self.title_font), color

    def operator(self, plane: Plane) -> str | None:
        """A non-airline plane's operator, if it's a business or agency (never
        a private owner's name). Several owners, or a bank holding it in
        trust, say nothing about who flies it."""
        owner = plane.owner.upper()
        if not owner or "," in owner or "TRUSTEE" in owner:
            return None
        words = [w for w in plane.owner.split() if w.lower().strip(".") not in COMPANY_WORDS]
        if not words or " ".join(words).upper() == plane.registration.upper():
            return None  # ("N123AB LLC")
        if not BUSINESS.search(owner) and not (len(words) == 1 and len(plane.owner.split()) > 1):
            return None
        return " ".join(w.lower() if i and w.lower() in SMALL_WORDS
                        else w if len(w) <= 3 and w.isupper() and w != "AIR" else w.capitalize()
                        for i, w in enumerate(words))

    def lines(self, plane: Plane, distance: float, az: float, now: float) -> list:
        """The card for a plane: (row top, [(text or icon, color, x, font)])."""
        airline = CALLSIGN.match(plane.callsign) is not None
        found = self.feed.route(plane.callsign) if airline else None
        route = found if found and plausible(found, plane) else None
        title, color = self.title(plane, found)
        rows = [(TITLE_TOP, [(title, color, MARGIN, self.title_font)])]
        room = WIDTH - 2 * MARGIN

        # Route, an emergency, or who operates it
        emergency = EMERGENCIES.get(plane.squawk)
        if emergency:
            if (now % 1.0) < 0.7:
                rows.append((LINE_TOPS[0], [(f"{emergency} {plane.squawk}", ALERT, MARGIN, None)]))
        elif route:
            a = self.width(route["from"])
            rows.append((LINE_TOPS[0], [(route["from"], WHITE, MARGIN, None),
                                        ("route", DIM, MARGIN + a + 2, None),
                                        (route["to"], WHITE, MARGIN + a + 6, None)]))
        elif not airline:
            who = self.operator(plane)
            if who:
                rows.append((LINE_TOPS[0], [(self.fit([who], room), DIM, MARGIN, None)]))

        # Type, and how far and which way
        miles = distance / METERS_PER_MILE
        where = f"{miles:.1f}mi {COMPASS[round(az / 45) % 8]}" if miles < 9.95 else \
            f"{miles:.0f}mi {COMPASS[round(az / 45) % 8]}"
        right = WIDTH - MARGIN - self.width(where)
        kind = self.fit([type_name(plane), plane.type], right - MARGIN - 3)
        rows.append((LINE_TOPS[1], [(kind, DIM, MARGIN, None), (where, DIM, right, None)]))

        # Altitude, climbing or descending, and speed
        parts = []
        x = MARGIN
        if abs(plane.climb) >= CLIMBING:
            up = plane.climb > 0
            parts.append(("up" if up else "down", CLIMB_COLOR if up else DESCEND_COLOR, x, None))
            x += 4
        altitude = f"{round(plane.feet / 100) * 100}ft"
        parts.append((altitude, WHITE, x, None))
        speed = f"{round(plane.knots * MPH_PER_KNOT)}mph"
        parts.append((speed, WHITE, WIDTH - MARGIN - self.width(speed), None))
        rows.append((LINE_TOPS[2], parts))
        return rows

    def idle(self, now: float) -> list:
        """With no plane in view: the time, and why there's no card."""
        clock = datetime.fromtimestamp(now, TIMEZONE)
        label = f"{clock.hour % 12 or 12}:{clock.minute:02d} {'AM' if clock.hour < 12 else 'PM'}"
        quiet = "no flight data" if time.time() - self.feed.polled_at > NO_DATA_SECONDS else "no planes west"
        return [(TITLE_TOP + 6, [(label, WHITE, (WIDTH - self.width(label, self.title_font)) // 2, self.title_font)]),
                (LINE_TOPS[1] + 1, [(quiet, DIM, (WIDTH - self.width(quiet)) // 2, None)])]

    def render(self, rows: list) -> np.ndarray:
        frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
        for top, parts in rows:
            for part, color, x, font in parts:
                art = self.icons.get(part) if font is None else None
                y = top
                if art is None:
                    art = self.text(part, font)
                else:  # an icon sits on the small text's baseline
                    y += self.font.ascent - art.shape[0]
                w = min(art.shape[1], WIDTH - x)
                if w > 0 and x >= 0:
                    frame[y:y + art.shape[0], x:x + w] += art[:, :w, None] * np.array(color, np.float32)
        return frame

    def dots(self, frame: np.ndarray, seen: list) -> None:
        """Each plane in view as a dot along the bottom, where it is across
        the window (south at the left); the one on the card lit."""
        span = VIEW_RIGHT - VIEW_LEFT
        for plane, distance, az in seen:
            x = MARGIN + round(min(1.0, max(0.0, (az - VIEW_LEFT) / span)) * (WIDTH - 1 - 2 * MARGIN))
            if plane is self.current:
                left = min(x, WIDTH - MARGIN - 2)  # (2 wide, inside the margin)
                frame[DOTS_ROW, left:left + 2] = WHITE
            else:
                frame[DOTS_ROW, x] = np.maximum(frame[DOTS_ROW, x], FAINT)

    # The frame ------------------------------------------------------------

    def frame(self) -> np.ndarray:
        now = time.time()
        self.track(now)
        seen = self.in_view(now)
        self.choose(seen, now)
        if self.current is not None:
            distance, az = next((d, a) for p, d, a in seen if p is self.current)
            rows = self.lines(self.current, distance, az, now)
        else:
            rows = self.idle(now)
        key = repr(rows)
        if key != self.card_key:
            self.card, self.card_key = self.render(rows), key
        fade = min(1.0, (now - self.changed) / FADE_SECONDS)
        frame = self.card * fade if fade < 1 else self.card.copy()
        self.dots(frame, seen)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


SCENE = Flights


def main() -> None:
    run_scene(Flights, "Flights out the west window LED matrix display")


if __name__ == "__main__":
    main()
