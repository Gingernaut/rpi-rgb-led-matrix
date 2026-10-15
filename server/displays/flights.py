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
the two airports, since its routes are often out of date. A plane squawking
an emergency code says so in red.

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
FETCH_RETRY = 30
FETCH_RETRY_MAX = 600
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "flights"
USER_AGENT = "pixel-display (home LED matrix; github.com/Gingernaut/rpi-rgb-led-matrix)"

# A route is shown only if the plane is within ROUTE_SLACK of the great
# circle between its endpoints
ROUTE_SLACK = 250_000

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
TITLE_TOP = 0
LINE_TOPS = (9, 16, 23)
DOTS_ROW = 30
MARGIN = 1  # columns left clear at each side

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
# Words that mark a non-airline operator as a business (a private owner's
# name isn't shown)
BUSINESS = re.compile(r"\b(INC|LLC|CO|CORP|CORPORATION|COMPANY|LTD|LP|AIRLINES?|AIR|AVIATION|FLIGHT|"
                      r"UNIVERSITY|COUNTY|STATE|POLICE|SHERIFF|PATROL|NETWORK|SERVICES?|BANK|TRUST)\b")
COMPANY_WORDS = {"inc", "llc", "co", "corp", "ltd", "lp"}


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
        self.climb = report.get("geom_rate", report.get("baro_rate")) or 0
        self.knots = report.get("gs") or 0
        track = math.radians(report.get("track") or 0)
        speed = self.knots * KNOTS
        self.anchor = (*local(report["lat"], report["lon"]), self.feet * METERS_PER_FOOT - HOME_HEIGHT)
        self.velocity = (speed * math.sin(track), speed * math.cos(track), self.climb * FEET_PER_MINUTE)
        self.heard = report["at"]

    def position(self, now: float) -> tuple[float, float, float]:
        dt = now - self.heard
        return tuple(a + v * dt for a, v in zip(self.anchor, self.velocity))


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
                "flight": route.get("callsign_iata"),
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
        self.since = -math.inf
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
            return
        close = [plane for plane, distance, az in seen if distance < CLOSE]
        choice = close[0] if close else min(seen, key=lambda s: (s[0].shown, s[1]))[0]
        if choice is not current:
            self.current = choice
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
        """The flight (airline and number) or registration, and its color."""
        match = re.match(r"^([A-Z]{3})0*(\d[0-9A-Z]*)$", plane.callsign)
        room = WIDTH - 2 * MARGIN
        if not match:
            name = plane.registration or plane.callsign or plane.id.upper()
            return name, MILITARY if plane.military else WHITE
        prefix, number = match.groups()
        known = AIRLINES.get(prefix)
        color = known[1] if known else MILITARY if plane.military else DEFAULT_AIRLINE
        airline = known[0] if known else short_airline(route["airline"]) if route and route.get("airline") else None
        flight = route.get("flight") if route else None
        options = [f"{airline} {number}" if airline else None,
                   f"{flight[:2]} {flight[2:]}" if flight else None, plane.callsign]
        return self.fit([o for o in options if o], room, self.title_font), color

    def operator(self, plane: Plane) -> str | None:
        """A non-airline plane's operator, if it's a business or agency."""
        if not plane.owner or "," in plane.owner or not BUSINESS.search(plane.owner.upper()):
            return None
        words = [w for w in plane.owner.split() if w.lower().strip(".") not in COMPANY_WORDS]
        return " ".join(w if len(w) <= 3 and w.isupper() and w not in ("AIR",) else w.title() for w in words)

    def lines(self, plane: Plane, distance: float, az: float, now: float) -> list:
        """The card for a plane: (row top, [(text or icon, color, x, font)])."""
        route = None
        airline = re.match(r"^[A-Z]{3}\d", plane.callsign) is not None
        if airline:
            found = self.feed.route(plane.callsign)
            if found:
                lat = HOME_LAT + plane.anchor[1] / M_PER_LAT
                lon = HOME_LON + plane.anchor[0] / M_PER_LON
                if off_route(found, lat, lon) < ROUTE_SLACK:
                    route = found
        title, color = self.title(plane, route)
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
        where = f"{miles:.1f}mi {COMPASS[round(az / 45) % 8]}" if miles < 10 else \
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
            x = min(WIDTH - 1, max(0, round((az - VIEW_LEFT) / span * (WIDTH - 1))))
            if plane is self.current:
                frame[DOTS_ROW:DOTS_ROW + 2, max(0, x - 1):x + 1] = WHITE
            else:
                frame[DOTS_ROW + 1, x] = np.maximum(frame[DOTS_ROW + 1, x], FAINT)

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
        fade = min(1.0, (now - self.since) / FADE_SECONDS)
        frame = self.card * fade if fade < 1 else self.card.copy()
        self.dots(frame, seen)
        self.frame_count += 1
        return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)


SCENE = Flights


def main() -> None:
    run_scene(Flights, "Flights out the west window LED matrix display")


if __name__ == "__main__":
    main()
