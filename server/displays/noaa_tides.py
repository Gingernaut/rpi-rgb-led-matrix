"""NOAA tide predictions for a station: its highs and lows, fetched by a
background thread (twice a day, cached in .cache/tides/<station>.json), and
the tide's height at any moment, with cosine curves between them.

tide_pool.py (the water in its pool) and tides.py (a 48-hour tide chart)
share this.
"""

import bisect
import json
import math
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
MAX_AGE = 12 * 3600  # refetch twice a day
DAYS = 4  # of predictions fetched, starting yesterday
TIDE_DAY = 24.84 * 3600  # a lunar day: the tides come round again about this often
FETCH_RETRY = 300
FETCH_RETRY_MAX = 3600
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "tides"


def log(message: str) -> None:
    print(f"noaa_tides: {message}", file=sys.stderr, flush=True)


class Predictions:
    """A station's predicted highs and lows, kept fresh by a background
    thread. Readers take `points`, (times, heights, kinds) with kinds "H" or
    "L", swapped in whole so the thread can't be caught halfway through
    replacing them."""

    def __init__(self, station: str) -> None:
        self.station = station
        self.cache = CACHE_DIR / f"{station}.json"
        self.points: tuple[list[float], list[float], list[str]] = ([], [], [])
        self.fetched = 0.0  # when the predictions in hand were fetched
        self.load()
        threading.Thread(target=self.loop, name=f"tides-{station}", daemon=True).start()

    def feet(self, t: float) -> float | None:
        """The tide's height (feet above mean lower low water) at t. Past the
        end of the predictions (NOAA unreachable for days), the same tide a
        whole number of lunar days earlier; None before any have arrived."""
        times, heights, _ = self.points
        if len(times) < 2:
            return None
        if t >= times[-1]:
            t -= (math.floor((t - times[-1]) / TIDE_DAY) + 1) * TIDE_DAY
        i = bisect.bisect_right(times, t)
        if not 0 < i < len(times):
            return None
        k = (t - times[i - 1]) / (times[i] - times[i - 1])
        return heights[i - 1] + (heights[i] - heights[i - 1]) * (1 - math.cos(math.pi * k)) / 2

    def events(self, start: float, end: float) -> list[tuple[float, float, str]]:
        """The highs and lows from start to end: (time, feet, "H" or "L")."""
        times, heights, kinds = self.points
        i = bisect.bisect_left(times, start)
        j = bisect.bisect_right(times, end)
        return list(zip(times[i:j], heights[i:j], kinds[i:j]))

    def load(self) -> None:
        try:
            self.parse(json.loads(self.cache.read_text()))
            self.fetched = self.cache.stat().st_mtime
        except (OSError, ValueError, KeyError, TypeError):
            pass

    def parse(self, data: dict) -> None:
        points = []
        for p in data["predictions"]:
            when = datetime.strptime(p["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            points.append((when.timestamp(), float(p["v"]), p.get("type", "")))
        points.sort()
        if len(points) < 2:
            raise ValueError("no predictions")
        self.points = ([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])

    def loop(self) -> None:
        retry = FETCH_RETRY
        while True:
            if time.time() - self.fetched < MAX_AGE:
                time.sleep(600)
                continue
            try:
                start = datetime.now(timezone.utc) - timedelta(days=1)
                r = requests.get(URL, timeout=15, params={
                    "product": "predictions", "application": "pixel_display",
                    "begin_date": start.strftime("%Y%m%d"), "range": DAYS * 24,
                    "datum": "MLLW", "station": self.station, "time_zone": "gmt",
                    "units": "english", "interval": "hilo", "format": "json"})
                r.raise_for_status()
                data = r.json()
                self.parse(data)
                self.fetched = time.time()
                retry = FETCH_RETRY
            except Exception as e:  # keep what we have; try again later
                log(f"station {self.station} fetch failed, retrying: {type(e).__name__}: {e}")
                time.sleep(retry)
                retry = min(retry * 2, FETCH_RETRY_MAX)
                continue
            try:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                self.cache.write_text(json.dumps(data))
            except OSError as e:  # the predictions are in hand; only the next start misses them
                log(f"couldn't cache station {self.station}'s tides: {e}")
