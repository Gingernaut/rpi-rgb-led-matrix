#!/usr/bin/env python
"""Spotify "now playing" display for the RGB LED matrix.

Shows album art, artist name, and song title when music is playing.
Shows the Spotify logo when idle.

Requires CLIENT_ID and CLIENT_SECRET environment variables.
Requires rgbmatrix Python bindings installed (make build-python && make install-python).
"""

import argparse
import io
import os
import signal
import sys
import threading
import time
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from PIL import Image
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = Path(__file__).resolve().parent / "media"
FONTS_DIR = PROJECT_ROOT / "fonts"
SPOTIFY_TOKEN_CACHE = PROJECT_ROOT / ".spotify_cache"

# How often to ask Spotify what's playing, and the cap for backing off on errors
POLL_SECONDS = 5.0
MAX_POLL_BACKOFF_SECONDS = 60.0

# Frame period. Scrolling moves 1 px per frame and was tuned at the old cadence:
# sleep(0.035) rounded up to three refreshes of the panel's 75 Hz refresh =
# 40 ms (25 fps). Frames follow a fixed deadline schedule, so the scroll speed
# stays the same at other refresh rates too.
FRAME_SECONDS = 3 / 75
# Wake this long before each deadline; SwapOnVSync then waits for the next
# vsync. Keep it under one refresh period (13.3 ms at 75 Hz).
VSYNC_MARGIN_SECONDS = 0.003

ICON_SIZE = 30

# Lazy import — rgbmatrix only exists on the Pi after make install-python
rgbmatrix = None
graphics = None


def import_rgbmatrix():
    global rgbmatrix, graphics
    from rgbmatrix import RGBMatrix, RGBMatrixOptions, FrameCanvas
    from rgbmatrix import graphics as _graphics

    rgbmatrix = type(sys)("rgbmatrix")
    rgbmatrix.RGBMatrix = RGBMatrix
    rgbmatrix.RGBMatrixOptions = RGBMatrixOptions
    rgbmatrix.FrameCanvas = FrameCanvas
    graphics = _graphics


class CurrentSong(BaseModel):
    artist: str
    title: str
    album_cover: str

    @field_validator("artist")
    @classmethod
    def shorten_long_names(cls, v: str) -> str:
        if v.lower().strip() == "red hot chili peppers":
            return "RHCP"
        return v

    def should_combine_text(self) -> bool:
        return len(self.artist) > 6

    def should_scroll_title(self) -> bool:
        return len(self.title) > 6

    def get_font_name(self) -> str:
        return "6x9" if self.should_combine_text() else "5x8"


def describe_poll_error(e: Exception) -> str:
    if isinstance(e, EOFError):
        # spotipy falls back to an interactive login prompt when the saved token
        # can't be refreshed, and there's no stdin under the service
        return "Spotify login expired; run server/displays/spotify_auth.py to log in again"
    return f"{type(e).__name__}: {e}"


class NowPlaying:
    """A song plus its pre-rendered album-art block, published by the poller."""

    __slots__ = ("song", "art_rgb")

    def __init__(self, song: CurrentSong, art_rgb: bytes) -> None:
        self.song = song
        self.art_rgb = art_rgb


class SpotifyDisplay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Spotify client, only used from the poller thread once run() starts
        self.sp = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=os.environ["CLIENT_ID"],
                client_secret=os.environ["CLIENT_SECRET"],
                redirect_uri="http://127.0.0.1:8888/callback/",
                scope="user-read-currently-playing user-library-read",
                cache_path=str(SPOTIFY_TOKEN_CACHE),
                open_browser=False,
            )
        )
        self.http = requests.Session()

        # Spotify idle logo
        icon = Image.open(MEDIA_DIR / "spotify.png").convert("RGB")
        self.icon_rgb = icon.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS).tobytes()

        # Matrix setup
        import_rgbmatrix()
        options = rgbmatrix.RGBMatrixOptions()
        options.rows = args.rows
        options.cols = args.cols
        options.hardware_mapping = args.gpio_mapping
        options.brightness = args.brightness
        options.pwm_lsb_nanoseconds = args.pwm_lsb_nanoseconds
        options.limit_refresh_rate_hz = args.limit_refresh_rate_hz
        options.drop_privileges = False
        options.gpio_slowdown = args.slowdown_gpio

        self.matrix = rgbmatrix.RGBMatrix(options=options)
        self.fonts = {}
        for name in ("5x8", "6x9"):
            font = graphics.Font()
            font.LoadFont(str(FONTS_DIR / f"{name}.bdf"))
            self.fonts[name] = font

        # Album art sits at (1, 1) inside a black block covering the left of the
        # panel, so scrolling text doesn't show through around it
        self.art_size = self.matrix.height - 2
        self.art_block_width = self.art_size + 2

        # Latest state from the poller: None when nothing is playing
        self.now_playing: NowPlaying | None = None
        self.changed = threading.Event()
        self.stop = threading.Event()

    def fetch_current_song(self) -> CurrentSong | None:
        track = self.sp.current_user_playing_track()
        if not track or not track.get("item"):
            return None

        item = track["item"]
        artist = item["artists"][0]["name"]
        title = item["name"]
        album = item.get("album", {})
        images = album.get("images", [])
        cover_url = images[0]["url"] if images else None

        if not cover_url:
            return None

        return CurrentSong(artist=artist, title=title, album_cover=cover_url)

    def render_album_art(self, url: str) -> bytes:
        r = self.http.get(url, timeout=10)
        r.raise_for_status()
        art = Image.open(io.BytesIO(r.content)).convert("RGB")
        art = art.resize((self.art_size, self.art_size), Image.LANCZOS)
        block = Image.new("RGB", (self.art_block_width, self.matrix.height), (0, 0, 0))
        block.paste(art, (1, 1))
        return block.tobytes()

    def poll_once(self) -> None:
        """Fetch the current song and publish it if it changed. Raises on errors."""
        song = self.fetch_current_song()
        current = self.now_playing
        if song is None:
            if current is not None:
                self.now_playing = None
                self.changed.set()
        elif current is None or current.song != song:
            self.now_playing = NowPlaying(song, self.render_album_art(song.album_cover))
            self.changed.set()

    def poll_loop(self) -> None:
        delay = POLL_SECONDS
        last_error = None
        while not self.stop.wait(delay):
            try:
                self.poll_once()
            except Exception as e:  # keep showing the last state; retry with backoff
                error = describe_poll_error(e)
                if error != last_error:
                    print(f"spotify: poll failed, retrying: {error}", file=sys.stderr, flush=True)
                last_error = error
                delay = min(delay * 2, MAX_POLL_BACKOFF_SECONDS)
            else:
                if last_error is not None:
                    print("spotify: poll recovered", file=sys.stderr, flush=True)
                last_error = None
                delay = POLL_SECONDS

    def run(self) -> None:
        # First poll before the first frame, so a playing song doesn't flash the logo
        try:
            self.poll_once()
        except Exception as e:
            print(f"spotify: poll failed, retrying: {describe_poll_error(e)}", file=sys.stderr, flush=True)
        threading.Thread(target=self.poll_loop, name="spotify-poll", daemon=True).start()

        canvas = self.matrix.CreateFrameCanvas()
        text_color = graphics.Color(255, 255, 255)
        scroll_x = canvas.width
        font = None
        shown = object()  # what's on the panel: a NowPlaying, None (logo), or unset
        next_frame = time.monotonic()

        while True:
            now_playing = self.now_playing

            if now_playing is None:
                # No song playing — draw the Spotify logo once; the panel keeps
                # showing the last swapped frame, so just wait for a change
                if shown is not None:
                    canvas.Clear()
                    canvas.SetPixelsRGB(self.icon_rgb, ICON_SIZE, ICON_SIZE, 1, 1)
                    canvas = self.matrix.SwapOnVSync(canvas)
                    shown = None
                self.changed.wait(timeout=1.0)
                self.changed.clear()
                next_frame = time.monotonic()
                continue

            song = now_playing.song
            if now_playing is not shown:
                shown = now_playing
                font = self.fonts[song.get_font_name()]
                scroll_x = canvas.width

            # Render frame
            canvas.Clear()

            if song.should_combine_text():
                combined = f"{song.artist} - {song.title}"
                text_len = graphics.DrawText(
                    canvas, font, scroll_x, 18, text_color, combined
                )
                scroll_x -= 1
                if scroll_x + text_len < 0:
                    scroll_x = canvas.width
            else:
                # Static artist
                artist_x = 32 + (
                    1
                    if len(song.artist) == 6
                    else int(24 / max(len(song.artist), 1))
                )
                graphics.DrawText(
                    canvas, font, artist_x, 12, text_color, song.artist
                )

                # Title — static or scrolling
                if not song.should_scroll_title():
                    graphics.DrawText(
                        canvas, font, artist_x, 26, text_color, song.title
                    )
                else:
                    text_len = graphics.DrawText(
                        canvas, font, scroll_x, 26, text_color, song.title
                    )
                    scroll_x -= 1
                    if scroll_x + text_len < 0:
                        scroll_x = canvas.width

            # Album art in its black block, drawn over any text that scrolled under it
            canvas.SetPixelsRGB(now_playing.art_rgb, self.art_block_width, self.matrix.height)

            wait = next_frame - VSYNC_MARGIN_SECONDS - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            canvas = self.matrix.SwapOnVSync(canvas)
            next_frame += FRAME_SECONDS
            now = time.monotonic()
            if now > next_frame:  # fell a whole frame behind; don't try to catch up
                next_frame = now + FRAME_SECONDS

    def cleanup(self) -> None:
        self.stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spotify LED matrix display")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cols", type=int, default=64)
    parser.add_argument("--gpio-mapping", default="adafruit-hat")
    parser.add_argument("--brightness", type=int, default=50)
    parser.add_argument("--slowdown-gpio", type=int, default=4)
    parser.add_argument("--pwm-lsb-nanoseconds", type=int, default=300)
    parser.add_argument("--limit-refresh-rate-hz", type=int, default=150)
    args = parser.parse_args()

    display = SpotifyDisplay(args)

    def handle_signal(signum, frame):
        display.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        display.run()
    finally:
        display.cleanup()


if __name__ == "__main__":
    main()
