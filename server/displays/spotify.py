#!/usr/bin/env python
"""Spotify "now playing" display for the RGB LED matrix.

Shows album art, artist name, and song title when music is playing.
Shows the Spotify logo when idle.

Requires CLIENT_ID and CLIENT_SECRET environment variables (log in once with
spotify_auth.py) and the rgbmatrix Python bindings (make build-python).
"""

import argparse
import io
import os
import sys
import threading
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyOauthError
from PIL import Image
from pydantic import BaseModel, field_validator

from matrix_common import FramePacer, create_matrix, parse_matrix_args, run_display
from spotify_auth import PROJECT_ROOT, make_auth_manager

MEDIA_DIR = Path(__file__).resolve().parent / "media"
FONTS_DIR = PROJECT_ROOT / "fonts"

# How often to ask Spotify what's playing, and the cap for backing off on errors
POLL_SECONDS = 5.0
MAX_POLL_BACKOFF_SECONDS = 60.0

# Frame period. Scrolling moves 1 px per frame and was tuned at the old cadence:
# sleep(0.035) rounded up to three refreshes of the panel's 75 Hz refresh =
# 40 ms (25 fps). Frames follow a fixed deadline schedule, so the scroll speed
# stays the same at other refresh rates too.
FRAME_SECONDS = 3 / 75

ICON_SIZE = 30

COMBINED_FONT = "6x9"  # "artist - title" on one scrolling line
SPLIT_FONT = "5x8"     # artist and title on separate lines


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
        return COMBINED_FONT if self.should_combine_text() else SPLIT_FONT


def describe_poll_error(e: Exception) -> str:
    # invalid_grant: the saved refresh token was revoked or expired. EOFError:
    # there's no usable saved token, so spotipy fell back to an interactive login
    # prompt, and there's no stdin under the service.
    if isinstance(e, EOFError) or (isinstance(e, SpotifyOauthError) and e.error == "invalid_grant"):
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
        # Spotify client, only used from the poller thread once run() starts
        self.sp = spotipy.Spotify(
            auth_manager=make_auth_manager(os.environ["CLIENT_ID"], os.environ["CLIENT_SECRET"])
        )
        self.http = requests.Session()

        # Spotify idle logo
        icon = Image.open(MEDIA_DIR / "spotify.png").convert("RGB")
        self.icon_rgb = icon.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS).tobytes()

        self.matrix = create_matrix(args)
        from rgbmatrix import graphics  # only on the Pi, like create_matrix's import
        self.graphics = graphics
        self.fonts = {}
        for name in (COMBINED_FONT, SPLIT_FONT):
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
        self.stopping = threading.Event()
        self.last_poll_error: str | None = None

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

    def try_poll(self) -> bool:
        """poll_once(), logging failures once each (and recovery). Returns success."""
        try:
            self.poll_once()
        except Exception as e:  # keep showing the last state; the caller retries
            error = describe_poll_error(e)
            if error != self.last_poll_error:
                print(f"spotify: poll failed, retrying: {error}", file=sys.stderr, flush=True)
            self.last_poll_error = error
            return False
        if self.last_poll_error is not None:
            print("spotify: poll recovered", file=sys.stderr, flush=True)
        self.last_poll_error = None
        return True

    def poll_loop(self, delay: float) -> None:
        while not self.stopping.wait(delay):
            delay = POLL_SECONDS if self.try_poll() else min(delay * 2, MAX_POLL_BACKOFF_SECONDS)

    def run(self) -> None:
        graphics = self.graphics
        # First poll before the first frame, so a playing song doesn't flash the logo
        delay = POLL_SECONDS if self.try_poll() else 2 * POLL_SECONDS
        threading.Thread(target=self.poll_loop, args=(delay,), name="spotify-poll", daemon=True).start()

        canvas = self.matrix.CreateFrameCanvas()
        text_color = graphics.Color(255, 255, 255)
        scroll_x = canvas.width
        font = None
        shown = object()  # what's on the panel: a NowPlaying, None (logo), or unset
        pacer = FramePacer(self.matrix, FRAME_SECONDS)

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
                pacer.reset()
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
            canvas = pacer.swap(canvas)

    def cleanup(self) -> None:
        self.stopping.set()


def main() -> None:
    run_display(SpotifyDisplay(parse_matrix_args("Spotify LED matrix display")))


if __name__ == "__main__":
    main()
