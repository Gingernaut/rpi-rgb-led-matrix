#!/usr/bin/env python
"""Spotify "now playing" display for the RGB LED matrix.

Shows album art, artist name, and song title when music is playing.
Shows the Spotify logo when idle.

Requires CLIENT_ID and CLIENT_SECRET environment variables.
Requires rgbmatrix Python bindings installed (make build-python && make install-python).
"""

import argparse
import os
import signal
import shutil
import sys
import tempfile
import time
from pathlib import Path

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from PIL import Image
from pydantic import BaseModel, field_validator
from cachetools import cached, TTLCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = Path(__file__).resolve().parent / "media"
FONTS_DIR = PROJECT_ROOT / "fonts"
SPOTIFY_TOKEN_CACHE = PROJECT_ROOT / ".spotify_cache"

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

    def download_album_art(self, dest_dir: Path) -> Path:
        safe_name = self.title.replace(" ", "_").replace("/", "_")
        filepath = dest_dir / f"{safe_name}.png"
        r = requests.get(self.album_cover)
        filepath.write_bytes(r.content)
        return filepath


class SpotifyDisplay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="spotify_art_"))
        self.current_song: CurrentSong | None = None
        self.current_album_image: Image.Image | None = None

        # Spotify client
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

        # Spotify idle logo
        icon = Image.open(MEDIA_DIR / "spotify.png").convert("RGB")
        self.spotify_icon = icon.resize((30, 30), Image.LANCZOS)

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
        self.font = graphics.Font()

        # Pre-built black image for fast album-region clearing
        img_size = self.matrix.height - 2
        self.black_rect = Image.new("RGB", (img_size + 2, self.matrix.height), (0, 0, 0))

    @cached(cache=TTLCache(maxsize=1, ttl=5))
    def get_current_song(self) -> CurrentSong | None:
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

    def run(self) -> None:
        canvas = self.matrix.CreateFrameCanvas()
        text_color = graphics.Color(255, 255, 255)
        scroll_x = canvas.width
        delay = 0.035

        while True:
            song = self.get_current_song()

            if song:
                # Song changed — download new art
                if self.current_song != song:
                    # Clean old art
                    for f in self.tmp_dir.glob("*.png"):
                        f.unlink()

                    self.current_song = song
                    art_path = song.download_album_art(self.tmp_dir)

                    image = Image.open(art_path).convert("RGB")
                    img_size = self.matrix.height - 2
                    self.current_album_image = image.resize(
                        (img_size, img_size), Image.LANCZOS
                    )

                    font_file = FONTS_DIR / f"{song.get_font_name()}.bdf"
                    self.font.LoadFont(str(font_file))
                    scroll_x = canvas.width

                # Render frame
                canvas.Clear()

                if song.should_combine_text():
                    combined = f"{song.artist} - {song.title}"
                    text_len = graphics.DrawText(
                        canvas, self.font, scroll_x, 18, text_color, combined
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
                        canvas, self.font, artist_x, 12, text_color, song.artist
                    )

                    # Title — static or scrolling
                    if not song.should_scroll_title():
                        graphics.DrawText(
                            canvas, self.font, artist_x, 26, text_color, song.title
                        )
                    else:
                        text_len = graphics.DrawText(
                            canvas, self.font, scroll_x, 26, text_color, song.title
                        )
                        scroll_x -= 1
                        if scroll_x + text_len < 0:
                            scroll_x = canvas.width

                # Black out the album art region so scrolling text doesn't show through
                canvas.SetImage(self.black_rect, 0, 0)

                # Album art (drawn on top of the blacked-out region)
                canvas.SetImage(self.current_album_image, 1, 1)

            else:
                # No song playing — show Spotify logo
                if self.current_song:
                    canvas.Clear()
                self.current_song = None
                canvas.SetImage(self.spotify_icon, 1, 1)

            canvas = self.matrix.SwapOnVSync(canvas)
            time.sleep(delay)

    def cleanup(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


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
