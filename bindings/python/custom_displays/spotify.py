#!/usr/bin/env python
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import requests
from pydantic import BaseModel
import time
from samplebase import SampleBase
from rgbmatrix import graphics
from PIL import Image
from typing import Optional
from cachetools import cached, TTLCache
import os

from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8888/callback/"
SCOPE = "user-read-currently-playing user-library-read"

sp = spotipy.Spotify(
    auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
    )
)
# results = sp.current_user_saved_tracks()
# for idx, item in enumerate(results['items']):
#     track = item['track']
#     print(idx, track['artists'][0]['name'], " – ", track['name'])


class CurrentSong(BaseModel):
    artist: str
    title: str
    album_cover: str

    def should_combine_text(self) -> bool:
        return len(self.artist) > 6

    def should_scroll_title(self) -> bool:
        return len(self.title) > 6

    def get_font_size(self) -> str:
        if self.should_combine_text():
            return "6x9"
        return "5x8"

    def get_art_filename(self) -> str:
        return f"{self.title.replace(' ','_')}"

    def download_album_art(self) -> str:
        print(f"downloading album art for {self}")
        # To save to a relative path.
        r = requests.get(self.album_cover)
        filepath = self.get_album_art_path()
        with open(filepath, "wb") as f:
            f.write(r.content)
        return filepath

    def get_album_art_path(self):
        return f"media/{self.get_art_filename()}.png"

    def __repr__(self):
        return f"{self.artist} - {self.title}"


class SongScroller(SampleBase):
    current_song = None

    current_album_image = None

    @cached(cache=TTLCache(maxsize=1, ttl=5))
    def get_current_playing_song(self) -> Optional[CurrentSong]:
        print("checking spotify for song")

        track = sp.current_user_playing_track()

        if track:

            track = track["item"]
            artist = track.get("artists")[0]["name"]
            song_title = track.get("name")

            album = track.get("album")
            album_art = None
            if album.get("images") and len(album.get("images")) > 0:
                album_art = album.get("images")[0]["url"]

            return CurrentSong(artist=artist, title=song_title, album_cover=album_art)
        else:
            print("spotify returned no track :(")
        return None

    def __init__(self, *args, **kwargs):
        super(SongScroller, self).__init__(*args, **kwargs)
        self.parser.add_argument(
            "-d", "--delay", help="how long to pause before scrolling", default=0.035
        )
        self.parser.add_argument(
            "-f", "--font", help="which font size to choose", default="5x8"
        )

        self.current_song = None

    def run(self):

        double_buffer = self.matrix.CreateFrameCanvas()
        delay = float(self.args.delay)
        font = graphics.Font()

        textColor = graphics.Color(255, 255, 255)

        scroll_text_start_x = double_buffer.width
        while True:
            # only check current track every 5/10 seconds
            song = self.get_current_playing_song()

            if song:
                if self.current_song == song:
                    print("already have the right song info!", end="\r")
                else:
                    print("have new song")

                    old_art = [f for f in os.listdir("media/") if f.endswith(".png")]
                    for f in old_art:
                        print(f"deleting {f}")
                        os.remove(os.path.join("media", f))

                    self.current_song = song
                    self.current_song.download_album_art()

                    img_path = self.current_song.get_album_art_path()
                    image = Image.open(img_path).convert("RGB")
                    img_size = self.matrix.height - 2
                    self.current_album_image = image.resize(
                        (img_size, img_size), Image.LANCZOS
                    )

                    chosen_font = f"{song.get_font_size()}.bdf"
                    font.LoadFont(f"../../fonts/{chosen_font}")

            else:
                # show spotify icon?
                print("no playing song", end="\r")
                self.current_song = None

            double_buffer.Clear()

            if self.current_song:

                if self.current_song.should_combine_text():
                    vertical_offset = 18

                    combined_text = (
                        f"{self.current_song.artist} - {self.current_song.title}"
                    )
                    i = graphics.DrawText(
                        double_buffer,
                        font,
                        scroll_text_start_x,
                        vertical_offset,
                        textColor,
                        combined_text,
                    )

                    scroll_text_start_x -= 1
                    if scroll_text_start_x + i < 0:
                        scroll_text_start_x = double_buffer.width
                else:
                    artist_vertical_offset = 12
                    song_vertical_offset = 26

                    artist_start_pos = 32

                    if len(self.current_song.artist) == 6:
                        artist_start_pos += 1
                    else:
                        artist_start_pos += int(24 / len(self.current_song.artist))

                    # artist
                    i = graphics.DrawText(
                        double_buffer,
                        font,
                        artist_start_pos,
                        artist_vertical_offset,
                        textColor,
                        self.current_song.artist,
                    )

                    if not self.current_song.should_scroll_title():
                        i = graphics.DrawText(
                            double_buffer,
                            font,
                            artist_start_pos,
                            song_vertical_offset,
                            textColor,
                            self.current_song.title,
                        )
                    else:
                        s = graphics.DrawText(
                            double_buffer,
                            font,
                            scroll_text_start_x,
                            song_vertical_offset,
                            textColor,
                            self.current_song.title,
                        )

                        scroll_text_start_x -= 1
                        if scroll_text_start_x + s < 0:
                            scroll_text_start_x = double_buffer.width

                # Image
                double_buffer.SetImage(self.current_album_image, 1, 1)

                # left border
                for x in range(0, 1):
                    for y in range(double_buffer.height):
                        double_buffer.SetPixel(x, y, 0, 0, 0)

            time.sleep(delay)
            double_buffer = self.matrix.SwapOnVSync(double_buffer)


# Main function
# e.g. call with
#  sudo ./image-scroller.py --chain=4
# if you have a chain of four
if __name__ == "__main__":
    song_scroller = SongScroller()
    if not song_scroller.process():
        song_scroller.print_help()
