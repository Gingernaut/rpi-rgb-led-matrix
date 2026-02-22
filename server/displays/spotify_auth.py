#!/usr/bin/env python
"""One-time Spotify OAuth setup for headless systems.

Run this interactively to authorize the Spotify display:
    source ~/dotfiles/credentials.sh
    cd ~/rpi-rgb-led-matrix
    .venv/bin/python server/displays/spotify_auth.py

It will print a URL to visit in your browser. After authorizing,
copy the full redirect URL from your browser's address bar
(even if the page shows "connection refused") and paste it here.
The token is saved to .spotify_cache for the display to reuse.
"""

import os
import sys
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPOTIFY_TOKEN_CACHE = PROJECT_ROOT / ".spotify_cache"
REDIRECT_URI = "http://127.0.0.1:8888/callback/"
SCOPE = "user-read-currently-playing user-library-read"


def main() -> None:
    client_id = os.environ.get("CLIENT_ID")
    client_secret = os.environ.get("CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Error: CLIENT_ID and CLIENT_SECRET must be set")
        sys.exit(1)

    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=str(SPOTIFY_TOKEN_CACHE),
        open_browser=False,
    )

    # Check if we already have a valid cached token
    token_info = auth_manager.cache_handler.get_cached_token()
    if token_info and not auth_manager.is_token_expired(token_info):
        print("Already authenticated! Token is cached at .spotify_cache")
        sp = spotipy.Spotify(auth_manager=auth_manager)
        user = sp.current_user()
        print(f"Logged in as: {user['display_name']}")
        return

    # Manual auth flow — no local server needed
    auth_url = auth_manager.get_authorize_url()
    print()
    print("1. Open this URL in your browser:")
    print(f"   {auth_url}")
    print()
    print("2. Authorize the app")
    print()
    print("3. You'll be redirected to a page that won't load (connection refused).")
    print("   That's OK — copy the FULL URL from your browser's address bar.")
    print()

    response_url = input("4. Paste the redirect URL here: ").strip()

    code = auth_manager.parse_response_code(response_url)
    auth_manager.get_access_token(code, as_dict=False)

    print()
    print("Success! Token saved to .spotify_cache")

    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    print(f"Logged in as: {user['display_name']}")

    track = sp.current_user_playing_track()
    if track and track.get("item"):
        name = track["item"]["name"]
        artist = track["item"]["artists"][0]["name"]
        print(f"Currently playing: {artist} - {name}")
    else:
        print("No track currently playing.")


if __name__ == "__main__":
    main()
