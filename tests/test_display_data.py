import os
import random
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "displays"))

import iss
import noaa_tides
import planets
import spotify
import spotify_auth
import tide_pool
import tides


TLE = (
    "1 25544U 98067A   24001.00000000  .00012715  00000-0  22751-3 0  9992",
    "2 25544  51.6400  32.8800 0005040  53.2235  68.2180 15.49500000432011",
)


class DisplayDataTest(unittest.TestCase):
    def predictions(self):
        predictions = noaa_tides.Predictions.__new__(noaa_tides.Predictions)
        predictions.parse({"predictions": [
            {"t": "2026-11-21 00:00", "v": "2", "type": "L"},
            {"t": "2026-11-21 06:00", "v": "11", "type": "H"},
            {"t": "2026-11-21 12:00", "v": "1", "type": "L"},
            {"t": "2026-11-22 00:00", "v": "12", "type": "H"},
        ]})
        return predictions

    def test_noaa_predictions_end_at_the_last_published_value(self):
        predictions = self.predictions()
        first, _, _, last = predictions.points[0]
        self.assertEqual(predictions.feet(first), 2)
        self.assertEqual(predictions.feet(last), 12)
        self.assertIsNone(predictions.feet(last + 3600))
        with patch.dict(os.environ, clear=True), patch.object(tide_pool, "Predictions", return_value=predictions):
            artistic_tide = tide_pool.Tide()
        self.assertEqual(artistic_tide.feet(last + tide_pool.TIDE_DAY), 12)

    def test_tide_chart_labels_incomplete_and_expired_data(self):
        predictions = self.predictions()
        last = predictions.points[0][-1]
        with patch.object(tides, "Predictions", return_value=predictions):
            scene = tides.Tides(64, 32)
        with patch.object(scene, "put", wraps=scene.put) as put:
            for now in (last - 600, last - 600 + tides.EVENT_SECONDS):
                with patch.object(tides.time, "time", return_value=now):
                    scene.frame()
            self.assertIn("partial data", [call.args[1] for call in put.call_args_list])
            put.reset_mock()
            with patch.object(tides.time, "time", return_value=last + 3600):
                scene.frame()
            self.assertIn("tide expired", [call.args[1] for call in put.call_args_list])

    def test_iss_cloud_cache_obeys_forecast_expiry(self):
        tracker = iss.IssTracker.__new__(iss.IssTracker)
        p = SimpleNamespace(start=1000, end=1200)
        tracker.feed = SimpleNamespace(
            prediction=None, forecast=iss.Forecast(np.array([0, 2000]), np.array([90, 90]), 0)
        )
        tracker.clouds_for = (None, None)
        tracker.clouds = {}
        with patch.object(iss.time, "time", return_value=iss.FORECAST_STALE - 1):
            self.assertEqual(tracker.cloud(p), 90)
        with patch.object(iss.time, "time", return_value=iss.FORECAST_STALE + 1):
            self.assertIsNone(tracker.cloud(p))

    def test_iss_stale_elements_hide_the_ground_track_and_passes(self):
        orbit = iss.Orbit(*TLE)
        now = orbit.epoch + iss.TLE_STALE + 1
        prediction = iss.OrbitPrediction(orbit, [SimpleNamespace(start=now, end=now + 60)], 31, now)
        feed = SimpleNamespace(prediction=prediction, forecast=None)
        with patch.dict(os.environ, clear=True), patch.object(iss, "Feed", return_value=feed):
            tracker = iss.IssTracker(64, 32)
        with (
            patch.object(iss.time, "time", return_value=now),
            patch.object(tracker, "line", wraps=tracker.line) as line,
            patch.object(tracker, "ground_track", wraps=tracker.ground_track) as ground_track,
        ):
            tracker.frame()
        ground_track.assert_not_called()
        line.assert_called_once_with([("STALE ISS DATA", iss.GRAY)], 64)

    def test_iss_tle_refresh_publishes_the_new_orbit_and_passes_together(self):
        feed = iss.Feed.__new__(iss.Feed)
        orbit = iss.Orbit(*TLE)
        feed.clock = lambda: orbit.epoch
        previous = iss.OrbitPrediction(orbit, [], 2, orbit.epoch)
        feed.prediction = previous
        response = Mock(text="\n".join(TLE))
        passes = [SimpleNamespace(start=orbit.epoch + 3600)]

        def search(new_orbit, now, days):
            self.assertIs(feed.prediction, previous)
            self.assertIsNot(new_orbit, orbit)
            return passes

        with (
            TemporaryDirectory() as directory,
            patch.object(iss.requests, "get", return_value=response),
            patch.object(iss, "find_passes", side_effect=search),
        ):
            feed.fetch_tle(Path(directory, "tle.txt"))
        self.assertIsNot(feed.prediction.orbit, orbit)
        self.assertIs(feed.prediction.passes, passes)
        self.assertEqual(feed.prediction.days, iss.PREDICT_DAYS)

    def test_iss_shows_pending_prediction_after_the_clock_is_corrected(self):
        feed = iss.Feed.__new__(iss.Feed)
        orbit = iss.Orbit(*TLE)
        feed.forecast = None
        feed.clock = lambda: orbit.epoch - iss.TLE_STALE - 1
        feed.prediction = feed.predict(orbit, days=2)
        with patch.dict(os.environ, clear=True), patch.object(iss, "Feed", return_value=feed):
            tracker = iss.IssTracker(64, 32)
        with (
            patch.object(iss.time, "time", return_value=orbit.epoch),
            patch.object(tracker, "line", wraps=tracker.line) as line,
        ):
            tracker.frame()
        line.assert_called_once_with([("ISS PREDICTING", iss.GRAY)], 64)

    def test_spotify_paused_track_is_idle(self):
        display = spotify.SpotifyDisplay.__new__(spotify.SpotifyDisplay)
        payload = {
            "is_playing": False,
            "item": {"name": "Song", "artists": [{"name": "Artist"}],
                     "album": {"images": [{"url": "https://example.com/cover"}]}},
        }
        display.sp = SimpleNamespace(current_user_playing_track=lambda: payload)
        self.assertIsNone(display.fetch_current_song())
        payload["is_playing"] = True
        self.assertEqual(display.fetch_current_song().title, "Song")

    def test_spotify_display_auth_never_prompts_for_a_token(self):
        auth = spotify_auth.make_auth_manager("test-id", "test-secret", interactive=False)
        with patch.object(auth.cache_handler, "get_cached_token", return_value=None):
            with self.assertRaises(spotify_auth.SpotifyLoginRequired):
                auth.get_access_token(as_dict=False)
        self.assertGreater(auth.requests_timeout, 0)

    def test_spotify_swaps_the_logo_before_polling_then_waits_on_static_text(self):
        display = spotify.SpotifyDisplay.__new__(spotify.SpotifyDisplay)
        display.matrix = Mock()
        canvas = display.matrix.CreateFrameCanvas.return_value
        canvas.width = 64
        display.matrix.height = 32
        display.matrix.SwapOnVSync.return_value = canvas
        display.graphics = Mock()
        display.fonts = {"5x8": object()}
        display.art_block_width = 32
        display.icon_rgb = bytes(30 * 30 * 3)
        song = spotify.CurrentSong(artist="ABBA", title="SOS", album_cover="unused")
        display.now_playing = spotify.NowPlaying(song, bytes(32 * 32 * 3))
        display.changed = Mock()
        display.changed.wait.side_effect = StopIteration
        with (
            patch.object(spotify, "notify_ready") as ready,
            patch.object(spotify.threading, "Thread") as thread,
        ):
            thread.return_value.start.side_effect = lambda: ready.assert_called_once()
            with self.assertRaises(StopIteration):
                display.run()
        canvas.SetPixelsRGB.assert_any_call(display.icon_rgb, 30, 30, 1, 1)
        self.assertEqual(display.matrix.SwapOnVSync.call_count, 2)
        display.changed.wait.assert_called_once()

    def test_seeded_planet_has_repeatable_texture(self):
        def frame():
            random.seed(19)
            np.random.seed(19)
            with patch.object(planets.random, "randrange", return_value=0):
                scene = planets.Planets(64, 32)
            scene.frames = 80
            return scene.frame()

        np.testing.assert_array_equal(frame(), frame())

    def test_planets_cycle_through_the_four_planets(self):
        with (
            patch.object(planets, "Planet", wraps=planets.Planet) as build,
            patch.object(planets.random, "randrange", return_value=0),
        ):
            scene = planets.Planets(64, 32)
            for _ in range(4):
                scene.frames = planets.PLANET_FRAMES
                scene.frame()

        self.assertEqual(
            [call.args[2] for call in build.call_args_list],
            [
                planets.earth_surface,
                planets.mars_surface,
                planets.jupiter_surface,
                planets.saturn_surface,
                planets.earth_surface,
            ],
        )

    def test_planet_frames_continue_while_next_body_is_prepared(self):
        started = Event()
        release = Event()
        first, second = planets.BODIES[:2]

        def prepare(width, height, seed):
            started.set()
            if not release.wait(3):
                raise TimeoutError("the renderer blocked on the planet build")
            return second(width, height, seed)

        with (
            ThreadPoolExecutor(max_workers=1) as worker,
            patch.object(planets, "BODIES", [first, prepare]),
            patch.object(planets.random, "randrange", return_value=0),
        ):
            scene = planets.Planets(64, 32, worker=worker)
            deadline = round(planets.PLANET_SECONDS / scene.FRAME_SECONDS)
            scene.frames = deadline - round(planets.FADE_SECONDS / scene.FRAME_SECONDS)
            try:
                self.assertEqual(scene.frame().nbytes, 64 * 32 * 3)
                self.assertTrue(started.wait(3))
                scene.frames = deadline
                self.assertEqual(scene.frame().nbytes, 64 * 32 * 3)
                self.assertEqual(scene.body, 0)
            finally:
                release.set()
            scene.next_planet.result(timeout=3)
            self.assertEqual(scene.frame().nbytes, 64 * 32 * 3)
            self.assertEqual(scene.body, 1)


if __name__ == "__main__":
    unittest.main()
