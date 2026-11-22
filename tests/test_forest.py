import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from server import app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "displays"))

from forest import HEIGHT, WIDTH, Forest


class ForestTest(unittest.TestCase):
    def scene(self, visitor=""):
        with patch.dict(os.environ, {"FOREST_EVENT": visitor}), patch.object(random, "getrandbits", return_value=19):
            return Forest(WIDTH, HEIGHT)

    def test_mode_is_listed_and_launches_forest(self):
        self.assertIn("forest", app.display_modes()["animations"])
        with patch.object(app.display, "start_python", return_value=123) as start:
            self.assertEqual(app.display_animation("forest"), {
                "status": "starting", "pid": 123, "mode": "forest",
            })
        start.assert_called_once_with(script="server/displays/forest.py", mode="forest")

    def test_every_visitor_crosses_and_clears_the_frame(self):
        for visitor in ("deer", "fox", "rabbit", "squirrel", "slug"):
            with self.subTest(visitor=visitor):
                scene = self.scene(visitor)
                quiet = self.scene(visitor)
                quiet.next_visit = float("inf")
                appeared = False
                for _ in range(round(22 / scene.FRAME_SECONDS)):
                    frame = scene.frame()
                    background = quiet.frame()
                    if scene.visitor == visitor:
                        appeared |= not np.array_equal(frame, background)
                    elif appeared:
                        np.testing.assert_array_equal(frame, background)
                        break
                else:
                    self.fail(f"{visitor} never appeared and cleared the frame")

    def test_bright_shafts_pass_through_a_green_ambient_scene(self):
        scene = self.scene()
        scene.next_visit = float("inf")
        green = []
        for _ in range(round(23 / scene.FRAME_SECONDS)):
            frame = scene.frame()
            self.assertEqual((frame.dtype, frame.shape), (np.uint8, (WIDTH * HEIGHT * 3,)))
            self.assertTrue(frame.flags.c_contiguous)
            green.append(frame.reshape(HEIGHT, WIDTH, 3)[7:18, 18:28, 1].mean())

        self.assertGreater(green[0], 70)
        self.assertGreater(max(green), green[0] + 15)
        self.assertAlmostEqual(green[-1], green[0], delta=2)

    def test_sun_shower_keeps_its_motion_after_weeks_of_uptime(self):
        scene = self.scene()
        scene.shower = True
        light = np.ones((HEIGHT, WIDTH), np.float32)

        def rain_at(t):
            scene.sun_start, scene.sun_end = t - 4, t + 4
            frame = np.zeros((HEIGHT, WIDTH, 3), np.float32)
            scene.air(frame, light, t)
            return frame

        np.testing.assert_array_equal(rain_at(4.06), rain_at(4_000_004.06))


if __name__ == "__main__":
    unittest.main()
