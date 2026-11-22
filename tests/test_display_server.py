import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from server.app import display_modes
from server.config import DisplayConfig
from server.display import DisplayManager, DisplayStartError


class DisplayServerTest(unittest.TestCase):
    def setUp(self):
        self.manager = DisplayManager(DisplayConfig())
        self.addCleanup(self.manager.stop)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.gate = Path(directory.name, "ready")
        self.child = Path(directory.name, "display.py")
        self.child.write_text(
            "import os, time\n"
            "from pathlib import Path\n"
            f"while not Path({str(self.gate)!r}).exists():\n"
            "    time.sleep(0.01)\n"
            "fd = int(os.environ.pop('PIXEL_DISPLAY_READY_FD'))\n"
            "os.write(fd, b'1')\n"
            "os.close(fd)\n"
            "time.sleep(60)\n"
        )

    def await_status(self, expected):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status = self.manager.status()
            if status["status"] == expected:
                return status
            time.sleep(0.01)
        self.fail(f"expected {expected}, got {status!r}")

    def test_lists_spotify(self):
        self.assertIn("spotify", display_modes()["animations"])

    def test_status_tracks_readiness_failure_and_off(self):
        pid = self.manager.start_python(str(self.child), mode="sample")
        self.assertEqual(self.manager.status()["status"], "starting")
        self.gate.touch()
        self.assertEqual(self.await_status("running")["pid"], pid)
        self.assertEqual(self.manager.status()["mode"], "sample")
        self.manager.start_python(str(self.child.with_name("absent.py")), mode="missing")
        failed = self.await_status("failed")
        self.assertFalse(failed["running"])
        self.assertEqual(failed["mode"], "missing")
        self.assertIsNone(failed["pid"])
        self.assertEqual(failed["exit_code"], 2)
        self.manager.stop()
        self.assertEqual(self.manager.status()["status"], "off")

    def test_spawn_error_leaves_a_failed_status_without_a_pid_or_exit_code(self):
        with self.assertRaisesRegex(DisplayStartError, "Could not start demo"):
            self.manager.start([str(self.child.with_name("absent-demo"))])
        status = self.manager.status()
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["mode"], "demo")
        self.assertIsNone(status["pid"])
        self.assertIsNone(status["exit_code"])

    def test_off_keeps_the_child_started_after_it(self):
        self.manager.start_python(str(self.child), mode="old")
        self.gate.touch()
        self.await_status("running")
        old = self.manager._process
        waited = threading.Event()
        release = threading.Event()
        started = threading.Event()
        real_wait = old.wait

        def pause_after_wait(*args, **kwargs):
            result = real_wait(*args, **kwargs)
            waited.set()
            release.wait(3)
            return result

        replacement = []

        def start_replacement():
            replacement.append(self.manager.start_python(str(self.child), mode="replacement"))
            started.set()

        stopping = threading.Thread(target=self.manager.stop)
        starting = threading.Thread(target=start_replacement)
        with patch.object(old, "wait", pause_after_wait):
            stopping.start()
            try:
                self.assertTrue(waited.wait(3))
                starting.start()
                self.assertFalse(started.wait(0.1))
            finally:
                release.set()
                stopping.join(3)
                if starting.ident is not None:
                    starting.join(3)
        self.assertTrue(started.is_set())
        self.assertEqual(self.await_status("running")["pid"], replacement[0])


if __name__ == "__main__":
    unittest.main()
