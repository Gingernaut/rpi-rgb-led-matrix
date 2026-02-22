import signal
import subprocess
from pathlib import Path

from server.config import DisplayConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DisplayManager:
    """Manages a single display subprocess. Starting a new display kills the current one."""

    def __init__(self, config: DisplayConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_pid(self) -> int | None:
        return self._process.pid if self.is_running else None

    def start(self, cmd: list[str], extra_args: list[str] | None = None) -> int:
        """Kill any running display, then start a new one. Returns the new PID."""
        self.stop()
        full_cmd = cmd + self._config.to_args() + (extra_args or [])
        self._process = subprocess.Popen(full_cmd, cwd=PROJECT_ROOT)
        return self._process.pid

    def stop(self) -> None:
        """Stop the current display process if one is running."""
        if not self.is_running:
            return
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
        self._process = None
