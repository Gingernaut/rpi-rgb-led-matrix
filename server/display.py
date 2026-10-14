import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from server.config import DisplayConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DisplayManager:
    """Manages a single display subprocess. Starting a new display kills the current one."""

    def __init__(self, config: DisplayConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_pid(self) -> int | None:
        return self._process.pid if self.is_running else None

    def start(
        self,
        cmd: list[str],
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Kill any running display, then start a new one. Returns the new PID."""
        with self._lock:
            self.stop()
            full_cmd = cmd + self._config.to_args() + (extra_args or [])
            proc_env = {**os.environ, **(env or {})}
            self._process = subprocess.Popen(
                full_cmd, cwd=PROJECT_ROOT, env=proc_env, start_new_session=True,
            )
            return self._process.pid

    def start_python(
        self,
        script: str,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a Python display script using the current interpreter. The
        project's editable install puts the rgbmatrix bindings on its path."""
        with self._lock:
            self.stop()
            script_path = str(PROJECT_ROOT / script)
            full_cmd = (
                [sys.executable, script_path]
                + self._config.to_python_args()
                + (extra_args or [])
            )
            proc_env = {**os.environ, **(env or {})}
            self._process = subprocess.Popen(
                full_cmd, cwd=PROJECT_ROOT, env=proc_env, start_new_session=True,
            )
            return self._process.pid

    def stop(self) -> None:
        """Stop the current display process if one is running."""
        if self._process is None:
            return

        # Reap a process that already exited on its own (prevents zombies)
        if self._process.poll() is not None:
            self._process = None
            return

        # Kill the entire process group so child processes are cleaned up too
        pgid = os.getpgid(self._process.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._process.wait()
        self._process = None
