import os
import signal
import subprocess
import sys
from pathlib import Path

from server.config import DisplayConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINDINGS_DIR = str(PROJECT_ROOT / "bindings" / "python")


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

    def start(
        self,
        cmd: list[str],
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Kill any running display, then start a new one. Returns the new PID."""
        self.stop()
        full_cmd = cmd + self._config.to_args() + (extra_args or [])
        proc_env = {**os.environ, **(env or {})}
        self._process = subprocess.Popen(full_cmd, cwd=PROJECT_ROOT, env=proc_env)
        return self._process.pid

    def start_python(
        self,
        script: str,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        """Start a Python display script using the current interpreter."""
        script_path = str(PROJECT_ROOT / script)
        full_cmd = [sys.executable, script_path] + self._config.to_python_args() + (extra_args or [])
        self.stop()
        # Add rgbmatrix bindings to PYTHONPATH so the C extension is importable
        python_path = os.environ.get("PYTHONPATH", "")
        if BINDINGS_DIR not in python_path:
            python_path = f"{BINDINGS_DIR}:{python_path}" if python_path else BINDINGS_DIR
        proc_env = {**os.environ, "PYTHONPATH": python_path, **(env or {})}
        self._process = subprocess.Popen(full_cmd, cwd=PROJECT_ROOT, env=proc_env)
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
