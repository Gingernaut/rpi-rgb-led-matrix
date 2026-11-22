import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from server.config import DisplayConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
READY_FD_ENV = "PIXEL_DISPLAY_READY_FD"


class DisplayStartError(Exception):
    pass


class DisplayManager:
    """Manages a single display subprocess. Starting a new display kills the current one."""

    def __init__(self, config: DisplayConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None
        self._mode: str | None = None
        self._ready_fd: int | None = None
        self._ready = False
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            process = self._process
            exit_code = process.poll() if process is not None else None
            alive = process is not None and exit_code is None
            if alive and self._ready_fd is not None:
                try:
                    self._ready = os.read(self._ready_fd, 1) == b"1"
                except BlockingIOError:
                    pass
                else:
                    self._close_ready_fd()
            if alive:
                state = "running" if self._ready else "starting"
            else:
                self._close_ready_fd()
                if self._mode is None:
                    state = "off"
                else:
                    state = "exited" if exit_code == 0 else "failed"
            return {
                "status": state,
                "running": alive,
                "pid": process.pid if alive else None,
                "mode": self._mode,
                "exit_code": exit_code,
            }

    def start(
        self,
        cmd: list[str],
        extra_args: list[str] | None = None,
    ) -> int:
        return self._start(cmd + self._config.to_args() + (extra_args or []), "demo")

    def start_python(
        self,
        script: str,
        *,
        mode: str,
    ) -> int:
        cmd = [sys.executable, str(PROJECT_ROOT / script)] + self._config.to_python_args()
        return self._start(cmd, mode)

    def _start(self, cmd: list[str], mode: str) -> int:
        with self._lock:
            self._stop_locked()
            self._mode = mode
            try:
                read_fd, write_fd = os.pipe()
                try:
                    os.set_blocking(read_fd, False)
                    self._process = subprocess.Popen(
                        cmd, cwd=PROJECT_ROOT, start_new_session=True, pass_fds=(write_fd,),
                        env={**os.environ, READY_FD_ENV: str(write_fd)},
                    )
                except BaseException:
                    os.close(read_fd)
                    raise
                finally:
                    os.close(write_fd)
            except OSError as error:
                raise DisplayStartError(f"Could not start {mode}: {error.strerror}") from error
            self._ready_fd = read_fd
            return self._process.pid

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            self._signal_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(process.pid, signal.SIGKILL)
                process.wait()
        self._close_ready_fd()
        self._process = None
        self._mode = None
        self._ready = False

    def _close_ready_fd(self) -> None:
        if self._ready_fd is not None:
            os.close(self._ready_fd)
            self._ready_fd = None

    @staticmethod
    def _signal_group(pid: int, sig: signal.Signals) -> None:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
