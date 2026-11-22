"""Cancelable read pacing and explicit recovery after a disconnected volume."""

from dataclasses import dataclass
from contextvars import ContextVar
import errno
import os
from pathlib import Path
import sqlite3
from threading import Event
from time import monotonic

from .domain import ManagerError

current_session = ContextVar("current_read_session", default=None)


class ComparisonCancelled(Exception):
    pass


def check_cancelled(cancel):
    if cancel.is_set():
        raise ComparisonCancelled()


@dataclass(frozen=True)
class ReadOptions:
    gentle: bool = False
    mib_per_second: int = 8
    pause_ms: int = 100

    def __post_init__(self):
        if not 1 <= self.mib_per_second <= 500 or not 0 <= self.pause_ms <= 5000:
            raise ValueError("Ungültige Einstellungen für den Schonmodus.")


def volume_available(path):
    try:
        return Path(Path(path).anchor).is_dir()
    except OSError:
        return False


class ReadSession:
    def __init__(self, cancel, options=None, progress=None):
        self.cancel = cancel
        self.options = options or ReadOptions()
        self.progress = progress or (lambda current, total, message: None)
        self.paused = Event()
        self.resume_requested = Event()
        self.statistics = {}
        self.stat_errors = {}
        self.generation = 0
        self._next_read = monotonic()

    def wait(self, seconds):
        if self.cancel.wait(max(0, seconds)):
            raise ComparisonCancelled()

    def pace(self, count):
        if self.options.gentle:
            now = monotonic()
            self._next_read = max(now, self._next_read) + count / (self.options.mib_per_second * 1024 * 1024)
            self.wait(self._next_read - now)
        check_cancelled(self.cancel)

    def file_finished(self):
        if self.options.gentle:
            self.wait(self.options.pause_ms / 1000)

    def disconnected(self, path, exc):
        cause = exc
        while cause is not None:
            if (getattr(cause, "winerror", None) in {21, 23, 31, 55, 64, 121, 433, 483, 1117, 1167}
                    or getattr(cause, "errno", None) in {errno.EIO, errno.ENODEV, errno.ENXIO}):
                return True
            cause = cause.__cause__
        return not volume_available(path)

    def pause(self, path):
        self.resume_requested.clear()
        self.paused.set()
        self.progress(0, 0, f"Laufwerk {Path(path).anchor} nicht erreichbar. Gespeicherte Daten bleiben erhalten. "
                           "Laufwerk wieder verbinden und „Fortsetzen“ anklicken.")
        try:
            while not self.resume_requested.wait(0.1):
                check_cancelled(self.cancel)
            check_cancelled(self.cancel)
            self.statistics.clear()
            self.stat_errors.clear()
            self.generation += 1
        finally:
            self.paused.clear()

    def resume(self):
        self.resume_requested.set()

    def call(self, path, operation):
        while True:
            check_cancelled(self.cancel)
            try:
                return operation()
            except (OSError, ManagerError, sqlite3.Error) as exc:
                if not self.disconnected(path, exc):
                    raise
                self.pause(path)

    def stat(self, path, *, refresh=False):
        key = os.path.normcase(os.path.abspath(path))
        if not refresh and key in self.stat_errors:
            raise self.stat_errors[key]
        if refresh or key not in self.statistics:
            try:
                self.statistics[key] = self.call(path, lambda: Path(path).stat())
                self.stat_errors.pop(key, None)
            except OSError as exc:
                self.stat_errors[key] = exc
                raise
        return self.statistics[key]

    def scan(self, operation):
        while True:
            generation = self.generation
            result = operation()
            if generation == self.generation or self.cancel.is_set():
                return result
            # Revalidate stamps after a reconnect before exposing any matches.
            self.progress(0, 0, "Laufwerk wieder verbunden · Gespeicherte Daten und Dateistand werden geprüft …")


class PacedFile:
    """Small reads bound bursts, including FFmpeg probing and fallback decoding."""
    def __init__(self, path, session):
        self.file = Path(path).open("rb")
        self.session = session

    def read(self, size=-1):
        check_cancelled(self.session.cancel)
        size = min(size, 64 * 1024) if size >= 0 else 64 * 1024
        self.session.pace(size)
        return self.file.read(size)

    def seek(self, offset, whence=0):
        check_cancelled(self.session.cancel)
        return self.file.seek(offset, whence)

    def tell(self):
        return self.file.tell()

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
