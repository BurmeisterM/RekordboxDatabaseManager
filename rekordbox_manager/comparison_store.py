"""Persistent, disposable audio cache, separate from the Rekordbox database."""

from contextlib import closing
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

import numpy as np

from .domain import ManagerError


# Bump when decoding, sample representation or the similarity algorithm changes.
ALGORITHM_VERSION = "middle-5s-mono-8000-f32-correlation-250ms-v1"


def comparison_database_path(library_path: Path) -> Path:
    return local_comparison_database_path(library_path)


def prepare_comparison_database(library_path: Path, session=None) -> Path:
    """Move work to the user profile; copy a legacy cache without changing it."""
    target = comparison_database_path(library_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ManagerError(f"Die lokale Vergleichsdatenbank kann nicht angelegt werden:\n{target}\n"
                           f"Bitte Schreibrechte und freien Speicher prüfen.\n{exc}") from exc
    if target.exists():
        return target
    legacy = Path(library_path).with_suffix(".trackvergleich.sqlite3")
    temporary = target.with_name(f"migration-{uuid4().hex}.sqlite3")

    def migrate():
        if not legacy.is_file():
            return
        if session is not None:
            session.progress(0, 0, "Vorhandene Vergleichsdaten lokal übernehmen …")
        try:
            with closing(sqlite3.connect(legacy.resolve().as_uri() + "?mode=ro", uri=True)) as source:
                with closing(sqlite3.connect(temporary)) as destination:
                    def progress(status, remaining, total):
                        if session is not None:
                            session.pace(64 * 4096)
                    source.backup(destination, pages=64, progress=progress)
            if not target.exists():
                os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    try:
        if session is None:
            migrate()
        else:
            session.call(legacy, migrate)
    except sqlite3.Error as exc:
        raise ManagerError(f"Die vorhandene Vergleichsdatenbank konnte nicht übernommen werden:\n{legacy}\n"
                           f"Die Originaldatei bleibt erhalten.\n{exc}") from exc
    return target


def local_comparison_database_path(library_path: Path) -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    identity = sha256(path_key(str(library_path)).encode("utf-8")).hexdigest()
    return root / "RekordboxDatabaseManager" / "Trackvergleich" / identity / "master.trackvergleich.sqlite3"


def open_comparison_store(path: Path, library_id: str = "", fallback_path: Path | None = None):
    """Fall back only for access errors during opening, never for corrupt data."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        return ComparisonStore(path, library_id)
    except (sqlite3.OperationalError, OSError) as exc:
        code = getattr(exc, "sqlite_errorcode", 0) & 0xff
        if fallback_path is None or Path(path) == Path(fallback_path) or (not isinstance(exc, OSError) and code not in (
                sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_READONLY, sqlite3.SQLITE_PERM)):
            raise
        try:
            fallback_path = Path(fallback_path)
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            return ComparisonStore(fallback_path, library_id)
        except (sqlite3.Error, OSError) as fallback_error:
            raise OSError(f"Sammlungsordner: {path}\n{exc}\n"
                          f"Lokale Ausweichablage: {fallback_path}\n{fallback_error}\n"
                          "Bitte Erreichbarkeit, Schreibrechte und freien Speicher prüfen.") from fallback_error


def path_key(path: str) -> str:
    # No file access: this also works for a temporarily disconnected drive.
    return os.path.normcase(os.path.abspath(path)) if path else ""


class ComparisonStore:
    def __init__(self, path: Path, library_id: str = ""):
        self.path = Path(path)
        self.connection = sqlite3.connect(str(self.path), timeout=5)
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, artist TEXT NOT NULL,
                    size INTEGER, mtime_ns INTEGER, samples BLOB, error TEXT NOT NULL DEFAULT '',
                    pending INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS matches (
                    first_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                    second_id TEXT NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
                    score REAL NOT NULL,
                    PRIMARY KEY (first_id, second_id)
                );
                CREATE INDEX IF NOT EXISTS matches_second ON matches(second_id);
                CREATE INDEX IF NOT EXISTS tracks_file ON tracks(path, size, mtime_ns);
                CREATE TABLE IF NOT EXISTS audio_files (
                    path TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
                    samples BLOB NOT NULL, PRIMARY KEY(path, size, mtime_ns)
                );
            """)
            metadata = dict(self.connection.execute("SELECT key, value FROM metadata").fetchall())
            # An existing schema can be read from a read-only file. Verify that
            # writes (including the rollback journal) work before accepting it.
            with self.connection:
                self.connection.execute("UPDATE metadata SET value=value WHERE key='algorithm'")
            identity = {"algorithm": ALGORITHM_VERSION, "library_id": library_id}
            if any(metadata.get(key) != value for key, value in identity.items()):
                with self.connection:
                    self.connection.execute("DELETE FROM tracks")
                    self.connection.execute("DELETE FROM audio_files")
                    self.connection.execute("DELETE FROM metadata")
                    self.connection.executemany("INSERT INTO metadata VALUES (?, ?)", identity.items())
            if self.connection.execute("SELECT 1 FROM metadata WHERE key='shared_audio'").fetchone() is None:
                with self.connection:
                    self.connection.execute("INSERT OR IGNORE INTO audio_files "
                        "SELECT path,size,mtime_ns,samples FROM tracks WHERE samples IS NOT NULL "
                        "AND size IS NOT NULL AND mtime_ns IS NOT NULL")
                    self.connection.execute("UPDATE tracks SET samples=NULL WHERE samples IS NOT NULL "
                                            "AND size IS NOT NULL AND mtime_ns IS NOT NULL")
                    self.connection.execute("INSERT INTO metadata VALUES ('shared_audio','1')")
        except Exception:
            self.connection.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.connection.close()

    def entries(self):
        # Do not pull every audio blob into RAM just to inspect metadata.
        return {row["id"]: row for row in self.connection.execute(
            "SELECT id, path, artist, size, mtime_ns, error, pending, "
            "(samples IS NOT NULL OR EXISTS (SELECT 1 FROM audio_files a WHERE a.path=tracks.path "
            "AND a.size=tracks.size AND a.mtime_ns=tracks.mtime_ns)) AS ready FROM tracks")}

    @property
    def initialized(self):
        return self.connection.execute("SELECT 1 FROM metadata WHERE key='initialized'").fetchone() is not None

    def mark_initialized(self):
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('initialized', '1')")

    def prune(self, ids):
        with self.connection:
            self.connection.execute("CREATE TEMP TABLE current_ids (id TEXT PRIMARY KEY)")
            self.connection.executemany("INSERT INTO current_ids VALUES (?)", ((key,) for key in ids))
            self.connection.execute("DELETE FROM tracks WHERE id NOT IN (SELECT id FROM current_ids)")
            self.connection.execute("DROP TABLE current_ids")

    def put(self, track_id, path, artist, stamp, samples=None, error=""):
        # A changed signature/artist invalidates every old pair involving this ID.
        size, mtime = stamp if stamp is not None else (None, None)
        blob = samples.astype("<f4").tobytes() if samples is not None else None
        with self.connection:
            if samples is not None:
                self._save_audio(path, stamp, blob)
            self.connection.execute("DELETE FROM matches WHERE first_id=? OR second_id=?", (track_id, track_id))
            self.connection.execute("""
                INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(id) DO UPDATE SET path=excluded.path, artist=excluded.artist,
                    size=excluded.size, mtime_ns=excluded.mtime_ns, samples=excluded.samples,
                    error=excluded.error, pending=1
            """, (track_id, path, artist, size, mtime, None, error))

    def _save_audio(self, path, stamp, blob):
        self.connection.execute("DELETE FROM audio_files WHERE path=? AND (size<>? OR mtime_ns<>?)", (path, *stamp))
        self.connection.execute("INSERT OR IGNORE INTO audio_files VALUES (?, ?, ?, ?)", (path, *stamp, blob))

    def save_audio(self, path, stamp, samples):
        with self.connection:
            self._save_audio(path, stamp, samples.astype("<f4").tobytes())

    def cached_samples(self, path, stamp):
        row = self.connection.execute("SELECT samples FROM audio_files WHERE path=? AND size=? AND mtime_ns=?",
                                      (path, *stamp)).fetchone()
        if row is None:
            # Reuse pre-1.1.6 rows without decoding or rebuilding their matches.
            row = self.connection.execute("SELECT samples FROM tracks WHERE path=? AND size=? AND mtime_ns=? "
                                          "AND samples IS NOT NULL LIMIT 1", (path, *stamp)).fetchone()
        return np.frombuffer(row[0], dtype="<f4") if row is not None else None

    def samples(self, track_id):
        row = self.connection.execute("SELECT samples,path,size,mtime_ns FROM tracks WHERE id=?", (track_id,)).fetchone()
        if row[0] is not None:
            return np.frombuffer(row[0], dtype="<f4")
        return self.cached_samples(row[1], (row[2], row[3]))

    def find_file(self, path, stamp):
        return self.connection.execute(
            "SELECT id FROM tracks WHERE path=? AND size=? AND mtime_ns=? AND samples IS NOT NULL LIMIT 1",
            (path, *stamp)).fetchone()

    def save_matches(self, matches):
        with self.connection:
            self.connection.executemany("INSERT OR REPLACE INTO matches VALUES (?, ?, ?)", matches)

    def mark_compared(self, ids):
        with self.connection:
            self.connection.executemany("UPDATE tracks SET pending=0 WHERE id=? AND pending<>0",
                                        ((key,) for key in ids))

    def defer(self, track_id):
        # If a drive disappears during a scan, check this row against any newly
        # added tracks when it returns, even if its audio itself did not change.
        with self.connection:
            self.connection.execute("UPDATE tracks SET pending=1 WHERE id=? AND pending=0", (track_id,))

    def matches(self):
        return self.connection.execute("SELECT first_id, second_id, score FROM matches ORDER BY score DESC")


def remove_cached_tracks(library_path: Path, ids):
    """After a successful Rekordbox commit, remove cache rows and dependent pairs."""
    ids = tuple(ids)
    errors = []
    # Legacy caches are read only. A later scan prunes removed IDs after import.
    for path in (local_comparison_database_path(library_path),):
        try:
            if not path.is_file():
                continue
            with closing(sqlite3.connect(str(path), timeout=5)) as connection:
                connection.execute("PRAGMA foreign_keys=ON")
                with connection:
                    connection.executemany("DELETE FROM tracks WHERE id=?", ((key,) for key in ids))
                    # The shared audio is keyed by file rather than ID and can
                    # still belong to retained entries using the same file.
        except (sqlite3.Error, OSError) as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise OSError("\n".join(errors))
