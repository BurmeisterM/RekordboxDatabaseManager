from contextlib import closing
import errno
from pathlib import Path
import sqlite3
from threading import Event
from time import monotonic

import numpy as np
import pytest

from rekordbox_manager import comparison, database
from rekordbox_manager.comparison import ComparisonCancelled, compare_tracks, read_middle, read_samples
from rekordbox_manager.comparison_store import ComparisonStore, comparison_database_path, path_key
from rekordbox_manager.io_control import ReadOptions, ReadSession


def test_title_and_track_comparisons_share_audio_in_both_directions(collection, monkeypatch):
    path, _ = collection
    reads, real_read = [], comparison.read_middle

    def read(file, cancel):
        reads.append(Path(file))
        return real_read(file, cancel)

    monkeypatch.setattr(comparison, "read_middle", read)
    library = database.load_library(path)
    assert len(reads) == 2
    cache = comparison_database_path(path)
    result = compare_tracks(library.tracks, Event(), cache_path=cache, library_id=library.db_id)
    assert len(reads) == 3 and result.updated == 1 and result.reused == 2
    monkeypatch.setattr(comparison, "read_middle", lambda *args: pytest.fail("Unchanged audio must be shared"))
    assert database.load_library(path).groups == library.groups
    assert compare_tracks(library.tracks, Event(), cache_path=cache, library_id=library.db_id).reused == 3
    with closing(sqlite3.connect(cache)) as conn:
        assert conn.execute("SELECT count(*) FROM audio_files").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM tracks WHERE samples IS NOT NULL").fetchone()[0] == 0


@pytest.mark.parametrize("resume", [False, True])
def test_disconnected_scan_preserves_saved_audio_and_matches(collection, monkeypatch, resume):
    library = database.load_library(collection[0], verify_audio=False)
    cache = comparison_database_path(library.path)
    compare_tracks(library.tracks, Event(), cache_path=cache, library_id=library.db_id)
    with ComparisonStore(cache, library.db_id) as store:
        saved = tuple(store.matches())
        blobs = {key: store.samples(key).tobytes() for key in store.entries()}
    cancel = Event()
    disconnected = True
    attempted = []
    actual_stat = Path.stat
    failing = Path(library.tracks[0].path)

    def stat(file, *args, **kwargs):
        if file == failing:
            attempted.append(file)
            if disconnected:
                raise OSError(errno.ENODEV, "Laufwerk getrennt", str(file))
        return actual_stat(file, *args, **kwargs)

    def progress(current, total, message):
        nonlocal disconnected
        if not session.paused.is_set():
            return
        assert "bleiben erhalten" in message
        with ComparisonStore(cache, library.db_id) as store:
            assert tuple(store.matches()) == saved
            assert {key: store.samples(key).tobytes() for key in store.entries()} == blobs
        if resume:
            disconnected = False
            session.resume()
        else:
            cancel.set()

    session = ReadSession(cancel, progress=progress)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(comparison, "read_middle", lambda *args: pytest.fail("Preserve and reuse cached audio"))
    result = compare_tracks(library.tracks, cancel, progress, cache_path=cache, library_id=library.db_id,
                            update=True, session=session)
    assert result.cancelled is not resume and not result.errors
    assert len(attempted) <= 3
    if resume:
        assert result.reused == 3 and len(result.matches) == 3
    else:
        with ComparisonStore(cache, library.db_id) as store:
            assert tuple(store.matches()) == saved
            assert {key: store.samples(key).tobytes() for key in store.entries()} == blobs


def test_reconnection_rechecks_files_processed_before_the_outage(collection, monkeypatch):
    library = database.load_library(collection[0], verify_audio=False)
    cache = comparison_database_path(library.path)
    compare_tracks(library.tracks, Event(), cache_path=cache, library_id=library.db_id)
    first, last = Path(library.tracks[0].path), Path(library.tracks[-1].path)
    actual_stat, disconnected, reads = Path.stat, True, []
    real_read = comparison.read_middle

    def stat(file, *args, **kwargs):
        if file == last and disconnected:
            raise OSError(errno.ENODEV, "Laufwerk getrennt")
        return actual_stat(file, *args, **kwargs)

    def progress(current, total, message):
        nonlocal disconnected
        if session.paused.is_set():
            # Emulate a file changed while the volume was unavailable.
            first.write_bytes(first.read_bytes() + b"\0")
            disconnected = False
            session.resume()

    def read(file, cancel):
        reads.append(Path(file))
        return real_read(file, cancel)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(comparison, "read_middle", read)
    session = ReadSession(Event(), progress=progress)
    result = compare_tracks(library.tracks, session.cancel, progress, cache_path=cache,
                            library_id=library.db_id, session=session)
    assert not result.cancelled and not result.errors and reads == [first]
    assert result.updated == 1 and result.reused == 2 and len(result.matches) == 3


def test_duplicate_file_metadata_is_checked_once_on_cached_scan(collection, monkeypatch):
    from dataclasses import replace
    library = database.load_library(collection[0], verify_audio=False)
    first = library.tracks[0]
    tracks = tuple(replace(first, id=str(i)) for i in range(25))
    cache = comparison_database_path(library.path)
    compare_tracks(tracks, Event(), cache_path=cache)
    actual_stat, checks = Path.stat, []

    def stat(file, *args, **kwargs):
        if file == Path(first.path):
            checks.append(file)
        return actual_stat(file, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    result = compare_tracks(tracks, Event(), cache_path=cache)
    assert result.reused == 25 and len(checks) == 1


def test_gentle_decoding_matches_normal_audio_and_paces_reads(collection, monkeypatch):
    path = collection[0].parent / "copy.mp3"
    normal = read_middle(path, Event())
    session = ReadSession(Event(), ReadOptions(True, 1, 150))
    waits = []
    monkeypatch.setattr(session, "wait", waits.append)
    gentle = read_samples(path, session.cancel, session)
    assert np.array_equal(normal, gentle)
    assert len(waits) > 2 and all(delay >= 0 for delay in waits)
    assert waits[-1] == 0.15


def test_gentle_fallback_reads_are_paced_and_cancellable(collection, monkeypatch):
    path = collection[0].parent / "copy.mp3"
    session = ReadSession(Event(), ReadOptions(True, 1, 0))
    waits = []

    def no_seek(*args):
        raise ValueError("bad seek index")

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            session.cancel.set()
        if session.cancel.is_set():
            raise ComparisonCancelled()

    monkeypatch.setattr(comparison, "_read_middle_seek", no_seek)
    monkeypatch.setattr(session, "wait", wait)
    with pytest.raises(ComparisonCancelled):
        read_samples(path, session.cancel, session)
    assert len(waits) == 3
    path.rename(path.with_name("released.mp3"))


def test_read_pacing_can_be_cancelled_during_a_wait():
    cancel = Event()
    session = ReadSession(cancel, ReadOptions(True, 1, 100))
    cancel.set()
    start = monotonic()
    with pytest.raises(ComparisonCancelled):
        session.pace(100 * 1024 * 1024)
    assert monotonic() - start < 0.5
