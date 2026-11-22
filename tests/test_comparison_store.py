from contextlib import closing
from pathlib import Path
import sqlite3
from threading import Event

import pytest

from rekordbox_manager.comparison import compare_tracks
from rekordbox_manager.comparison_store import (
    comparison_database_path, local_comparison_database_path, remove_cached_tracks,
)
from rekordbox_manager.domain import ManagerError, Track


def test_inaccessible_adjacent_cache_falls_back_and_survives_restart(tmp_path, monkeypatch, synthetic_mp3_audio):
    library = tmp_path / "unavailable-drive" / "master.db"
    adjacent = comparison_database_path(library)
    local = local_comparison_database_path(library)
    audio = tmp_path / "music.mp3"
    audio.write_bytes(synthetic_mp3_audio)
    tracks = tuple(Track(key, key, "Artist", path=str(audio)) for key in ("a", "b"))
    result = compare_tracks(tracks, Event(), cache_path=adjacent, fallback_cache_path=local)
    assert result.cache_path == str(local) and len(result.matches) == 1
    assert not adjacent.exists()

    # Simulate the external drive returning with an old cache of its own.
    library.parent.mkdir()
    compare_tracks(tracks, Event(), cache_path=adjacent)
    assert comparison_database_path(library) == local
    monkeypatch.setattr("rekordbox_manager.comparison.read_middle",
                        lambda *args: pytest.fail("Saved fallback must be reused"))
    result = compare_tracks(tracks, Event(), cache_path=comparison_database_path(library),
                            fallback_cache_path=local)
    assert result.reused == 2 and result.updated == 0 and len(result.matches) == 1

    remove_cached_tracks(library, iter(("a",)))
    for cache in (adjacent, local):
        with closing(sqlite3.connect(cache)) as connection:
            assert connection.execute("SELECT id FROM tracks").fetchall() == [("b",)]
            assert connection.execute("SELECT * FROM matches").fetchall() == []


def test_existing_readonly_cache_uses_local_fallback(tmp_path, monkeypatch):
    library = tmp_path / "master.db"
    adjacent = comparison_database_path(library)
    local = local_comparison_database_path(library)
    compare_tracks((), Event(), cache_path=adjacent)
    original = adjacent.read_bytes()
    connect = sqlite3.connect

    def readonly(path, **kwargs):
        if Path(path) == adjacent:
            return connect(adjacent.as_uri() + "?mode=ro", uri=True, **kwargs)
        return connect(path, **kwargs)

    monkeypatch.setattr("rekordbox_manager.comparison_store.sqlite3.connect", readonly)
    result = compare_tracks((), Event(), cache_path=adjacent, fallback_cache_path=local)
    assert result.cache_path == str(local) and local.is_file()
    assert adjacent.read_bytes() == original


def test_writable_adjacent_cache_still_uses_original_location(tmp_path):
    library = tmp_path / "master.db"
    adjacent = comparison_database_path(library)
    local = local_comparison_database_path(library)
    result = compare_tracks((), Event(), cache_path=adjacent, fallback_cache_path=local)
    assert result.cache_path == str(adjacent) and adjacent.is_file()
    assert not local.exists()


def test_corrupt_cache_is_reported_without_falling_back(tmp_path):
    adjacent = tmp_path / "master.trackvergleich.sqlite3"
    adjacent.write_bytes(b"not a database")
    local = local_comparison_database_path(tmp_path / "master.db")
    with pytest.raises(ManagerError, match="Vergleichsdatenbank"):
        compare_tracks((), Event(), cache_path=adjacent, fallback_cache_path=local)
    assert not local.exists()
    assert adjacent.read_bytes() == b"not a database"


def test_failure_of_both_locations_reports_paths_and_access_hint(tmp_path):
    adjacent = tmp_path / "missing" / "cache.sqlite3"
    blocked = tmp_path / "blocked"
    blocked.write_bytes(b"file instead of directory")
    local = blocked / "cache.sqlite3"
    with pytest.raises(ManagerError) as error:
        compare_tracks((), Event(), cache_path=adjacent, fallback_cache_path=local)
    assert str(adjacent) in str(error.value) and str(local) in str(error.value)
    assert "Schreibrechte" in str(error.value)


def test_local_paths_keep_collections_separate_and_do_not_create_directories(tmp_path):
    first = tmp_path / "first" / "master.db"
    second = tmp_path / "second" / "master.db"
    local = local_comparison_database_path(first)
    assert local != local_comparison_database_path(second)
    assert local == local_comparison_database_path(first.parent / "." / first.name)
    assert not local.parent.exists()


def test_cleanup_attempts_both_caches_if_one_is_corrupt(tmp_path):
    library = tmp_path / "master.db"
    local = local_comparison_database_path(library)
    local.parent.mkdir(parents=True)
    local.write_bytes(b"broken cache")
    adjacent = library.with_suffix(".trackvergleich.sqlite3")
    compare_tracks((), Event(), cache_path=adjacent)
    with closing(sqlite3.connect(adjacent)) as connection:
        with connection:
            connection.execute("INSERT INTO tracks (id,path,artist) VALUES ('a','','Artist')")
    with pytest.raises(OSError, match="broken|database"):
        remove_cached_tracks(library, ["a"])
    with closing(sqlite3.connect(adjacent)) as connection:
        assert not connection.execute("SELECT id FROM tracks").fetchall()
