from dataclasses import replace
from contextlib import closing
import os
from pathlib import Path
import sqlite3
from threading import Event
import wave

import av
import numpy as np
import pytest

from rekordbox_manager.comparison import (
    AudioMatch, ComparisonCancelled, ComparisonResult, MIN_SIMILARITY, SAMPLE_COUNT, SAMPLE_RATE,
    compare_tracks, comparison_groups, read_middle, signature, similarity, verify_duplicate_groups,
)
from rekordbox_manager.domain import ManagerError, Track, find_duplicates
from rekordbox_manager.comparison_store import ComparisonStore, comparison_database_path, remove_cached_tracks


def write_wav(path, samples, rate=SAMPLE_RATE):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def write_mp3(path, samples, rate=SAMPLE_RATE):
    with av.open(str(path), "w", format="mp3") as output:
        stream = output.add_stream("libmp3lame", rate=rate)
        stream.layout = "mono"
        stream.bit_rate = 64000
        frame = av.AudioFrame.from_ndarray(
            (np.clip(samples, -1, 1) * 32767).astype("<i2").reshape(1, -1),
            format="s16", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return path


def test_reads_exact_center_window_and_releases_file(tmp_path):
    # Edges differ from the middle; DB metadata is deliberately not involved.
    rng = np.random.default_rng(12)
    samples = rng.normal(0, 0.15, 12 * SAMPLE_RATE)
    path = write_wav(tmp_path / "Mitte mit Umlaut ä.wav", samples)
    result = read_middle(path, Event())
    expected = samples[int(3.5 * SAMPLE_RATE):int(8.5 * SAMPLE_RATE)]
    assert len(result) == SAMPLE_COUNT
    assert np.corrcoef(expected, result)[0, 1] > 0.99999
    path.rename(tmp_path / "released.wav")


@pytest.mark.parametrize("rate", [8000, 44100, 48000])
def test_mp3_center_window_at_common_sample_rates(tmp_path, rate):
    path = write_mp3(tmp_path / "Modular Phaze - Æxus.mp3",
                     np.random.default_rng(12).normal(0, 0.1, rate * 12), rate)
    samples = read_middle(path, Event())
    assert len(samples) == SAMPLE_COUNT and np.isfinite(samples).all()
    path.rename(tmp_path / "released.mp3")


def test_failed_seek_recovers_exact_middle_without_padding(tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    samples = np.random.default_rng(12).normal(0, 0.15, 12 * SAMPLE_RATE)
    path = write_wav(tmp_path / "bad-index.wav", samples)

    def failed_seek(*args):
        raise ManagerError("Der mittlere Fünf-Sekunden-Ausschnitt konnte nicht vollständig gelesen werden.")

    monkeypatch.setattr(comparison, "_read_middle_seek", failed_seek)
    result = read_middle(path, Event())
    expected = samples[int(3.5 * SAMPLE_RATE):int(8.5 * SAMPLE_RATE)]
    assert len(result) == SAMPLE_COUNT
    assert np.corrcoef(expected, result)[0, 1] > 0.99999
    short = write_wav(tmp_path / "short.wav", samples[:4 * SAMPLE_RATE])
    with pytest.raises(ManagerError, match="kürzer"):
        read_middle(short, Event())


def test_sequential_recovery_can_be_cancelled_and_releases_files(tmp_path):
    from rekordbox_manager import comparison
    path = write_mp3(tmp_path / "fallback.mp3", np.ones(12 * SAMPLE_RATE) * 0.1)

    class CancelDuringDecode(Event):
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 10

    # The recovery is used directly to ensure cancellation occurs while decoding.
    with pytest.raises(ComparisonCancelled):
        comparison._read_middle_sequential(path, CancelDuringDecode())
    path.rename(tmp_path / "released.mp3")


def test_detects_same_artist_audio_with_different_titles_gain_and_file_edges(tmp_path):
    rng = np.random.default_rng(9)
    samples = rng.normal(0, 0.1, 12 * SAMPLE_RATE)
    other = samples * 2
    other[:3 * SAMPLE_RATE] = rng.normal(0, 0.1, 3 * SAMPLE_RATE)
    other[9 * SAMPLE_RATE:] = rng.normal(0, 0.1, 3 * SAMPLE_RATE)
    a = Track("a", "First title", "First artist", path=str(write_mp3(tmp_path / "a.mp3", samples)), duration=999)
    b = Track("b", "Unrelated title", "First artist", path=str(write_mp3(tmp_path / "b.mp3", other)), duration=1)
    c = Track("c", "First title", "First artist", path=str(write_mp3(
        tmp_path / "c.mp3", rng.normal(0, 0.1, len(samples)))))
    progress = []
    result = compare_tracks((a, b, c), Event(), lambda *args: progress.append(args))
    assert result.checked == 3 and not result.skipped
    assert [(match.first.id, match.second.id) for match in result.matches] == [("a", "b")]
    assert result.matches[0].similarity > 99.9
    assert progress[-1][:2] == (3, 3)


def test_only_compares_mp3_pairs_within_normalized_artist(tmp_path):
    from rekordbox_manager import comparison
    path = write_mp3(tmp_path / "same.MP3", np.random.default_rng(4).normal(0, 0.1, 12 * SAMPLE_RATE))
    first = Track("a", "A", "Ärtist Name", path=str(path))
    tracks = (first, replace(first, id="b", artist="  A\u0308RTIST   NAME  "),
              replace(first, id="c", artist="Other"), replace(first, id="d", artist="OTHER"),
              replace(first, id="e", artist="Solo"))
    progress = []
    result = comparison.compare_tracks(tracks, Event(), lambda *args: progress.append(args))
    assert result.checked == 5 and not result.skipped
    assert [(match.first.id, match.second.id) for match in result.matches] == [("a", "b"), ("c", "d")]
    assert progress[-1][:2] == (2, 2)


def test_skips_other_formats_and_missing_artists_before_reading_audio(tmp_path, monkeypatch):
    from rekordbox_manager import comparison

    def unexpected_read(*args):
        pytest.fail("Excluded files must not be decoded")

    monkeypatch.setattr(comparison, "read_middle", unexpected_read)
    tracks = tuple(Track(str(i), "A", artist, path=str(tmp_path / filename))
                   for i, (filename, artist) in enumerate((
                       ("a.wav", "Artist"), ("b.flac", "Artist"),
                       ("c.mp3", ""), ("d.mp3", "   "))))
    result = compare_tracks(tracks, Event())
    assert not result.matches and result.checked == 0
    assert len(result.skipped) == 4
    assert not result.errors
    assert all("Nur MP3" in reason for reason in result.skipped[:2])
    assert all("Kein Interpret" in reason for reason in result.skipped[2:])


def test_track_groups_are_disjoint_and_do_not_assume_transitive_similarity():
    a, b, c, d = (Track(key, key, "Artist") for key in "abcd")
    result = ComparisonResult((AudioMatch(a, b, 95), AudioMatch(b, c, 99), AudioMatch(c, d, 90)))
    groups = comparison_groups(result)
    assert [tuple(t.id for t in group.tracks) for group in groups] == [("a", "b"), ("c", "d")]
    assert [group.similarity for group in groups] == [95, 90]
    assert comparison_groups(replace(result, matches=tuple(reversed(result.matches)))) == groups
    assert not comparison_groups(replace(result, cancelled=True))


def test_track_groups_prefer_bitrate_then_playlists_and_show_minimum_score():
    a = Track("a", "First", "Artist", bitrate=128, playlists=2)
    b = Track("b", "Second", "Artist", bitrate=320)
    c = Track("c", "Third", "Artist", bitrate=320, playlists=1)
    groups = comparison_groups(ComparisonResult((AudioMatch(a, b, 98), AudioMatch(a, c, 90), AudioMatch(b, c, 96))))
    assert len(groups) == 1
    assert groups[0].recommended_id == "c" and groups[0].similarity == 90
    assert [track.id for track in groups[0].tracks] == ["c", "b", "a"]


@pytest.mark.parametrize("score, expected", [(89.9999, False), (90.0, False), (90.0001, True)])
@pytest.mark.parametrize("title", ["Song (Original)", "Song (Original Mix)"])
def test_duplicate_audio_requires_strictly_more_than_ninety_percent(tmp_path, monkeypatch, score, expected, title):
    monkeypatch.setattr("rekordbox_manager.comparison.similarity", lambda *args: score)
    samples = np.random.default_rng(5).normal(0, 0.1, 12 * SAMPLE_RATE)
    tracks = (Track("a", title, "Alice + Bob", path=str(write_mp3(tmp_path / "a.mp3", samples))),
              Track("b", "Song", "Alice & Bob", path=str(write_mp3(tmp_path / "b.mp3", samples))))
    groups, notices = verify_duplicate_groups(find_duplicates(tracks))
    assert bool(groups) == expected
    assert not notices
    if expected:
        assert groups[0].similarity == score


def test_duplicate_audio_splits_different_recordings_and_preserves_recommendations(tmp_path):
    rng = np.random.default_rng(45)
    first, second = (rng.normal(0, 0.1, 12 * SAMPLE_RATE) for _ in range(2))
    tracks = tuple(Track(track_id, "Song", "Artist", bitrate=bitrate,
                         path=str(write_mp3(tmp_path / f"{track_id}.mp3", audio)))
                   for track_id, bitrate, audio in (("a", 128, first), ("b", 320, first),
                                                   ("c", 320, second), ("d", 128, second)))
    groups, notices = verify_duplicate_groups(find_duplicates(tracks))
    assert not notices
    assert [tuple(track.id for track in group.tracks) for group in groups] == [("b", "a"), ("c", "d")]
    assert [group.recommended_id for group in groups] == ["b", "c"]
    assert all(group.similarity > 99.9 for group in groups)


def test_duplicate_similarity_is_not_treated_as_transitive(tmp_path, monkeypatch):
    scores = iter((95.0, 80.0, 95.0))  # A~B and B~C do not imply A~C.
    monkeypatch.setattr("rekordbox_manager.comparison.similarity", lambda *args: next(scores))
    audio = np.random.default_rng(5).normal(0, 0.1, 12 * SAMPLE_RATE)
    tracks = tuple(Track(track_id, "Song", "Artist", path=str(write_mp3(tmp_path / f"{track_id}.mp3", audio)))
                   for track_id in "abc")
    groups, notices = verify_duplicate_groups(find_duplicates(tracks))
    assert not notices
    assert len(groups) == 1 and tuple(track.id for track in groups[0].tracks) == ("a", "b")
    assert groups[0].similarity == 95.0


def test_duplicate_verification_only_reads_metadata_candidates_and_reports_unreadable_files(tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    path = write_mp3(tmp_path / "a.mp3", np.random.default_rng(8).normal(0, 0.1, 12 * SAMPLE_RATE))
    tracks = (Track("a", "Song", "Artist", path=str(path)),
              Track("b", "Song (Original)", "Artist", path=str(tmp_path / "missing.mp3")),
              Track("c", "Different title", "Artist", path=str(tmp_path / "unrelated.mp3")))
    calls = []
    real_read = comparison.read_middle

    def read(path, cancel):
        calls.append(path)
        return real_read(path, cancel)

    monkeypatch.setattr(comparison, "read_middle", read)
    groups, notices = verify_duplicate_groups(find_duplicates(tracks))
    assert not groups
    assert calls == [path.resolve()]
    assert len(notices) == 1 and "missing.mp3" in notices[0]


def test_duplicate_verification_cancellation_discards_results(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    path = write_mp3(tmp_path / "a.mp3", np.random.default_rng(8).normal(0, 0.1, 12 * SAMPLE_RATE))
    tracks = tuple(Track(str(i), "Song", "Artist", path=str(path)) for i in range(10))
    cancel = Event()

    def progress(current, total, message):
        if message.startswith("Trackpaare"):
            cancel.set()

    with pytest.raises(ComparisonCancelled):
        verify_duplicate_groups(find_duplicates(tracks), cancel, progress)
    assert not list(tmp_path.glob("rekordbox-trackvergleich-*"))


@pytest.mark.parametrize("correlation, expected", [(0.899, False), (0.9, True), (0.901, True)])
def test_ninety_percent_boundary(correlation, expected):
    rng = np.random.default_rng(28)
    first = rng.normal(size=SAMPLE_COUNT)
    first -= first.mean()
    first /= np.linalg.norm(first)
    noise = rng.normal(size=SAMPLE_COUNT)
    noise -= noise.mean()
    noise -= np.dot(noise, first) * first
    noise /= np.linalg.norm(noise)
    second = correlation * first + np.sqrt(1 - correlation ** 2) * noise
    score = similarity(signature(first), signature(second))
    assert score == pytest.approx(correlation * 100, abs=0.0001)
    assert (score >= MIN_SIMILARITY) == expected


def test_small_time_shift_is_tolerated_and_silence_is_not_a_match(tmp_path):
    rng = np.random.default_rng(2)
    first = rng.normal(size=SAMPLE_COUNT)
    shift = SAMPLE_RATE // 10
    second = np.concatenate((rng.normal(size=shift), first[:-shift]))
    assert similarity(signature(first), signature(second)) > 99.99
    path = write_wav(tmp_path / "silence.wav", np.zeros(SAMPLE_COUNT * 2))
    with pytest.raises(ManagerError, match="Stille"):
        read_middle(path, Event())


def test_skips_missing_short_corrupt_and_silent_files_with_reasons(tmp_path):
    short = write_mp3(tmp_path / "short.mp3", np.zeros(4 * SAMPLE_RATE))
    silent = write_mp3(tmp_path / "silent.mp3", np.zeros(12 * SAMPLE_RATE))
    corrupt = tmp_path / "corrupt.mp3"
    corrupt.write_bytes(b"not an audio file")
    tracks = tuple(Track(str(i), path.stem, "Artist", path=str(path))
                   for i, path in enumerate((short, silent, corrupt, tmp_path / "missing.mp3")))
    result = compare_tracks(tracks, Event())
    assert result.checked == 0 and not result.matches
    assert len(result.skipped) == 4
    assert "kürzer" in result.skipped[0] and "Stille" in result.skipped[1]
    assert result.errors == result.skipped


def test_same_file_for_multiple_track_entries_is_read_once(tmp_path, monkeypatch):
    path = write_mp3(tmp_path / "same.mp3", np.random.default_rng(4).normal(0, 0.1, 12 * SAMPLE_RATE))
    first = Track("a", "A", "Artist", path=str(path))
    from rekordbox_manager import comparison
    real_read = comparison.read_middle
    calls = []

    def read(*args):
        calls.append(args)
        return real_read(*args)

    monkeypatch.setattr(comparison, "read_middle", read)
    result = compare_tracks((first, replace(first, id="b"), replace(first, id="c")), Event())
    assert len(calls) == 1 and result.checked == 3
    assert len(result.matches) == 3
    assert all(match.similarity == 100 for match in result.matches)


def test_cancellation_discards_partial_results_and_cleans_temporary_files(tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    cancel = Event()
    path = write_mp3(tmp_path / "same.mp3", np.random.default_rng(3).normal(0, 0.1, 12 * SAMPLE_RATE))
    first = Track("a", "A", "Artist", path=str(path))

    def progress(current, total, message):
        if message.startswith("Trackpaare"):
            cancel.set()

    tracks = tuple(replace(first, id=str(i)) for i in range(10))
    result = comparison.compare_tracks(tracks, cancel, progress)
    assert result.cancelled and not result.matches
    assert not list(tmp_path.glob("rekordbox-trackvergleich-*"))


@pytest.mark.parametrize("codec, extension, rate", [("libmp3lame", "mp3", 44100), ("flac", "flac", 48000)])
def test_reads_real_encoded_audio_across_formats_and_sample_rates(tmp_path, codec, extension, rate):
    # A changing multitone signal, encoded by the real codec, not renamed WAV bytes.
    time = np.arange(rate * 12) / rate
    audio = (0.15 * np.sin(2 * np.pi * (310 * time + 8 * time ** 2))
             + 0.1 * np.sin(2 * np.pi * 731 * time) * np.sin(2 * np.pi * 2 * time))
    source = write_wav(tmp_path / "source.wav", audio, rate)
    encoded = tmp_path / f"encoded.{extension}"
    with av.open(str(source)) as input_file, av.open(str(encoded), "w") as output_file:
        stream = output_file.add_stream(codec, rate=rate)
        stream.layout = "mono"
        if extension == "mp3":
            stream.bit_rate = 128000
        for frame in input_file.decode(audio=0):
            for packet in stream.encode(frame):
                output_file.mux(packet)
        for packet in stream.encode(None):
            output_file.mux(packet)
    assert similarity(signature(read_middle(source, Event())),
                      signature(read_middle(encoded, Event()))) >= 90


@pytest.fixture
def indexed_audio(tmp_path):
    rng = np.random.default_rng(2026)
    first, other = (rng.normal(0, 0.1, 12 * SAMPLE_RATE) for _ in range(2))
    tracks = tuple(Track(key, key, "Artist", path=str(write_mp3(tmp_path / f"{key}.mp3", samples)))
                   for key, samples in (("a", first), ("b", first), ("c", other)))
    cache = comparison_database_path(tmp_path / "master.db")
    result = compare_tracks(tracks, Event(), cache_path=cache)
    assert result.updated == 3 and result.checked == 3
    assert [(m.first.id, m.second.id) for m in result.matches] == [("a", "b")]
    return tracks, cache, first, other


def test_persistent_scan_and_unchanged_update_never_decode_recompare_or_rewrite(indexed_audio, monkeypatch):
    tracks, cache, _, _ = indexed_audio
    before = cache.read_bytes()

    def unexpected(*args):
        pytest.fail("Unchanged tracks must use persisted results without decoding or comparing")

    monkeypatch.setattr("rekordbox_manager.comparison.read_middle", unexpected)
    monkeypatch.setattr("rekordbox_manager.comparison.similarity", unexpected)
    for update in (False, True):
        result = compare_tracks(tracks, Event(), cache_path=cache, update=update)
        assert result.reused == 3 and result.updated == 0 and result.checked == 3
        assert len(result.matches) == 1 and not result.skipped
        assert cache.read_bytes() == before


def test_update_indexes_only_new_files_and_compares_only_new_pairs(indexed_audio, tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    tracks, cache, samples, _ = indexed_audio
    extra = Track("d", "New title", "Artist", path=str(write_mp3(tmp_path / "d.mp3", samples)))
    reads, scores = [], []
    real_read, real_score = comparison.read_middle, comparison.similarity

    def read(path, cancel):
        reads.append(path)
        return real_read(path, cancel)

    def score(*args):
        scores.append(True)
        return real_score(*args)

    monkeypatch.setattr(comparison, "read_middle", read)
    monkeypatch.setattr(comparison, "similarity", score)
    result = compare_tracks((*tracks, extra), Event(), cache_path=cache)
    assert not reads and not scores and len(result.skipped) == 1
    assert "Aktualisieren" in result.skipped[0]
    result = compare_tracks((*tracks, extra), Event(), cache_path=cache, update=True)
    assert reads == [tmp_path / "d.mp3"] and len(scores) == 3
    assert result.updated == 1 and result.reused == 3
    assert {(m.first.id, m.second.id) for m in result.matches} == {("a", "b"), ("a", "d"), ("b", "d")}


def test_changed_file_is_excluded_until_update_and_old_matches_are_invalidated(indexed_audio, monkeypatch):
    from rekordbox_manager import comparison
    tracks, cache, _, other = indexed_audio
    changed = tracks[0]
    old_stat = os.stat(changed.path)
    write_mp3(changed.path, other)
    os.utime(changed.path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))
    reads = []
    real_read = comparison.read_middle

    def read(path, cancel):
        reads.append(str(path))
        return real_read(path, cancel)

    monkeypatch.setattr(comparison, "read_middle", read)
    result = compare_tracks(tracks, Event(), cache_path=cache)
    assert not result.matches and not reads and len(result.skipped) == 1
    result = compare_tracks(tracks, Event(), cache_path=cache, update=True)
    assert reads == [changed.path] and result.updated == 1
    assert [(m.first.id, m.second.id) for m in result.matches] == [("a", "c")]


def test_artist_changes_reuse_audio_and_title_changes_use_current_metadata(indexed_audio, monkeypatch):
    tracks, cache, _, _ = indexed_audio
    changed = (replace(tracks[0], artist="Other"), *tracks[1:])
    monkeypatch.setattr("rekordbox_manager.comparison.read_middle", lambda *args: pytest.fail("Reuse audio for metadata edits"))
    result = compare_tracks(changed, Event(), cache_path=cache, update=True)
    assert not result.matches and result.updated == 0
    renamed = replace(tracks[0], title="Renamed")
    result = compare_tracks((renamed, *tracks[1:]), Event(), cache_path=cache, update=True)
    assert len(result.matches) == 1 and result.matches[0].first.title == "Renamed"


def test_cached_deletion_cascades_and_external_removal_is_pruned(indexed_audio):
    tracks, cache, _, _ = indexed_audio
    remove_cached_tracks(Path(tracks[0].path).parent / "master.db", ["a"])
    with closing(sqlite3.connect(cache)) as connection:
        assert connection.execute("SELECT id FROM tracks ORDER BY id").fetchall() == [("b",), ("c",)]
        assert connection.execute("SELECT * FROM matches").fetchall() == []
    result = compare_tracks(tracks[2:], Event(), cache_path=cache)
    assert result.checked == 1 and not result.matches
    with ComparisonStore(cache) as store:
        assert set(store.entries()) == {"c"}


@pytest.mark.parametrize("phase", ["index", "pairs"])
def test_cancelled_persistent_scan_resumes_without_rereading_indexed_files(tmp_path, monkeypatch, phase):
    from rekordbox_manager import comparison
    samples = np.random.default_rng(123).normal(0, 0.1, 12 * SAMPLE_RATE)
    tracks = tuple(Track(str(index), str(index), "Artist",
                         path=str(write_mp3(tmp_path / f"{index}.mp3", samples))) for index in range(3))
    cache = tmp_path / "cache.sqlite3"
    cancel, reads = Event(), []
    real_read = comparison.read_middle

    def read(path, event):
        reads.append(path)
        return real_read(path, event)

    def report(current, total, message):
        if (phase == "index" and current == 1 and message.startswith("Vergleichsdatenbank")) or message.startswith("Trackpaare"):
            cancel.set()

    monkeypatch.setattr(comparison, "read_middle", read)
    result = compare_tracks(tracks, cancel, report, cache_path=cache)
    assert result.cancelled and not result.matches
    result = compare_tracks(tracks, Event(), cache_path=cache)
    assert len(result.matches) == 3 and not result.cancelled
    # The interrupted decoder checks cancellation before opening a file.
    assert sum(path == tmp_path / "0.mp3" for path in reads) == 1
    if phase == "pairs":
        assert len(reads) == 3


def test_cancellation_commits_small_match_batch_and_keeps_audio_for_resume(tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    samples = np.random.default_rng(5).normal(0, 0.1, 12 * SAMPLE_RATE)
    tracks = tuple(Track(str(i), str(i), "Artist",
                         path=str(write_mp3(tmp_path / f"{i}.mp3", samples))) for i in range(3))
    cache, cancel = tmp_path / "cache.sqlite3", Event()

    # Request cancellation after the first pair, before the normal 25-pair commit.
    score = comparison.similarity

    def stop_after_score(*args):
        value = score(*args)
        cancel.set()
        return value

    monkeypatch.setattr(comparison, "similarity", stop_after_score)
    result = compare_tracks(tracks, cancel, cache_path=cache)
    assert result.cancelled and not result.matches
    with ComparisonStore(cache) as store:
        assert len(tuple(store.matches())) == 1
        assert len(store.entries()) == 3
    monkeypatch.setattr(comparison, "similarity", score)
    monkeypatch.setattr(comparison, "read_middle", lambda *args: pytest.fail("Saved audio must survive cancellation"))
    result = compare_tracks(tracks, Event(), cache_path=cache)
    assert result.reused == 3 and len(result.matches) == 3


def test_update_retries_old_incomplete_window_errors_without_discarding_good_audio(indexed_audio, monkeypatch):
    from rekordbox_manager import comparison
    from rekordbox_manager.comparison_store import path_key
    tracks, cache, _, _ = indexed_audio
    failed = tracks[0]
    stat = os.stat(failed.path)
    with ComparisonStore(cache) as store:
        store.put(failed.id, path_key(failed.path), "artist", (stat.st_size, stat.st_mtime_ns),
                  error="Der mittlere Fünf-Sekunden-Ausschnitt konnte nicht vollständig gelesen werden.")
    reads, real_read = [], comparison.read_middle

    def read(path, cancel):
        reads.append(str(path))
        return real_read(path, cancel)

    monkeypatch.setattr(comparison, "read_middle", read)
    assert compare_tracks(tracks, Event(), cache_path=cache).errors
    assert not reads
    result = compare_tracks(tracks, Event(), cache_path=cache, update=True)
    assert not reads and result.reused == 3
    assert len(result.matches) == 1 and not result.errors


def test_duplicate_file_references_use_persisted_audio_even_for_new_ids(indexed_audio, monkeypatch):
    tracks, cache, _, _ = indexed_audio
    extra = replace(tracks[0], id="d")
    monkeypatch.setattr("rekordbox_manager.comparison.read_middle", lambda *args: pytest.fail("File already indexed"))
    result = compare_tracks((extra, *tracks), Event(), cache_path=cache, update=True)
    assert result.updated == 0 and result.reused == 4 and len(result.matches) == 3


def test_cached_errors_are_reused_and_repaired_files_are_indexed(tmp_path, monkeypatch):
    from rekordbox_manager import comparison
    path = tmp_path / "broken.mp3"
    path.write_bytes(b"corrupt")
    track = Track("a", "A", "Artist", path=str(path))
    cache = tmp_path / "cache.sqlite3"
    result = compare_tracks((track,), Event(), cache_path=cache)
    assert result.errors
    real_read = comparison.read_middle
    monkeypatch.setattr(comparison, "read_middle", lambda *args: pytest.fail("Unchanged broken file must not be decoded again"))
    assert compare_tracks((track,), Event(), cache_path=cache, update=True).errors == result.errors
    write_mp3(path, np.random.default_rng(8).normal(0, 0.1, 12 * SAMPLE_RATE))
    monkeypatch.setattr(comparison, "read_middle", real_read)
    result = compare_tracks((track,), Event(), cache_path=cache, update=True)
    assert result.updated == 1 and not result.errors


def test_algorithm_or_library_identity_change_rebuilds_the_cache(indexed_audio, monkeypatch):
    tracks, cache, _, _ = indexed_audio
    monkeypatch.setattr("rekordbox_manager.comparison_store.ALGORITHM_VERSION", "new-version")
    assert compare_tracks(tracks, Event(), cache_path=cache).updated == 3
    assert compare_tracks(tracks, Event(), cache_path=cache, library_id="other-library").updated == 3


def test_unusable_persistent_database_reports_its_path(tmp_path):
    path = tmp_path / "cache.sqlite3"
    path.write_bytes(b"not a database")
    with pytest.raises(ManagerError, match="Vergleichsdatenbank") as error:
        compare_tracks((), Event(), cache_path=path)
    assert str(path) in str(error.value)
