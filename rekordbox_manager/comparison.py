"""Read-only comparison of five-second audio windows centered on the file midpoint."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

import av
import numpy as np

from .domain import DuplicateGroup, ManagerError, Track, normalize


SAMPLE_RATE = 8000
WINDOW_SECONDS = 5
SAMPLE_COUNT = SAMPLE_RATE * WINDOW_SECONDS
MAX_SHIFT = SAMPLE_RATE // 4
MIN_SIMILARITY = 90.0
FFT_SIZE = 1 << (2 * SAMPLE_COUNT - 1).bit_length()


class ComparisonCancelled(Exception):
    pass


def check_cancelled(cancel: Event):
    if cancel.is_set():
        raise ComparisonCancelled()


@dataclass(frozen=True)
class AudioMatch:
    first: Track
    second: Track
    similarity: float


@dataclass(frozen=True)
class ComparisonResult:
    matches: tuple[AudioMatch, ...] = ()
    skipped: tuple[str, ...] = ()
    checked: int = 0
    cancelled: bool = False


def read_middle(path: Path, cancel: Event) -> np.ndarray:
    """Decode only the center window (plus codec preroll), using the file's duration."""
    check_cancelled(cancel)
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ManagerError("Die Datei enthält keine Audiospur.")
            stream = container.streams.audio[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None
                        else float(container.duration / av.time_base) if container.duration is not None else 0)
            if not np.isfinite(duration) or duration <= 0:
                raise ManagerError("Die tatsächliche Dateidauer konnte nicht ermittelt werden.")
            if duration < WINDOW_SECONDS:
                raise ManagerError("Die Datei ist kürzer als fünf Sekunden.")
            origin = float((stream.start_time or 0) * stream.time_base)
            start = (duration - WINDOW_SECONDS) / 2 + origin
            end = start + WINDOW_SECONDS
            # Compressed formats need a little preroll for their decoder/resampler.
            container.seek(int(max(origin, start - 0.5) / float(stream.time_base)), stream=stream)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
            samples = np.zeros(SAMPLE_COUNT, dtype=np.float32)
            covered = np.zeros(SAMPLE_COUNT, dtype=bool)

            def consume(frame):
                if frame.time is None:
                    raise ManagerError("Die Audiospur enthält keine verlässlichen Zeitangaben.")
                offset = round((frame.time - start) * SAMPLE_RATE)
                left, right = max(0, offset), min(SAMPLE_COUNT, offset + frame.samples)
                if right > left:
                    samples[left:right] = frame.to_ndarray().reshape(-1)[left - offset:right - offset]
                    covered[left:right] = True

            for frame in container.decode(stream):
                check_cancelled(cancel)
                for output in resampler.resample(frame):
                    consume(output)
                if frame.time is not None and frame.time >= end + 0.1:
                    break
            for output in resampler.resample(None):
                consume(output)
            if not covered.all():
                raise ManagerError("Der mittlere Fünf-Sekunden-Ausschnitt konnte nicht vollständig gelesen werden.")
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise ManagerError(f"Die Musikdatei konnte nicht gelesen werden: {exc}") from exc
    if not np.isfinite(samples).all():
        raise ManagerError("Die Audiospur enthält ungültige Abtastwerte.")
    samples -= samples.mean()
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-5:
        raise ManagerError("Der mittlere Ausschnitt enthält nur Stille oder ein zu leises Signal.")
    return samples / rms


def signature(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(samples.astype(np.float64), n=FFT_SIZE)
    energy = np.concatenate(([0.0], np.cumsum(samples.astype(np.float64) ** 2)))
    return spectrum, energy


def similarity(first, second) -> float:
    """Absolute normalized cross-correlation; allow up to 250 ms codec/time offset.

    A score is a signal similarity measure, not a probability of track identity.
    Every alignment uses at least 4.75 seconds of the two five-second windows.
    """
    first_fft, first_energy = first
    second_fft, second_energy = second
    correlation = np.fft.irfft(first_fft * np.conj(second_fft), n=FFT_SIZE)
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    first_left, second_left = np.maximum(shifts, 0), np.maximum(-shifts, 0)
    length = SAMPLE_COUNT - np.abs(shifts)
    energy = ((first_energy[first_left + length] - first_energy[first_left])
              * (second_energy[second_left + length] - second_energy[second_left]))
    scores = np.divide(np.abs(correlation[shifts % FFT_SIZE]), np.sqrt(energy),
                       out=np.zeros(len(shifts)), where=energy > 1e-12)
    # Suppress floating-point noise at the inclusive 90 % boundary.
    return round(float(np.clip(scores.max(), 0, 1) * 100), 8)


def compare_tracks(tracks: tuple[Track, ...], cancel: Event, progress=None) -> ComparisonResult:
    """Compare readable MP3 files within each artist, regardless of title. Never write to source files.

    Spill signatures to temporary storage and keep a bounded cache in memory so a
    large library does not retain every decoded sample/FFT in RAM.
    """
    return _compare_tracks(tracks, cancel, progress, lambda track: normalize(track.artist), mp3_only=True)


def _compare_tracks(tracks, cancel, progress, group_key, *, mp3_only=False, strict=False):
    matches, skipped, checked = [], [], 0
    report = progress or (lambda current, total, message: None)
    try:
        with TemporaryDirectory(prefix="rekordbox-trackvergleich-") as directory:
            paths = {}
            artists = {}
            for index, track in enumerate(tracks):
                check_cancelled(cancel)
                report(index, len(tracks), f"Audio lesen: {index + 1}/{len(tracks)} · {track.title}")
                try:
                    if not track.path:
                        raise ManagerError("Kein Musikdateipfad vorhanden.")
                    if mp3_only and Path(track.path).suffix.casefold() != ".mp3":
                        raise ManagerError("Nur MP3-Dateien werden verglichen.")
                    key_for_group = group_key(track)
                    if not key_for_group:
                        raise ManagerError("Kein Interpret vorhanden.")
                    path = Path(track.path).resolve(strict=True)
                    if path not in paths:
                        samples = read_middle(path, cancel)
                        key = len(paths)
                        spectrum, energy = signature(samples)
                        np.savez(Path(directory) / f"{key}.npz", spectrum=spectrum, energy=energy)
                        paths[path] = key
                    artists.setdefault(key_for_group, []).append((track, paths[path]))
                    checked += 1
                except (ManagerError, OSError) as exc:
                    skipped.append(f"{track.title or 'Ohne Titel'} · ID {track.id} · {track.path}\n{exc}")

            @lru_cache(maxsize=32)
            def load(key):
                with np.load(Path(directory) / f"{key}.npz") as data:
                    return data["spectrum"], data["energy"]

            total = sum(len(group) * (len(group) - 1) // 2 for group in artists.values())
            current = 0
            try:
                for group in artists.values():
                    for index, (first, first_key) in enumerate(group[:-1]):
                        check_cancelled(cancel)
                        first_signature = load(first_key)
                        for second, second_key in group[index + 1:]:
                            check_cancelled(cancel)
                            score = 100.0 if first_key == second_key else similarity(first_signature, load(second_key))
                            if score > MIN_SIMILARITY or (not strict and score == MIN_SIMILARITY):
                                matches.append(AudioMatch(first, second, score))
                            current += 1
                            if current % 25 == 0 or current == total:
                                report(current, total, f"Trackpaare vergleichen: {current}/{total} · {len(matches)} Treffer")
            finally:
                load.cache_clear()
    except ComparisonCancelled:
        # Discard partial matches so they cannot be mistaken for a complete scan.
        return ComparisonResult(skipped=tuple(skipped), checked=checked, cancelled=True)
    return ComparisonResult(tuple(sorted(matches, key=lambda match: -match.similarity)), tuple(skipped), checked)


def verify_duplicate_groups(groups: tuple[DuplicateGroup, ...], cancel: Event | None = None,
                            progress=None) -> tuple[tuple[DuplicateGroup, ...], tuple[str, ...]]:
    """Confirm metadata candidates using strictly more than 90 % audio similarity.

    Partition each candidate group into disjoint cliques. Every possible keeper
    in a confirmed group must match every other member; similarity is not transitive.
    """
    cancel = cancel if cancel is not None else Event()
    tracks = tuple(track for group in groups for track in group.tracks)
    if not tracks:
        return (), ()
    result = _compare_tracks(tracks, cancel, progress, lambda track: track.duplicate_key, strict=True)
    if result.cancelled:
        raise ComparisonCancelled()
    scores = {frozenset((match.first.id, match.second.id)): match.similarity for match in result.matches}
    confirmed = []
    for group in groups:
        partitions = []
        for track in group.tracks:
            check_cancelled(cancel)
            for partition in partitions:
                if all(frozenset((track.id, other.id)) in scores for other in partition):
                    partition.append(track)
                    break
            else:
                partitions.append([track])
        for partition in partitions:
            if len(partition) > 1:
                minimum = min(scores[frozenset((first.id, second.id))]
                              for index, first in enumerate(partition) for second in partition[index + 1:])
                confirmed.append(DuplicateGroup(group.key, tuple(partition), partition[0].id, minimum))
    return tuple(confirmed), result.skipped
