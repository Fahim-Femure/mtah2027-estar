"""Contract v1: normalized pitch coordinates, zero-based tactical video frames.

Arrays are copied and marked read-only at construction. NaN represents missing
observations; partial coordinate pairs and infinity are rejected, never guessed.
Validation is explicit (not assert-based) so it also operates under python -O.
"""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray

Side = Literal["H", "A"]
Label = Literal["H", "A", "D", "X"]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def array(value: NDArray, dtype: object, shape: tuple, name: str) -> NDArray:
    require(isinstance(value, np.ndarray), f"{name}: expected ndarray")
    require(value.dtype == np.dtype(dtype), f"{name}: incorrect dtype")
    require(value.shape == shape, f"{name}: incorrect shape")
    result = value.copy()
    result.flags.writeable = False
    return result


def strings(value: NDArray, shape: tuple, name: str) -> NDArray:
    require(isinstance(value, np.ndarray), f"{name}: expected ndarray")
    require(value.dtype.kind == "U" and value.shape == shape, f"{name}: expected Unicode array of shape {shape}")
    result = value.copy()
    result.flags.writeable = False
    return result


def coordinates(xy: NDArray, observed: NDArray, name: str) -> None:
    require(not np.isinf(xy).any(), f"{name}: infinity")
    finite = np.isfinite(xy)
    require(np.array_equal(finite[..., 0], finite[..., 1]), f"{name}: partial coordinate pair")
    require(np.array_equal(finite.all(axis=-1), observed), f"{name}: observation mask mismatch")


@dataclass(frozen=True)
class Event:
    event_id: str
    start_frame: int
    end_frame: int
    code: str
    team: Side | None
    tags: Mapping[str, str]

    def __post_init__(self):
        require(bool(self.event_id) and bool(self.code), "event identifiers must be nonempty")
        require(self.code == self.code.strip().upper(), "event code must be normalized")
        require(integer(self.start_frame) and integer(self.end_frame), "event indices must be integers")
        require(0 <= self.start_frame < self.end_frame, "event interval must be nonempty and half-open")
        require(self.team in ("H", "A", None), "unknown event team")
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in self.tags.items()), "event tags must be strings")
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame


@dataclass(frozen=True)
class Geometry:
    matrices: FloatArray
    valid: NDArray[np.bool_]
    reprojection_error: FloatArray

    def __post_init__(self):
        n = len(self.valid)
        for name, dtype, shape in (("matrices", np.float64, (n, 3, 3)), ("valid", bool, (n,)), ("reprojection_error", np.float64, (n,))):
            object.__setattr__(self, name, array(getattr(self, name), dtype, shape, name))
        require(np.isfinite(self.matrices[self.valid]).all(), "valid matrices must be finite")
        require(np.isnan(self.matrices[~self.valid]).all(), "invalid matrices must be NaN")
        if self.valid.any():
            require(np.all(np.linalg.cond(self.matrices[self.valid]) < 1e12), "singular or ill-conditioned geometry")
        require(np.isfinite(self.reprojection_error[self.valid]).all(), "valid geometry needs finite error")
        require(not np.isinf(self.reprojection_error).any(), "infinite reprojection error")
        require(np.all(self.reprojection_error[np.isfinite(self.reprojection_error)] >= 0), "negative reprojection error")


@dataclass(frozen=True)
class MatchInput:
    schema_version: int
    match_id: str
    source: Literal["tactical"]
    n_frames: int
    fps: int
    pitch_length_m: float
    pitch_width_m: float
    coordinate_system: str
    team_names: Mapping[Side, str]
    segments: tuple[tuple[int, int, int], ...]
    frame: IntArray
    half: NDArray[np.int8]
    slot_id: NDArray[np.str_]
    slot_team: NDArray[np.str_]
    player_xy: FloatArray
    player_observed: NDArray[np.bool_]
    player_speed_mps: FloatArray
    ball_xy: FloatArray
    ball_observed: NDArray[np.bool_]
    ball_speed_mps: FloatArray
    events: tuple[Event, ...] = ()
    geometry: Geometry | None = None

    def __post_init__(self):
        require(integer(self.schema_version) and self.schema_version == 1, "unsupported schema")
        require(self.source == "tactical", "only tactical timeline supported")
        require(integer(self.n_frames) and self.n_frames > 0, "invalid frame count")
        require(integer(self.fps) and self.fps == 25, "contract v1 requires 25 fps")
        require(self.match_id == "BAR-ATH" or self.match_id.startswith("SYNTH-"), "unknown match kind")
        require(self.match_id != "BAR-ATH" or self.n_frames == 199319, "competition frame count must be 199319")
        require(self.coordinate_system == "pitch_normalized_0_1", "incorrect coordinate convention")
        require(np.isfinite([self.pitch_length_m, self.pitch_width_m]).all() and min(self.pitch_length_m, self.pitch_width_m) > 0, "invalid pitch")
        require(set(self.team_names) == {"H", "A"} and all(isinstance(v, str) and v.strip() for v in self.team_names.values()), "two named teams required")
        object.__setattr__(self, "team_names", MappingProxyType(dict(self.team_names)))
        n, s = self.n_frames, len(self.slot_id)
        require(s > 0, "no slots")
        for name, dtype, shape in (
            ("frame", np.int64, (n,)), ("half", np.int8, (n,)),
            ("player_xy", np.float64, (n, s, 2)), ("player_observed", bool, (n, s)),
            ("player_speed_mps", np.float64, (n, s)), ("ball_xy", np.float64, (n, 2)),
            ("ball_observed", bool, (n,)), ("ball_speed_mps", np.float64, (n,)),
        ):
            object.__setattr__(self, name, array(getattr(self, name), dtype, shape, name))
        for name in ("slot_id", "slot_team"):
            object.__setattr__(self, name, strings(getattr(self, name), (s,), name))
        require(len(set(self.slot_id)) == s and all(self.slot_id != ""), "slot IDs must be unique and nonempty")
        require(set(self.slot_team) <= {"H", "A"}, "unknown slot team")
        require(np.array_equal(self.frame, np.arange(n)), "frame must be contiguous 0..N-1")
        expected = np.zeros(n, dtype=np.int8)
        previous = 0
        segments = tuple(tuple(block) for block in self.segments)
        for start, stop, half in segments:
            require(all(integer(v) for v in (start, stop, half)), "segment values must be integers")
            require(previous <= start < stop <= n and half in (1, 2), "invalid or overlapping segment")
            expected[start:stop] = half
            previous = stop
        object.__setattr__(self, "segments", segments)
        require(np.array_equal(self.half, expected), "half differs from segments")
        coordinates(self.player_xy, self.player_observed, "players")
        coordinates(self.ball_xy, self.ball_observed, "ball")
        for speed in (self.player_speed_mps, self.ball_speed_mps):
            require(not np.isinf(speed).any() and np.all(speed[np.isfinite(speed)] >= 0), "invalid speed")
        events = tuple(self.events)
        require(all(isinstance(e, Event) and e.end_frame <= n for e in events), "event outside timeline")
        require(len({e.event_id for e in events}) == len(events), "duplicate event IDs")
        require(list(events) == sorted(events, key=lambda e: (e.start_frame, e.end_frame, e.event_id)), "events must be ordered")
        object.__setattr__(self, "events", events)
        require(self.geometry is None or (isinstance(self.geometry, Geometry) and len(self.geometry.valid) == n), "geometry length mismatch")


@dataclass(frozen=True)
class PredictionResult:
    schema_version: int
    match_id: str
    source: str
    frame: IntArray
    label: NDArray[np.str_]
    engine_version: str
    config_digest: str
    confidence: FloatArray | None = None
    reason: NDArray[np.str_] | None = None

    def __post_init__(self):
        n = len(self.frame)
        require(integer(self.schema_version) and self.schema_version == 1, "unsupported prediction schema")
        require(self.source == "tactical" and bool(self.match_id), "invalid prediction metadata")
        require(n > 0 and bool(self.engine_version) and bool(self.config_digest), "missing prediction provenance")
        object.__setattr__(self, "frame", array(self.frame, np.int64, (n,), "frame"))
        object.__setattr__(self, "label", strings(self.label, (n,), "label"))
        require(np.array_equal(self.frame, np.arange(n)), "prediction frames must be contiguous")
        require(set(self.label) <= {"H", "A", "D", "X"}, "invalid prediction label")
        if self.confidence is not None:
            confidence = array(self.confidence, np.float64, (n,), "confidence")
            abstain = self.label == "X"
            require(np.isnan(confidence[abstain]).all(), "X confidence must be NaN")
            require(np.isfinite(confidence[~abstain]).all() and np.all((confidence[~abstain] >= 0) & (confidence[~abstain] <= 1)), "invalid confidence")
            object.__setattr__(self, "confidence", confidence)
        if self.reason is not None:
            reason = strings(self.reason, (n,), "reason")
            require(set(reason) <= {"nearest_player", "baseline_missing", "synthetic_stub"}, "unknown reason code")
            object.__setattr__(self, "reason", reason)

    def validate_for(self, match: MatchInput) -> None:
        require((self.schema_version, self.match_id, self.source) == (match.schema_version, match.match_id, match.source), "prediction/input metadata mismatch")
        require(np.array_equal(self.frame, match.frame), "prediction/input frames differ")
