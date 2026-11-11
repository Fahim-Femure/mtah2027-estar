"""Per-frame spatial possession evidence. Pure functions of MatchInput.

Missing observations stay missing: the ball is never interpolated, an unobserved
player never gets a position, and a missing ball is never evidence of a dead
ball. Distances use metres (normalized * pitch size), team-level minima only,
so slot identity switches and duplicated tracks do not matter.
"""
from dataclasses import dataclass

import numpy as np

from ..contracts import MatchInput
from .config import EngineConfig

# control codes
NONE, HOME, AWAY, CONTESTED = 0, 1, 2, 3
# why a frame has no single-team control (NONE/CONTESTED detail)
WHY_CONTROL, WHY_BALL_MISSING, WHY_NO_PLAYERS, WHY_IN_FLIGHT, WHY_LOOSE, WHY_CONTESTED = 0, 1, 2, 3, 4, 5


@dataclass(frozen=True)
class SpatialEvidence:
    ball_xy_m: np.ndarray      # (N,2) metres, NaN where unobserved (never filled)
    ball_speed_mps: np.ndarray  # (N,) source speed, else position-derived, else NaN
    d_home_m: np.ndarray       # (N,) nearest observed home player to ball, inf if unknown
    d_away_m: np.ndarray       # (N,) nearest observed away player to ball, inf if unknown
    control: np.ndarray        # (N,) int8 NONE/HOME/AWAY/CONTESTED
    why: np.ndarray            # (N,) int8 WHY_* detail
    # Dead-ball motion evidence: the larger of source and position-derived speed
    # (either independent channel showing motion counts), NaN if neither exists.
    # A stuck or zero source channel therefore cannot hide observed movement.
    motion_speed_mps: np.ndarray


def lag_speed(xy: np.ndarray, obs: np.ndarray, half: np.ndarray, fps: int, lag: int) -> np.ndarray:
    """speed[t] = |xy[t] - xy[t-lag]| * fps / lag where both ends are observed in
    the same half, else NaN. Never bridges a frame it cannot see."""
    n = len(obs)
    speed = np.full(n, np.nan)
    if lag >= n:
        return speed
    ok = obs[lag:] & obs[:-lag] & (half[lag:] == half[:-lag])
    idx = np.flatnonzero(ok) + lag
    speed[idx] = np.linalg.norm(xy[idx] - xy[idx - lag], axis=1) * fps / lag
    return speed


def derived_ball_speed(match: MatchInput, config: EngineConfig) -> np.ndarray:
    """Position-derived speed, NaN where no observed lag supports one.

    The value is the smaller of the backward and forward lag speeds, so the frame
    a ball is received (fast before, slow after) and the frame it is struck (slow
    before, fast after) both read as slow, while a ball in flight reads fast in
    both directions. Lag `speed_lag_frames` damps jitter; lag 1 is the fallback.

    Symmetric (S2) rule: a finite speed requires bilateral support. When exactly one
    direction is finite because the other endpoint is unobserved inside the same half
    (a tracking gap / reacquisition edge), the estimate is suppressed to NaN rather
    than trusted one-sided — isolated ``np.fmin`` ignoring NaN let a ~4 m / 40 ms
    re-lock read as >100 m/s and fire false ``IN_FLIGHT`` evidence. A one-sided
    estimate is kept only at timeline/segment edges (counterpart out of range or in
    another half), where no bilateral estimate can exist by construction. NaN is
    never treated as stationary: downstream checks require finite speed.
    """
    scale = np.array([match.pitch_length_m, match.pitch_width_m])
    xy = match.ball_xy * scale
    half = match.half
    n = match.n_frames
    derived = np.full(n, np.nan)
    for lag in (config.speed_lag_frames, 1):
        back = lag_speed(xy, match.ball_observed, half, match.fps, lag)
        fwd = np.full(n, np.nan)
        fwd[:-lag] = back[lag:]
        back_ok = np.isfinite(back)
        fwd_ok = np.isfinite(fwd)
        combined = np.full(n, np.nan)
        both_ok = back_ok & fwd_ok
        with np.errstate(invalid="ignore"):
            combined[both_ok] = np.fmin(back, fwd)[both_ok]
        # Edge missing: counterpart index out of range or across a half/segment
        # boundary. Any other missing counterpart lies in the same half but is
        # unobserved -> insufficient evidence -> NaN (no one-sided value).
        fwd_edge = np.zeros(n, dtype=bool)
        fwd_edge[:-lag] = half[:-lag] != half[lag:]
        fwd_edge[-lag:] = True
        back_edge = np.zeros(n, dtype=bool)
        back_edge[lag:] = half[lag:] != half[:-lag]
        back_edge[:lag] = True
        only_back = back_ok & ~fwd_ok & fwd_edge
        combined[only_back] = back[only_back]
        only_fwd = fwd_ok & ~back_ok & back_edge
        combined[only_fwd] = fwd[only_fwd]
        derived = np.where(np.isnan(derived), combined, derived)

    # Bilateral fallback (P2-B.2). The fixed lags above miss frames beside an
    # isolated dropout: with the ball missing every third frame, no frame has an
    # observed partner at exactly lag 1 or lag 5 on both sides, so every speed
    # became NaN. A frame that has an observed frame on BOTH sides within
    # speed_lag_frames (same half) still has bilateral support and takes the
    # smaller of the two speeds. Each side uses its FARTHEST observed frame inside
    # the window, so jitter is damped as close to lag speed_lag_frames as the
    # observations allow. A frame next to a longer gap (no observation within
    # speed_lag_frames on one side) stays NaN, exactly as S2 requires.
    k = config.speed_lag_frames
    obs = match.ball_observed
    idx = np.arange(n)
    last_at_or_before = np.maximum.accumulate(np.where(obs, idx, -1))
    first_at_or_after = np.minimum.accumulate(np.where(obs, idx, n)[::-1])[::-1]
    back_i = first_at_or_after[np.clip(idx - k, 0, n - 1)]      # earliest observed in [t-k, t]
    fwd_i = last_at_or_before[np.clip(idx + k, 0, n - 1)]       # latest observed in [t, t+k]
    b, f = np.clip(back_i, 0, n - 1), np.clip(fwd_i, 0, n - 1)
    fill = (obs & np.isnan(derived) & (back_i < idx) & (fwd_i > idx)
            & (half[b] == half) & (half[f] == half))
    t = np.flatnonzero(fill)
    if t.size:
        back = np.linalg.norm(xy[t] - xy[back_i[t]], axis=1) * match.fps / (t - back_i[t])
        fwd = np.linalg.norm(xy[fwd_i[t]] - xy[t], axis=1) * match.fps / (fwd_i[t] - t)
        derived[t] = np.fmin(back, fwd)
    return derived


def ball_speed(match: MatchInput, config: EngineConfig) -> np.ndarray:
    """Source speed where finite; otherwise the position-derived speed; else NaN."""
    provided = match.ball_speed_mps
    return np.where(np.isfinite(provided), provided, derived_ball_speed(match, config))


def nearest_team_distance(match: MatchInput, team: str, scale: np.ndarray) -> np.ndarray:
    cols = np.flatnonzero(match.slot_team == team)
    if cols.size == 0:
        return np.full(match.n_frames, np.inf)
    delta = (match.player_xy[:, cols, :] - match.ball_xy[:, None, :]) * scale
    dist = np.sqrt(np.sum(delta * delta, axis=2))
    usable = match.player_observed[:, cols] & match.ball_observed[:, None]
    return np.where(usable, dist, np.inf).min(axis=1)


def spatial_evidence(match: MatchInput, config: EngineConfig) -> SpatialEvidence:
    scale = np.array([match.pitch_length_m, match.pitch_width_m])
    derived = derived_ball_speed(match, config)
    provided = match.ball_speed_mps
    speed = np.where(np.isfinite(provided), provided, derived)
    motion = np.fmax(np.where(match.ball_observed, provided, np.nan), derived)
    d_home = nearest_team_distance(match, "H", scale)
    d_away = nearest_team_distance(match, "A", scale)
    nearest = np.minimum(d_home, d_away)
    radius = config.control_radius_m

    control = np.full(match.n_frames, NONE, dtype=np.int8)
    why = np.full(match.n_frames, WHY_LOOSE, dtype=np.int8)
    in_flight = np.isfinite(speed) & (speed > config.max_control_ball_speed_mps)
    both_close = (d_home <= radius) & (d_away <= radius)   # both finite, so the difference is defined
    with np.errstate(invalid="ignore"):          # inf - inf on frames masked out below
        gap = np.abs(np.where(both_close, d_home - d_away, np.inf))
    contested = both_close & (gap < config.contest_margin_m)
    close = nearest <= radius

    home = close & ~contested & (d_home < d_away)
    away = close & ~contested & (d_away < d_home)
    tie = close & ~contested & (d_home == d_away)  # exact equality: undecidable, treat as contested
    control[home], control[away] = HOME, AWAY
    control[contested | tie] = CONTESTED
    why[home | away] = WHY_CONTROL
    why[contested | tie] = WHY_CONTESTED

    # Overrides, strongest last: travelling ball, then nobody visible, then no ball.
    for mask, code in ((in_flight, WHY_IN_FLIGHT),
                       (match.ball_observed & ~match.player_observed.any(axis=1), WHY_NO_PLAYERS),
                       (~match.ball_observed, WHY_BALL_MISSING)):
        control[mask] = NONE
        why[mask] = code

    ball_xy_m = match.ball_xy * scale
    for arr in (ball_xy_m, speed, d_home, d_away, control, why, motion):
        arr.flags.writeable = False
    return SpatialEvidence(ball_xy_m, speed, d_home, d_away, control, why, motion)
