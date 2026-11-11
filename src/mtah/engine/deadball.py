"""Dead-ball inference from documented features only.

Evidence used (all described in DATA.md / CHALLENGE.md):
  * ball observed beyond a touch or goal line (finite out-of-pitch coordinates
    are preserved by the contract);
  * a stoppage persists until the ball is observed back inside the pitch and
    moving (a placed ball for a corner, goal kick or free kick is still dead).
    Motion is shown by either the source or the position-derived speed; with
    sparse observations (no per-frame speed), by displacement between in-pitch
    observations at least `speed_lag_frames` apart within the same stoppage;
  * Smart Tagging set-piece clusters establish restarts only with movement
    evidence in [S - window, E); without evidence while a stoppage is in
    progress, the window abstains as DEAD_UNCONFIRMED (X) and the hold ends
    at E (docs/ENGINE_DECISION_TAGGING_NO_ONSET.md);
  * a stationary ball with nobody near it for several seconds is awaiting a
    restart.

A missing ball on its own is never dead-ball evidence. Dead state is only
*held* through missing frames after positive evidence of a stoppage.
"""
from bisect import bisect_right

import numpy as np

from ..contracts import MatchInput
from .config import EngineConfig
from .evidence import AWAY, HOME, SpatialEvidence

LIVE, DEAD_OUT, DEAD_HOLD, DEAD_SET_PIECE, DEAD_STATIONARY, DEAD_UNCONFIRMED = 0, 1, 2, 3, 4, 5


def segment_bounds(match: MatchInput):
    return [(start, stop) for start, stop, _ in match.segments]


def runs_of(mask: np.ndarray):
    """Half-open (start, stop) runs of True in a 1-D bool array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def out_of_pitch(match: MatchInput, ev: SpatialEvidence, config: EngineConfig) -> np.ndarray:
    x, y = ev.ball_xy_m[:, 0], ev.ball_xy_m[:, 1]
    m = config.out_margin_m
    with np.errstate(invalid="ignore"):
        out = (x < -m) | (x > match.pitch_length_m + m) | (y < -m) | (y > match.pitch_width_m + m)
    return match.ball_observed & out


def cluster_set_pieces(events, set_piece_codes: tuple, seg_start: int, seg_stop: int):
    """Group overlapping set-piece events in one segment into [(S, E, team)].

    Events are assumed sorted by (start_frame, end_frame, event_id). Overlaps
    merge transitively; abutting events (e0 == s0') remain separate clusters.
    """
    clusters = []
    cur_s = cur_e = None
    cur_teams = set()
    for event in events:
        if event.code not in set_piece_codes:
            continue
        s0 = event.start_frame
        if not (seg_start <= s0 < seg_stop):
            continue
        e0 = min(event.end_frame, seg_stop)
        if cur_s is None:
            cur_s, cur_e = s0, e0
            cur_teams = {event.team} if event.team in ("H", "A") else set()
        elif s0 < cur_e:
            cur_e = max(cur_e, e0)
            if event.team in ("H", "A"):
                cur_teams.add(event.team)
        else:
            team = next(iter(cur_teams)) if len(cur_teams) == 1 else None
            clusters.append((cur_s, cur_e, team))
            cur_s, cur_e = s0, e0
            cur_teams = {event.team} if event.team in ("H", "A") else set()
    if cur_s is not None:
        team = next(iter(cur_teams)) if len(cur_teams) == 1 else None
        clusters.append((cur_s, cur_e, team))
    return clusters


def _superseded_rest_start(rest: np.ndarray, onset_frame: int, cluster_end: int, min_dead_frames: int):
    """First frame of >= min_dead_frames consecutive rest in (onset_frame, cluster_end), or None."""
    for ra, rb in runs_of(rest[onset_frame + 1:cluster_end]):
        if rb - ra >= min_dead_frames:
            return onset_frame + 1 + ra
    return None


def _sparse_moved(t: int, ball_observed: np.ndarray, speed: np.ndarray, xy: np.ndarray,
                  seen: list, lag: int, fps: int, restart_speed_mps: float) -> bool:
    if not (ball_observed[t] and not np.isfinite(speed[t])):
        return False
    k = bisect_right(seen, t - lag)
    if not k:
        return False
    u = seen[k - 1]
    return bool(np.hypot(*(xy[t] - xy[u])) * fps / (t - u) >= restart_speed_mps)


def _advance_hold(start_t: int, stop_t: int, dead: bool, seen: list, reason: np.ndarray,
                  out: np.ndarray, ball_observed: np.ndarray, speed: np.ndarray, xy: np.ndarray,
                  moving_inside: np.ndarray, superseded_motion: np.ndarray, restart_flag: np.ndarray,
                  lag: int, fps: int, restart_speed_mps: float, on_moved=None, write: bool = True):
    for t in range(start_t, stop_t):
        if out[t]:
            dead = True
            if write:
                reason[t] = DEAD_OUT
            seen = []
        elif dead:
            moved = False
            if not superseded_motion[t] and _sparse_moved(
                t, ball_observed, speed, xy, seen, lag, fps, restart_speed_mps
            ):
                moved = True if on_moved is None else not on_moved(t)
            if ball_observed[t]:
                seen.append(t)
            if (moving_inside[t] and not superseded_motion[t]) or moved or restart_flag[t]:
                dead = False
                seen = []
            elif write:
                reason[t] = DEAD_HOLD
    return dead, seen


def _lead_in_length(anchor: int, seg_start: int, max_lead: int,
                    base_live: np.ndarray, superseded_motion: np.ndarray) -> int:
    t = anchor - 1
    steps = 0
    while t >= seg_start and anchor - t <= max_lead and not (base_live[t] and not superseded_motion[t]):
        steps += 1
        t -= 1
    return steps


def _apply_lead_in(anchor: int, seg_start: int, max_lead: int,
                   base_live: np.ndarray, superseded_motion: np.ndarray, reason: np.ndarray) -> None:
    steps = _lead_in_length(anchor, seg_start, max_lead, base_live, superseded_motion)
    if steps:
        block = reason[anchor - steps:anchor]
        block[block == LIVE] = DEAD_SET_PIECE


def _mark_superseded(superseded_motion: np.ndarray, added: list, start_t: int, stop_t: int) -> None:
    for k in range(start_t, stop_t):
        if not superseded_motion[k]:
            superseded_motion[k] = True
            added.append(k)


def infer_dead(match: MatchInput, ev: SpatialEvidence, config: EngineConfig):
    """Return (dead_reason int8 (N,), restarts list, restart_kinds list).
    LIVE outside segments too; the temporal layer labels those frames X."""
    n = match.n_frames
    out = out_of_pitch(match, ev, config)
    speed = ev.motion_speed_mps   # either channel showing motion counts (see SpatialEvidence)
    with np.errstate(invalid="ignore"):
        moving_inside = match.ball_observed & ~out & np.isfinite(speed) & (speed >= config.restart_speed_mps)
        rest = match.ball_observed & np.isfinite(speed) & (speed < config.stationary_speed_mps)
        controlled_play = (match.ball_observed & ~out & np.isin(ev.control, (HOME, AWAY))
                           & np.isfinite(speed) & (speed >= config.stationary_speed_mps))

    onset = moving_inside.copy()
    onset[1:] &= ~moving_inside[:-1]
    base_live = moving_inside | controlled_play
    superseded_motion = np.zeros(n, dtype=bool)
    restart_flag = np.zeros(n, dtype=bool)
    reason = np.full(n, LIVE, dtype=np.int8)

    xy = ev.ball_xy_m
    lag = config.speed_lag_frames
    w = config.set_piece_window_frames
    restarts = []
    restart_kinds = []

    for seg_start, seg_stop in segment_bounds(match):
        onset[seg_start] = False  # a segment's first frame is not an observed onset
        clusters = cluster_set_pieces(match.events, config.set_piece_codes, seg_start, seg_stop)
        dead = False
        seen = []   # in-pitch observations since the stoppage began (or since the last out frame)
        cursor = seg_start
        seg_moved = []

        def record_moved(t: int) -> bool:
            seg_moved.append(t)
            return False

        for S, E, team in clusters:
            lo = max(seg_start, S - w)
            added_superseded = []

            # Classify moving_inside onsets in [lo, E) as superseded or candidate.
            valid_mi = []
            for h in (lo + np.flatnonzero(onset[lo:E])).tolist():
                if h <= seg_start:
                    continue
                rest_start = _superseded_rest_start(rest, h, E, config.min_dead_frames)
                if rest_start is not None:
                    _mark_superseded(superseded_motion, added_superseded, max(h, cursor), rest_start)
                else:
                    valid_mi.append(h)

            moved_cands = [
                t for t in seg_moved
                if lo <= t < cursor and _superseded_rest_start(rest, t, E, config.min_dead_frames) is None
            ]

            def on_moved_before(t: int) -> bool:
                if t >= lo:
                    rest_start = _superseded_rest_start(rest, t, E, config.min_dead_frames)
                    if rest_start is not None:
                        _mark_superseded(superseded_motion, added_superseded, max(t, cursor), rest_start)
                        return True
                    moved_cands.append(t)
                seg_moved.append(t)
                return False

            dead, seen = _advance_hold(
                cursor, S, dead, seen, reason, out, match.ball_observed, speed, xy,
                moving_inside, superseded_motion, restart_flag, lag, match.fps, config.restart_speed_mps,
                on_moved=on_moved_before, write=True,
            )

            stoppage_at_S = (
                (S > seg_start and reason[S - 1] in (DEAD_OUT, DEAD_HOLD))
                or _lead_in_length(S, seg_start, config.set_piece_max_lead_frames, base_live, superseded_motion)
                >= config.min_dead_frames
            )

            def on_moved_inside(t: int) -> bool:
                rest_start = _superseded_rest_start(rest, t, E, config.min_dead_frames)
                if rest_start is not None:
                    _mark_superseded(superseded_motion, added_superseded, max(t, cursor), rest_start)
                    return True
                moved_cands.append(t)
                return False

            _advance_hold(
                S, E, dead, list(seen), reason, out, match.ball_observed, speed, xy,
                moving_inside, superseded_motion, restart_flag, lag, match.fps, config.restart_speed_mps,
                on_moved=on_moved_inside, write=False,
            )

            cands = sorted(set(valid_mi) | set(moved_cands))
            early = [t for t in cands if S <= t <= S + w]
            before = [t for t in cands if lo <= t < S]
            late = [t for t in cands if S + w < t < E]

            if early:
                kind, restart = "early", min(early)
            elif before:
                kind, restart = "before", max(before)
            elif late and stoppage_at_S:
                kind, restart = "late", min(late)
            elif stoppage_at_S:
                kind, restart = "unconfirmed", E
            else:
                kind, restart = "ignored", None

            if kind == "ignored":
                for k in added_superseded:
                    superseded_motion[k] = False
                dead, seen = _advance_hold(
                    S, E, dead, seen, reason, out, match.ball_observed, speed, xy,
                    moving_inside, superseded_motion, restart_flag, lag, match.fps, config.restart_speed_mps,
                    on_moved=record_moved, write=True,
                )
            elif kind == "unconfirmed":
                reason[S:E] = DEAD_UNCONFIRMED
                dead, seen = False, []
                _apply_lead_in(S, seg_start, config.set_piece_max_lead_frames,
                               base_live, superseded_motion, reason)
                restarts.append((E, team, seg_start, seg_stop))
            else:
                restart_flag[restart] = True
                dead, seen = _advance_hold(
                    S, E, dead, seen, reason, out, match.ball_observed, speed, xy,
                    moving_inside, superseded_motion, restart_flag, lag, match.fps, config.restart_speed_mps,
                    on_moved=record_moved, write=True,
                )
                _apply_lead_in(restart, seg_start, config.set_piece_max_lead_frames,
                               base_live, superseded_motion, reason)
                restarts.append((restart, team, seg_start, seg_stop))

            restart_kinds.append(kind)
            cursor = E

        dead, seen = _advance_hold(
            cursor, seg_stop, dead, seen, reason, out, match.ball_observed, speed, xy,
            moving_inside, superseded_motion, restart_flag, lag, match.fps, config.restart_speed_mps,
            on_moved=record_moved, write=True,
        )

    # Placed ball: stationary, observed, nobody within control radius.
    with np.errstate(invalid="ignore"):
        still = (match.ball_observed & np.isfinite(speed) & (speed < config.stationary_speed_mps)
                 & (np.minimum(ev.d_home_m, ev.d_away_m) > config.control_radius_m))
    for start, stop in segment_bounds(match):
        for a, b in runs_of(still[start:stop]):
            if b - a >= config.stationary_frames:
                block = reason[start + a:start + b]
                block[block == LIVE] = DEAD_STATIONARY

    # Drop dead runs too short to be a real stoppage (tracking noise at a line).
    for start, stop in segment_bounds(match):
        for a, b in runs_of(reason[start:stop] != LIVE):
            if b - a < config.min_dead_frames:
                reason[start + a:start + b] = LIVE
    reason.flags.writeable = False
    return reason, restarts, restart_kinds
