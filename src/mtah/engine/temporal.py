"""Deterministic temporal possession reasoning over spatial and dead-ball evidence.

Rules (CHALLENGE.md, "Possession rules"):
  * possession changes only when the other team gains *controlled* possession:
    a team-level control run with at least `confirm_frames` evidence frames;
    shorter runs are touches/deflections and change nothing;
  * while the ball is loose, contested, in flight or unobserved, possession
    stays with the last team that had it;
  * the change is placed at the onset of the confirming run (offline batch
    lookahead), not at the frame the confirmation threshold is reached;
  * a dead ball is D and ends the possession; the live stretch after it starts
    with the first team to establish control (the restart possessor), else a
    set-piece team prior, else X;
  * state resets at every published segment boundary; frames outside segments
    are X (ungraded per DATA.md).
"""
from dataclasses import dataclass

import numpy as np

from ..contracts import MatchInput
from .config import EngineConfig
from .deadball import (DEAD_HOLD, DEAD_OUT, DEAD_SET_PIECE, DEAD_STATIONARY, DEAD_UNCONFIRMED, LIVE, runs_of)
from .evidence import AWAY, HOME, WHY_BALL_MISSING, SpatialEvidence

SIDE = {HOME: "H", AWAY: "A"}

# Engine-internal reason vocabulary (contract v1 cannot carry it; see docs).
REASONS = (
    "outside_segment", "dead_ball_out", "dead_hold", "dead_set_piece_lead_in", "dead_stationary_ball",
    "abstain_unconfirmed_restart",
    "controlled", "carried", "carried_ball_missing", "backfilled_restart",
    "backfilled_unconfirmed_restart", "set_piece_team_prior",
    "abstain_no_evidence",
)
DEAD_REASON = {DEAD_OUT: "dead_ball_out", DEAD_HOLD: "dead_hold", DEAD_SET_PIECE: "dead_set_piece_lead_in",
               DEAD_STATIONARY: "dead_stationary_ball"}
DEAD_CONFIDENCE = {DEAD_OUT: 0.9, DEAD_HOLD: 0.7, DEAD_SET_PIECE: 0.6, DEAD_STATIONARY: 0.6}


@dataclass(frozen=True)
class ControlRun:
    side: int      # HOME or AWAY
    onset: int     # first evidence frame
    last: int      # last evidence frame
    count: int     # evidence frames

    def confirmed(self, config: EngineConfig) -> bool:
        return self.count >= config.confirm_frames


def control_runs(control: np.ndarray, start: int, stop: int, config: EngineConfig):
    """Team-level runs inside one live stretch [start, stop).

    A run continues across frames without evidence for up to run_gap_frames and
    ends at the first opposing-team evidence frame.
    """
    runs, side, onset, last, count = [], None, 0, 0, 0
    for t in range(start, stop):
        c = int(control[t])
        if c not in (HOME, AWAY):
            continue
        if side == c and t - last <= config.run_gap_frames:
            last, count = t, count + 1
            continue
        if side is not None:
            runs.append(ControlRun(side, onset, last, count))
        side, onset, last, count = c, t, t, 1
    if side is not None:
        runs.append(ControlRun(side, onset, last, count))
    return runs


def restart_prior(restarts, a: int, config: EngineConfig, active_kinds=None):
    """Team tag of the set piece whose restart opens the live stretch at `a`.

    The opening restart is the one nearest `a` within the window (ties prefer
    confirmed restarts over unconfirmed cluster ends, then the earlier restart,
    then event order). Its tag decides: a nearest restart tagged N/A (e.g. a
    throw-in) gives no prior, rather than falling back to an older tagged
    restart.
    """
    if active_kinds is None:
        near = [(abs(restart - a), 0, restart, i) for i, (restart, _t, _s, _e) in enumerate(restarts)
                if abs(restart - a) <= config.set_piece_window_frames]
    else:
        near = [(abs(restart - a), 1 if kind == "unconfirmed" else 0, restart, i)
                for i, ((restart, _t, _s, _e), kind) in enumerate(zip(restarts, active_kinds))
                if abs(restart - a) <= config.set_piece_window_frames]
    if not near:
        return None
    team = restarts[min(near)[3]][1]
    return team if team in ("H", "A") else None


def decide(match: MatchInput, ev: SpatialEvidence, dead: np.ndarray, restarts, config: EngineConfig,
           restart_kinds=()):
    n = match.n_frames
    label = np.full(n, "X", dtype="<U1")
    confidence = np.full(n, np.nan)
    reason = np.full(n, "outside_segment", dtype=f"<U{max(map(len, REASONS))}")

    for code, name in DEAD_REASON.items():
        hit = dead == code
        label[hit], confidence[hit], reason[hit] = "D", DEAD_CONFIDENCE[code], name
    # Unconfirmed set-piece windows remain X with NaN confidence (Option A).
    reason[dead == DEAD_UNCONFIRMED] = "abstain_unconfirmed_restart"

    active_kinds = [k for k in restart_kinds if k != "ignored"] if restart_kinds else None
    confirmed_restarts = (
        {r[0] for r, k in zip(restarts, active_kinds) if k != "unconfirmed"}
        if active_kinds is not None else set()
    )

    for seg_start, seg_stop, _half in match.segments:
        live = dead[seg_start:seg_stop] == LIVE
        for a, b in runs_of(live):
            a, b = a + seg_start, b + seg_start
            confirmed = [r for r in control_runs(ev.control, a, b, config) if r.confirmed(config)]
            if not confirmed:
                team = restart_prior(restarts, a, config, active_kinds) if config.use_set_piece_team_prior else None
                if team is None:
                    reason[a:b] = "abstain_no_evidence"
                else:
                    label[a:b], confidence[a:b], reason[a:b] = team, 0.4, "set_piece_team_prior"
                continue
            # Back-fill the restart possessor up to the first confirmed control.
            first = confirmed[0]
            label[a:first.onset] = SIDE[first.side]
            after_unconfirmed = (
                a > seg_start and dead[a - 1] == DEAD_UNCONFIRMED and a not in confirmed_restarts
            )
            if after_unconfirmed:
                confidence[a:first.onset], reason[a:first.onset] = 0.4, "backfilled_unconfirmed_restart"
            else:
                confidence[a:first.onset], reason[a:first.onset] = 0.5, "backfilled_restart"
            for i, run in enumerate(confirmed):
                end = confirmed[i + 1].onset if i + 1 < len(confirmed) else b
                side = SIDE[run.side]
                label[run.onset:end] = side
                last_support = run.onset
                for t in range(run.onset, end):
                    if ev.control[t] == run.side:
                        last_support = t
                        confidence[t], reason[t] = 0.9, "controlled"
                    else:
                        gap = t - last_support
                        confidence[t] = 0.35 + 0.55 * 0.5 ** (gap / 50.0)
                        reason[t] = "carried_ball_missing" if ev.why[t] == WHY_BALL_MISSING else "carried"
    return label, confidence, reason
