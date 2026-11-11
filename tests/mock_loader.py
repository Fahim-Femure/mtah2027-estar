"""TEST-ONLY stand-in for Data Engineering's future real-data loader.

It implements the seam `mtah.pipeline.prepare_match(input_root: Path) -> MatchInput`
with an invented, deterministic match. It never reads `input_root` (not even an
existence check), never touches the filesystem, and uses a `SYNTH-` match id so
its output can never pass as the competition match.

This module lives under tests/ and is installed only inside `installed_mock_loader()`.
Production code must never import it; tests/test_integration_harness.py enforces that.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
import types

import numpy as np

from mtah.contracts import Event, MatchInput

MOCK_MATCH_ID = "SYNTH-MOCK-ATD"
N_FRAMES = 199319                                   # documented tactical frame count
SEGMENTS = ((11185, 90652, 1), (110808, 190198, 2))  # published half blocks (DATA.md, games.py)
BLOCK = 500          # invented possession block length (frames)
OUT_FRAMES = 100     # every 4th block opens with the ball out of play
GAP_FRAMES = 50      # then unobserved (retrieval) before a tagged restart
L, W = 105.3, 68.0
HOME_XY, AWAY_XY = np.array([0.30, 0.50]), np.array([0.70, 0.50])
OFFSET = np.array([0.5 / L, 0.0])                   # ball 0.5 m ahead of the carrier


def block_layout(n_frames=N_FRAMES, segments=SEGMENTS):
    """(team, phase, out mask, gap mask, restarts). Pure and hand-checkable: inside a
    segment, block b = (t - start) // BLOCK and phase = (t - start) % BLOCK; even
    blocks home, odd away; blocks with b % 4 == 3 start with OUT_FRAMES out, then
    GAP_FRAMES missing, then a SET PIECES restart by that block's team."""
    team = np.full(n_frames, "", dtype="<U1")
    phase = np.zeros(n_frames, dtype=np.int64)
    out = np.zeros(n_frames, dtype=bool)
    gap = np.zeros(n_frames, dtype=bool)
    restarts = []
    for start, stop, _ in segments:
        for b, a in enumerate(range(start, stop, BLOCK)):
            z = min(a + BLOCK, stop)
            team[a:z] = "H" if b % 2 == 0 else "A"
            phase[a:z] = np.arange(z - a)
            if b % 4 == 3 and a + OUT_FRAMES + GAP_FRAMES < z:
                out[a:a + OUT_FRAMES] = True
                gap[a + OUT_FRAMES:a + OUT_FRAMES + GAP_FRAMES] = True
                restarts.append((a + OUT_FRAMES + GAP_FRAMES, team[a]))
    return team, phase, out, gap, restarts


def build_mock_match(n_frames=N_FRAMES, segments=SEGMENTS, *, frame=None, match_id=MOCK_MATCH_ID) -> MatchInput:
    team, phase, out, gap, restarts = block_layout(n_frames, segments)
    t = np.arange(n_frames)
    # The carrier runs along +x at 1.5 m/s within its block, so the ball genuinely moves.
    drift = (phase * 1.5 / 25 / L)[:, None] * np.array([1.0, 0.0])
    players = np.empty((n_frames, 4, 2))
    players[:, 0] = HOME_XY + np.where((team == "H")[:, None], drift, 0)
    players[:, 1] = HOME_XY + [0, 0.3]
    players[:, 2] = AWAY_XY + np.where((team == "A")[:, None], drift, 0)
    players[:, 3] = AWAY_XY + [0, 0.3]
    ball = np.full((n_frames, 2), np.nan)
    home, away = team == "H", team == "A"
    ball[home] = players[home, 0] + OFFSET
    ball[away] = players[away, 2] + OFFSET
    ball[out] = (0.5, 1.05)                         # 3.4 m beyond the touchline
    ball[gap] = np.nan
    ball[(t % 7 == 3) & ~out] = np.nan              # invented occlusions
    events = tuple(Event(f"mock-{i}", r, r + 10, "SET PIECES", side, {"Type": "Throw-in"})
                   for i, (r, side) in enumerate(restarts))
    half = np.zeros(n_frames, dtype=np.int8)
    for a, b, h in segments:
        half[a:b] = h
    return MatchInput(
        schema_version=1, match_id=match_id, source="tactical", n_frames=n_frames, fps=25,
        pitch_length_m=L, pitch_width_m=W, coordinate_system="pitch_normalized_0_1",
        team_names={"H": "Mock Home", "A": "Mock Away"}, segments=tuple(segments),
        frame=np.arange(n_frames, dtype=np.int64) if frame is None else frame, half=half,
        slot_id=np.array(["Track 1", "Track 2", "Track 3", "Track 4"]),
        slot_team=np.array(["H", "H", "A", "A"]), player_xy=players,
        player_observed=np.ones((n_frames, 4), dtype=bool), player_speed_mps=np.full((n_frames, 4), np.nan),
        ball_xy=ball, ball_observed=np.isfinite(ball).all(axis=1), ball_speed_mps=np.full(n_frames, np.nan),
        events=events,
    )


class MockLoader:
    """Records calls; returns `build()` output. Never opens `input_root`."""

    def __init__(self, build=build_mock_match):
        self.build = build
        self.calls = []

    def prepare_match(self, input_root):
        if not isinstance(input_root, Path):
            raise TypeError("prepare_match expects a pathlib.Path")
        self.calls.append(input_root)
        return self.build()


@contextmanager
def installed_mock_loader(loader=None):
    """Install a mock `mtah.pipeline` for the duration of the block only."""
    loader = MockLoader() if loader is None else loader
    module = types.ModuleType("mtah.pipeline")
    module.prepare_match = loader.prepare_match
    module.__mock__ = True
    previous = sys.modules.get("mtah.pipeline")
    sys.modules["mtah.pipeline"] = module
    try:
        yield loader
    finally:
        if previous is None:
            sys.modules.pop("mtah.pipeline", None)
        else:
            sys.modules["mtah.pipeline"] = previous
