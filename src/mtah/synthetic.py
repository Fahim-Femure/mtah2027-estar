"""Invented fixture, not an extract of competition data."""
import numpy as np
from .contracts import Event, MatchInput


def make_match(n_frames: int = 12) -> MatchInput:
    if not isinstance(n_frames, int) or isinstance(n_frames, bool) or n_frames < 4:
        raise ValueError("synthetic fixture needs at least four frames")
    players = np.tile(np.array([[[0.2, 0.5], [0.8, 0.5]]]), (n_frames, 1, 1))
    ball = np.tile([0.21, 0.5], (n_frames, 1))
    ball[n_frames // 2:] = [0.79, 0.5]
    ball[1] = np.nan
    return MatchInput(
        schema_version=1, match_id="SYNTH-FOUNDATION", source="tactical", n_frames=n_frames,
        fps=25, pitch_length_m=105.3, pitch_width_m=68.0,
        coordinate_system="pitch_normalized_0_1", team_names={"H": "Synthetic Home", "A": "Synthetic Away"},
        segments=((0, n_frames, 1),), frame=np.arange(n_frames, dtype=np.int64),
        half=np.ones(n_frames, dtype=np.int8), slot_id=np.array(["slot-1", "slot-2"]),
        slot_team=np.array(["H", "A"]), player_xy=players,
        player_observed=np.ones((n_frames, 2), dtype=bool), player_speed_mps=np.full((n_frames, 2), np.nan),
        ball_xy=ball, ball_observed=np.isfinite(ball).all(axis=1), ball_speed_mps=np.full(n_frames, np.nan),
        events=(Event("synthetic-1", 0, 2, "SYNTHETIC", None, {}),),
    )
