"""Programmatic invented football scenarios for engine tests.

Nothing here is copied, sampled or derived from competition data or labels.
Positions are normalized pitch coordinates; helpers speak metres where useful.
"""
import numpy as np

from mtah.contracts import Event, MatchInput

L, W = 105.3, 68.0
FPS = 25
# Default slot layout: two home, two away, far apart and far from the centre spot.
HOME_START = [(0.30, 0.30), (0.30, 0.70)]
AWAY_START = [(0.70, 0.30), (0.70, 0.70)]


class Scenario:
    def __init__(self, n, segments=None, extra_home=0, extra_away=0):
        self.n = n
        starts = HOME_START + [(0.2, 0.5)] * extra_home + AWAY_START + [(0.8, 0.5)] * extra_away
        self.teams = ["H"] * (2 + extra_home) + ["A"] * (2 + extra_away)
        self.players = np.tile(np.array(starts, dtype=float)[None], (n, 1, 1))
        self.ball = np.tile(np.array([0.5, 0.5]), (n, 1))  # centre spot, nobody near
        self.ball_speed = np.full(n, np.nan)
        self.events = []
        self.segments = segments if segments is not None else ((0, n, 1),)

    def slot(self, team, index):
        return [i for i, t in enumerate(self.teams) if t == team][index]

    # --- tracking helpers -------------------------------------------------
    def dribble(self, start, stop, team, index, speed_mps=3.0, offset_m=0.5):
        """Player runs along +x at speed_mps with the ball offset_m ahead."""
        s = self.slot(team, index)
        origin = self.players[start, s].copy()
        steps = np.arange(stop - start)[:, None]
        pos = origin + steps * np.array([speed_mps / FPS / L, 0.0])
        self.players[start:stop, s] = pos
        self.players[stop:, s] = pos[-1]
        self.ball[start:stop] = pos + np.array([offset_m / L, 0.0])

    def ball_path(self, start, stop, a, b):
        t = np.linspace(0.0, 1.0, stop - start, endpoint=False)[:, None]
        self.ball[start:stop] = np.array(a) + t * (np.array(b) - np.array(a))

    def ball_still(self, start, stop, xy):
        self.ball[start:stop] = xy

    def ball_missing(self, start, stop):
        self.ball[start:stop] = np.nan

    def player_at(self, start, stop, team, index, xy):
        self.players[start:stop, self.slot(team, index)] = xy

    def player_missing(self, start, stop, team, index):
        self.players[start:stop, self.slot(team, index)] = np.nan

    def near(self, team, index, frame, offset_m=0.5):
        return self.players[frame, self.slot(team, index)] + np.array([offset_m / L, 0.0])

    def event(self, start, stop, code="SET PIECES", team=None, **tags):
        self.events.append(Event(f"synth-{len(self.events)}", start, stop, code, team, tags))

    def build(self) -> MatchInput:
        n, s = self.n, len(self.teams)
        half = np.zeros(n, dtype=np.int8)
        for a, b, h in self.segments:
            half[a:b] = h
        events = tuple(sorted(self.events, key=lambda e: (e.start_frame, e.end_frame, e.event_id)))
        return MatchInput(
            schema_version=1, match_id="SYNTH-ENGINE", source="tactical", n_frames=n, fps=FPS,
            pitch_length_m=L, pitch_width_m=W, coordinate_system="pitch_normalized_0_1",
            team_names={"H": "Synthetic Home", "A": "Synthetic Away"}, segments=tuple(self.segments),
            frame=np.arange(n, dtype=np.int64), half=half,
            slot_id=np.array([f"slot-{i}" for i in range(s)]), slot_team=np.array(self.teams),
            player_xy=self.players.copy(), player_observed=np.isfinite(self.players).all(axis=2),
            player_speed_mps=np.full((n, s), np.nan), ball_xy=self.ball.copy(),
            ball_observed=np.isfinite(self.ball).all(axis=1), ball_speed_mps=self.ball_speed.copy(),
            events=events,
        )
