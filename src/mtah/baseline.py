"""Unsmooth nearest-player adapter, matching the official raw baseline semantics.

Missing ball/no observed player becomes D for compatibility, not as a claim that
occlusion means dead ball. No training, advanced inference or GT access.
"""
import hashlib
import numpy as np
from .contracts import MatchInput, PredictionResult


def predict(match: MatchInput) -> PredictionResult:
    delta = (match.player_xy - match.ball_xy[:, None, :]) * [match.pitch_length_m, match.pitch_width_m]
    distance = np.sum(delta * delta, axis=2)
    usable = match.player_observed & match.ball_observed[:, None]
    distance = np.where(usable, distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    seen = usable.any(axis=1)
    labels = np.where(seen, match.slot_team[nearest], "D")
    result = PredictionResult(1, match.match_id, match.source, match.frame, labels,
                              "nearest-raw-v1", hashlib.sha256(b"nearest-raw-v1:no-smoothing").hexdigest(),
                              reason=np.where(seen, "nearest_player", "baseline_missing"))
    result.validate_for(match)
    return result
