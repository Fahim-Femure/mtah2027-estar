"""Possession Intelligence Engine entry point: MatchInput -> PredictionResult.

Consumes only the shared contract. Reads no files, assets or ground truth; no
network, no randomness. Same input and config -> identical output.
"""
from dataclasses import dataclass

import numpy as np

from ..contracts import MatchInput, PredictionResult
from .config import ENGINE_VERSION, EngineConfig
from .deadball import infer_dead
from .evidence import SpatialEvidence, spatial_evidence
from .temporal import decide


@dataclass(frozen=True)
class EngineTrace:
    """Engine-internal diagnostics. Local only: never exported or published."""
    evidence: SpatialEvidence
    dead_reason: np.ndarray
    reason: np.ndarray           # engine vocabulary, see temporal.REASONS
    restarts: tuple              # (restart_frame, team, segment_start, segment_stop)
    restart_kinds: tuple = ()    # per-cluster kind: early / before / late / unconfirmed / ignored


def predict_with_trace(match: MatchInput, config: EngineConfig | None = None):
    if not isinstance(match, MatchInput):
        raise TypeError("predict expects a validated MatchInput")
    config = EngineConfig() if config is None else config
    if not isinstance(config, EngineConfig):
        raise TypeError("config must be an EngineConfig")
    ev = spatial_evidence(match, config)
    dead, restarts, restart_kinds = infer_dead(match, ev, config)
    label, confidence, reason = decide(match, ev, dead, restarts, config, restart_kinds=restart_kinds)
    # Contract v1 restricts `reason` to the baseline vocabulary, so engine
    # reasons stay in the trace and the contract field is omitted.
    result = PredictionResult(1, match.match_id, match.source, match.frame, label,
                              ENGINE_VERSION, config.digest(), confidence=confidence)
    result.validate_for(match)
    reason.flags.writeable = False
    return result, EngineTrace(ev, dead, reason, tuple(restarts), tuple(restart_kinds))


def predict(match: MatchInput, config: EngineConfig | None = None) -> PredictionResult:
    return predict_with_trace(match, config)[0]
