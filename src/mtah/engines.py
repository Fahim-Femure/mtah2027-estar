"""Engine selector. Every engine maps a validated MatchInput to a PredictionResult.

  baseline  unsmoothed nearest-player adapter (mtah.baseline), official raw semantics
  hybrid    possession engine (mtah.engine): spatial evidence + dead-ball inference
            + temporal reasoning, with Smart Tagging set pieces as restart evidence
"""
from typing import Callable

from . import baseline, engine
from .contracts import MatchInput, PredictionResult

ENGINES: dict[str, Callable[[MatchInput], PredictionResult]] = {
    "baseline": baseline.predict,
    "hybrid": engine.predict,
}


def predict_with(name: str, match: MatchInput) -> PredictionResult:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; choose one of {', '.join(sorted(ENGINES))}")
    result = ENGINES[name](match)
    result.validate_for(match)
    return result
