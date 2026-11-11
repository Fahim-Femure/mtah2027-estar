"""ML-owned possession engine. Consumes contract v1 MatchInput only."""
from .config import ENGINE_VERSION, EngineConfig
from .possession import EngineTrace, predict, predict_with_trace

__all__ = ["ENGINE_VERSION", "EngineConfig", "EngineTrace", "predict", "predict_with_trace"]
