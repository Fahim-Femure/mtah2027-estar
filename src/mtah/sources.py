"""Input sources for the CLI. This module dispatches; it parses no match data.

  synthetic  invented fixture (mtah.synthetic.make_match)
  atd        Data Engineering's real-data loader, expected at
             mtah.pipeline.prepare_match(input_root: Path) -> MatchInput

The real-data loader is owned by Data Engineering and lives in
mtah.pipeline.prepare_match. When mtah.pipeline cannot be imported or exposes
no callable prepare_match, `atd` fails with LoaderUnavailableError before any
input path is opened or even checked for existence.
"""
import importlib
import importlib.util
from pathlib import Path

from .contracts import MatchInput
from .synthetic import make_match

LOADER_MODULE = "mtah.pipeline"
LOADER_FUNCTION = "prepare_match"


class LoaderUnavailableError(RuntimeError):
    """The Data Engineering loader is not installed in this checkout."""


def _loader():
    try:
        module = importlib.import_module(LOADER_MODULE)
        function = getattr(module, LOADER_FUNCTION, None)
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name in (LOADER_MODULE, "mtah"):
            function = None  # not installed
        else:
            # Installed but broken (e.g. a missing optional dependency): say so, never fall back.
            raise LoaderUnavailableError(f"real-data loader failed to load: {exc}") from exc
    if not callable(function):
        raise LoaderUnavailableError(
            f"real-data loader is not available: {LOADER_MODULE}.{LOADER_FUNCTION}(input_root) -> MatchInput "
            "is owned by Data Engineering and is not installed in this checkout. "
            "Use --source synthetic, or install the Data Engineering pipeline.")
    return function


def load_match(source: str, *, frames: int = 12, input_root: Path | None = None) -> MatchInput:
    if source == "synthetic":
        return make_match(frames)
    if source == "atd":
        match = _loader()(Path(input_root))
        if not isinstance(match, MatchInput):
            raise TypeError(f"{LOADER_MODULE}.{LOADER_FUNCTION} must return a MatchInput, got {type(match).__name__}")
        return match
    raise ValueError(f"unknown source {source!r}")
