"""Canonical atomic CSV export through the unchanged official validator."""
from dataclasses import dataclass
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile

from .contracts import MatchInput, PredictionResult


@dataclass(frozen=True)
class ExportReceipt:
    path: Path
    rows: int
    sha256: str
    validated: bool


def official_metrics():
    # Deliberately bind to this checkout's authoritative implementation.
    path = Path(__file__).resolve().parents[2] / "evaluation" / "possession_metrics.py"
    spec = importlib.util.spec_from_file_location("mtah_official_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_submission(result: PredictionResult, match: MatchInput, path: Path) -> ExportReceipt:
    result.validate_for(match)
    path = Path(path)
    root = Path(__file__).resolve().parents[2]
    if path.resolve().is_relative_to((root / "evaluation").resolve()):
        raise ValueError("write generated outputs outside official evaluation files")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write("frame,label\n")
            handle.writelines(f"{frame},{label}\n" for frame, label in zip(result.frame, result.label))
        pm = official_metrics()
        reason = pm.validate_submission(pm.read_label_csv(temp_path), match.n_frames)
        if reason:
            raise ValueError(f"official validation failed: {reason}")
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        # Windows rename refuses existing destinations. POSIX rename overwrites,
        # so use an exclusive hard link there. Both publish complete bytes.
        if os.name == "nt":
            os.rename(temp_path, path)
        else:
            os.link(temp_path, path)
        return ExportReceipt(path, match.n_frames, digest, True)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
