"""Metadata-only staging guard. Never opens staged file contents.

This detects likely assets, not embedded secrets or copied data disguised as code.
Use alongside explicit staging and human source review. No Git config is changed.
"""
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCK_DIRS = {"data", "datasets", "private", "restricted", "exports", "artifacts", "outputs", "runs", "cache", ".cache", ".venv", "venv", "env", "__pycache__", ".ipynb_checkpoints", ".aws", ".ssh"}
BLOCK_EXT = {".csv", ".xml", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".zip", ".7z", ".rar", ".tar", ".gz", ".gzip", ".pl3container", ".npz", ".npy", ".parquet", ".pkl", ".pickle", ".pdf", ".ipynb", ".pem", ".key", ".sqlite", ".db", ".bin", ".exe", ".dll", ".pt", ".pth"}
SAFE_EXT = {".py", ".md", ".toml", ".ps1", ".sh", ".cff", ".yml", ".yaml", ".json"}


def path_reason(path: str) -> str | None:
    p = PurePosixPath(path.replace('\\', '/').lower())
    synthetic = p.parts[:3] == ("tests", "fixtures", "synthetic")
    if p.parts[0] == "evaluation" or str(p) == "games.py":
        return "official reference files must remain unchanged"
    if any(part in BLOCK_DIRS or part.startswith('.venv') or part.endswith('-private') for part in p.parts):
        return "restricted/generated/environment path"
    if p.name.startswith('.env') and p.name != '.env.example':
        return "local environment credentials"
    if any(word in p.name for word in ('credential', 'service-account', 'secret', '_atd', '_homography', '_smart_tagging', 'possession_gt')):
        return "likely credential or competition asset"
    if p.suffix in BLOCK_EXT and not (synthetic and p.suffix in {'.csv', '.xml'}):
        return "likely data, archive, binary or output-bearing notebook"
    if p.suffix == '.txt' and not (p.name.startswith(('requirements', 'constraints')) or synthetic):
        return "unexpected text export"
    # requirements.lock (repository root only) is the team's dependency lock.
    if (p.suffix not in SAFE_EXT | {'.txt', '.csv', '.xml'} and str(p) != 'requirements.lock'
            and p.name not in {'.gitignore', '.gitattributes', '.env.example', 'license'}):
        return "unreviewed file extension"
    return None


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main() -> int:
    paths = [p.decode('utf-8') for p in git('diff', '--cached', '--name-only', '--no-renames', '-z').split(b'\0') if p]
    reasons = {p: path_reason(p) for p in paths}
    entries = {}
    for entry in git('ls-files', '--stage', '-z').split(b'\0'):
        if entry:
            meta, name = entry.split(b'\t', 1)
            mode, oid, stage = meta.split()
            entries[name.decode('utf-8')] = (mode, oid, stage)
    for path in paths:
        if path not in entries:  # deletion; still enforce path protection above
            continue
        mode, oid, stage = entries[path]
        if stage != b'0' or mode not in (b'100644', b'100755'):
            reasons[path] = 'unmerged entry, symlink or submodule'
        size = int(git('cat-file', '-s', oid.decode()).strip())
        if size > 1_000_000:
            reasons[path] = 'staged blob exceeds 1 MB'
    rejected = {p: r for p, r in reasons.items() if r}
    for path, reason in rejected.items():
        print(f'REJECT {path!r}: {reason}')
    print(f'Staged safety: {len(paths)} paths checked; {len(rejected)} rejected (metadata only)')
    return int(bool(rejected))


if __name__ == '__main__':
    sys.exit(main())
