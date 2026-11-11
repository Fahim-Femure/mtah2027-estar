"""Submission preflight: strict checks on a frame,label CSV before it is submitted.

    python scripts/submission_preflight.py SUBMISSION.csv [--expect-rows 199319] [--compare OTHER.csv]

Checks (all reported, not just the first):
  * UTF-8, no byte-order mark, LF line endings only, final newline
  * header exactly `frame,label`
  * exactly --expect-rows data rows (default 199319, the tactical video)
  * every row `<frame>,<label>`, frame sequence exactly 0..N-1, label in H/A/D/X
  * the unchanged official validator (evaluation/possession_metrics.py) accepts it
  * optional: byte-identical to --compare (reproducibility)

Reads only the CSV(s) named. Never opens match assets or ground truth; does not
score. Exit 0 = PASS, 1 = FAIL.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtah.export import official_metrics  # noqa: E402  (binds to this checkout's evaluator)

ROW = re.compile(r"^(\d+),([A-Z])$")
LABELS = set("HADX")
MAX_ROW_ERRORS = 5


def check(path: Path, expect_rows: int, compare: Path | None = None):
    problems, info = [], {}
    if not path.is_file():
        return ["file not found"], info
    raw = path.read_bytes()
    info["sha256"] = hashlib.sha256(raw).hexdigest()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("UTF-8 byte-order mark present (must be UTF-8 without BOM)")
    if b"\r" in raw:
        problems.append("CR characters present (line endings must be LF only)")
    if not raw.endswith(b"\n"):
        problems.append("missing final newline")
    try:
        text = raw.decode("utf-8").lstrip("﻿").replace("\r\n", "\n")
    except UnicodeDecodeError:
        return problems + ["not valid UTF-8"], info
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != "frame,label":
        problems.append("header must be exactly 'frame,label'")
    rows = lines[1:]
    info["rows"] = len(rows)
    if len(rows) != expect_rows:
        problems.append(f"expected {expect_rows} data rows, found {len(rows)}")
    counts, row_errors = Counter(), []
    for k, line in enumerate(rows, 1):
        match = ROW.match(line)
        if not match:
            row_errors.append(f"row {k}: malformed {line[:20]!r}")
            continue
        frame, label = int(match.group(1)), match.group(2)
        counts[label] += 1
        if frame != k - 1:
            row_errors.append(f"row {k}: expected frame {k - 1}, found {frame}")
        if label not in LABELS:
            row_errors.append(f"row {k}: invalid label {label!r}")
    if row_errors:
        shown = row_errors[:MAX_ROW_ERRORS]
        more = len(row_errors) - len(shown)
        problems += shown + ([f"... {more} more row errors"] if more else [])
    info["labels"] = " ".join(f"{k}={counts.get(k, 0)}" for k in "HADX")
    try:
        pm = official_metrics()
        reason = pm.validate_submission(pm.read_label_csv(path), expect_rows)
    except Exception as exc:  # the official reader itself refused the file
        reason = f"could not read file ({type(exc).__name__}: {exc})"
    info["official"] = "valid" if not reason else reason
    if reason:
        problems.append(f"official validator: {reason}")
    if compare is not None:
        if not compare.is_file():
            problems.append("compare file not found")
        elif compare.read_bytes() != raw:
            problems.append(f"not byte-identical to {compare.name}")
        else:
            info["compare"] = f"byte-identical to {compare.name}"
    return problems, info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--expect-rows", type=int, default=199319)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    problems, info = check(args.csv, args.expect_rows, args.compare)
    if "rows" in info:
        print(f"rows={info['rows']} {info['labels']} sha256={info['sha256']}")
    if "official" in info:
        print(f"official validator: {info['official']}")
    if "compare" in info:
        print(info["compare"])
    if problems:
        print("FAIL")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
