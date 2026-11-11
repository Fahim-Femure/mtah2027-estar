"""CLI: MatchInput source -> selected engine -> PredictionResult -> official CSV.

Exit codes: 0 success; 1 output refused (exists, or inside official evaluation
files); 2 usage, input or loader error. Errors never echo input paths.
"""
import argparse
from pathlib import Path
import re
import sys

from .engines import ENGINES, predict_with
from .export import export_submission
from .sources import LoaderUnavailableError, load_match


def _loader_error_message(exc):
    """Drop quoted loader-message fragments at the CLI boundary.

    The loader never includes raw ATD values in its sanitized messages. This
    replaces quoted exception fragments before printing so an unexpected
    loader error cannot expose an ATD field excerpt through the CLI, while
    ignoring word-internal apostrophes and trailing possessives so English
    contractions (such as "can't") cannot invert quote pairing.
    """
    message = exc if isinstance(exc, str) else str(exc)
    message = re.sub(r'"(?:\\.|[^"])*"', '"<redacted>"', message)
    return re.sub(
        r"(?<!\w)'(?:\\.|[^']|(?<=\w)'(?=\w))*?'(?!\w)",
        "'<redacted>'",
        message,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m mtah",
        description="Run a possession engine and write a validated frame,label CSV. "
                    "The synthetic source opens no match assets.")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="baseline",
                        help="baseline: nearest-player adapter (default, P0.3 behaviour); hybrid: possession engine")
    parser.add_argument("--source", choices=("synthetic", "atd"), default="synthetic",
                        help="synthetic: invented fixture (default); atd: real session records via the Data Engineering loader")
    parser.add_argument("--frames", type=int, default=12, help="synthetic fixture length")
    parser.add_argument("--input", type=Path, help="local asset root for --source atd, passed to the loader unopened")
    parser.add_argument("--out", type=Path, required=True, help="new CSV path; existing files are never overwritten")
    return parser


def redact(message: str, input_root) -> str:
    """Replace every spelling of the input root in a (loader-supplied) message.

    Only whole path occurrences are replaced: the match must not sit inside a
    longer word or path component (so `--input x` leaves "expected" alone and
    `D:/root` leaves `D:/rooted` alone). A trailing separator, punctuation or
    sentence-ending period still counts as a boundary.
    """
    if input_root is None:
        return message
    spellings = {str(input_root), Path(input_root).as_posix()}
    try:
        resolved = Path(input_root).resolve()
        spellings |= {str(resolved), resolved.as_posix()}
    except OSError:
        pass
    for spelling in sorted((s for s in spellings if s and s != "."), key=len, reverse=True):
        pattern = r"(?<![\w.~-])" + re.escape(spelling) + r"(?![\w~-]|\.\w)"
        message = re.sub(pattern, "<input>", message)
    return message


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.source == "atd" and args.input is None:
        parser.error("--input is required with --source atd")
    if args.source == "synthetic" and args.input is not None:
        parser.error("--input is only valid with --source atd")
    try:
        match = load_match(args.source, frames=args.frames, input_root=args.input)
        result = predict_with(args.engine, match)
    except LoaderUnavailableError as exc:
        print(f"error: {redact(str(exc), args.input)}", file=sys.stderr)
        return 2
    except (ValueError, TypeError) as exc:
        print(f"error: {_loader_error_message(redact(str(exc), args.input))}", file=sys.stderr)
        return 2
    try:
        receipt = export_submission(result, match, args.out)
    except FileExistsError:
        print("error: output already exists; choose a new --out path", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{args.source} engine={args.engine} rows={receipt.rows} validated={receipt.validated} sha256={receipt.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
