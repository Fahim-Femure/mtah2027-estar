"""TEST-ONLY launcher: `python tests/run_cli_with_mock_loader.py <mtah CLI args>`.

Runs the real CLI (`mtah.__main__.main`) in a fresh process with the mock
Data Engineering loader installed, so `--source atd` can be exercised end to end.
Never use it for a submission: the mock match id is SYNTH-MOCK-ATD.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_loader import installed_mock_loader  # noqa: E402
from mtah.__main__ import main  # noqa: E402

if __name__ == "__main__":
    with installed_mock_loader():
        sys.exit(main(sys.argv[1:]))
