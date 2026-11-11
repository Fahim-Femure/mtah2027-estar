"""Environment lock checker tests (scripts/check_environment.py). No network, no installs."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_environment", ROOT / "scripts" / "check_environment.py")
check_environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_environment)


class EnvironmentLockTests(unittest.TestCase):
    def test_parse_lock_file(self):
        pins = check_environment.parse_constraints("# comment\nnumpy==2.4.6\n\npandas==3.1.0  # pinned\n")
        self.assertEqual(pins, {"numpy": "2.4.6", "pandas": "3.1.0"})

    def test_unpinned_lines_are_rejected(self):
        for bad in ("numpy>=2\n", "numpy\n", "numpy==\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_environment.parse_constraints(bad)

    def test_compare_reports_drift_and_missing(self):
        problems = check_environment.compare({"numpy": "2.4.6", "pandas": "3.1.0", "tzdata": "2026.6"},
                                             {"numpy": "2.4.6", "pandas": "3.0.0"})
        self.assertEqual(problems, ["pandas: locked 3.1.0, installed 3.0.0", "tzdata: locked 2026.6, not installed"])
        self.assertEqual(check_environment.compare({"numpy": "2.4.6"}, {"numpy": "2.4.6"}), [])

    def test_python_version_rule(self):
        self.assertEqual(check_environment.python_problems((3, 11, 9)), [])
        self.assertEqual(check_environment.python_problems((3, 11, 4)), ["python: locked 3.11.9, running 3.11.4"])
        self.assertEqual(check_environment.python_problems((3, 14, 4)), ["python: locked 3.11.9, running 3.14.4"])

    def test_byte_order_mark_is_ignored(self):
        # Windows PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM.
        self.assertEqual(check_environment.parse_constraints("﻿numpy==2.4.6\n"), {"numpy": "2.4.6"})

    def test_team_lock_is_requirements_lock(self):
        # Team decision (P2-B.2): requirements.lock from Data Engineering is the one lock.
        self.assertEqual(check_environment.LOCK, ROOT / "requirements.lock")
        self.assertFalse((ROOT / "constraints-py311.txt").exists())
        pins = check_environment.read_lock()
        self.assertEqual(pins, {"numpy": "2.4.6", "pandas": "3.1.1", "python-dateutil": "4.0.0", "tzdata": "2026.6"})


if __name__ == "__main__":
    unittest.main()
