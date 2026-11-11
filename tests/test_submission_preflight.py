"""Submission preflight checker tests (scripts/submission_preflight.py).

Inputs are invented CSVs written here; the good file is a synthetic CLI output.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submission_preflight.py"
N = 199319


def temporary_directory():
    root = ROOT / "artifacts" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def preflight(*args):
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def body(labels=None, n=N):
    labels = labels or ["X"] * n
    return "frame,label\n" + "".join(f"{i},{lab}\n" for i, lab in enumerate(labels))


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = temporary_directory()
        cls.dir = Path(cls.tmp.name)
        cls.good = cls.dir / "good.csv"
        cls.good.write_bytes(body(["H", "A", "D", "X"] * (N // 4) + ["X"] * (N % 4)).encode())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write(self, name, data: bytes):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_good_file_passes_and_reports(self):
        code, out = preflight(self.good)
        self.assertEqual(code, 0, out)
        self.assertIn("PASS", out)
        self.assertIn("rows=199319", out)
        self.assertIn("official validator: valid", out)
        self.assertRegex(out, r"sha256=[0-9a-f]{64}")
        self.assertIn("H=49829 A=49829 D=49829 X=49832", out)   # 4 * 49829 + 3 trailing X = 199319

    def test_byte_identical_compare(self):
        twin = self.write("twin.csv", self.good.read_bytes())
        self.assertEqual(preflight(self.good, "--compare", twin)[0], 0)
        other = self.write("other.csv", body().encode())
        code, out = preflight(self.good, "--compare", other)
        self.assertEqual(code, 1)
        self.assertIn("not byte-identical to", out)

    def test_each_defect_is_named(self):
        good = self.good.read_bytes()
        rows = good.decode().splitlines(keepends=True)
        cases = {
            "rows.csv": (body(n=N - 1).encode(), "expected 199319 data rows, found 199318"),
            "bom.csv": (b"\xef\xbb\xbf" + good, "UTF-8 byte-order mark"),
            "crlf.csv": (good.replace(b"\n", b"\r\n"), "CR characters"),
            "nonl.csv": (good[:-1], "missing final newline"),
            "header.csv": (good.replace(b"frame,label", b"frame,possession", 1), "header must be exactly 'frame,label'"),
            "label.csv": ("".join(rows[:5] + ["4,Z\n"] + rows[6:]).encode(), "row 5: invalid label 'Z'"),
            "gap.csv": ("".join(rows[:5] + ["5,H\n"] + rows[6:]).encode(), "row 5: expected frame 4, found 5"),
            "onebased.csv": (("frame,label\n" + "".join(f"{i + 1},X\n" for i in range(N))).encode(),
                             "row 1: expected frame 0, found 1"),
            "spaces.csv": (good.replace(b"0,H\n", b"0, H\n", 1), "row 1: malformed"),
        }
        for name, (data, message) in cases.items():
            with self.subTest(case=name):
                code, out = preflight(self.write(name, data))
                self.assertEqual(code, 1, out)
                self.assertIn(message, out)
                self.assertIn("FAIL", out)

    def test_missing_file(self):
        code, out = preflight(self.dir / "absent.csv")
        self.assertEqual(code, 1)
        self.assertIn("file not found", out)

    def test_official_validator_disagreement_is_reported(self):
        # 199318 rows: the official validator must reject it too, and its reason is shown.
        code, out = preflight(self.write("short.csv", body(n=N - 1).encode()))
        self.assertEqual(code, 1)
        self.assertIn("official validator:", out)
        self.assertNotIn("official validator: valid", out)


if __name__ == "__main__":
    unittest.main()
