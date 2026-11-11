"""CLI integration tests: synthetic MatchInput -> selected engine -> PredictionResult -> CSV.

Invented fixtures only. No competition asset, ground truth or video is read.
The real-data path is exercised only through a fake, test-local loader module.
"""
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from mtah.__main__ import main
from mtah.synthetic import make_match

ROOT = Path(__file__).resolve().parents[1]
# Synthetic reference SHA-256 recorded in docs/SUBMISSION_PREFLIGHT.md for `python -m mtah --engine baseline --frames 199319`.
P03_BASELINE_FULL_SHA = "13151c4c89668b48f718881fa9e921d4d2220154199407880a607a6fa8f630f0"


def temporary_directory():
    root = ROOT / "artifacts" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:          # argparse usage errors
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def labels_of(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "".join(line.split(",")[1] for line in lines[1:])


class EngineSelectionTests(unittest.TestCase):
    def test_hybrid_end_to_end_labels(self):
        # make_match(40): home slot beside the ball for frames 0-19 (ball missing
        # at frame 1), away slot beside it for 20-39. Hybrid carries possession
        # through the missing frame; baseline marks it D.
        with temporary_directory() as tmp:
            code, out, err = cli("--engine", "hybrid", "--frames", "40", "--out", f"{tmp}/h.csv")
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(labels_of(f"{tmp}/h.csv"), "H" * 20 + "A" * 20)
            self.assertIn("engine=hybrid", out)
            self.assertIn("rows=40 validated=True", out)

    def test_baseline_end_to_end_labels(self):
        with temporary_directory() as tmp:
            code, out, _ = cli("--engine", "baseline", "--frames", "40", "--out", f"{tmp}/b.csv")
            self.assertEqual(code, 0)
            self.assertEqual(labels_of(f"{tmp}/b.csv"), "HD" + "H" * 18 + "A" * 20)
            self.assertIn("engine=baseline", out)

    def test_default_engine_is_baseline_and_bytes_match_p03(self):
        # Backward compatibility: the pre-selector command must produce the exact
        # file published in P0.3.
        with temporary_directory() as tmp:
            code, out, _ = cli("--frames", "199319", "--out", f"{tmp}/d.csv")
            self.assertEqual(code, 0)
            digest = hashlib.sha256(Path(f"{tmp}/d.csv").read_bytes()).hexdigest()
            self.assertEqual(digest, P03_BASELINE_FULL_SHA)
            self.assertIn(f"sha256={P03_BASELINE_FULL_SHA}", out)

    def test_hybrid_full_length_is_deterministic_and_official_format(self):
        with temporary_directory() as tmp:
            for name in ("a", "b"):
                code, _, _ = cli("--engine", "hybrid", "--frames", "199319", "--out", f"{tmp}/{name}.csv")
                self.assertEqual(code, 0)
            a, b = Path(f"{tmp}/a.csv").read_bytes(), Path(f"{tmp}/b.csv").read_bytes()
            self.assertEqual(a, b)
            self.assertTrue(a.startswith(b"frame,label\n0,"))
            self.assertFalse(a.startswith(b"\xef\xbb\xbf"))
            self.assertNotIn(b"\r", a)
            self.assertTrue(a.endswith(b"\n"))
            self.assertEqual(a.count(b"\n"), 199320)

    def test_unknown_engine_is_a_usage_error(self):
        with temporary_directory() as tmp:
            code, _, err = cli("--engine", "magic", "--out", f"{tmp}/x.csv")
            self.assertEqual(code, 2)
            self.assertIn("invalid choice: 'magic'", err)
            self.assertFalse(Path(f"{tmp}/x.csv").exists())

    def test_module_entry_point_runs(self):
        with temporary_directory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            proc = subprocess.run([sys.executable, "-B", "-m", "mtah", "--engine", "hybrid", "--frames", "40",
                                   "--out", f"{tmp}/m.csv"], cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(labels_of(f"{tmp}/m.csv"), "H" * 20 + "A" * 20)


class RealDataSeamTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("mtah.pipeline", None)

    def test_missing_loader_is_a_clear_error_and_input_is_not_opened(self):
        with temporary_directory() as tmp:
            never = Path(tmp) / "does-not-exist"     # an existence check would give a different message

            # Simulate a checkout where the real-data loader is absent.
            with patch.dict(sys.modules, {"mtah.pipeline": None}):
                code, out, err = cli("--source", "atd", "--input", str(never), "--engine", "hybrid",
                                     "--out", f"{tmp}/r.csv")

            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("real-data loader is not available", err)
            self.assertIn("mtah.pipeline.prepare_match", err)
            self.assertNotIn(str(never), err)          # no local paths echoed into shareable logs
            self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_missing_assets_are_a_path_free_error(self):
        # The real loader is installed now; a wrong --input must still be a
        # clear exit-2 error: no traceback, no CSV, no path echoed.
        with temporary_directory() as tmp:
            empty = Path(tmp) / "empty-root"
            empty.mkdir()

            code, out, err = cli(
                "--source", "atd",
                "--input", str(empty),
                "--engine", "hybrid",
                "--out", f"{tmp}/r.csv"
            )

            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("DATA.md", err)
            self.assertNotIn(str(empty), err)
            self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_atd_source_requires_input(self):
        with temporary_directory() as tmp:
            code, _, err = cli("--source", "atd", "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertIn("--input is required with --source atd", err)

    def test_synthetic_source_rejects_input(self):
        with temporary_directory() as tmp:
            code, _, err = cli("--input", tmp, "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertIn("--input is only valid with --source atd", err)

    def test_registered_loader_is_used(self):
        calls = []

        def prepare_match(input_root):
            calls.append(input_root)
            return make_match(40)

        sys.modules["mtah.pipeline"] = types.SimpleNamespace(prepare_match=prepare_match)
        with temporary_directory() as tmp:
            code, out, err = cli("--source", "atd", "--input", "local-root", "--engine", "hybrid",
                                 "--out", f"{tmp}/r.csv")
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(calls, [Path("local-root")])
            self.assertEqual(labels_of(f"{tmp}/r.csv"), "H" * 20 + "A" * 20)
            self.assertTrue(out.startswith("atd engine=hybrid"))

    def test_loader_excerpt_is_redacted_at_the_cli_boundary(self):
        sys.modules["mtah.pipeline"] = types.SimpleNamespace(
            prepare_match=lambda root: (_ for _ in ()).throw(
                ValueError("invalid literal for int() with base 10: 'SENSITIVE_TOKEN'")
            )
        )
        with temporary_directory() as tmp:
            code, out, err = cli("--source", "atd", "--input", "x", "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertNotIn("SENSITIVE_TOKEN", err)
            self.assertIn("<redacted>", err)
            self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_unquoted_input_path_and_quoted_excerpt_are_both_redacted(self):
        sys.modules["mtah.pipeline"] = types.SimpleNamespace(
            prepare_match=lambda root: (_ for _ in ()).throw(
                ValueError(f"missing file under {root}: bad 'SENSITIVE_TOKEN'")
            )
        )
        with temporary_directory() as tmp:
            secret = Path(tmp) / "BAR-ATH-secret"
            secret.mkdir()
            code, out, err = cli("--source", "atd", "--input", str(secret), "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertNotIn(str(secret), err)
            self.assertNotIn("SENSITIVE_TOKEN", err)
            self.assertIn("<input>", err)
            self.assertIn("<redacted>", err)
            self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_contractions_and_double_quotes_do_not_bypass_redaction(self):
        cases = (
            ("can't parse value: 'SENSITIVE_TOKEN'", "can't parse value", "SENSITIVE_TOKEN", "'<redacted>'"),
            ("isn't valid: 'SENSITIVE_TOKEN'", "isn't valid", "SENSITIVE_TOKEN", "'<redacted>'"),
            ('invalid value "SENSITIVE_DOUBLE"', "invalid value", "SENSITIVE_DOUBLE", '"<redacted>"'),
        )
        for template, safe_phrase, secret_token, marker in cases:
            with self.subTest(template=template), temporary_directory() as tmp:
                sys.modules["mtah.pipeline"] = types.SimpleNamespace(
                    prepare_match=lambda root, msg=template: (_ for _ in ()).throw(ValueError(msg))
                )
                code, out, err = cli("--source", "atd", "--input", "x", "--out", f"{tmp}/r.csv")
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertNotIn(secret_token, err)
                self.assertIn(safe_phrase, err)
                self.assertIn(marker, err)
                self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_contraction_with_unquoted_input_path_and_quoted_token_redacts_both(self):
        with temporary_directory() as tmp:
            secret_root = Path(tmp) / "BAR-ATH-secret"
            secret_root.mkdir()
            sys.modules["mtah.pipeline"] = types.SimpleNamespace(
                prepare_match=lambda root: (_ for _ in ()).throw(
                    ValueError(f"can't read under {root}: bad 'SENSITIVE_TOKEN'")
                )
            )
            code, out, err = cli("--source", "atd", "--input", str(secret_root), "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertNotIn(str(secret_root), err)
            self.assertNotIn("SENSITIVE_TOKEN", err)
            self.assertIn("can't read under <input>", err)
            self.assertIn("'<redacted>'", err)
            self.assertFalse(Path(f"{tmp}/r.csv").exists())

    def test_safe_module_diagnostics_remain_visible(self):
        class BrokenPipeline(types.ModuleType):
            def __getattr__(self, name):
                raise ModuleNotFoundError("No module named 'kloppy'", name="kloppy")

        sys.modules["mtah.pipeline"] = BrokenPipeline("mtah.pipeline")
        with temporary_directory() as tmp:
            secret = Path(tmp) / "secret-dir"
            code, out, err = cli("--source", "atd", "--input", str(secret), "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertEqual(out, "")
            self.assertIn("real-data loader failed to load: No module named 'kloppy'", err)
            self.assertNotIn(str(secret), err)

    def test_unsupported_xml_encodings_exit_two_without_leaking_encoding_or_path(self):
        bad_xml = '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><TrackData/>'
        with temporary_directory() as tmp:
            root = Path(tmp) / "bundle-secret"
            tactical = root / "tactical"
            tactical.mkdir(parents=True)
            (tactical / "BAR-ATH_tactical_atd.txt").write_text("", encoding="utf-8")
            (tactical / "BAR-ATH_tactical_atd.xml").write_text(bad_xml, encoding="ascii")
            (tactical / "BAR-ATH_tactical_homography.csv").write_text("", encoding="utf-8")
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text("<SportsCode/>", encoding="utf-8")

            code1, out1, err1 = cli("--source", "atd", "--input", str(root), "--out", f"{tmp}/o1.csv")
            self.assertEqual(code1, 2)
            self.assertEqual(out1, "")
            self.assertIn("unsupported XML encoding", err1)
            self.assertNotIn("SENSITIVE_ENCODING", err1)
            self.assertNotIn(str(root), err1)
            self.assertFalse(Path(f"{tmp}/o1.csv").exists())

            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                '<TrackData><Teams><Team id="t1"><Name>Barcelona</Name></Team></Teams>'
                '<Players><Player id="1" teamId="t1"/></Players></TrackData>',
                encoding="utf-8",
            )
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><SportsCode/>',
                encoding="ascii",
            )
            with patch("mtah.pipeline._load_atd", return_value=(
                __import__("numpy").zeros((199319, 1, 2)),
                __import__("numpy").ones((199319, 1), dtype=bool),
                __import__("numpy").zeros((199319, 1)),
                __import__("numpy").zeros((199319, 2)),
                __import__("numpy").ones(199319, dtype=bool),
                __import__("numpy").full(199319, float("nan")),
            )), patch("mtah.pipeline._load_geometry", return_value=None):
                code2, out2, err2 = cli("--source", "atd", "--input", str(root), "--out", f"{tmp}/o2.csv")
            self.assertEqual(code2, 2)
            self.assertEqual(out2, "")
            self.assertIn("unsupported XML encoding", err2)
            self.assertNotIn("SENSITIVE_ENCODING", err2)
            self.assertNotIn(str(root), err2)
            self.assertFalse(Path(f"{tmp}/o2.csv").exists())

    def test_loader_returning_wrong_type_is_rejected(self):
        sys.modules["mtah.pipeline"] = types.SimpleNamespace(prepare_match=lambda root: {"frames": []})
        with temporary_directory() as tmp:
            code, _, err = cli("--source", "atd", "--input", "x", "--out", f"{tmp}/r.csv")
            self.assertEqual(code, 2)
            self.assertIn("prepare_match must return a MatchInput", err)
            self.assertFalse(Path(f"{tmp}/r.csv").exists())


class OutputErrorTests(unittest.TestCase):
    def test_existing_output_is_refused(self):
        with temporary_directory() as tmp:
            target = Path(tmp) / "exists.csv"
            target.write_bytes(b"keep")
            code, _, err = cli("--engine", "hybrid", "--frames", "40", "--out", str(target))
            self.assertEqual(code, 1)
            self.assertIn("output already exists", err)
            self.assertEqual(target.read_bytes(), b"keep")

    def test_official_evaluation_directory_is_refused(self):
        target = ROOT / "evaluation" / "results" / "cli-should-not-exist.csv"
        code, _, err = cli("--engine", "hybrid", "--frames", "40", "--out", str(target))
        self.assertEqual(code, 1)
        self.assertIn("outside official evaluation files", err)
        self.assertFalse(target.exists())

    def test_too_few_synthetic_frames_is_a_clear_error(self):
        with temporary_directory() as tmp:
            code, _, err = cli("--frames", "3", "--out", f"{tmp}/s.csv")
            self.assertEqual(code, 2)
            self.assertIn("synthetic fixture needs at least four frames", err)


if __name__ == "__main__":
    unittest.main()
