"""P2-B.1 synthetic integration harness for the future Data Engineering loader.

Exercises `--source atd` end to end through a TEST-ONLY mock of
`mtah.pipeline.prepare_match` (tests/mock_loader.py). No competition asset,
ground truth or video is read. Expected labels are hand-derived from the mock's
documented block layout, never computed by the engine under test.
"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import mtah
from mtah.__main__ import main
from mtah.contracts import MatchInput
from mtah.engine import predict
from mtah.engines import predict_with
from mtah.export import export_submission, official_metrics
from mtah.sources import LoaderUnavailableError, load_match
from mock_loader import (BLOCK, GAP_FRAMES, MOCK_MATCH_ID, N_FRAMES, OUT_FRAMES, SEGMENTS, MockLoader,
                         block_layout, build_mock_match, installed_mock_loader)
from scenarios import Scenario

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "run_cli_with_mock_loader.py"
S1 = SEGMENTS[0][0]          # 11185, first graded frame


def temporary_directory():
    root = ROOT / "artifacts" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def labels_of(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "".join(line.split(",")[1] for line in lines[1:])


def run_subprocess(*argv):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([sys.executable, "-B", str(RUNNER), *map(str, argv)], cwd=ROOT, env=env,
                          capture_output=True, text=True)


class HarnessIsolationTests(unittest.TestCase):
    def test_tests_import_this_checkout(self):
        # The shared .venv may carry an editable install of another checkout.
        self.assertEqual(Path(mtah.__file__).resolve().parent, (ROOT / "src" / "mtah").resolve())

    def test_production_code_never_references_the_mock(self):
        pattern = re.compile(r"mock_loader|installed_mock_loader|MockLoader|SYNTH-MOCK|run_cli_with_mock")
        for path in (ROOT / "src").rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    # The next two tests hold whether or not Data Engineering's real loader is
    # installed: they check that the mock is never what runs outside its block.
    def test_mock_is_removed_after_the_block(self):
        before = sys.modules.get("mtah.pipeline")
        with installed_mock_loader():
            self.assertEqual(load_match("atd", input_root=Path("x")).match_id, MOCK_MATCH_ID)
        self.assertIs(sys.modules.get("mtah.pipeline"), before)
        with temporary_directory() as tmp:
            # No loader -> LoaderUnavailableError; real loader -> ValueError for
            # an empty root. Never the mock's invented match.
            with self.assertRaises((LoaderUnavailableError, ValueError)):
                load_match("atd", input_root=Path(tmp) / "no-assets")

    def test_plain_cli_process_never_uses_the_mock(self):
        with temporary_directory() as tmp:
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            proc = subprocess.run([sys.executable, "-B", "-m", "mtah", "--source", "atd",
                                   "--input", str(Path(tmp) / "no-assets"), "--engine", "hybrid",
                                   "--out", f"{tmp}/p.csv"],
                                  cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertEqual(proc.stdout, "")
            self.assertTrue(proc.stderr.startswith("error: "), proc.stderr)
            self.assertNotIn("SYNTH-MOCK", proc.stderr)
            self.assertFalse(Path(f"{tmp}/p.csv").exists())

    def test_this_checkout_reports_a_missing_loader(self):
        # Simulate absence explicitly so this also holds once mtah.pipeline exists.
        with patch.dict(sys.modules, {"mtah.pipeline": None}):
            with self.assertRaisesRegex(LoaderUnavailableError, "real-data loader is not available"):
                load_match("atd", input_root=Path("x"))

    def test_mock_never_reads_the_input_root(self):
        with temporary_directory() as tmp, installed_mock_loader() as loader:
            root = Path(tmp) / "no-such-asset-root"
            code, _, err = cli("--source", "atd", "--input", root, "--engine", "baseline", "--out", f"{tmp}/o.csv")
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(loader.calls, [root])
            self.assertFalse(root.exists())

    def test_mock_output_is_marked_synthetic(self):
        match = build_mock_match()
        self.assertEqual(match.match_id, MOCK_MATCH_ID)
        self.assertTrue(match.match_id.startswith("SYNTH-"))
        self.assertEqual(match.n_frames, N_FRAMES)


class MockAtdEngineTests(unittest.TestCase):
    """Both engines through `--source atd` with the mock loader installed."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = temporary_directory()
        cls.out = {}
        with installed_mock_loader():
            for engine in ("baseline", "hybrid"):
                path = Path(cls.tmp.name) / f"{engine}.csv"
                code, stdout, err = cli("--source", "atd", "--input", "local-root", "--engine", engine, "--out", path)
                assert (code, err) == (0, ""), err
                cls.out[engine] = (path, stdout)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_frames_and_alphabet(self):
        for engine, (path, stdout) in self.out.items():
            with self.subTest(engine=engine):
                lines = path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], "frame,label")
                self.assertEqual(len(lines) - 1, 199319)
                frames = [int(line.split(",")[0]) for line in lines[1:]]
                self.assertEqual(frames, list(range(199319)))
                self.assertTrue(set(labels_of(path)) <= set("HADX"))
                self.assertTrue(stdout.startswith(f"atd engine={engine} rows=199319 validated=True sha256="))

    def test_hybrid_hand_derived_frames(self):
        lab = labels_of(self.out["hybrid"][0])
        expected = {
            0: "X", 11184: "X", 100000: "X", 199318: "X",        # outside published segments
            S1 + 250: "H",                                         # block 0, home carrier
            S1 + BLOCK + 250: "A",                                 # block 1, away carrier
            S1 + 3 * BLOCK + 50: "D",                              # block 3, ball out
            S1 + 3 * BLOCK + OUT_FRAMES + 20: "D",                 # block 3, unobserved retrieval
            S1 + 3 * BLOCK + 300: "A",                             # block 3 after the tagged restart
        }
        for frame, label in expected.items():
            with self.subTest(frame=frame):
                self.assertEqual(lab[frame], label)
        outside = 11185 + (110808 - 90652) + (199319 - 190198)
        self.assertEqual(lab.count("X"), outside)

    def test_hybrid_dead_ball_only_where_the_mock_stops_play(self):
        lab = np.array(list(labels_of(self.out["hybrid"][0])))
        _, _, out, gap, _ = block_layout()
        stoppage = out | gap
        allowed = stoppage.copy()
        observed = build_mock_match().ball_observed
        # Before a stoppage (documented lead-in rule): unobserved frames directly
        # before it may be dead (the ball could have left play there).
        for start in np.flatnonzero(stoppage & ~np.concatenate(([False], stoppage[:-1]))):
            t = start - 1
            while t >= 0 and not observed[t]:
                allowed[t] = True
                t -= 1
        # After a stoppage (P2-B.2, S2): the FIRST observed frame after the gap is a
        # reacquisition frame with forward-only speed evidence, which S2 does not
        # trust, and unobserved frames stay held (documented hold rule). The dead
        # ball must end by the SECOND observed frame after the gap, the first one
        # with observations on both sides.
        for end in np.flatnonzero(stoppage & ~np.concatenate((stoppage[1:], [False]))):
            t, seen = end + 1, 0
            while t < len(lab):
                seen += bool(observed[t])
                if seen == 2:
                    break
                allowed[t] = True
                t += 1
        self.assertTrue((lab == "D").any())
        self.assertFalse(((lab == "D") & ~allowed).any())

    def test_baseline_hand_derived_frames(self):
        lab = labels_of(self.out["baseline"][0])
        # Baseline has no segment logic: missing ball -> D everywhere, nearest player otherwise.
        self.assertEqual(lab[0], "D")
        self.assertEqual(lab[S1 + 250], "H")
        self.assertEqual(lab[S1 + BLOCK + 250], "A")
        self.assertEqual(lab[S1 + 3 * BLOCK + OUT_FRAMES + 20], "D")
        self.assertNotIn("X", lab)

    def test_cli_bytes_equal_in_process_pipeline(self):
        match = build_mock_match()
        with temporary_directory() as tmp:
            for engine in ("baseline", "hybrid"):
                with self.subTest(engine=engine):
                    receipt = export_submission(predict_with(engine, match), match, Path(tmp) / f"{engine}.csv")
                    self.assertEqual((Path(tmp) / f"{engine}.csv").read_bytes(), self.out[engine][0].read_bytes())
                    self.assertTrue(receipt.validated)


class SubprocessDeterminismTests(unittest.TestCase):
    def test_fresh_processes_are_byte_identical(self):
        with temporary_directory() as tmp:
            for engine in ("baseline", "hybrid"):
                with self.subTest(engine=engine):
                    digests = []
                    for run in ("a", "b"):
                        proc = run_subprocess("--source", "atd", "--input", "local-root", "--engine", engine,
                                              "--out", f"{tmp}/{engine}-{run}.csv")
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        digests.append(hashlib.sha256(Path(f"{tmp}/{engine}-{run}.csv").read_bytes()).hexdigest())
                        self.assertIn(f"sha256={digests[-1]}", proc.stdout)
                    self.assertEqual(digests[0], digests[1])


class LoaderErrorTests(unittest.TestCase):
    def run_with(self, loader, engine="hybrid", input_root="local-root"):
        with temporary_directory() as tmp, installed_mock_loader(loader):
            target = Path(tmp) / "o.csv"
            code, out, err = cli("--source", "atd", "--input", input_root, "--engine", engine, "--out", target)
            return code, out, err, target.exists()

    def test_loader_error_is_reported_without_the_input_path(self):
        def failing(root):
            raise ValueError(f"missing tracking file under {root}")
        loader = MockLoader()
        loader.build = None
        loader.prepare_match = failing
        secret = r"C:\local\assets\BAR-ATH-root"
        code, out, err, written = self.run_with(loader, input_root=secret)
        self.assertEqual((code, out, written), (2, "", False))
        self.assertIn("missing tracking file under <input>", err)
        self.assertNotIn(secret, err)

    def failing_loader(self, message):
        def failing(root):
            raise ValueError(message.format(root=root))
        loader = MockLoader()
        loader.prepare_match = failing
        return loader

    def test_redaction_never_edits_inside_words(self):
        # A short --input must not be replaced inside other words of the message
        # (observed with --input x: "e<input>pected ... atd.t<input>t").
        loader = self.failing_loader("expected tactical/BAR-ATH_tactical_atd.txt and exactly x files under {root}")
        code, _, err, _ = self.run_with(loader, input_root="x")
        self.assertEqual(code, 2)
        self.assertEqual(err, "error: expected tactical/BAR-ATH_tactical_atd.txt and exactly <input> files "
                              "under <input>\n")

    def test_redaction_covers_path_prefixes_and_sentence_ends(self):
        root = r"C:\assets\root"
        loader = self.failing_loader(r"cannot read {root}\tactical\a.txt; root was {root}.")
        code, _, err, _ = self.run_with(loader, input_root=root)
        self.assertEqual(code, 2)
        self.assertEqual(err, "error: cannot read <input>\\tactical\\a.txt; root was <input>.\n")

    def test_redaction_leaves_longer_sibling_paths_alone(self):
        loader = self.failing_loader("found D:/rooted/file instead of {root}/file")
        code, _, err, _ = self.run_with(loader, input_root="D:/root")
        self.assertEqual(code, 2)
        self.assertEqual(err, "error: found D:/rooted/file instead of <input>/file\n")

    def test_noncontiguous_frames_are_rejected_by_the_contract(self):
        shifted = np.arange(1, N_FRAMES + 1, dtype=np.int64)        # 1-based ATD ids not converted
        code, out, err, written = self.run_with(MockLoader(lambda: build_mock_match(frame=shifted)))
        self.assertEqual((code, written), (2, False))
        self.assertIn("frame must be contiguous 0..N-1", err)

    def test_loader_import_failure_is_a_clear_error(self):
        class Broken(types.ModuleType):
            def __getattr__(self, name):
                raise ModuleNotFoundError("No module named 'kloppy'", name="kloppy")
        sys.modules["mtah.pipeline"] = Broken("mtah.pipeline")
        try:
            with temporary_directory() as tmp:
                code, _, err = cli("--source", "atd", "--input", "x", "--out", f"{tmp}/o.csv")
        finally:
            sys.modules.pop("mtah.pipeline", None)
        self.assertEqual(code, 2)
        self.assertIn("real-data loader failed to load: No module named 'kloppy'", err)

    def test_short_match_writes_a_csv_that_preflight_rejects(self):
        short = MockLoader(lambda: build_mock_match(n_frames=20000, segments=((1000, 9000, 1), (11000, 19000, 2))))
        with temporary_directory() as tmp, installed_mock_loader(short):
            target = Path(tmp) / "short.csv"
            code, _, _ = cli("--source", "atd", "--input", "x", "--engine", "hybrid", "--out", target)
            self.assertEqual(code, 0)                       # valid for its own length...
            proc = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / "submission_preflight.py"), str(target)],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)            # ...but not a submission
            self.assertIn("expected 199319 data rows, found 20000", proc.stdout)


class OutputPathSafetyTests(unittest.TestCase):
    def test_new_nested_directory_and_no_temp_leftovers(self):
        with temporary_directory() as tmp, installed_mock_loader():
            target = Path(tmp) / "a b" / "nested" / "sub.csv"          # space in path, new dirs
            code, _, err = cli("--source", "atd", "--input", "x", "--engine", "hybrid", "--out", target)
            self.assertEqual((code, err), (0, ""))
            self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["sub.csv"])

    def test_refused_overwrite_leaves_no_temp_file(self):
        with temporary_directory() as tmp, installed_mock_loader():
            target = Path(tmp) / "keep.csv"
            target.write_bytes(b"keep")
            code, _, err = cli("--source", "atd", "--input", "x", "--engine", "hybrid", "--out", target)
            self.assertEqual(code, 1)
            self.assertIn("output already exists", err)
            self.assertEqual(target.read_bytes(), b"keep")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["keep.csv"])


class ScorerEquivalenceTests(unittest.TestCase):
    """T12: verify official possession_metrics.fill_abstentions equivalence."""

    def test_unconfirmed_window_is_dead_after_official_abstention_fill(self):
        pm = official_metrics()
        s = Scenario(600)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 600)
        s.event(250, 375, team="A", Type="Throw-in")
        match = s.build()
        result = predict(match)

        runs = [(start, stop) for start, stop, _ in match.segments]
        codes = pm.label_codes(result.label)
        filled = pm.fill_abstentions(codes, runs)
        self.assertTrue(np.all(filled[250:375] == pm.LABELS.index("D")))

        # Equivalent to replacing the unconfirmed X window with D for spellF1 / transitions.
        d_labels = result.label.copy()
        d_labels[250:375] = "D"
        d_filled = pm.fill_abstentions(pm.label_codes(d_labels), runs)
        np.testing.assert_array_equal(filled, d_filled)
        gt = pm.label_codes(np.array(["H"] * 100 + ["D"] * 200 + ["A"] * 300))
        self.assertEqual(pm.spell_scores(gt, filled, runs)["mean_f1"],
                         pm.spell_scores(gt, d_filled, runs)["mean_f1"])


if __name__ == "__main__":
    unittest.main()
