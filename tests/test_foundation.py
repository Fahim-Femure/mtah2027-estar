"""Invented fixtures only. Never loads competition assets or public labels."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import unittest

import numpy as np
from mtah.alignment import atd_to_video, tagging_to_video, validate_frame_sequence, project
from mtah.baseline import predict
from mtah.contracts import Event, Geometry, PredictionResult
from mtah.export import export_submission
from mtah.synthetic import make_match

ROOT = Path(__file__).resolve().parents[1]


def temporary_directory():
    # Keep all invented test files in the ignored workspace on sandboxed Windows.
    root = ROOT / 'artifacts' / 'test-tmp'
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlignmentTests(unittest.TestCase):
    def test_atd_endpoints_no_offsets(self):
        self.assertEqual(atd_to_video(1), 0)
        self.assertEqual(atd_to_video(199319), 199318)
        self.assertEqual(atd_to_video(11186), 11185)
        for bad in (0, -1, 1.2, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError): atd_to_video(bad)

    def test_tagging_rounding(self):
        self.assertEqual(tagging_to_video(0), 0)
        self.assertEqual(tagging_to_video(.1), 2)  # Python tie-to-even, as official round()
        self.assertEqual(tagging_to_video(.12), 3)
        for bad in (-1, float('nan'), float('inf'), True):
            with self.subTest(bad=bad), self.assertRaises(ValueError): tagging_to_video(bad)

    def test_exact_continuity(self):
        validate_frame_sequence([1, 2, 3, 4], 4, atd=True)
        validate_frame_sequence([0, 1, 2, 3], 4)
        for bad in ([1, 3, 3, 4], [1, 3, 2, 4], [1, 2], [], [1., 2., 3., 4.]):
            with self.subTest(bad=bad), self.assertRaises(ValueError): validate_frame_sequence(bad, 4, atd=True)
        with self.assertRaises(ValueError): validate_frame_sequence([1, 2, 3, 4], 4)

    def test_projection_guards(self):
        np.testing.assert_allclose(project(np.eye(3), .2, .7), [.2, .7])
        with self.assertRaises(ValueError): project(np.zeros((3, 3)), .2, .7)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.match = make_match()

    def test_copy_and_read_only(self):
        original = self.match.ball_xy.copy()
        m = replace(self.match, ball_xy=original)
        original[0] = .8
        self.assertEqual(m.ball_xy[0, 0], .21)
        with self.assertRaises(ValueError): m.ball_xy[0] = 0
        with self.assertRaises(TypeError): m.team_names['H'] = 'changed'

    def test_missingness_and_outside_pitch(self):
        self.assertFalse(self.match.ball_observed[1])
        b = self.match.ball_xy.copy(); b[0] = [-.1, 1.1]
        replace(self.match, ball_xy=b)  # not clipped
        for pair in ([np.nan, .3], [np.inf, .3]):
            b = self.match.ball_xy.copy(); b[0] = pair
            with self.assertRaises(ValueError): replace(self.match, ball_xy=b)
        with self.assertRaises(ValueError): replace(self.match, ball_observed=np.ones(12, dtype=bool))

    def test_shapes_types_teams(self):
        cases = dict(frame=np.arange(12, dtype=np.float64), player_xy=np.zeros((12, 1, 2)),
                     slot_team=np.array(['H', '?']), slot_id=np.array(['x', 'x']),
                     fps=30, schema_version=2, source='broadcast', match_id='BAR-ATH')
        for key, value in cases.items():
            with self.subTest(key=key), self.assertRaises(ValueError): replace(self.match, **{key:value})

    def test_half_gaps(self):
        half = np.array([1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2], dtype=np.int8)
        replace(self.match, half=half, segments=((0, 4, 1), (8, 12, 2)))
        with self.assertRaises(ValueError): replace(self.match, segments=((0, 8, 1), (7, 12, 2)))
        with self.assertRaises(ValueError): replace(self.match, half=half)

    def test_event_end_exclusion(self):
        e = Event('e', 1, 3, 'TEST', None, {})
        self.assertTrue(e.contains(1)); self.assertFalse(e.contains(3))
        with self.assertRaises(ValueError): Event('e', 3, 3, 'TEST', None, {})
        with self.assertRaises(ValueError): replace(self.match, events=(Event('e', 0, 13, 'TEST', None, {}),))
        with self.assertRaises(ValueError): replace(self.match, events=(e, e))

    def test_geometry(self):
        g = Geometry(np.tile(np.eye(3), (12, 1, 1)), np.ones(12, dtype=bool), np.zeros(12))
        replace(self.match, geometry=g)
        with self.assertRaises(ValueError): Geometry(np.zeros((12, 3, 3)), np.ones(12, dtype=bool), np.zeros(12))
        with self.assertRaises(ValueError): Geometry(np.tile(np.eye(3), (12, 1, 1)), np.zeros(12, dtype=bool), np.zeros(12))
        with self.assertRaises(ValueError): replace(self.match, geometry=Geometry(np.eye(3)[None], np.ones(1, dtype=bool), np.zeros(1)))

    def test_prediction_validation(self):
        p = predict(self.match)
        for change in ({'label':np.array(['Z']*12)}, {'frame':np.array([0]*12,dtype=np.int64)},
                       {'label':np.array(['H']*11)}, {'confidence':np.full(12, 2.)}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError): replace(p, **change)
        with self.assertRaises(ValueError): replace(p, match_id='different').validate_for(self.match)
        with self.assertRaises(ValueError): replace(p, frame=np.arange(11,dtype=np.int64), label=p.label[:11], reason=None).validate_for(self.match)
        x = replace(p, label=np.full(12, 'X'), confidence=np.full(12, np.nan))
        x.validate_for(self.match)


class IntegrationTests(unittest.TestCase):
    def test_missing_players_and_tie(self):
        m = make_match(4)
        players = m.player_xy.copy(); players[0] = np.nan
        observed = m.player_observed.copy(); observed[0] = False
        m = replace(m, player_xy=players, player_observed=observed)
        self.assertEqual(predict(m).label[0], 'D')
        self.assertEqual(predict(m).label[1], 'D')
        players = m.player_xy.copy(); players[2] = [[.5, .5], [.5, .5]]
        self.assertEqual(predict(replace(m, player_xy=players)).label[2], 'H')

    def test_raw_baseline_compatibility(self):
        official = load_file('official_baseline', ROOT/'starter/baseline.py')
        m = make_match(4)
        with temporary_directory() as tmp:
            raw = Path(tmp)/'invented.txt'
            raw.write_text('1:0.2,0.5,0;0.8,0.5,0:0.21,0.5,0\n2:0.2,0.5,0;0.8,0.5,0:NaN,NaN,NaN\n3:0.2,0.5,0;0.8,0.5,0:0.79,0.5,0\n4:0.2,0.5,0;0.8,0.5,0:0.79,0.5,0\n')
            codes = official.nearest_player_labels(raw, np.array([0, 1]), 4, (105.3, 68.0))
            np.testing.assert_array_equal(predict(m).label, official.smooth(codes, 0))

    def test_deterministic_full_length_export(self):
        m = make_match(199319)
        with temporary_directory() as tmp:
            a = export_submission(predict(m), m, Path(tmp)/'a.csv')
            b = export_submission(predict(m), m, Path(tmp)/'b.csv')
            self.assertEqual(a.sha256, b.sha256)
            data = a.path.read_bytes()
            self.assertEqual(data, b.path.read_bytes())
            self.assertTrue(data.startswith(b'frame,label\n0,H\n'))
            self.assertTrue(data.endswith(b'199318,A\n'))
            self.assertNotIn(b'\r', data)
            self.assertEqual(len(data.splitlines()), 199320)
            with self.assertRaises(FileExistsError): export_submission(predict(m), m, a.path)

    def test_reject_reference_output_path(self):
        m = make_match()
        with self.assertRaises(ValueError): export_submission(predict(m), m, ROOT/'evaluation/results/baseline_raw.csv')


class SafetyTests(unittest.TestCase):
    def test_real_staged_metadata(self):
        guard = load_file('staged_guard_integration', ROOT/'scripts/check_staged.py')
        with temporary_directory() as tmp:
            repo = Path(tmp)
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            guard.ROOT = repo
            (repo/'safe.py').write_text('# invented test source\n', encoding='utf-8')
            subprocess.run(['git', '-C', str(repo), 'add', 'safe.py'], check=True)
            self.assertEqual(guard.main(), 0)
            (repo/'invented.zip').write_bytes(b'not an actual archive')
            subprocess.run(['git', '-C', str(repo), 'add', 'invented.zip'], check=True)
            self.assertEqual(guard.main(), 1)

    def test_path_policy(self):
        guard = load_file('staged_guard', ROOT/'scripts/check_staged.py')
        for p in ('data/x.txt', 'copy/game_atd.txt', 'exports/a.json', '.env', 'a.mp4', 'a.zip',
                  'evaluation/possession_metrics.py', 'evaluation/results/baseline_new.csv',
                  'features.npz', 'notebook.ipynb', '.venv311/x.py', 'secrets.json', 'other.lock',
                  'data/requirements.lock', 'workspace-private/a.py', 'dir/Team-Private/notes.md'):
            with self.subTest(path=p): self.assertIsNotNone(guard.path_reason(p))
        for p in ('src/mtah/contracts.py', 'tests/test_foundation.py', 'docs/ESTAR_TECHNICAL_DOCUMENT.md', 'non_private/safe.py',
                  'tests/fixtures/synthetic/short.csv', '.env.example', 'requirements.txt',
                  'requirements.lock'):
            with self.subTest(path=p): self.assertIsNone(guard.path_reason(p))


if __name__ == '__main__':
    unittest.main()
