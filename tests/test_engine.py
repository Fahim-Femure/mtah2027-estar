"""Possession engine tests on invented scenarios only.

No competition assets, public ground truth or scores are read. Expected labels
come from the scenario construction and the documented rules, never from data.
"""
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from mtah.engine import ENGINE_VERSION, EngineConfig, predict, predict_with_trace
from mtah.engine.deadball import DEAD_HOLD, DEAD_OUT, DEAD_SET_PIECE, DEAD_STATIONARY, DEAD_UNCONFIRMED, LIVE
from mtah.engine.evidence import (AWAY, CONTESTED, HOME, NONE, WHY_BALL_MISSING, WHY_IN_FLIGHT,
                                  WHY_NO_PLAYERS, ball_speed, derived_ball_speed)
from mtah.engine.temporal import REASONS
from mtah.export import export_submission
from mtah.synthetic import make_match
from scenarios import FPS, L, W, Scenario

ROOT = Path(__file__).resolve().parents[1]


def labels(result):
    return "".join(result.label)


def run(scenario, config=None):
    return predict_with_trace(scenario.build(), config)


class SpatialEvidenceTests(unittest.TestCase):
    def test_dribble_is_home_control(self):
        s = Scenario(60)
        s.dribble(0, 60, "H", 0)
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control == HOME))
        self.assertEqual(labels(result), "H" * 60)
        np.testing.assert_allclose(result.confidence, 0.9)

    def test_contested_duel_is_not_control(self):
        s = Scenario(100)
        s.dribble(0, 100, "H", 0)
        # An away defender arrives at the ball: equal distance from both players.
        for t in range(40, 70):
            ball = s.ball[t]
            s.player_at(t, t + 1, "H", 0, ball - [0.6 / L, 0])
            s.player_at(t, t + 1, "A", 0, ball + [0.6 / L, 0])
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[40:70] == CONTESTED))
        self.assertEqual(labels(result), "H" * 100)

    def test_ball_in_flight_past_defender_is_not_control(self):
        s = Scenario(120)
        s.dribble(0, 40, "H", 0)
        a, b = s.ball[39], s.near("H", 1, 0)
        s.ball_path(40, 70, a, b)
        mid = (a + b) / 2
        s.player_at(0, 120, "A", 0, mid)              # defender standing on the passing lane
        s.ball_speed[40:70] = 20.0                    # source speed says travelling
        s.dribble(70, 120, "H", 1)
        result, trace = run(s)
        self.assertFalse(np.any(trace.evidence.control[40:70] == AWAY))
        self.assertTrue(np.any(trace.evidence.why[40:70] == WHY_IN_FLIGHT))
        self.assertEqual(labels(result), "H" * 120)

    def test_identity_switch_and_duplicate_track(self):
        s = Scenario(100, extra_home=1)
        s.dribble(0, 100, "H", 0)
        s.player_missing(50, 100, "H", 0)             # track lost...
        s.players[50:, s.slot("H", 2)] = s.ball[50:] - [0.5 / L, 0]   # ...re-acquired in another slot
        s.players[20:40, s.slot("H", 1)] = s.players[20:40, s.slot("H", 0)]  # duplicated track
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 100)

    def test_ball_observed_but_no_players(self):
        s = Scenario(40)
        s.ball_path(0, 40, (0.4, 0.4), (0.6, 0.6))
        s.players[:] = np.nan
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.why == WHY_NO_PLAYERS))
        self.assertEqual(labels(result), "X" * 40)
        self.assertTrue(np.isnan(result.confidence).all())

    def test_speed_derivation_never_bridges_a_gap(self):
        s = Scenario(30)
        s.ball_path(0, 30, (0.2, 0.5), (0.8, 0.5))
        s.ball_missing(10, 12)
        match = s.build()
        speed = ball_speed(match, EngineConfig())
        self.assertTrue(np.isnan(speed[10:12]).all())
        expected = 0.6 * L / 30 * FPS
        finite = np.isfinite(speed)
        np.testing.assert_allclose(speed[finite], expected, rtol=1e-9)
        # Source speed always wins where present.
        s.ball_speed[:] = 4.0
        np.testing.assert_allclose(ball_speed(s.build(), EngineConfig()), 4.0)


class DerivedSpeedSymmetryTests(unittest.TestCase):
    def test_reacquisition_one_sided_fast_speed_is_suppressed(self):
        # Ball is unobserved just before frame 20; the forward displacement alone
        # looks like a >100 m/s relock, but the backward lag is unobserved in the
        # same half, so S2 must return NaN at the reacquisition frame.
        s = Scenario(50)
        s.ball_path(0, 10, (0.1, 0.5), (0.2, 0.5))
        s.ball_missing(10, 20)
        s.ball_path(20, 35, (0.8, 0.5), (1.0, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        self.assertTrue(np.isnan(speed[20]))
        self.assertTrue(np.isnan(speed[9]))
        # The next frame has both lag-1 endpoints observed in the same half.
        self.assertGreater(speed[21], 12.0)

    def test_genuine_both_direction_flight_is_preserved(self):
        s = Scenario(100)
        s.ball_path(0, 100, (0.1, 0.5), (0.9, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        expected = 0.8 * L / 100 * FPS
        self.assertAlmostEqual(speed[50], expected, places=6)
        self.assertGreater(speed[50], 12.0)

    def test_one_sided_missing_is_not_stationary_evidence(self):
        s = Scenario(50)
        s.ball_still(0, 10, (0.5, 0.5))
        s.ball_missing(10, 20)
        s.ball_still(20, 50, (0.5, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        self.assertTrue(np.isnan(speed[20]))
        self.assertTrue(np.isnan(speed[9]))
        # Away from the missing counterpart, bilateral stillness is retained.
        self.assertEqual(speed[30], 0.0)

    def test_moderate_one_sided_speed_is_not_trusted_as_motion(self):
        s = Scenario(50)
        s.ball_path(0, 10, (0.1, 0.5), (0.2, 0.5))
        s.ball_missing(10, 20)
        s.ball_path(20, 50, (0.5, 0.5), (0.55, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        self.assertTrue(np.isnan(speed[20]))
        self.assertTrue(np.isnan(speed[9]))
        self.assertGreater(speed[21], 0.0)

    def test_terminal_and_half_boundary_one_sided_estimates_are_kept(self):
        # No bilateral estimate exists at the timeline edge, so the available
        # direction is retained; the same applies across a segment boundary.
        s = Scenario(100, segments=((0, 50, 1), (50, 100, 2)))
        s.ball_path(0, 100, (0.1, 0.5), (0.9, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        expected = 0.8 * L / 100 * FPS
        self.assertAlmostEqual(speed[0], expected, places=6)
        self.assertAlmostEqual(speed[49], expected, places=6)
        self.assertAlmostEqual(speed[50], expected, places=6)
        self.assertAlmostEqual(speed[99], expected, places=6)

    def test_stationary_and_transition_frames_remain_finite(self):
        s = Scenario(50)
        s.ball_still(0, 25, (0.5, 0.5))
        s.ball_path(25, 50, (0.5, 0.5), (0.6, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        self.assertEqual(speed[20], 0.0)
        # At the transition, the smaller (stationary) direction wins, not the
        # fast forward direction.
        self.assertEqual(speed[25], 0.0)


class MissingBallTests(unittest.TestCase):
    def test_short_gap_carries_possession_without_inventing_ball(self):
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        s.ball_missing(40, 90)
        match = s.build()
        result, trace = predict_with_trace(match)
        self.assertEqual(labels(result), "H" * 150)
        self.assertTrue(np.isnan(trace.evidence.ball_xy_m[40:90]).all())   # nothing interpolated
        self.assertTrue(np.all(trace.evidence.control[40:90] == NONE))
        self.assertTrue(np.all(trace.evidence.why[40:90] == WHY_BALL_MISSING))
        self.assertTrue(np.all(trace.reason[40:90] == "carried_ball_missing"))
        self.assertTrue(np.all(np.diff(result.confidence[40:90]) < 0))     # trust decays

    def test_long_gap_is_never_dead_ball_by_itself(self):
        s = Scenario(500)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 450)
        s.dribble(450, 500, "H", 1)
        result, trace = run(s)
        self.assertNotIn("D", labels(result))
        self.assertTrue(np.all(trace.dead_reason == LIVE))
        self.assertEqual(labels(result), "H" * 500)

    def test_possession_change_inside_occlusion_lands_on_reappearance(self):
        s = Scenario(200)
        s.dribble(0, 60, "H", 0)
        s.ball_missing(60, 120)
        s.dribble(120, 200, "A", 0)
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 120 + "A" * 80)


class TemporalTests(unittest.TestCase):
    def test_handover_at_onset_of_controlled_possession(self):
        s = Scenario(120)
        s.dribble(0, 40, "H", 0)
        s.ball_path(40, 60, s.ball[39], s.near("A", 0, 60))
        s.dribble(60, 120, "A", 0)
        result, trace = run(s)
        lab = labels(result)
        change = lab.index("A")
        # The ball is within control radius one frame before the receiver's
        # first dribble frame; a one-frame offset is free on spellF1.
        self.assertIn(change, (59, 60))
        self.assertEqual(lab, "H" * change + "A" * (120 - change))
        self.assertEqual(trace.reason[change], "controlled")

    def test_brief_touch_is_not_a_change(self):
        s = Scenario(120)
        s.dribble(0, 120, "H", 0)
        s.ball_speed[:] = 3.0                         # source speed: the touch is real control evidence
        for t in range(50, 55):                       # 5 frames < confirm_frames
            s.ball[t] = s.near("A", 0, t)
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[50:55] == AWAY))
        self.assertEqual(labels(result), "H" * 120)

    def test_confirm_threshold_is_respected(self):
        for frames, expected in ((9, "H" * 120), (10, "H" * 50 + "A" * 10 + "H" * 60)):
            with self.subTest(frames=frames):
                s = Scenario(120)
                s.dribble(0, 120, "H", 0)
                s.ball_speed[:] = 3.0                 # isolate the threshold from speed edges
                for t in range(50, 50 + frames):
                    s.ball[t] = s.near("A", 0, t)
                result, _ = run(s)
                self.assertEqual(labels(result), expected)

    def test_team_passes_keep_one_possession(self):
        s = Scenario(160)
        s.dribble(0, 40, "H", 0)
        s.ball_path(40, 70, s.ball[39], s.near("H", 1, 70))
        s.dribble(70, 160, "H", 1)
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 160)

    def test_segments_reset_state_and_outside_is_abstain(self):
        s = Scenario(200, segments=((20, 90, 1), (110, 180, 2)))
        s.dribble(20, 90, "H", 0)
        s.ball_missing(90, 200)
        result, trace = run(s)
        lab = labels(result)
        self.assertEqual(lab[:20] + lab[90:110] + lab[180:], "X" * 60)
        self.assertEqual(lab[20:90], "H" * 70)
        self.assertEqual(lab[110:180], "X" * 70)      # no carry across halftime
        self.assertTrue(np.isnan(result.confidence[110:180]).all())
        self.assertTrue(np.all(trace.reason[:20] == "outside_segment"))

    def test_no_evidence_abstains(self):
        s = Scenario(80)
        s.ball_missing(0, 80)
        result, trace = run(s)
        self.assertEqual(labels(result), "X" * 80)
        self.assertTrue(np.all(trace.reason == "abstain_no_evidence"))


class DeadBallTests(unittest.TestCase):
    def build_out_and_restart(self):
        s = Scenario(260)
        s.dribble(0, 50, "H", 0)
        out_point = (s.ball[49][0], 1.05)               # 3.4 m beyond the touchline
        s.ball_path(50, 60, s.ball[49], out_point)
        s.ball_still(60, 100, out_point)                # observed out, rolling to a stop
        s.ball_missing(100, 150)                        # retrieved, not visible
        spot = (0.6, 0.99)
        s.ball_still(150, 180, spot)                    # placed inside, taker beside it
        s.player_at(150, 260, "A", 0, (spot[0] - 0.5 / L, spot[1]))
        s.ball_path(180, 200, spot, s.near("A", 1, 200))
        s.dribble(200, 260, "A", 1)
        return s

    def test_out_of_play_hold_and_restart(self):
        result, trace = run(self.build_out_and_restart())
        lab = labels(result)
        first_out = int(np.flatnonzero(trace.dead_reason == DEAD_OUT)[0])
        self.assertTrue(55 <= first_out <= 60, first_out)
        self.assertEqual(lab[:50], "H" * 50)
        self.assertEqual(lab[60:180], "D" * 120)
        self.assertTrue(np.all(trace.dead_reason[100:150] == DEAD_HOLD))   # held through missing ball
        self.assertTrue(np.all(trace.dead_reason[150:180] == DEAD_HOLD))   # placed ball still dead
        restart = lab.index("A")
        self.assertTrue(180 <= restart <= 182, restart)
        self.assertEqual(lab[restart:], "A" * (260 - restart))
        self.assertIn(trace.reason[restart], ("backfilled_restart", "controlled"))
        self.assertTrue(np.all(result.confidence[60:180] > 0))

    def test_brief_out_blip_is_noise(self):
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        s.ball[70:75] = (0.5, 1.2)
        result, trace = run(s)
        self.assertNotIn("D", labels(result))
        self.assertTrue(np.all(trace.dead_reason == LIVE))

    def test_stationary_unattended_ball(self):
        s = Scenario(260)
        s.dribble(0, 60, "H", 0)
        s.ball_path(60, 70, s.ball[59], (0.5, 0.5))
        s.ball_still(70, 180, (0.5, 0.5))              # nobody within the control radius
        s.player_at(180, 260, "A", 1, (0.5 - 0.5 / L, 0.5))
        s.ball_path(180, 200, (0.5, 0.5), (0.55, 0.5))
        s.player_at(180, 200, "A", 1, s.ball[180:200] - [0.5 / L, 0])
        s.dribble(200, 260, "A", 1)
        result, trace = run(s)
        lab = labels(result)
        self.assertTrue(np.all(trace.dead_reason[72:178] == DEAD_STATIONARY))
        self.assertEqual(lab[:60], "H" * 60)
        self.assertEqual(lab[200:], "A" * 60)

    def build_set_piece(self, team="A", restart_control=None):
        s = Scenario(300)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 300)
        s.event(250, 260, team=team, Type="Corner Kick")
        if restart_control:
            s.dribble(255, 300, restart_control, 1)
        return s

    def test_set_piece_extends_dead_back_and_supplies_prior(self):
        result, trace = run(self.build_set_piece())
        lab = labels(result)
        # S2: the last observed frame before the same-half tracking gap has no
        # bilateral speed support, so it is not live-motion evidence.
        self.assertEqual(lab, "H" * 99 + "D" * 151 + "X" * 10 + "A" * 40)
        self.assertTrue(np.all(trace.dead_reason[100:250] == DEAD_SET_PIECE))
        self.assertTrue(np.all(trace.dead_reason[250:260] == DEAD_UNCONFIRMED))
        self.assertTrue(np.all(trace.reason[250:260] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[260:] == "set_piece_team_prior"))

    def test_set_piece_prior_can_be_disabled(self):
        result, trace = run(self.build_set_piece(), EngineConfig(use_set_piece_team_prior=False))
        self.assertEqual(labels(result)[250:], "X" * 50)
        self.assertTrue(np.isnan(result.confidence[250:]).all())
        self.assertTrue(np.all(trace.reason[250:260] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[260:] == "abstain_no_evidence"))

    def test_throw_in_without_team_has_no_prior(self):
        result, trace = run(self.build_set_piece(team=None))
        self.assertEqual(labels(result)[250:], "X" * 50)
        self.assertTrue(np.all(trace.reason[250:260] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[260:] == "abstain_no_evidence"))

    def test_tracking_control_overrides_set_piece_team(self):
        result, trace = run(self.build_set_piece(team="A", restart_control="H"))
        # Restart = first frame with bilateral moving-speed support (256), not
        # merely the first observed frame after the gap (255).
        self.assertEqual(trace.restarts[0][0], 256)
        self.assertEqual(labels(result)[100:], "D" * 156 + "H" * 44)

    def test_set_piece_lead_in_stops_at_live_ball(self):
        s = Scenario(300)
        s.dribble(0, 200, "H", 0)                     # ball observed up to frame 199
        s.ball_missing(200, 300)
        s.event(250, 260, team="H")
        result, trace = run(s)
        # Movement before the window is live play, not the restart. Frame 199
        # is observed, but S2 gives it NaN derived speed because its forward
        # counterpart is unobserved in the same half.
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(trace.restarts[0][0], 260)
        np.testing.assert_array_equal(np.flatnonzero(trace.dead_reason != LIVE), np.arange(199, 260))
        self.assertEqual(labels(result), "H" * 199 + "D" * 51 + "X" * 10 + "H" * 40)


class OutputContractTests(unittest.TestCase):
    def test_result_metadata_and_determinism(self):
        sc = DeadBallTests().build_out_and_restart()
        match = sc.build()
        a, ta = predict_with_trace(match)
        b, tb = predict_with_trace(match)
        self.assertEqual(a.engine_version, ENGINE_VERSION)
        self.assertEqual(a.config_digest, EngineConfig().digest())
        np.testing.assert_array_equal(a.label, b.label)
        np.testing.assert_array_equal(a.confidence, b.confidence)
        np.testing.assert_array_equal(ta.reason, tb.reason)
        self.assertEqual(ta.restart_kinds, tb.restart_kinds)
        self.assertIsNone(a.reason)                     # contract v1 vocabulary cannot carry engine reasons
        self.assertTrue(set(ta.reason) <= set(REASONS))
        for arr in (a.label, a.confidence, ta.reason, ta.dead_reason, ta.evidence.control):
            self.assertFalse(arr.flags.writeable)
        a.validate_for(match)

    def test_config_digest_and_validation(self):
        self.assertEqual(EngineConfig().digest(), EngineConfig().digest())
        self.assertNotEqual(EngineConfig().digest(), EngineConfig(confirm_frames=11).digest())
        for bad in ({"confirm_frames": 0}, {"control_radius_m": -1}, {"control_radius_m": float("nan")},
                    {"run_gap_frames": True}, {"set_piece_codes": ("set pieces",)},
                    {"use_set_piece_team_prior": 1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EngineConfig(**bad)
        with self.assertRaises(TypeError):
            predict(make_match(), config={"confirm_frames": 3})
        with self.assertRaises(TypeError):
            predict("not a match")

    def test_foundation_fixture_and_official_export(self):
        match = make_match(40)
        result = predict(match)
        self.assertEqual(len(result.label), 40)
        root = ROOT / "artifacts" / "test-tmp"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            receipt = export_submission(result, match, Path(tmp) / "engine.csv")
        self.assertTrue(receipt.validated)
        self.assertEqual(receipt.rows, 40)

    def test_engine_source_is_isolated(self):
        """Engine code must not read files, assets, GT or the network."""
        banned = re.compile(r"\bopen\(|\bimport os\b|pathlib|games|evaluation|urllib|requests|socket|"
                            r"random|read_csv|np\.load|loadtxt|gt_public|subprocess")
        for path in (ROOT / "src" / "mtah" / "engine").glob("*.py"):
            text = "\n".join(line for line in path.read_text(encoding="utf-8").splitlines()
                             if not line.lstrip().startswith("#"))
            code = re.sub(r'"""[\s\S]*?"""', "", text)
            with self.subTest(path=path.name):
                self.assertIsNone(banned.search(code))


if __name__ == "__main__":
    unittest.main()
