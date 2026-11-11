"""P1-B.2 robustness audit: invented edge-case scenarios only.

No competition assets, public ground truth, video or scores are read. Expected
labels follow from scenario construction and the documented engine rules.
The three limitations recorded as expectedFailure in P1-B.2 were resolved in
P1-B.3; DeadBallReleaseRegressionTests holds them as ordinary regression tests
together with counterexamples against premature restart.
"""
from dataclasses import replace
import unittest
import warnings

import numpy as np

from mtah.engine import EngineConfig, predict, predict_with_trace
from mtah.engine.deadball import DEAD_SET_PIECE, DEAD_UNCONFIRMED, LIVE
from mtah.engine.evidence import AWAY, CONTESTED, HOME, NONE, derived_ball_speed
from scenarios import L, Scenario


def labels(result):
    return "".join(result.label)


def run(scenario, config=None):
    return predict_with_trace(scenario.build(), config)


def assert_valid_output(test, result):
    """Contract-level sanity that every scenario must satisfy."""
    conf, lab = result.confidence, result.label
    test.assertTrue(np.isnan(conf[lab == "X"]).all())
    finite = conf[lab != "X"]
    test.assertTrue(np.isfinite(finite).all())
    test.assertTrue(((finite >= 0) & (finite <= 1)).all())


class LongMissingBallTests(unittest.TestCase):
    def test_ten_minute_gap_inside_possession(self):
        n = 15_100
        s = Scenario(n)
        s.dribble(0, 50, "H", 0)
        s.ball_missing(50, n - 50)
        s.dribble(n - 50, n, "H", 1)
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * n)
        self.assertNotIn("D", labels(result))
        self.assertTrue(np.all(trace.dead_reason == LIVE))
        gap = result.confidence[50:n - 50]
        self.assertTrue(np.all(np.diff(gap) <= 0))
        self.assertGreaterEqual(gap.min(), 0.35)   # floor, no underflow below it
        assert_valid_output(self, result)

    def test_whole_half_missing_is_abstain_not_dead(self):
        s = Scenario(1000, segments=((0, 500, 1), (500, 1000, 2)))
        s.dribble(0, 500, "H", 0)
        s.ball_missing(500, 1000)
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * 500 + "X" * 500)
        self.assertTrue(np.all(trace.reason[500:] == "abstain_no_evidence"))

    def test_long_gap_at_segment_start_backfills_first_possessor(self):
        # Documented rule: a live stretch starts with the first team to establish
        # control (restart possessor), however long the leading gap.
        s = Scenario(3100)
        s.ball_missing(0, 3000)
        s.dribble(3000, 3100, "A", 0)
        result, trace = run(s)
        self.assertEqual(labels(result), "A" * 3100)
        self.assertTrue(np.all(trace.reason[:3000] == "backfilled_restart"))
        np.testing.assert_allclose(result.confidence[:3000], 0.5)

    def test_long_gap_then_other_team(self):
        s = Scenario(10_100)
        s.dribble(0, 50, "H", 0)
        s.ball_missing(50, 10_050)
        s.dribble(10_050, 10_100, "A", 0)
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 10_050 + "A" * 50)

    def test_sparse_ball_observations_still_give_control(self):
        # Ball seen only every third frame. Since P2-B.2 the bilateral fallback
        # derives the true 3 m/s from observations 3 frames apart on both sides;
        # the end frames have one side only and stay NaN. Control is unaffected.
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        for t in range(150):
            if t % 3:
                s.ball[t] = np.nan
        result, trace = run(s)
        speed = trace.evidence.ball_speed_mps
        np.testing.assert_allclose(speed[3:147:3], 3.0, rtol=1e-9)
        self.assertTrue(np.isnan(speed[0]) and np.isnan(speed[147]))
        self.assertTrue(np.all(trace.evidence.control[::3] == HOME))
        self.assertEqual(labels(result), "H" * 150)


class RapidAlternationTests(unittest.TestCase):
    def alternating(self, block, start=50, stop=150, n=200, lead="H"):
        s = Scenario(n)
        s.dribble(0, n, "H", 0)
        s.ball_speed[:] = 3.0      # source speed: every touch is real control evidence
        if lead is None:
            s.ball_missing(0, start)
        for t in range(start, stop):
            if ((t - start) // block) % 2 == 0:
                s.ball[t] = s.near("A", 0, t)
        return s

    def test_frame_by_frame_alternation_keeps_possession(self):
        result, trace = run(self.alternating(1))
        self.assertTrue(np.all(trace.evidence.control[50:150:2] == AWAY))
        self.assertTrue(np.all(trace.evidence.control[51:150:2] == HOME))
        self.assertEqual(labels(result), "H" * 200)

    def test_sub_threshold_blocks_never_change_possession(self):
        result, _ = run(self.alternating(9))
        self.assertEqual(labels(result), "H" * 200)

    def test_threshold_blocks_change_every_block(self):
        # Ten-frame blocks are confirmed by rule, so possession flips per block.
        result, _ = run(self.alternating(10))
        self.assertEqual(labels(result), "H" * 50 + ("A" * 10 + "H" * 10) * 5 + "H" * 50)

    def test_alternation_without_prior_possession_abstains(self):
        s = self.alternating(1, start=0, stop=100, n=100)
        result, _ = run(s)
        self.assertEqual(labels(result), "X" * 100)

    def test_contest_oscillation_then_confirmed_takeover(self):
        s = Scenario(160)
        s.dribble(0, 160, "H", 0)
        s.ball_speed[:] = 3.0
        for t in range(40, 80):                     # alternate contested / home
            if t % 2 == 0:
                s.player_at(t, t + 1, "A", 0, s.ball[t] + [0.5 / L, 0])
                s.player_at(t, t + 1, "H", 0, s.ball[t] - [0.5 / L, 0])
        for t in range(100, 160):
            s.ball[t] = s.near("A", 0, t)
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[40:80:2] == CONTESTED))
        self.assertEqual(labels(result), "H" * 100 + "A" * 60)


class TrackQualityTests(unittest.TestCase):
    def test_controller_track_lost_with_far_opponent(self):
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        s.player_missing(40, 90, "H", 0)
        s.player_at(40, 90, "A", 0, s.ball[60] + [0, 3.0 / 68.0])   # 3 m to the side of the run
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[40:90] == NONE))
        self.assertEqual(labels(result), "H" * 150)

    def test_controller_track_lost_brief_opponent_proximity(self):
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        s.player_missing(40, 60, "H", 0)
        for t in range(40, 49):                     # 9 frames < confirm_frames
            s.player_at(t, t + 1, "A", 0, s.ball[t] + [1.0 / L, 0])
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 150)

    def test_cross_team_duplicate_track_is_contested_not_turnover(self):
        s = Scenario(150)
        s.dribble(0, 150, "H", 0)
        # Tracker duplicates the home carrier into an away slot.
        s.players[40:80, s.slot("A", 1)] = s.players[40:80, s.slot("H", 0)]
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[40:80] == CONTESTED))
        self.assertEqual(labels(result), "H" * 150)

    def test_all_players_lost_while_ball_moves(self):
        s = Scenario(200)
        s.dribble(0, 200, "H", 0)
        s.players[60:160] = np.nan
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 200)

    def test_forty_six_slots_mostly_empty(self):
        s = Scenario(120, extra_home=21, extra_away=21)
        s.players[:, 4:] = np.nan
        s.players[30:60, 10] = s.players[30:60, s.slot("H", 0)]   # one stray duplicate
        s.dribble(0, 120, "H", 0)
        result, _ = run(s)
        self.assertEqual(len(s.teams), 46)
        self.assertEqual(labels(result), "H" * 120)

    def test_match_with_no_away_slots(self):
        s = Scenario(80)
        s.teams = ["H"] * 4
        s.dribble(0, 80, "H", 0)
        result, trace = run(s)
        self.assertTrue(np.isinf(trace.evidence.d_away_m).all())
        self.assertEqual(labels(result), "H" * 80)

    def build_mixed(self):
        s = Scenario(300)
        s.dribble(0, 60, "H", 0)
        s.ball_path(60, 80, s.ball[59], s.near("A", 0, 80))
        s.dribble(80, 160, "A", 0)
        s.ball_missing(160, 200)
        s.dribble(200, 300, "H", 1)
        return s

    def test_slot_order_does_not_matter(self):
        match = self.build_mixed().build()
        perm = np.array([3, 1, 0, 2])
        shuffled = replace(match, slot_id=match.slot_id[perm], slot_team=match.slot_team[perm],
                           player_xy=match.player_xy[:, perm], player_observed=match.player_observed[:, perm],
                           player_speed_mps=match.player_speed_mps[:, perm])
        a, b = predict(match), predict(shuffled)
        np.testing.assert_array_equal(a.label, b.label)
        np.testing.assert_array_equal(a.confidence, b.confidence)

    def test_team_swap_mirrors_labels(self):
        match = self.build_mixed().build()
        swapped = replace(match, slot_team=np.where(match.slot_team == "H", "A", "H"))
        a, b = labels(predict(match)), labels(predict(swapped))
        self.assertEqual(b, a.translate(str.maketrans("HA", "AH")))


class HalfTimeBoundaryTests(unittest.TestCase):
    def test_adjacent_segments_reset_without_gap(self):
        s = Scenario(200, segments=((0, 100, 1), (100, 200, 2)))
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 200)
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * 100 + "X" * 100)

    def test_run_split_by_boundary_is_not_confirmed(self):
        s = Scenario(200, segments=((0, 100, 1), (100, 200, 2)))
        s.dribble(0, 90, "H", 0)
        s.ball_speed[:] = 3.0
        s.ball_missing(90, 200)
        for t in range(95, 105):                    # 5 + 5 frames either side of the reset
            s.ball[t] = s.near("A", 0, t)
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 100 + "X" * 100)

    def test_speed_is_not_derived_across_the_boundary(self):
        s = Scenario(200, segments=((0, 100, 1), (100, 200, 2)))
        s.ball_path(0, 100, (0.1, 0.5), (0.2, 0.5))
        s.ball_path(100, 200, (0.9, 0.5), (0.8, 0.5))   # teleport at the reset
        _, trace = run(s)
        speed = trace.evidence.ball_speed_mps
        self.assertLess(speed[99], 10)
        self.assertLess(speed[100], 10)

    def test_dead_hold_does_not_cross_halftime(self):
        s = Scenario(260, segments=((0, 100, 1), (160, 260, 2)))
        s.dribble(0, 50, "H", 0)
        s.ball_path(50, 60, s.ball[49], (s.ball[49][0], 1.05))
        s.ball_still(60, 160, (0.4, 1.05))           # still out at the half's end and beyond
        s.dribble(160, 260, "A", 0)
        result, trace = run(s)
        lab = labels(result)
        self.assertEqual(lab[60:100], "D" * 40)
        self.assertEqual(lab[100:160], "X" * 60)
        self.assertEqual(lab[160:], "A" * 100)
        self.assertTrue(np.all(trace.dead_reason[100:] == LIVE))

    def test_set_piece_lead_in_does_not_reach_previous_half(self):
        s = Scenario(200, segments=((0, 100, 1), (100, 200, 2)))
        s.dribble(0, 90, "H", 0)
        s.ball_missing(90, 130)
        s.dribble(130, 200, "H", 1)
        s.event(102, 110, team="A")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("ignored",))
        self.assertEqual(trace.restarts, ())
        self.assertEqual(labels(result), "H" * 200)

    def test_set_piece_in_halftime_gap_is_ignored(self):
        s = Scenario(250, segments=((0, 100, 1), (150, 250, 2)))
        s.dribble(0, 100, "H", 0)
        s.dribble(150, 250, "H", 1)
        s.event(120, 130, team="A")
        result, trace = run(s)
        self.assertEqual(trace.restarts, ())
        self.assertEqual(labels(result), "H" * 100 + "X" * 50 + "H" * 100)

    def test_no_dead_or_possession_outside_segments(self):
        s = Scenario(300, segments=((50, 150, 1), (200, 250, 2)))
        s.dribble(0, 300, "H", 0)
        s.ball_still(150, 200, (0.5, 1.1))
        s.event(10, 20, team="A")
        s.event(170, 180, team="A")
        result, _ = run(s)
        lab = labels(result)
        outside = lab[:50] + lab[150:200] + lab[250:]
        self.assertEqual(outside, "X" * len(outside))


class SetPieceTransitionTests(unittest.TestCase):
    def test_tagged_restart_ends_out_of_play_hold(self):
        s = Scenario(300)
        s.dribble(0, 50, "H", 0)
        s.ball_path(50, 60, s.ball[49], (s.ball[49][0], 1.05))
        s.ball_still(60, 100, (0.4, 1.05))
        s.ball_missing(100, 300)
        s.event(200, 210, team="A")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(trace.restarts[0][0], 210)
        self.assertEqual(labels(result), "H" * 60 + "D" * 140 + "X" * 10 + "A" * 90)

    def test_false_positive_tag_in_live_play_changes_nothing(self):
        s = Scenario(200)
        s.dribble(0, 200, "H", 0)
        s.event(100, 110, team="A")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("ignored",))
        self.assertEqual(trace.restarts, ())
        self.assertTrue(np.all(trace.dead_reason == LIVE))
        self.assertEqual(labels(result), "H" * 200)

    def test_back_to_back_set_pieces_use_the_restart_that_opens_play(self):
        s = Scenario(400)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 400)
        s.event(250, 260, team="H")                 # first tag
        s.event(300, 310, team="A")                 # retaken / second restart opens play
        result, trace = run(s)
        self.assertTrue(np.all(trace.dead_reason[100:250] == DEAD_SET_PIECE))
        self.assertTrue(np.all(trace.dead_reason[250:260] == DEAD_UNCONFIRMED))
        self.assertTrue(np.all(trace.dead_reason[260:300] == DEAD_SET_PIECE))
        self.assertTrue(np.all(trace.dead_reason[300:310] == DEAD_UNCONFIRMED))
        self.assertEqual(labels(result), "H" * 99 + "D" * 151 + "X" * 10 + "D" * 40 + "X" * 10 + "A" * 90)

    def test_nearest_untagged_restart_gives_no_prior(self):
        s = Scenario(400)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 400)
        s.event(260, 270, team="H")
        s.event(300, 310, team=None)                # throw-in, Team N/A, opens play
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 99 + "D" * 161 + "X" * 10 + "D" * 30 + "X" * 100)

    def test_tag_team_then_opponent_wins_ball(self):
        s = Scenario(300)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 200)
        s.event(198, 205, team="H")
        s.ball_path(200, 210, (0.45, 0.5), s.near("A", 0, 210))
        s.dribble(210, 300, "A", 0)
        result, _ = run(s)
        lab = labels(result)
        self.assertEqual(lab, "H" * 99 + "D" * 102 + "A" * 99)

    def test_set_piece_at_segment_end(self):
        s = Scenario(200)
        s.dribble(0, 150, "H", 0)
        s.ball_missing(150, 200)
        s.event(195, 200, team="A")
        result, _ = run(s)
        self.assertEqual(labels(result), "H" * 149 + "D" * 46 + "X" * 5)

    def test_overlapping_set_piece_events_are_deterministic(self):
        def build():
            s = Scenario(300)
            s.dribble(0, 100, "H", 0)
            s.ball_missing(100, 300)
            s.event(250, 280, team="A", Type="Free Kick")
            s.event(250, 260, team="A", Type="Free Kick")
            return s
        a, b = run(build())[0], run(build())[0]
        self.assertEqual(labels(a), labels(b))
        self.assertEqual(labels(a), "H" * 99 + "D" * 151 + "X" * 30 + "A" * 20)


class InvalidAndIncompleteInputTests(unittest.TestCase):
    def test_config_rejects_string_codes(self):
        for bad in ("SET PIECES", "SETPIECES", ("SET PIECES", 3), ("",)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EngineConfig(set_piece_codes=bad)
        self.assertEqual(EngineConfig(set_piece_codes=["SET PIECES"]).set_piece_codes, ("SET PIECES",))

    def test_config_rejects_non_finite_and_wrong_types(self):
        for bad in ({"out_margin_m": float("inf")}, {"restart_speed_mps": "1"}, {"confirm_frames": 2.0},
                    {"stationary_frames": -5}, {"speed_lag_frames": 0}, {"contest_margin_m": None}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                EngineConfig(**bad)

    def test_contract_rejects_corrupt_ball_features(self):
        match = Scenario(20).build()
        ball = match.ball_xy.copy()
        cases = []
        inf_ball = ball.copy(); inf_ball[3, 0] = np.inf
        cases.append({"ball_xy": inf_ball})
        partial = ball.copy(); partial[4, 1] = np.nan
        cases.append({"ball_xy": partial})
        cases.append({"ball_observed": ~match.ball_observed})
        neg = match.ball_speed_mps.copy(); neg[2] = -1.0
        cases.append({"ball_speed_mps": neg})
        cases.append({"player_xy": match.player_xy[:, :, :1]})
        cases.append({"ball_xy": ball.astype(np.float32)})
        for change in cases:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                replace(match, **change)

    def test_single_frame_and_tiny_inputs(self):
        for n in (1, 2, 4, 6):
            with self.subTest(n=n):
                s = Scenario(n)
                s.dribble(0, n, "H", 0)
                result, _ = run(s)
                self.assertEqual(len(result.label), n)
                assert_valid_output(self, result)

    def test_no_segments_means_all_abstain(self):
        s = Scenario(50, segments=())
        s.dribble(0, 50, "H", 0)
        result, trace = run(s)
        self.assertEqual(labels(result), "X" * 50)
        self.assertTrue(np.all(trace.reason == "outside_segment"))

    def test_source_speed_without_ball_is_ignored(self):
        s = Scenario(120)
        s.dribble(0, 120, "H", 0)
        s.ball_missing(40, 80)
        s.ball_speed[40:80] = 30.0
        result, trace = run(s)
        self.assertTrue(np.all(trace.evidence.control[40:80] == NONE))
        self.assertEqual(labels(result), "H" * 120)

    def test_extreme_finite_coordinates_without_warnings(self):
        s = Scenario(200)
        s.dribble(0, 50, "H", 0)
        s.ball_still(50, 150, (50.0, -40.0))        # absurd but finite tracker output
        s.players[60:70, 1] = (1e6, -1e6)
        s.dribble(150, 200, "A", 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result, _ = run(s)
        self.assertEqual(labels(result)[50:150], "D" * 100)
        assert_valid_output(self, result)

    def test_off_pitch_player_near_out_ball_does_not_revive_play(self):
        s = Scenario(200)
        s.dribble(0, 50, "H", 0)
        s.ball_still(50, 150, (0.5, 1.05))
        s.player_at(50, 150, "A", 1, (0.5 - 0.5 / L, 1.05))   # bench/staff beside the ball
        s.dribble(150, 200, "H", 0)
        result, _ = run(s)
        self.assertEqual(labels(result)[50:150], "D" * 100)

    def test_engine_does_not_mutate_input(self):
        match = TrackQualityTests().build_mixed().build()
        before = {k: getattr(match, k).copy() for k in ("player_xy", "ball_xy", "ball_speed_mps", "player_observed")}
        predict(match)
        predict(match, EngineConfig(confirm_frames=3))
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(match, key), value)


def sparse(s, start, stop, keep_every=3):
    """Ball observed only on every `keep_every`-th frame: no lag-1/lag-5 speed exists."""
    for t in range(start, stop):
        if t % keep_every:
            s.ball[t] = np.nan


class DeadBallReleaseRegressionTests(unittest.TestCase):
    """P1-B.2 limitations resolved in P1-B.3, each with counterexamples guarding
    against premature restart or false dead-ball termination."""

    def out_then_play(self):
        s = Scenario(300)
        s.dribble(0, 50, "H", 0)
        s.ball_path(50, 60, s.ball[49], (s.ball[49][0], 1.05))
        s.ball_still(60, 100, (0.4, 1.05))
        s.dribble(100, 300, "A", 0)
        return s

    def out_then_placed(self):
        """Out at ~60, retrieved (missing) 100-120, placed inside with the taker
        beside it 120-220, kicked at 220 to a teammate who dribbles on."""
        s = Scenario(320)
        s.dribble(0, 50, "H", 0)
        s.ball_path(50, 60, s.ball[49], (s.ball[49][0], 1.05))
        s.ball_still(60, 100, (0.4, 1.05))
        s.ball_missing(100, 120)
        spot = (0.4, 0.99)
        s.ball_still(120, 220, spot)
        s.player_at(120, 320, "A", 0, (spot[0] - 0.5 / L, spot[1]))
        s.ball_path(220, 240, spot, s.near("A", 1, 240))
        s.dribble(240, 320, "A", 1)
        return s

    # --- (a) speed unknown -------------------------------------------------
    def test_hold_releases_when_speed_is_unknown(self):
        s = self.out_then_play()
        s.ball_missing(100, 110)                    # no lag may reach back to the out frames
        sparse(s, 110, 300)
        self.assertEqual(labels(run(s)[0])[150:], "A" * 150)

    def test_sparse_placed_ball_waits_for_the_kick(self):
        s = self.out_then_placed()
        sparse(s, 120, 320)
        lab = labels(run(s)[0])
        self.assertEqual(lab[60:220], "D" * 160)    # taker beside a still ball: no restart
        self.assertEqual(lab[225:], "A" * 95)

    def test_sparse_stoppage_never_compares_across_an_out_observation(self):
        s = Scenario(300)
        s.dribble(0, 50, "H", 0)
        s.ball_still(50, 60, (0.4, 1.05))           # seen out
        s.ball_missing(60, 150)
        s.ball_still(150, 300, (0.48, 0.95))        # placed ~8 m away, never kicked
        s.player_at(150, 300, "A", 0, (0.48 - 0.5 / L, 0.95))
        sparse(s, 150, 300)
        self.assertEqual(labels(run(s)[0])[50:], "D" * 250)

    def test_sparse_jitter_on_a_placed_ball_is_not_a_restart(self):
        # Every other frame observed, +/-5 cm alternating jitter: adjacent
        # observations differ by 10 cm in 2 frames (1.25 m/s), but observations
        # at least speed_lag_frames apart give 10 cm / 6 frames (0.42 m/s).
        s = self.out_then_placed()
        s.ball_path(220, 240, (0.4, 0.99), (0.4, 0.99))   # no kick: placed ball stays put
        s.ball_still(240, 320, (0.4, 0.99))
        for t in range(120, 320):
            if t % 2:
                s.ball[t] = np.nan
            else:
                s.ball[t, 0] += (0.05 if t % 4 == 0 else -0.05) / L
        self.assertEqual(labels(run(s)[0])[60:], "D" * 260)

    def test_sparse_comparison_restarts_after_a_second_out_observation(self):
        s = Scenario(400)
        s.dribble(0, 50, "H", 0)
        # Retrievals are unobserved gaps: an out frame followed directly by an
        # in-pitch frame would be indistinguishable from a throw-in (a restart).
        s.ball_still(50, 60, (0.4, 1.05))           # out
        s.ball_missing(60, 70)
        s.ball_still(70, 140, (0.40, 0.99))         # placed at spot A
        s.ball_missing(140, 150)                    # picked up again, unobserved (not played)
        s.ball_still(150, 160, (0.45, 1.05))        # seen out again
        s.ball_missing(160, 170)
        s.ball_still(170, 400, (0.48, 0.99))        # placed at spot B, ~8 m from A, never kicked
        s.player_at(60, 400, "A", 0, (0.48 - 0.5 / L, 0.99))
        sparse(s, 60, 150)
        sparse(s, 160, 400)
        self.assertEqual(labels(run(s)[0])[50:], "D" * 350)

    # --- (b) source speed stuck at zero ------------------------------------
    def test_hold_releases_when_source_speed_is_stuck_at_zero(self):
        s = self.out_then_play()
        s.ball_speed[:] = 0.0
        self.assertEqual(labels(run(s)[0])[150:], "A" * 150)

    def test_stuck_zero_source_placed_ball_waits_for_the_kick(self):
        s = self.out_then_placed()
        s.ball_speed[:] = 0.0
        lab = labels(run(s)[0])
        self.assertEqual(lab[60:220], "D" * 160)
        self.assertEqual(lab[225:], "A" * 95)

    def test_stuck_zero_source_does_not_make_a_rolling_ball_stationary(self):
        s = Scenario(250)
        s.dribble(0, 60, "H", 0)
        start = s.ball[59].copy()
        s.ball_path(60, 250, start, start + [190 * 2.0 / 25 / L, 0])   # rolls loose at 2 m/s
        s.ball_speed[:] = 0.0
        result, trace = run(s)
        self.assertTrue(np.all(trace.dead_reason == LIVE))
        self.assertEqual(labels(result), "H" * 250)

    # --- (c) set-piece lead-in over live possession --------------------------
    def test_set_piece_lead_in_spares_slow_controlled_play(self):
        s = Scenario(300)
        s.dribble(0, 250, "H", 0, speed_mps=0.5)    # shielding the ball, below restart speed
        s.ball_speed[:250] = 0.5
        s.ball_missing(250, 300)
        s.event(250, 260, team="H")
        self.assertNotIn("D", labels(run(s)[0])[:250])

    def test_set_piece_lead_in_still_covers_taker_beside_placed_ball(self):
        s = Scenario(340)
        s.dribble(0, 100, "H", 0)
        s.ball_missing(100, 150)
        spot = (0.995, 0.005)                        # corner arc
        s.ball_still(150, 245, spot)
        s.player_at(150, 340, "A", 0, (spot[0] - 0.5 / L, spot[1]))   # taker: control, ball still
        s.ball_path(245, 265, spot, s.near("A", 1, 265))
        s.dribble(265, 340, "A", 1)
        s.event(245, 255, team="A", Type="Corner Kick")
        result, trace = run(s)
        lab = labels(result)
        self.assertEqual(lab[100:245], "D" * 145)
        self.assertEqual(lab[250:], "A" * 90)


class S2GapEdgeTests(unittest.TestCase):
    """P2-B.2: S2 symmetric derived-speed evidence (3039d08) at gap edges.

    Invariant kept: a finite derived speed needs an observed frame on BOTH sides
    within speed_lag_frames in the same half; one-sided estimates survive only at
    timeline/segment edges. A single dropout beside a frame is not a gap edge.
    """

    def test_periodic_single_dropouts_keep_bilateral_speed(self):
        s = Scenario(60)
        s.ball_path(0, 60, (0.2, 0.5), (0.8, 0.5))     # 0.01 pitch lengths per frame
        for t in range(60):
            if t % 3 == 2:
                s.ball[t] = np.nan
        speed = derived_ball_speed(s.build(), EngineConfig())
        expected = 0.01 * L * 25                         # 26.325 m/s
        observed = [t for t in range(3, 57) if t % 3 != 2]
        np.testing.assert_allclose(speed[observed], expected, rtol=1e-9)

    def test_long_gap_reacquisition_stays_suppressed(self):
        s = Scenario(40)
        s.ball_path(0, 10, (0.2, 0.5), (0.3, 0.5))
        s.ball_missing(10, 16)                           # 6 frames > speed_lag_frames
        s.ball_path(16, 40, (0.5, 0.5), (0.6, 0.5))
        speed = derived_ball_speed(s.build(), EngineConfig())
        self.assertTrue(np.isnan(speed[16]))             # first frame after the gap
        self.assertTrue(np.isnan(speed[9]))              # last frame before it

    def test_periodic_dropouts_do_not_turn_live_play_dead_before_a_tag(self):
        s = Scenario(400)
        s.dribble(0, 250, "H", 0)                        # 3 m/s, ball seen 2 frames in 3
        for t in range(250):
            if t % 3 == 2:
                s.ball[t] = np.nan
        s.ball_missing(250, 400)
        # Tag window 280..380 holds no observation, so the restart is the tag start
        # and the lead-in runs back from 329 until it meets live evidence.
        s.event(330, 340, team="H")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(trace.restarts[0][0], 340)
        # 247 has observed frames at 246 and 249: bilateral 3 m/s with control -> live.
        # 249 is the last frame before a 150-frame gap: S2 gives no speed, so it
        # (and unobserved 248) joins the stoppage. Without bilateral fallback every
        # frame of the dribble is NaN and the lead-in reaches back to frame 80.
        np.testing.assert_array_equal(np.flatnonzero(trace.dead_reason != LIVE), np.arange(248, 340))
        self.assertEqual(labels(result), "H" * 248 + "D" * 82 + "X" * 10 + "H" * 60)

    def reacquired(self, dropout):
        s = Scenario(300)
        s.dribble(0, 50, "H", 0)
        s.ball_still(50, 100, (0.4, 1.05))               # seen out
        s.ball_missing(100, 150)                         # retrieval, unobserved
        s.dribble(150, 300, "A", 0)                      # reacquired at 150, moving
        if dropout:
            s.ball[151] = np.nan
        s.event(150, 160, team="A", Type="Throw-in")
        return run(s)

    def test_restart_one_frame_after_long_gap_reacquisition(self):
        result, trace = self.reacquired(dropout=False)
        # 150 has only forward evidence (S2): the restart is the first frame with
        # observed neighbours on both sides.
        self.assertEqual(trace.restarts[0][0], 151)
        self.assertEqual(labels(result), "H" * 50 + "D" * 101 + "A" * 149)

    def test_single_dropout_after_reacquisition_costs_no_extra_frame(self):
        result, trace = self.reacquired(dropout=True)
        # 151 is unobserved (held dead); 152 has observed 150 and 153 within the lag.
        self.assertEqual(trace.restarts[0][0], 152)
        self.assertEqual(labels(result), "H" * 50 + "D" * 102 + "A" * 148)

    def test_handover_reception_frame_is_not_in_flight(self):
        # The improvement S2 brought: a one-sided fast estimate at a reception
        # frame used to read as IN_FLIGHT and delay the new possession by a frame.
        s = Scenario(200)
        s.dribble(0, 100, "H", 0)
        s.dribble(100, 200, "A", 0)                      # 1-frame switch of carrier
        s.ball[105] = np.nan                             # removes the forward lag-5 pair at 100
        result, trace = run(s)
        self.assertEqual(trace.evidence.control[100], AWAY)
        self.assertEqual(labels(result), "H" * 100 + "A" * 100)


class TaggingWindowTests(unittest.TestCase):
    """P2 final review: Smart Tagging restart alignment.

    Public semantics (starter/load_smart_tagging.py): a set-piece tag marks a
    window of about 5 s AROUND the restart, verified by containment (the real
    kickoff lies inside [start_frame, end_frame)), not by start == restart.
    All tags below span [250, 375), i.e. 5 s.
    """

    TAG = (250, 375)

    def stoppage_then(self, n, reappear, extra=None):
        """Home dribble, ball seen out at 100-150, then unobserved until `reappear`,
        where the away carrier is first seen moving."""
        s = Scenario(n)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, reappear)
        s.dribble(reappear, n, "A", 0, speed_mps=1.5)            # stays inside the pitch
        s.event(*self.TAG, team="A", Type="Throw-in")
        return s

    # Guards: behaviour the current rule already gets right.
    def test_restart_near_tag_start(self):
        result, trace = run(self.stoppage_then(600, reappear=270))
        self.assertEqual(trace.restart_kinds, ("early",))
        self.assertEqual(trace.restarts[0][0], 271)              # 270 is the S2 reacquisition frame
        self.assertEqual(labels(result), "H" * 100 + "D" * 171 + "A" * 329)

    def test_false_positive_tag_in_live_play(self):
        s = Scenario(600)
        s.dribble(0, 600, "H", 0)
        s.event(*self.TAG, team="A")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("ignored",))
        self.assertEqual(trace.restarts, ())
        self.assertTrue(np.all(trace.dead_reason == LIVE))
        self.assertEqual(labels(result), "H" * 600)

    # Counterexamples: restart inside the tagged window but outside start +/- 2 s.
    def test_restart_late_inside_the_tag_window(self):
        # The ball is first seen moving at 325 (3 s after the tag start, inside the
        # 5 s window). The current rule finds no onset in [200, 300], falls back to
        # the tag start, ends the hold at 250 and back-fills A over 76 dead frames.
        result, trace = run(self.stoppage_then(600, reappear=325))
        self.assertEqual(trace.restart_kinds, ("late",))
        self.assertEqual(trace.restarts[0][0], 326)
        self.assertEqual(labels(result), "H" * 100 + "D" * 226 + "A" * 274)

    def test_restart_just_after_the_tag_window(self):
        # Tags are accurate to about a second (starter docstring): a restart seen at
        # end + 20 frames is still within the same tolerance used before the start.
        result, trace = run(self.stoppage_then(600, reappear=395))
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(trace.restarts[0][0], 375)
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 125 + "A" * 225)
        self.assertTrue(np.isnan(result.confidence[250:375]).all())
        self.assertTrue(np.all(trace.reason[250:375] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[375:395] == "backfilled_unconfirmed_restart"))
        np.testing.assert_allclose(result.confidence[375:395], 0.4)
        self.assertEqual(trace.reason[395], "controlled")

    def test_kick_inside_window_beats_a_placement_nudge_before_it(self):
        # Ball placed at 200, repositioned 1 m at 240-250 (2.5 m/s: a real movement
        # onset, but still the stoppage), kicked at 290. The current rule prefers the
        # onset nearest the tag START (the reposition, ~8 frames before it) over the
        # kick inside the window (~41 frames after it).
        s = Scenario(500)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 200)
        spot, moved = np.array([0.4, 0.99]), np.array([0.4 + 1.0 / L, 0.99])
        s.ball_still(200, 240, spot)
        s.ball_path(240, 250, spot, moved)
        s.ball_still(250, 290, moved)
        s.player_at(200, 250, "A", 0, spot - [0.5 / L, 0])     # taker beside the ball
        s.player_at(250, 500, "A", 0, moved - [0.5 / L, 0])
        s.ball_path(290, 310, moved, s.near("A", 1, 310))
        s.dribble(310, 500, "A", 1, speed_mps=1.5)
        s.event(*self.TAG, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("early",))
        self.assertIn(trace.restarts[0][0], (290, 291, 292))
        self.assertTrue(np.all(trace.dead_reason[251:290] != LIVE))   # still ball before the kick


    # T4-T7: No-onset and stoppage-gating cases.
    def test_no_onset_after_stoppage_abstains_in_window(self):
        s = Scenario(600)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 600)
        s.event(*self.TAG, team="A", Type="Throw-in")
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 125 + "A" * 225)
        self.assertTrue(np.isnan(result.confidence[250:375]).all())
        self.assertTrue(np.all(trace.reason[250:375] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[375:] == "set_piece_team_prior"))
        np.testing.assert_allclose(result.confidence[375:], 0.4)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(trace.restarts[0][0], 375)

    def test_no_onset_window_is_bounded_by_tag_end(self):
        result, trace = run(self.stoppage_then(600, reappear=475))
        self.assertTrue(np.all(trace.dead_reason[375:] == LIVE))
        self.assertEqual(labels(result)[250:375], "X" * 125)
        self.assertEqual(labels(result)[375:475], "A" * 100)
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 125 + "A" * 225)
        self.assertTrue(np.all(trace.reason[375:475] == "backfilled_unconfirmed_restart"))
        np.testing.assert_allclose(result.confidence[375:475], 0.4)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))

    def test_no_onset_without_stoppage_is_ignored(self):
        s = Scenario(600)
        s.dribble(0, 600, "H", 0)
        s.ball_missing(240, 400)
        s.event(*self.TAG, team="A")
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * 600)
        self.assertEqual(trace.restart_kinds, ("ignored",))
        self.assertEqual(trace.restarts, ())
        self.assertTrue(np.all(trace.dead_reason == LIVE))

    def test_late_onset_without_stoppage_is_ignored(self):
        s = Scenario(600)
        s.dribble(0, 600, "H", 0)
        s.ball_missing(240, 360)
        s.event(*self.TAG, team="A")
        result, trace = run(s)
        self.assertEqual(labels(result), "H" * 600)
        self.assertEqual(trace.restart_kinds, ("ignored",))
        self.assertEqual(trace.restarts, ())
        self.assertTrue(np.all(trace.dead_reason == LIVE))

    # Before-tag onset without an early onset.
    def test_before_tag_movement_onset_establishes_restart(self):
        result, trace = run(self.stoppage_then(600, reappear=235))
        self.assertEqual(trace.restart_kinds, ("before",))
        self.assertEqual(trace.restarts[0][0], 236)
        self.assertEqual(labels(result), "H" * 100 + "D" * 136 + "A" * 364)

    # T8-T10: Ambiguous movement (repositioning and slow controlled movement).
    def test_reposition_inside_window_is_not_the_restart(self):
        s = Scenario(500)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 200)
        spot, moved = np.array([0.4, 0.99]), np.array([0.4 + 1.0 / L, 0.99])
        s.ball_still(200, 260, spot)
        s.ball_path(260, 270, spot, moved)
        s.ball_still(270, 330, moved)
        s.player_at(200, 260, "A", 0, spot - [0.5 / L, 0])
        s.player_at(260, 500, "A", 0, moved - [0.5 / L, 0])
        s.ball_path(330, 350, moved, s.near("A", 1, 350))
        s.dribble(350, 500, "A", 1, speed_mps=1.5)
        s.event(*self.TAG, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("late",))
        self.assertIn(trace.restarts[0][0], (330, 331, 332))
        self.assertTrue(np.all(trace.dead_reason[200:330] != LIVE))
        self.assertEqual(labels(result)[340:], "A" * 160)

    def test_reposition_without_kick_in_window_abstains(self):
        s = Scenario(550)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 200)
        spot, moved = np.array([0.4, 0.99]), np.array([0.4 + 1.0 / L, 0.99])
        s.ball_still(200, 260, spot)
        s.ball_path(260, 270, spot, moved)
        s.ball_still(270, 400, moved)
        s.player_at(200, 260, "A", 0, spot - [0.5 / L, 0])
        s.player_at(260, 550, "A", 0, moved - [0.5 / L, 0])
        s.ball_path(400, 420, moved, s.near("A", 1, 420))
        s.dribble(420, 550, "A", 1, speed_mps=1.5)
        s.event(*self.TAG, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(labels(result)[:250], "H" * 100 + "D" * 150)
        self.assertEqual(labels(result)[250:375], "X" * 125)
        self.assertTrue(np.all(trace.reason[250:375] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.isnan(result.confidence[250:375]).all())
        self.assertEqual(labels(result)[375:], "A" * 175)

    def test_slow_controlled_movement_after_stoppage_abstains(self):
        s = Scenario(500)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 300)
        s.dribble(300, 500, "A", 0, speed_mps=0.5)
        s.event(*self.TAG, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(labels(result)[:250], "H" * 100 + "D" * 150)
        self.assertEqual(labels(result)[250:375], "X" * 125)
        self.assertTrue(np.all(trace.reason[250:375] == "abstain_unconfirmed_restart"))
        self.assertEqual(labels(result)[375:], "A" * 125)
        self.assertTrue(np.all(trace.reason[375:] == "controlled"))

    # T11a-T11b: Overlapping tags and cluster team resolution.
    def test_overlapping_tags_share_one_confirmed_restart(self):
        s = self.stoppage_then(400, reappear=255)
        s.events.clear()
        s.event(250, 280, team="A", Type="Throw-in")
        s.event(250, 260, team="A", Type="Throw-in")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("early",))
        self.assertEqual(len(trace.restarts), 1)
        self.assertEqual(trace.restarts[0][0], 256)
        self.assertNotIn("X", labels(result))
        self.assertEqual(labels(result), "H" * 100 + "D" * 156 + "A" * 144)

    def test_overlapping_tags_with_conflicting_teams_give_no_prior(self):
        s = Scenario(400)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 400)
        s.event(250, 300, team="A", Type="Throw-in")
        s.event(270, 330, team="H", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(len(trace.restarts), 1)
        self.assertEqual(trace.restarts[0][0], 330)
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 150)
        self.assertTrue(np.all(trace.reason[250:330] == "abstain_unconfirmed_restart"))
        self.assertTrue(np.all(trace.reason[330:] == "abstain_no_evidence"))

    def test_overlapping_tag_with_untagged_member_keeps_agreeing_team(self):
        s = Scenario(400)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 400)
        s.event(250, 300, team=None, Type="Throw-in")
        s.event(270, 330, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed",))
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 80 + "A" * 70)

    def test_abutting_tags_form_separate_clusters(self):
        s = Scenario(400)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 400)
        s.event(250, 280, team="H", Type="Free Kick")
        s.event(280, 310, team="A", Type="Free Kick")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("unconfirmed", "unconfirmed"))
        self.assertEqual(len(trace.restarts), 2)
        self.assertEqual(labels(result), "H" * 100 + "D" * 150 + "X" * 60 + "A" * 90)

    def test_sparse_displacement_hold_release_counts_as_movement_evidence(self):
        # Ball is seen only every 7 frames (> speed_lag_frames=5), so derived speed
        # stays NaN and sparse displacement (`moved`) releases the hold inside (S+50, E).
        s = Scenario(500)
        s.dribble(0, 100, "H", 0)
        s.ball_still(100, 150, (0.4, 1.05))
        s.ball_missing(150, 310)
        s.dribble(310, 420, "A", 0, speed_mps=1.5)
        sparse(s, 310, 420, keep_every=7)
        s.dribble(420, 500, "A", 0, speed_mps=1.5)
        s.event(*self.TAG, team="A", Type="Throw-in")
        result, trace = run(s)
        self.assertEqual(trace.restart_kinds, ("late",))
        # Observations at 315 and 322 (7 frames apart, 1.5 m/s) trigger `moved` at 322.
        self.assertEqual(trace.restarts[0][0], 322)
        self.assertEqual(labels(result), "H" * 100 + "D" * 222 + "A" * 178)


if __name__ == "__main__":
    unittest.main()
