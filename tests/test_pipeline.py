"""Focused tests for mtah.pipeline on invented fixtures only.

Every file here is written by the test into the ignored artifacts workspace.
No competition asset is read: the full dataset is validated by a separate,
manual local run, never by the ordinary suite.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mtah.pipeline import (
    _load_atd,
    _load_events,
    _load_geometry,
    _load_slot_teams,
    _segments_and_half,
    _without_paths,
    prepare_match,
)

ROOT = Path(__file__).resolve().parents[1]


def temporary_directory():
    # Keep all invented test files in the ignored workspace on sandboxed Windows.
    root = ROOT / "artifacts" / "test-tmp"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def atd_line(frame_id, slots, ball=(0.5, 0.5, 1.0)):
    """One raw ATD record: 46 is not special here, slots is the slot list."""
    groups = [
        "NaN,NaN,NaN" if slot is None else "{},{},{}".format(*slot)
        for slot in slots
    ]
    ball_text = "NaN,NaN,NaN" if ball is None else "{},{},{}".format(*ball)
    return "{}:{}:{}\n".format(frame_id, ";".join(groups), ball_text)


def tagging_xml(instances):
    """A minimal SportsCode file: (code, start_s, end_s, team) per instance."""
    blocks = []
    for code, start, end, team in instances:
        label = (
            ""
            if team is None
            else "<label><group>Team</group><text>{}</text></label>".format(team)
        )
        blocks.append(
            "<instance><start>{}</start><end>{}</end>"
            "<code>{}</code>{}</instance>".format(start, end, code, label)
        )
    return "<SportsCode>{}</SportsCode>".format("".join(blocks))


def homography_csv(rows):
    """One row per frame: (flat 3x3 or None for an invalid frame, error, valid)."""
    header = (
        "frame,"
        + ",".join(
            "h{}{}".format(r, c) for r in range(3) for c in range(3)
        )
        + ",reprojection_error,valid"
    )
    lines = [header]
    for frame, (matrix, error, valid) in enumerate(rows):
        cells = (
            ["NaN"] * 9
            if matrix is None
            else [repr(float(value)) for value in matrix]
        )
        lines.append(
            ",".join(
                [str(frame)] + cells + [repr(float(error)), str(int(valid))]
            )
        )
    return "\n".join(lines) + "\n"


def slot_metadata(players):
    """ATD metadata with Barcelona as team t1 and Atletic Club as t2."""
    return (
        "<TrackData><Teams>"
        '<Team id="t1"><Name>Barcelona</Name></Team>'
        '<Team id="t2"><Name>Atletic Club</Name></Team>'
        "</Teams><Players>{}</Players></TrackData>"
    ).format("".join(players))


def slot_metadata_with_specification(players, channels, refs):
    """ATD metadata plus a DataFormatSpecification.

    `players` are document-order `<Player>` elements, `channels` are
    `<PlayerChannel id="..." playerId="..."/>` elements, and `refs` are
    PlayerChannelRef ids in raw-file channel order.
    """
    players_text = "".join(players)
    channels_text = "".join(channels)
    groups_text = "".join(
        '<SplitRegister separator=",">'
        + "".join('<PlayerChannelRef playerChannelId="{}"/>'.format(ref) for ref in group)
        + "</SplitRegister>"
        for group in refs
    )
    return (
        "<TrackData><Teams>"
        '<Team id="t1"><Name>Barcelona</Name></Team>'
        '<Team id="t2"><Name>Atletic Club</Name></Team>'
        "</Teams><Players>{}</Players>"
        "<PlayerChannels>{}</PlayerChannels>"
        "<DataFormatSpecifications><DataFormatSpecification "
        'startFrame="1" endFrame="1"><SplitRegister separator=";">'
        + groups_text
        + "</SplitRegister></DataFormatSpecification>"
        "</DataFormatSpecifications></TrackData>"
    ).format(players_text, channels_text)


class SlotMappingVerifierTests(unittest.TestCase):
    def _kloppy_dataset(self, sides, unknown_channel=False, missing_ground=False):
        import types

        players = [
            types.SimpleNamespace(
                player_id=str(i + 1),
                team=types.SimpleNamespace(
                    ground=(
                        "HOME" if side == "H" else "AWAY"
                    )
                    if not (missing_ground and i == 0)
                    else None
                ),
            )
            for i, side in enumerate(sides)
        ]
        channels = [
            types.SimpleNamespace(
                player_channel_id=f"c{i + 1}",
                player=player,
            )
            for i, player in enumerate(players)
        ]
        refs = types.SimpleNamespace(
            children=[
                types.SimpleNamespace(player_channel_id=channel.player_channel_id)
                for channel in channels
            ]
        )
        if unknown_channel:
            refs.children.append(types.SimpleNamespace(player_channel_id="cUNKNOWN"))
        specifications = [
            types.SimpleNamespace(split_register=refs)
        ]
        return types.SimpleNamespace(
            metadata=types.SimpleNamespace(
                player_channels=channels,
                data_format_specifications=specifications,
                teams=(
                    types.SimpleNamespace(ground="HOME"),
                    types.SimpleNamespace(ground="AWAY"),
                ),
            )
        )

    def _load_verifier(self, loader_sides, reference_sides):
        import importlib.util
        import sys
        import types
        from unittest.mock import patch

        loader = types.SimpleNamespace(
            prepare_match=lambda root: types.SimpleNamespace(
                slot_team=np.array(loader_sides, dtype="<U1")
            )
        )
        result = self._kloppy_dataset(reference_sides)
        patcher = patch.dict(
            sys.modules,
            {
                "starter.load_atd": types.SimpleNamespace(
                    load=lambda game=None, data_dir=None, source=None: result
                ),
                "mtah.pipeline": loader,
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        spec = importlib.util.spec_from_file_location(
            "verify_slot_mapping_local",
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "verify_slot_mapping_local.py",
        )
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        sys.modules.pop("verify_slot_mapping_local", None)
        return verifier, result

    def test_permuted_equal_totals_are_rejected(self):
        verifier, _ = self._load_verifier(
            ["H", "A"] * 23,
            ["A", "H"] * 23,
        )
        result = verifier.compare_slot_mapping(
            np.array(["H", "A"] * 23, dtype="<U1"),
            np.array(["A", "H"] * 23, dtype="<U1"),
        )
        self.assertEqual(result.slots, 46)
        self.assertEqual(result.matches, 0)
        self.assertEqual(result.mismatches, 46)
        self.assertFalse(result.passed)

    def test_perfect_match_is_accepted(self):
        verifier, _ = self._load_verifier(
            ["H", "A"] * 23,
            ["H", "A"] * 23,
        )
        result = verifier.compare_slot_mapping(
            np.array(["H", "A"] * 23, dtype="<U1"),
            np.array(["H", "A"] * 23, dtype="<U1"),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.mismatches, 0)

    def test_single_mismatch_is_rejected(self):
        verifier, _ = self._load_verifier(
            ["H", "A"], ["A", "A"]
        )
        result = verifier.compare_slot_mapping(
            np.array(["H", "A"], dtype="<U1"),
            np.array(["H", "H"], dtype="<U1"),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)

    def test_kloppy_slot_sides_happy_path_uses_specification_order(self):
        verifier, _ = self._load_verifier(["H", "A"], ["H", "A"])
        sides = verifier._kloppy_slot_sides(self._kloppy_dataset(["A", "H"]))
        self.assertEqual(sides, ["A", "H"])  # helper preserves stub order

    def test_kloppy_slot_sides_unknown_channel_fails_closed(self):
        verifier, _ = self._load_verifier(["H"], ["H"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verifier._kloppy_slot_sides(
                self._kloppy_dataset(["H"], unknown_channel=True)
            )

    def test_kloppy_slot_sides_missing_team_ground_fails_closed(self):
        verifier, _ = self._load_verifier(["H"], ["H"])
        with self.assertRaisesRegex(ValueError, "missing team"):
            verifier._kloppy_slot_sides(
                self._kloppy_dataset(["H"], missing_ground=True)
            )

    def test_missing_identity_is_failed_closed(self):
        verifier, _ = self._load_verifier(
            ["H"], ["A"]
        )
        result = verifier.compare_slot_mapping(
            np.array(["H"], dtype="<U1"),
            np.array([None], dtype=object),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.mismatches, 1)

    def test_kloppy_slot_sides_conflicting_specifications_raises(self):
        import types

        verifier, dataset = self._load_verifier(["H", "A"], ["H", "A"])
        c1, c2 = dataset.metadata.player_channels
        reversed_refs = types.SimpleNamespace(
            children=[
                types.SimpleNamespace(player_channel_id=c2.player_channel_id),
                types.SimpleNamespace(player_channel_id=c1.player_channel_id),
            ]
        )
        dataset.metadata.data_format_specifications.append(
            types.SimpleNamespace(split_register=reversed_refs)
        )
        with self.assertRaisesRegex(ValueError, "different slot orders"):
            verifier._kloppy_slot_sides(dataset)

    def test_kloppy_slot_sides_missing_player_fails_closed(self):
        verifier, dataset = self._load_verifier(["H"], ["H"])
        dataset.metadata.player_channels[0].player = None
        self.assertIsNone(verifier._kloppy_team_name(dataset, None))
        with self.assertRaisesRegex(ValueError, "missing player"):
            verifier._kloppy_slot_sides(dataset)

    def test_kloppy_slot_sides_duplicate_channel_id_raises(self):
        import types

        verifier, dataset = self._load_verifier(["H"], ["H"])
        c1 = dataset.metadata.player_channels[0]
        dataset.metadata.player_channels.append(
            types.SimpleNamespace(player_channel_id=c1.player_channel_id, player=c1.player)
        )
        with self.assertRaisesRegex(ValueError, "duplicate player channel"):
            verifier._kloppy_slot_sides(dataset)

    def test_load_verifier_restores_sys_modules_on_teardown(self):
        import sys
        import types

        sentinel = types.ModuleType("sentinel.starter.load_atd")
        real_pipeline = sys.modules.get("mtah.pipeline")
        prior_starter = sys.modules.get("starter.load_atd", sentinel)

        class FailingSubTest(SlotMappingVerifierTests):
            def test_deliberate_failure(self):
                self._load_verifier(["H"], ["H"])
                raise AssertionError("deliberate test failure")

        sub = FailingSubTest("test_deliberate_failure")
        result = sub.run()
        self.assertFalse(result.wasSuccessful())
        self.assertIs(sys.modules.get("mtah.pipeline"), real_pipeline)
        self.assertIs(sys.modules.get("starter.load_atd", sentinel), prior_starter)

    def test_kloppy_slot_sides_none_or_blank_channel_id_raises(self):
        import types

        for bad_id in (None, "   "):
            with self.subTest(bad_id=bad_id):
                verifier, dataset = self._load_verifier(["H"], ["H"])
                dataset.metadata.player_channels.append(
                    types.SimpleNamespace(
                        player_channel_id=bad_id,
                        player=dataset.metadata.player_channels[0].player,
                    )
                )
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    verifier._kloppy_slot_sides(dataset)

    def test_kloppy_slot_sides_missing_split_register_and_empty_specs_raise(self):
        import types

        verifier, dataset = self._load_verifier(["H"], ["H"])
        dataset.metadata.data_format_specifications = [
            types.SimpleNamespace(split_register=None)
        ]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            verifier._kloppy_slot_sides(dataset)

        dataset.metadata.data_format_specifications = []
        with self.assertRaisesRegex(ValueError, "unavailable"):
            verifier._kloppy_slot_sides(dataset)

        dataset.metadata.data_format_specifications = [
            types.SimpleNamespace(split_register=types.SimpleNamespace(children=[]))
        ]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            verifier._kloppy_slot_sides(dataset)

    def test_verifier_main_reports_match_and_mismatch_without_leakage(self):
        from contextlib import redirect_stdout
        import io
        import sys
        from unittest.mock import patch

        verifier, _ = self._load_verifier(["H", "A"], ["H", "A"])
        out_ok = io.StringIO()
        with patch.object(sys, "argv", ["verify_slot_mapping_local.py", "--input-root", "local"]), redirect_stdout(out_ok):
            code_ok = verifier.main()
        self.assertEqual(code_ok, 0)
        self.assertIn("slots=2 matches=2 mismatches=0", out_ok.getvalue())

        verifier_bad, _ = self._load_verifier(["H", "A"], ["A", "A"])
        out_bad = io.StringIO()
        with patch.object(sys, "argv", ["verify_slot_mapping_local.py", "--input-root", "local"]), redirect_stdout(out_bad):
            code_bad = verifier_bad.main()
        self.assertEqual(code_bad, 1)
        self.assertIn("slots=2 matches=1 mismatches=1", out_bad.getvalue())


class AtdParsingTests(unittest.TestCase):
    def test_malformed_frame_id_does_not_echo_the_token(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "SENSITIVE_TOKEN:0.1,0.2,1.0:0.5,0.5,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid ATD frame id at record 2") as cm:
                _load_atd(path, 2, 1)
        self.assertNotIn("SENSITIVE_TOKEN", str(cm.exception))

    def test_malformed_numeric_player_coordinate_is_sanitized(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "2:0.1,BAD_PLAYER_VALUE,1.0:0.5,0.5,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid numeric ATD player coordinate at record 2"):
                _load_atd(path, 2, 1)

    def test_malformed_numeric_player_speed_is_sanitized(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "2:0.1,0.2,BAD_PLAYER_SPEED:0.5,0.5,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid numeric ATD player speed at record 2"):
                _load_atd(path, 2, 1)

    def test_malformed_numeric_ball_coordinate_is_sanitized(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "2:0.1,0.2,1.0:BAD_BALL_VALUE,0.5,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid numeric ATD ball coordinate at record 2"):
                _load_atd(path, 2, 1)

    def test_empty_numeric_player_field_is_sanitized(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "2:0.1,,1.0:0.5,0.5,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid numeric ATD player coordinate at record 2"):
                _load_atd(path, 2, 1)

    def test_empty_numeric_ball_field_is_sanitized(self):
        text = atd_line(1, [(0.1, 0.2, 1.0)]) + "2:0.1,0.2,1.0:0.5,,1.0\n"
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid numeric ATD ball coordinate at record 2"):
                _load_atd(path, 2, 1)

    def test_frame_ids_map_to_zero_based_rows_and_missing_stays_nan(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(
                atd_line(1, [(0.1, 0.2, 1.5), None])
                + atd_line(2, [(0.3, 0.4, 2.5), (0.6, 0.7, 3.5)], ball=None)
                + atd_line(3, [None, None]),
                encoding="utf-8",
            )
            xy, observed, speed, ball_xy, ball_observed, ball_speed = _load_atd(
                path, 3, 2
            )

        # 1-based id 1 lands on row 0; nothing shifts by one.
        np.testing.assert_array_equal(xy[0, 0], [0.1, 0.2])
        np.testing.assert_array_equal(xy[1, 1], [0.6, 0.7])
        self.assertEqual(xy.dtype, np.float64)
        self.assertEqual(observed.dtype, np.bool_)

        # Missing slots are NaN with a matching False mask.
        self.assertTrue(np.isnan(xy[0, 1]).all())
        self.assertTrue(np.isnan(xy[2]).all())
        np.testing.assert_array_equal(
            observed, [[True, False], [True, True], [False, False]]
        )

        # Player speed is read from the third value; ball speed is never invented.
        np.testing.assert_array_equal(speed[1], [2.5, 3.5])
        self.assertTrue(np.isnan(speed[0, 1]))
        np.testing.assert_array_equal(ball_xy[0], [0.5, 0.5])
        self.assertTrue(np.isnan(ball_xy[1]).all())
        np.testing.assert_array_equal(
            ball_observed, [True, False, True]
        )
        self.assertTrue(np.isnan(ball_speed).all())

    def test_rejects_non_contiguous_frame_ids(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(
                atd_line(1, [(0.1, 0.2, 1.0)])
                + atd_line(2, [(0.1, 0.2, 1.0)])
                + atd_line(4, [(0.1, 0.2, 1.0)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frame mismatch") as cm:
                _load_atd(path, 4, 1)
        self.assertNotIn("got 3", str(cm.exception))

    def test_rejects_record_count_that_differs_from_timeline(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(
                atd_line(1, [(0.1, 0.2, 1.0)]) + atd_line(2, [(0.1, 0.2, 1.0)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                _load_atd(path, 3, 1)

    def test_rejects_frame_beyond_the_timeline(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(
                atd_line(1, [(0.1, 0.2, 1.0)])
                + atd_line(2, [(0.1, 0.2, 1.0)])
                + atd_line(999, [(0.1, 0.2, 1.0)]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside the 2-frame") as cm:
                _load_atd(path, 2, 1)
        self.assertNotIn("998", str(cm.exception))
        self.assertNotIn("999", str(cm.exception))

    def test_invalid_utf8_atd_raises_sanitized_value_error(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_bytes(b"\xff\xfeinvalid\n")
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                _load_atd(path, 1, 1)

    def test_rejects_malformed_record(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text("1:0.1,0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed ATD record"):
                _load_atd(path, 1, 1)

    def test_rejects_wrong_slot_count(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "atd.txt"
            path.write_text(
                atd_line(1, [(0.1, 0.2, 1.0)] * 3), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "player slots"):
                _load_atd(path, 1, 2)

    def test_rejects_non_positive_and_non_integer_frame_ids(self):
        for frame_id in ("0", "-3", "abc"):
            with self.subTest(frame_id=frame_id), temporary_directory() as tmp:
                path = Path(tmp) / "atd.txt"
                path.write_text(
                    "{}:0.1,0.2,1.0:0.5,0.5,1.0\n".format(frame_id),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    _load_atd(path, 1, 1)


class SlotTeamTests(unittest.TestCase):
    def _metadata_with_two_specifications(self, first_groups, second_groups):
        players = [
            '<Player id="1" teamId="t1"/>',
            '<Player id="2" teamId="t2"/>',
        ]
        channels = [
            '<PlayerChannel id="c1" playerId="1"/>',
            '<PlayerChannel id="c2" playerId="2"/>',
        ]
        text = slot_metadata_with_specification(players, channels, first_groups)
        second = (
            '<DataFormatSpecification startFrame="2" endFrame="2">'
            '<SplitRegister separator=";">'
            + "".join(
                '<SplitRegister separator=",">'
                + "".join(
                    '<PlayerChannelRef playerChannelId="{}"/>'.format(ref)
                    for ref in group
                )
                + "</SplitRegister>"
                for group in second_groups
            )
            + "</SplitRegister></DataFormatSpecification>"
        )
        return text.replace("</DataFormatSpecifications>", second + "</DataFormatSpecifications>")

    def test_multiple_specifications_with_same_order_accepted(self):
        text = self._metadata_with_two_specifications(
            [["c1"], ["c2"]],
            [["c1"], ["c2"]],
        )
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(text, encoding="utf-8")
            np.testing.assert_array_equal(_load_slot_teams(path), ["H", "A"])

    def test_multiple_specifications_with_different_orders_raise(self):
        text = self._metadata_with_two_specifications(
            [["c1"], ["c2"]],
            [["c2"], ["c1"]],
        )
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "different slot orders"
            ):
                _load_slot_teams(path)

    def test_specification_order_defines_raw_slot_order(self):
        # Document order is 1(H), 2(A), 3(H), 4(A); the specification
        # orders the raw triples as 3, 1, 4, 2, so the sides must follow it.
        players = [
            '<Player id="1" teamId="t1"/>',
            '<Player id="2" teamId="t2"/>',
            '<Player id="3" teamId="t1"/>',
            '<Player id="4" teamId="t2"/>',
        ]
        channels = [
            '<PlayerChannel id="c1" playerId="1"/>',
            '<PlayerChannel id="c2" playerId="2"/>',
            '<PlayerChannel id="c3" playerId="3"/>',
            '<PlayerChannel id="c4" playerId="4"/>',
        ]
        groups = [["c3"], ["c1"], ["c4"], ["c2"]]
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(
                slot_metadata_with_specification(players, channels, groups),
                encoding="utf-8",
            )
            sides = _load_slot_teams(path)
        np.testing.assert_array_equal(sides, ["H", "H", "A", "A"])

    def test_specification_with_missing_player_channel_raises(self):
        players = [
            '<Player id="1" teamId="t1"/>',
            '<Player id="2" teamId="t2"/>',
        ]
        channels = ['<PlayerChannel id="c1" playerId="1"/>']
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(
                slot_metadata_with_specification(players, channels, [["c1"]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not cover"):
                _load_slot_teams(path)

    def test_specification_with_unknown_channel_raises(self):
        players = ['<Player id="1" teamId="t1"/>']
        channels = ['<PlayerChannel id="c1" playerId="1"/>']
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(
                slot_metadata_with_specification(players, channels, [["cX"]]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown player channel"):
                _load_slot_teams(path)

    def test_sides_follow_the_metadata_in_slot_order(self):
        players = [
            '<Player id="{}" teamId="{}"/>'.format(
                i, "t1" if i <= 3 else "t2"
            )
            for i in range(1, 6)
        ]
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(slot_metadata(players), encoding="utf-8")
            sides = _load_slot_teams(path)
        np.testing.assert_array_equal(sides, ["H", "H", "H", "A", "A"])
        self.assertEqual(sides.dtype.kind, "U")

    def test_unknown_team_name_raises(self):
        players = ['<Player id="1" teamId="t3"/>']
        text = slot_metadata(players).replace(
            "</Teams>",
            '<Team id="t3"><Name>SECRET_UNKNOWN_FC</Name></Team></Teams>',
        )
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown team") as cm:
                _load_slot_teams(path)
        self.assertNotIn("SECRET_UNKNOWN_FC", str(cm.exception))

    def test_team_missing_id_and_duplicate_team_ids_raise(self):
        players = ['<Player id="1" teamId="t1"/>']
        missing_id = slot_metadata(players).replace('<Team id="t2">', "<Team>")
        dup_id = slot_metadata(players).replace('<Team id="t2">', '<Team id="t1">')
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(missing_id, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "team without an id"):
                _load_slot_teams(path)
            path.write_text(dup_id, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate team ids"):
                _load_slot_teams(path)

    def test_channel_missing_id_and_duplicate_channel_ids_raise(self):
        players = ['<Player id="1" teamId="t1"/>', '<Player id="2" teamId="t2"/>']
        missing_chan = slot_metadata_with_specification(
            players,
            ['<PlayerChannel playerId="1"/>', '<PlayerChannel id="c2" playerId="2"/>'],
            [["c1"], ["c2"]],
        )
        dup_chan = slot_metadata_with_specification(
            players,
            ['<PlayerChannel id="c1" playerId="1"/>', '<PlayerChannel id="c1" playerId="2"/>'],
            [["c1"]],
        )
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(missing_chan, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "channel without an id"):
                _load_slot_teams(path)
            path.write_text(dup_chan, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate player channel ids"):
                _load_slot_teams(path)

    def test_mixed_nested_and_top_level_specifications_conflict_raises(self):
        players = ['<Player id="1" teamId="t1"/>', '<Player id="2" teamId="t2"/>']
        channels = [
            '<PlayerChannel id="c1" playerId="1"/>',
            '<PlayerChannel id="c2" playerId="2"/>',
        ]
        base = slot_metadata_with_specification(players, channels, [["c1"], ["c2"]])
        top_level_conflict = (
            '<DataFormatSpecification startFrame="2" endFrame="2">'
            '<SplitRegister separator=";">'
            '<SplitRegister separator=","><PlayerChannelRef playerChannelId="c2"/></SplitRegister>'
            '<SplitRegister separator=","><PlayerChannelRef playerChannelId="c1"/></SplitRegister>'
            "</SplitRegister></DataFormatSpecification>"
        )
        text = base.replace("</TrackData>", top_level_conflict + "</TrackData>")
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different slot orders"):
                _load_slot_teams(path)

    def test_missing_team_reference_raises(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(
                slot_metadata(['<Player id="1"/>']), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "unknown team"):
                _load_slot_teams(path)

    def test_non_contiguous_player_ids_raise(self):
        players = [
            '<Player id="1" teamId="t1"/>',
            '<Player id="2" teamId="t1"/>',
            '<Player id="4" teamId="t2"/>',
        ]
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(slot_metadata(players), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                _load_slot_teams(path)

    def test_metadata_without_players_raises(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(slot_metadata([]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no tracked players"):
                _load_slot_teams(path)

    def test_unsupported_xml_encoding_is_sanitized(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "meta.xml"
            path.write_text(
                '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><TrackData/>',
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "unsupported XML encoding") as cm:
                _load_slot_teams(path)
        self.assertNotIn("SENSITIVE_ENCODING", str(cm.exception))


class GeometryTests(unittest.TestCase):
    def write_csv(self, tmp, rows):
        tactical = Path(tmp) / "tactical"
        tactical.mkdir(exist_ok=True)
        (tactical / "BAR-ATH_tactical_homography.csv").write_text(
            homography_csv(rows), encoding="utf-8"
        )

    @staticmethod
    def scaled_rows(count, valid=True):
        return [
            (list(np.diag([i + 1, 1.0, 1.0]).ravel()), 0.1 * (i + 1), valid)
            for i in range(count)
        ]

    def test_matrices_stay_in_video_frame_order(self):
        with temporary_directory() as tmp:
            self.write_csv(tmp, self.scaled_rows(4))
            geometry = _load_geometry("BAR-ATH", Path(tmp), 4)

        self.assertEqual(geometry.matrices.shape, (4, 3, 3))
        diagonal = np.diagonal(geometry.matrices, axis1=1, axis2=2)
        np.testing.assert_array_equal(
            diagonal, [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]]
        )
        np.testing.assert_allclose(
            geometry.reprojection_error, [0.1, 0.2, 0.3, 0.4]
        )
        self.assertTrue(geometry.valid.all())

    def test_invalid_frames_keep_nan_matrices(self):
        rows = self.scaled_rows(3) + [(None, float("nan"), False)]
        with temporary_directory() as tmp:
            self.write_csv(tmp, rows)
            geometry = _load_geometry("BAR-ATH", Path(tmp), 4)

        np.testing.assert_array_equal(
            geometry.valid, [True, True, True, False]
        )
        self.assertTrue(np.isnan(geometry.matrices[3]).all())
        self.assertTrue(np.isnan(geometry.reprojection_error[3]))
        self.assertTrue(np.isfinite(geometry.matrices[:3]).all())

    def test_row_count_must_match_the_timeline(self):
        with temporary_directory() as tmp:
            self.write_csv(tmp, self.scaled_rows(3))
            with self.assertRaisesRegex(ValueError, "frame count"):
                _load_geometry("BAR-ATH", Path(tmp), 4)

    def test_singular_valid_matrix_is_rejected(self):
        rows = [([0.0] * 9, 0.5, True)] + self.scaled_rows(2)
        with temporary_directory() as tmp:
            self.write_csv(tmp, rows)
            with self.assertRaisesRegex(ValueError, "singular"):
                _load_geometry("BAR-ATH", Path(tmp), 3)

    def test_non_numeric_homography_and_missing_columns_are_sanitized(self):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir(exist_ok=True)
            csv_path = tactical / "BAR-ATH_tactical_homography.csv"
            csv_path.write_text(
                "frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n"
                "0,1,0,0,0,1,0,0,0,SENSITIVE_BAD_H,0.1,1\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as cm:
                _load_geometry("BAR-ATH", Path(tmp), 1)
            self.assertNotIn("SENSITIVE_BAD_H", str(cm.exception))
            self.assertNotIn(str(tmp), str(cm.exception))

            csv_path.write_text("0,1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns") as cm2:
                _load_geometry("BAR-ATH", Path(tmp), 1)
            self.assertNotIn(str(tmp), str(cm2.exception))


class SmartTaggingTests(unittest.TestCase):
    def load(self, instances, n_frames=2000):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                tagging_xml(instances), encoding="utf-8"
            )
            return _load_events("BAR-ATH", Path(tmp), 25, n_frames)

    def test_intervals_are_half_open_video_frames_with_sides_mapped(self):
        events = self.load(
            [
                ("BUILD UP", 10.0, 20.0, "Barcelona"),
                ("SET PIECES", 5.0, 9.6, "Atletic Club"),
                ("PROGRESSION", 30.0, 40.0, "N/A"),
                ("BUILD UP", 50.0, 60.0, None),  # no Team tag at all
            ]
        )

        self.assertEqual(len(events), 4)
        # Ordered by start; ids keep their file order, so they trace the source.
        self.assertEqual(
            [event.event_id for event in events],
            [
                "smart-tagging-0002",
                "smart-tagging-0001",
                "smart-tagging-0003",
                "smart-tagging-0004",
            ],
        )
        self.assertEqual(
            [(event.start_frame, event.end_frame) for event in events],
            [(125, 240), (250, 500), (750, 1000), (1250, 1500)],
        )
        self.assertEqual(
            [event.team for event in events], ["A", "H", None, None]
        )
        self.assertEqual(
            [event.code for event in events],
            ["SET PIECES", "BUILD UP", "PROGRESSION", "BUILD UP"],
        )

        # Half-open: end_frame is the first frame after the phase, not one less.
        self.assertTrue(events[1].contains(499))
        self.assertFalse(events[1].contains(500))

        # Tags keep their original key casing and values.
        self.assertEqual(events[1].tags["Team"], "Barcelona")
        self.assertEqual(events[3].tags, {})

    def test_lowercase_code_is_normalized_for_the_contract(self):
        events = self.load([("build up", 1.0, 2.0, None)])
        self.assertEqual(events[0].code, "BUILD UP")

    def test_unknown_team_label_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown team") as cm:
            self.load([("BUILD UP", 10.0, 20.0, "SECRET_SEVILLA")])
        self.assertNotIn("SECRET_SEVILLA", str(cm.exception))

    def test_non_numeric_timestamp_is_sanitized(self):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                "<SportsCode><instance><start>SENSITIVE_TS</start><end>2.0</end>"
                "<code>SET PIECES</code></instance></SportsCode>",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as cm:
                _load_events("BAR-ATH", Path(tmp), 25, 2000)
        self.assertNotIn("SENSITIVE_TS", str(cm.exception))

    def test_empty_interval_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid Smart Tagging interval"):
            self.load([("BUILD UP", 10.0, 10.0, None)])

    def test_interval_beyond_the_timeline_raises(self):
        with self.assertRaisesRegex(ValueError, "Invalid Smart Tagging interval"):
            self.load([("BUILD UP", 50.0, 60.0, None)], n_frames=1000)

    def test_missing_code_raises(self):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                "<SportsCode><instance><start>1.0</start>"
                "<end>2.0</end></instance></SportsCode>",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no code"):
                _load_events("BAR-ATH", Path(tmp), 25, 2000)

    def test_malformed_xml_becomes_a_contract_value_error(self):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                "<SportsCode><instance>", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "well-formed XML"):
                _load_events("BAR-ATH", Path(tmp), 25, 2000)

    def test_unsupported_xml_encoding_is_sanitized(self):
        with temporary_directory() as tmp:
            tactical = Path(tmp) / "tactical"
            tactical.mkdir()
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><SportsCode/>',
                encoding="ascii",
            )
            with self.assertRaisesRegex(ValueError, "unsupported XML encoding") as cm:
                _load_events("BAR-ATH", Path(tmp), 25, 2000)
        self.assertNotIn("SENSITIVE_ENCODING", str(cm.exception))


class SegmentTests(unittest.TestCase):
    def test_published_halves_become_half_open_segments(self):
        segments, half = _segments_and_half("BAR-ATH", 199319)
        self.assertEqual(
            segments, ((11185, 90652, 1), (110808, 190198, 2))
        )
        self.assertEqual(half.dtype, np.int8)
        self.assertEqual(len(half), 199319)
        for frame, value in (
            (0, 0),
            (11184, 0),
            (11185, 1),
            (90651, 1),
            (90652, 0),
            (110807, 0),
            (110808, 2),
            (190197, 2),
            (190198, 0),
            (199318, 0),
        ):
            self.assertEqual(int(half[frame]), value, frame)


class PrepareMatchErrorTests(unittest.TestCase):
    def test_missing_assets_raise_a_path_free_value_error(self):
        # input_root is the data directory (the parent of tactical/); an empty
        # one must fail before any asset is opened, with a message the CLI can
        # print verbatim: exit-code-2 type, no local path.
        with temporary_directory() as tmp:
            empty = Path(tmp) / "empty-root"
            empty.mkdir()
            with self.assertRaises(ValueError) as ctx:
                prepare_match(empty)

        message = str(ctx.exception)
        self.assertIn("not found under the provided input root", message)
        self.assertIn("DATA.md", message)
        self.assertNotIn(str(empty), message)
        self.assertNotIn(str(ROOT), message)

    def test_without_paths_strips_absolute_relative_windows_posix_and_resolved_paths(self):
        from pathlib import PureWindowsPath
        from unittest.mock import patch

        with temporary_directory() as tmp:
            abs_root = Path(tmp) / "secret-abs-root"
            abs_asset = abs_root / "tactical" / "BAR-ATH_tactical_atd.txt"
            rel_root = Path("rel-secret-bundle")
            rel_asset = rel_root / "tactical" / "BAR-ATH_tactical_atd.txt"
            win_asset = PureWindowsPath(r"C:\private\match_bundle\tactical\BAR-ATH_tactical_atd.txt")
            posix_asset = "/private/match_bundle/tactical/BAR-ATH_tactical_atd.txt"

            raw_msg = (
                f"invalid schema: abs={abs_asset} rel={rel_asset} "
                f"win={win_asset} posix={posix_asset} "
                f"raw_root={abs_root} root_posix={abs_root.as_posix()} "
                f"resolved_root={abs_root.resolve()}"
            )
            for exc_type in (ValueError, TypeError):
                with self.subTest(exc_type=exc_type.__name__):
                    cleaned = _without_paths(
                        exc_type(raw_msg),
                        (abs_asset, rel_asset, str(win_asset), posix_asset),
                        abs_root,
                    )
                    self.assertIsInstance(cleaned, exc_type)
                    msg = str(cleaned)
                    for secret in (
                        str(abs_asset),
                        abs_asset.as_posix(),
                        str(rel_asset),
                        rel_asset.as_posix(),
                        str(win_asset),
                        posix_asset,
                        str(abs_root),
                        abs_root.as_posix(),
                        str(abs_root.resolve()),
                        abs_root.resolve().as_posix(),
                    ):
                        self.assertNotIn(secret, msg)
                    self.assertIn("invalid schema:", msg)

            # Relative root spellings also strip when root itself is relative, without corrupting words.
            rel_resolved = rel_root.resolve()
            rel_cleaned = _without_paths(
                ValueError(
                    f"invalid input under {rel_root}, {rel_root.as_posix()}, "
                    f"and {rel_resolved} ({rel_asset})"
                ),
                (rel_asset,),
                rel_root,
            )
            rel_msg = str(rel_cleaned)
            self.assertIn("invalid input under <input-root>", rel_msg)
            self.assertNotIn(str(rel_resolved), rel_msg)
            self.assertNotIn(str(rel_asset), rel_msg)

            short_root = Path("a")
            short_cleaned = _without_paths(ValueError(f"invalid token in {short_root}/file"), (), short_root)
            self.assertEqual(str(short_cleaned), "invalid token in <input-root>/file")

            # Verify patch through prepare_match() end-to-end strips real paths on ValueError & TypeError.
            abs_asset.parent.mkdir(parents=True)
            abs_asset.write_text("", encoding="utf-8")
            (abs_root / "tactical" / "BAR-ATH_tactical_atd.xml").write_text("<TrackData/>", encoding="utf-8")
            (abs_root / "tactical" / "BAR-ATH_tactical_homography.csv").write_text("", encoding="utf-8")
            (abs_root / "tactical" / "BAR-ATH_tactical_smart_tagging.xml").write_text("<SportsCode/>", encoding="utf-8")

            for exc_type in (ValueError, TypeError):
                def _raise_with_paths(*args, exc_cls=exc_type, **kwargs):
                    raise exc_cls(
                        f"parse failure at {abs_asset} under {abs_root.as_posix()} ({abs_root.resolve()})"
                    )
                with self.subTest(prepare_exc=exc_type.__name__), patch.dict(
                    prepare_match.__globals__,
                    {"_load_slot_teams": _raise_with_paths},
                ):
                    with self.assertRaises(exc_type) as ctx:
                        prepare_match(abs_root)
                    out_msg = str(ctx.exception)
                    self.assertIn("parse failure at <input-root asset> under <input-root> (<input-root>)", out_msg)
                    self.assertNotIn(str(abs_root), out_msg)
                    self.assertNotIn(abs_root.as_posix(), out_msg)
                    self.assertNotIn(str(abs_asset), out_msg)

    def test_prepare_match_sanitizes_unknown_xml_encoding_for_both_assets(self):
        from unittest.mock import patch

        with temporary_directory() as tmp:
            root = Path(tmp) / "enc-root"
            tactical = root / "tactical"
            tactical.mkdir(parents=True)
            (tactical / "BAR-ATH_tactical_atd.txt").write_text("", encoding="utf-8")
            (tactical / "BAR-ATH_tactical_homography.csv").write_text("", encoding="utf-8")
            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><TrackData/>',
                encoding="ascii",
            )
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text("<SportsCode/>", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported XML encoding") as ctx1:
                prepare_match(root)
            self.assertNotIn("SENSITIVE_ENCODING", str(ctx1.exception))
            self.assertNotIn(str(root), str(ctx1.exception))

            (tactical / "BAR-ATH_tactical_atd.xml").write_text(
                slot_metadata(['<Player id="1" teamId="t1"/>']),
                encoding="utf-8",
            )
            (tactical / "BAR-ATH_tactical_smart_tagging.xml").write_text(
                '<?xml version="1.0" encoding="SENSITIVE_ENCODING"?><SportsCode/>',
                encoding="ascii",
            )
            fake_atd = (
                np.zeros((199319, 1, 2)),
                np.ones((199319, 1), dtype=bool),
                np.zeros((199319, 1)),
                np.zeros((199319, 2)),
                np.ones(199319, dtype=bool),
                np.full(199319, np.nan),
            )
            with patch.dict(
                prepare_match.__globals__,
                {
                    "_load_atd": lambda *a, **kw: fake_atd,
                    "_load_geometry": lambda *a, **kw: None,
                },
            ):
                with self.assertRaisesRegex(ValueError, "unsupported XML encoding") as ctx2:
                    prepare_match(root)
            self.assertNotIn("SENSITIVE_ENCODING", str(ctx2.exception))
            self.assertNotIn(str(root), str(ctx2.exception))


if __name__ == "__main__":
    unittest.main()
