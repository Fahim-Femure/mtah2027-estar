"""Prepare the tactical match data as a validated MatchInput.

Failure contract (see docs/DATA_PIPELINE.md, section 9):
raise ValueError or TypeError whose message contains no data excerpts and no
local paths, so the CLI reports the failure with exit code 2 and logs stay
shareable. Nothing here prints, touches the network, or reads the video.
"""

import re
import sys
from pathlib import Path

import numpy as np

# games.py lives at the repository root, beside this package. The starter
# modules resolve it the same way (see starter/load_atd.py); the guard keeps
# `python -m mtah --source atd` working from any working directory.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import games  # noqa: E402

from .alignment import atd_to_video  # noqa: E402
from .contracts import Event, Geometry, MatchInput  # noqa: E402

# Club names exactly as they appear in the ATD metadata and the Smart Tagging
# XML, mapped to the contract's sides. Anything else is an error, never a guess.
TEAM_SIDES = {"Barcelona": "H", "Atletic Club": "A"}

# Smart Tagging uses this literal for "no team". The contract represents the
# same absence as team=None.
NO_TEAM = "N/A"


def _data_format_player_order(root, channel_player):
    """The raw group order from the DataFormatSpecification, if present.

    The EPTS raw file lists player triples in the order produced by the
    specification's PlayerChannelRef elements. The document order of
    ``<Player>`` elements is not authoritative for that order.

    A metadata file may contain more than one DataFormatSpecification.
    The current MatchInput model carries one static slot-to-team mapping, so
    every specification must define the same slot order. Any disagreement is
    rejected rather than silently using the first specification.
    """
    specifications = root.findall(".//DataFormatSpecification")
    if not specifications:
        return None

    orders = []
    for specification in specifications:
        ordered = []
        seen = set()
        for ref in specification.iter("PlayerChannelRef"):
            reference = ref.get("playerChannelId")
            player_id = channel_player.get(reference) if reference is not None else None
            if player_id is None:
                raise ValueError(
                    "DataFormatSpecification references an unknown player channel"
                )
            if player_id not in seen:
                ordered.append(player_id)
                seen.add(player_id)
        orders.append(tuple(ordered))

    first_order = orders[0]
    if any(order != first_order for order in orders):
        raise ValueError(
            "DataFormatSpecification sections define different slot orders"
        )
    return list(first_order)


def _load_slot_teams(path):
    """The side of every tracking slot, in the specification-defined raw order.

    The raw ATD player-triple order is defined by the metadata's
    DataFormatSpecification order of PlayerChannelRef elements. If the
    metadata has no such specification, preserve the previous document-order
    convention. Unknown team names, missing or duplicate metadata ids, an
    unknown specification channel, and a specification that does not cover
    every player exactly once raise; the mapping is never guessed from position.
    """
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(path).getroot()
    except SyntaxError:
        raise ValueError("ATD metadata is not well-formed XML") from None
    except (LookupError, UnicodeDecodeError):
        raise ValueError("ATD metadata uses an unsupported XML encoding") from None

    teams = {}
    for team in root.findall(".//Teams/Team"):
        team_id = team.get("id")
        if not team_id or not team_id.strip():
            raise ValueError("ATD metadata contains a team without an id")
        if team_id in teams:
            raise ValueError("ATD metadata contains duplicate team ids")
        teams[team_id] = (team.findtext("Name") or "").strip()

    players = root.findall(".//Players/Player")
    if not players:
        raise ValueError("ATD metadata lists no tracked players")

    seen_players = set()
    player_side = {}
    for player in players:
        player_id = player.get("id")
        if not player_id or not player_id.strip():
            raise ValueError("ATD metadata contains a player without an id")
        if player_id in seen_players:
            raise ValueError("ATD metadata contains duplicate player ids")
        seen_players.add(player_id)

        team_id = player.get("teamId")
        if not team_id or team_id not in teams:
            raise ValueError("ATD metadata names an unknown team")
        name = teams[team_id]
        if name not in TEAM_SIDES:
            raise ValueError("ATD metadata names an unknown team")
        player_side[player_id] = TEAM_SIDES[name]

    channel_player = {}
    for channel in root.findall(".//PlayerChannels/PlayerChannel"):
        channel_id = channel.get("id")
        if not channel_id or not channel_id.strip():
            raise ValueError("ATD metadata contains a player channel without an id")
        if channel_id in channel_player:
            raise ValueError("ATD metadata contains duplicate player channel ids")
        channel_player[channel_id] = channel.get("playerId")
    specification_order = _data_format_player_order(root, channel_player)
    if specification_order is None:
        ordered_ids = [player.get("id") for player in players]
    else:
        ordered_ids = specification_order
        if set(ordered_ids) != set(player_side):
            raise ValueError(
                "DataFormatSpecification does not cover every tracked player"
            )
        if len(ordered_ids) != len(player_side):
            raise ValueError(
                "DataFormatSpecification contains duplicate or unknown players"
            )
    if set(ordered_ids) != {str(i) for i in range(1, len(ordered_ids) + 1)}:
        raise ValueError(
            "ATD metadata player ids are not contiguous slot order"
        )
    return np.array([player_side[player_id] for player_id in ordered_ids], dtype="<U1")


def _iter_atd_records(path):
    """Yield 1-based record numbers and stripped records from UTF-8 ATD text."""
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                yield count, line.rstrip("\r\n")
    except UnicodeDecodeError:
        raise ValueError("ATD asset is not valid UTF-8 text") from None


def _load_atd(path, n_frames, n_slots):
    """Read the raw ATD incrementally, one line (one frame) at a time.

    video_frame = frame_id - 1, via mtah.alignment.atd_to_video. A record whose
    id is not exactly its 1-based position, a malformed record, a slot count
    that disagrees with the metadata, or a file whose record count differs from
    n_frames all raise -- nothing is padded, trimmed or reordered.
    """
    player_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float64)
    player_speed = np.full((n_frames, n_slots), np.nan, dtype=np.float64)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float64)

    count = 0
    for count, raw_line in _iter_atd_records(path):
        expected_frame = count - 1
        parts = raw_line.split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed ATD record at line {count}")

        try:
            frame_id = int(parts[0])
        except ValueError:
            raise ValueError(
                f"invalid ATD frame id at record {count}"
            ) from None
        frame = atd_to_video(frame_id)  # ValueError on an id below 1

        if frame >= n_frames:
            raise ValueError(
                f"ATD frame at record {count} is outside the {n_frames}-frame timeline"
            )
        if frame != expected_frame:
            raise ValueError(
                f"ATD frame mismatch at record {count}: expected frame index {expected_frame}"
            )

        def parse_number(text, label, index):
            try:
                return float(text)
            except (TypeError, ValueError):
                raise ValueError(
                    f"invalid numeric ATD {label} at record {count}, field {index}"
                ) from None

        field_records = parts[1].split(";")
        if len(field_records) != n_slots:
            raise ValueError(
                f"Frame {frame}: expected {n_slots} player slots, "
                f"got {len(field_records)}"
            )

        player_values = []
        for index, field_record in enumerate(field_records, 1):
            fields = field_record.split(",")
            if len(fields) != 3:
                raise ValueError(
                    f"malformed ATD player data at record {count}, slot {index}: "
                    "expected 3 numeric fields"
                )
            player_values.append(
                np.array(
                    [
                        parse_number(fields[0], "player coordinate", 1),
                        parse_number(fields[1], "player coordinate", 2),
                        parse_number(fields[2], "player speed", 3),
                    ],
                    dtype=np.float64,
                )
            )
        players = np.array(player_values, dtype=np.float64)
        player_xy[frame] = players[:, :2]
        player_speed[frame] = players[:, 2]

        ball_fields = parts[2].split(",")
        if len(ball_fields) != 3:
            raise ValueError(f"Frame {frame}: malformed ball data")
        ball = np.array(
            [
                parse_number(ball_fields[0], "ball coordinate", 1),
                parse_number(ball_fields[1], "ball coordinate", 2),
                parse_number(ball_fields[2], "ball speed", 3),
            ],
            dtype=np.float64,
        )
        ball_xy[frame] = ball[:2]

    if count != n_frames:
        raise ValueError(
            f"ATD frame count mismatch: expected {n_frames}, got {count}"
        )

    player_observed = np.isfinite(player_xy).all(axis=2)
    ball_observed = np.isfinite(ball_xy).all(axis=1)

    # The ball group's third value is documented as "speed", but its units and
    # meaning are unverified, so ball speed stays NaN rather than being guessed.
    ball_speed = np.full(n_frames, np.nan, dtype=np.float64)

    return (
        player_xy,
        player_observed,
        player_speed,
        ball_xy,
        ball_observed,
        ball_speed,
    )


def _load_geometry(game, input_root, n_frames):
    """The per-frame homography via the starter loader, checked against the timeline.

    starter.load_homography returns (matrices, reprojection_error, valid) in
    video-frame order; this only checks the counts here. Geometry enforces the
    validity semantics: finite matrices where valid is True, NaN where it is
    False, finite non-negative errors, no singular matrices.
    """
    from starter.load_homography import load as load_homography

    try:
        matrices, reprojection_error, valid = load_homography(
            game=game,
            data_dir=input_root,
            source="tactical",
        )
    except OSError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, OverflowError) as exc:
        if "missing columns" in str(exc):
            raise ValueError("Homography CSV is missing required columns") from None
        raise ValueError("Homography asset contains invalid data") from None
    if not (len(matrices) == len(valid) == len(reprojection_error) == n_frames):
        raise ValueError("Homography frame count does not match video")

    return Geometry(
        matrices=matrices,
        valid=valid,
        reprojection_error=reprojection_error,
    )


def _load_events(game, input_root, fps, n_frames):
    """Smart Tagging intervals as contract Events: half-open, ordered, normalized.

    starter.load_smart_tagging already converts seconds to video frames and
    half-open intervals (end_frame is the first frame after the phase), so this
    validates rather than re-derives: every interval must be non-empty and lie
    inside the timeline, every code must be present, and every team label must
    be one the mapping knows. Unknown labels raise instead of becoming None.
    """
    from starter.load_smart_tagging import load

    try:
        raw_events = load(game=game, data_dir=input_root, fps=fps)
    except OSError:
        raise
    except SyntaxError:  # xml.etree.ElementTree.ParseError
        raise ValueError("Smart Tagging asset is not well-formed XML") from None
    except (LookupError, UnicodeDecodeError):
        raise ValueError("Smart Tagging asset uses an unsupported XML encoding") from None
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Smart Tagging asset contains invalid event timestamps") from None

    events = []
    for index, item in enumerate(raw_events):
        try:
            start = int(item["start_frame"])
            end = int(item["end_frame"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                f"Smart Tagging event {index} has an invalid frame interval"
            ) from None
        code = (item["code"] or "").strip().upper()

        if not code:
            raise ValueError(f"Smart Tagging event {index} has no code")

        if not (0 <= start < end <= n_frames):
            raise ValueError(
                f"Invalid Smart Tagging interval at event {index}"
            )

        raw_team = (item["tags"].get("Team") or "").strip() or NO_TEAM
        if raw_team == NO_TEAM:
            team = None
        elif raw_team in TEAM_SIDES:
            team = TEAM_SIDES[raw_team]
        else:
            raise ValueError(
                f"Smart Tagging event {index} names an unknown team"
            )

        tags = {
            str(key): str(value)
            for key, value in item["tags"].items()
            if key is not None and value is not None
        }

        events.append(
            Event(
                event_id=f"smart-tagging-{index + 1:04d}",
                start_frame=start,
                end_frame=end,
                code=code,
                team=team,
                tags=tags,
            )
        )

    return tuple(
        sorted(
            events,
            key=lambda event: (
                event.start_frame,
                event.end_frame,
                event.event_id,
            ),
        )
    )


def _segments_and_half(game, n_frames):
    """The published half blocks as half-open segments plus the frame->half array."""
    segments = tuple(
        (start, end + 1, half)
        for half, start, end in games.half_video_blocks(game)
    )
    half = np.zeros(n_frames, dtype=np.int8)
    for start, stop, half_number in segments:
        half[start:stop] = half_number
    return segments, half


def _without_paths(exc, paths, input_root=None):
    """Re-raise a loader error with every local asset or root path removed."""
    message = str(exc)
    replacements = []
    for label, entries in (("<input-root asset>", paths), ("<input-root>", () if input_root is None else (input_root,))):
        for entry in entries:
            p = Path(entry)
            for candidate in (str(p), p.as_posix()):
                if candidate and candidate != ".":
                    replacements.append((candidate, label))
            try:
                resolved = p.resolve()
                for candidate in (str(resolved), resolved.as_posix()):
                    if candidate and candidate != ".":
                        replacements.append((candidate, label))
            except OSError:
                pass
    for candidate, label in sorted(set(replacements), key=lambda pair: len(pair[0]), reverse=True):
        pattern = r"(?<![\w.~-])" + re.escape(candidate) + r"(?![\w~-]|\.\w)"
        message = re.sub(pattern, label, message)
    if isinstance(exc, TypeError):
        return TypeError(message)
    return ValueError(message)


def prepare_match(input_root: Path) -> MatchInput:
    """Load the tactical assets under input_root and build a validated MatchInput.

    input_root is the data directory itself (the one containing tactical/), the
    path the CLI passes through unchanged. All failures surface as ValueError
    or TypeError with messages free of local paths and data excerpts.
    """
    input_root = Path(input_root)
    game = games.GAME
    config = games.GAMES[game]

    n_frames = config["n_video_frames"]
    fps = config["fps"]
    pitch_length, pitch_width = config["pitch"]

    try:
        atd_path, meta_path, homography_path, tagging_path = games.require(
            ["atd_raw", "atd_meta", "homography", "smart_tagging"],
            game=game,
            data_dir=input_root,
            source="tactical",
        )
    except OSError as exc:
        raise ValueError(
            "tactical match data not found under the provided input root: "
            "expected tactical/BAR-ATH_tactical_atd.txt, "
            "tactical/BAR-ATH_tactical_atd.xml, "
            "tactical/BAR-ATH_tactical_homography.csv and "
            "tactical/BAR-ATH_tactical_smart_tagging.xml. "
            "The bundle is gated -- see DATA.md."
        ) from exc

    paths = (atd_path, meta_path, homography_path, tagging_path)

    try:
        slot_team = _load_slot_teams(meta_path)
        (
            player_xy,
            player_observed,
            player_speed,
            ball_xy,
            ball_observed,
            ball_speed,
        ) = _load_atd(atd_path, n_frames, len(slot_team))
        geometry = _load_geometry(game, input_root, n_frames)
        events = _load_events(game, input_root, fps, n_frames)
    except OSError as exc:
        raise ValueError(
            "a tactical match asset could not be read from the provided input root"
        ) from exc
    except SyntaxError as exc:
        raise ValueError("a tactical match asset is not well-formed XML") from exc
    except (ValueError, TypeError) as exc:
        raise _without_paths(exc, paths, input_root) from exc

    segments, half = _segments_and_half(game, n_frames)

    return MatchInput(
        schema_version=1,
        match_id=game,
        source="tactical",
        n_frames=n_frames,
        fps=fps,
        pitch_length_m=pitch_length,
        pitch_width_m=pitch_width,
        coordinate_system="pitch_normalized_0_1",
        team_names={side: name for name, side in TEAM_SIDES.items()},
        segments=segments,
        frame=np.arange(n_frames, dtype=np.int64),
        half=half,
        slot_id=np.array([str(i) for i in range(1, len(slot_team) + 1)]),
        slot_team=slot_team,
        player_xy=player_xy,
        player_observed=player_observed,
        player_speed_mps=player_speed,
        ball_xy=ball_xy,
        ball_observed=ball_observed,
        ball_speed_mps=ball_speed,
        events=events,
        geometry=geometry,
    )
