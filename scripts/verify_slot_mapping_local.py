"""Local-only EPTS slot-mapping cross-check.

This is not a test and must not be run in an AI-connected session against
restricted competition data. It is intended for the human operator in an
isolated environment with the starter optional dependencies installed.

The script loads the repository's Data Engineering loader and the authoritative
Kloppy EPTS adapter. It compares slot-to-team assignments element by element
and prints only aggregate summary counts (slots, matches, mismatches), never
player identities, tracking rows, raw values, or data excerpts.
"""

from __future__ import annotations

import argparse
import sys
from collections import namedtuple
from pathlib import Path

# Repository root and sources are placed on sys.path for local execution only.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


SlotMappingResult = namedtuple(
    "SlotMappingResult", ["slots", "matches", "mismatches", "passed"]
)


def compare_slot_mapping(loader_sides, reference_sides):
    """Compare loader and reference slot sides element by element.

    The reference must be independently established by player identity and
    channel order. This function never collapses the result to team totals.
    Missing or ambiguous reference identities fail closed.
    """
    if loader_sides is None or reference_sides is None:
        raise ValueError("Slot mapping reference is unavailable")
    if len(loader_sides) != len(reference_sides):
        raise ValueError("Slot mapping sizes differ")

    sides = list(loader_sides)
    reference = list(reference_sides)
    if not sides:
        raise ValueError("Slot mapping reference is unavailable")
    if any(side not in ("H", "A") for side in sides):
        raise ValueError("Invalid loader slot side")

    mismatches = sum(1 for a, b in zip(sides, reference) if b not in ("H", "A") or a != b)
    return SlotMappingResult(
        slots=len(sides),
        matches=len(sides) - mismatches,
        mismatches=mismatches,
        passed=mismatches == 0,
    )


def _kloppy_team_name(dataset, player):
    if player is None:
        return None
    team = getattr(player, "team", None)
    ground = getattr(team, "ground", None)
    if ground is not None and str(ground).upper().endswith("HOME"):
        return "H"
    if ground is not None and str(ground).upper().endswith("AWAY"):
        return "A"
    return None


def _iter_player_channel_refs(split_register):
    """Yield PlayerChannelRef-like objects in specification document order."""
    children = getattr(split_register, "children", None)
    if children is None:
        if hasattr(split_register, "player_channel_id"):
            yield split_register
        return
    for child in children:
        yield from _iter_player_channel_refs(child)


def _kloppy_slot_sides(dataset):
    """Extract the authoritative Kloppy slot order by player/channel identity.

    This intentionally does **not** use first-record `players_data` order or
    aggregate team counts. It follows each DataFormatSpecification's
    PlayerChannelRef order through Kloppy's ``PlayerChannel`` objects, so a
    slot receives the side of the exact identified player.
    """
    metadata = dataset.metadata
    player_channels = getattr(metadata, "player_channels", None)
    specifications = getattr(metadata, "data_format_specifications", None)
    if not player_channels or not specifications:
        raise ValueError("Kloppy EPTS channel mapping is unavailable")

    channel_map = {}
    for channel in player_channels:
        channel_id = getattr(channel, "player_channel_id", None)
        if channel_id is None or not str(channel_id).strip():
            raise ValueError("Kloppy EPTS channel mapping is incomplete")
        if channel_id in channel_map:
            raise ValueError("Kloppy EPTS metadata contains duplicate player channel ids")
        channel_map[channel_id] = channel

    orders = []
    for specification in specifications:
        split_register = getattr(specification, "split_register", None)
        if split_register is None:
            raise ValueError("Kloppy EPTS channel mapping is unavailable")
        ordered = []
        for ref in _iter_player_channel_refs(split_register):
            reference = getattr(ref, "player_channel_id", None)
            channel = channel_map.get(reference) if reference is not None else None
            if channel is None:
                raise ValueError("Kloppy EPTS channel mapping is incomplete")
            player = getattr(channel, "player", None)
            if player is None:
                raise ValueError("Kloppy EPTS channel mapping has a missing player")
            if player not in ordered:
                ordered.append(player)
        if not ordered:
            raise ValueError("Kloppy EPTS channel mapping is unavailable")
        orders.append(ordered)

    first_order = orders[0]
    if any(
        len(order) != len(first_order)
        or any(a != b for a, b in zip(order, first_order))
        for order in orders[1:]
    ):
        raise ValueError("Kloppy EPTS specifications define different slot orders")
    ordered_players = first_order

    sides = []
    for player in ordered_players:
        side = _kloppy_team_name(dataset, player)
        if side is None:
            raise ValueError("Kloppy EPTS channel mapping has a missing team")
        sides.append(side)
    return sides


def main() -> int:
    from starter.load_atd import load as load_kloppy_atd
    from mtah.pipeline import prepare_match

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="data",
        help="Parent directory containing tactical/ (human-local data root).",
    )
    args = parser.parse_args()

    match = prepare_match(Path(args.input_root))
    dataset = load_kloppy_atd(game="BAR-ATH", data_dir=args.input_root, source="tactical")

    try:
        kloppy_sides = _kloppy_slot_sides(dataset)
    except ValueError as exc:
        print(f"Invalid mapping: {exc}")
        return 1

    if len(kloppy_sides) != match.slot_team.size:
        print(f"Invalid mapping size: loader slots={match.slot_team.size}, kloppy tracks={len(kloppy_sides)}")
        return 1

    try:
        result = compare_slot_mapping(match.slot_team.tolist(), kloppy_sides)
    except ValueError as exc:
        print(f"Invalid mapping: {exc}")
        return 1

    if not result.passed:
        print(
            f"slots={result.slots} matches={result.matches} "
            f"mismatches={result.mismatches}"
        )
        print("INVALID: slot-to-team assignments differ")
        return 1

    print(f"slots={result.slots} matches={result.matches} mismatches=0")
    print("OK: loader slot-to-team mapping matches kloppy EPTS parsing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
