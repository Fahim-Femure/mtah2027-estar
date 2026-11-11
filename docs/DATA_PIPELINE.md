# Data pipeline — the real-data `MatchInput` handoff

Date: 2026-11-06. Real-data loader milestone shipped in commit `81dd361` (hardened in `166f35f`).
Owner: Data Engineering. Consumers: the possession engines and the CLI described in
`src/mtah/__main__.py`; the contract itself is defined in
`src/mtah/contracts.py` (`MatchInput` / `PredictionResult`).

> **Post-promotion status (`8a093bb` algorithm baseline / `d3faa90` reproduction baseline):** The Data Engineering loader (`166f35f`) and Smart Tagging `v0.2` (`d8a0dd0`, `possession-rules-v0.2`) are integrated. Current environment setup uses `requirements.lock` (see Section 1); historical references below to milestone commits (`81dd361`, `2026-11-06`) reflect their original development steps.

Everything below was checked against `src/mtah/pipeline.py`, `src/mtah/contracts.py` and
`src/mtah/sources.py` on the date above, by loading the real bundle. Asset formats and the
competition's own frame-alignment table stay in [DATA.md](../DATA.md) — this document only
covers what the pipeline exposes to the rest of the code.

## 1. Install and run

The package requires **Python 3.11 or newer** (`pyproject.toml` sets `requires-python = ">=3.11"`)
and `numpy` + `pandas`, both from the default requirements — the pipeline adds no further
dependency.

The official reproducibility environment for this handoff is **CPython 3.11.9 x64 on Windows**.
That environment was verified with:

- interpreter: `3.11.9 (tags/v3.11.9:de54cf5, Apr 2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)]`;
- NumPy `2.4.6`;
- pandas `3.1.1`;
- python-dateutil `4.0.0`;
- tzdata `2026.6`.

```powershell
# PowerShell, repository root
.\scripts\setup.ps1 -Python311 <path-to-cpython-3.11.9-x64>   # creates .venv, installs requirements.lock + editable package, pip check

# or by hand
& "<path-to-cpython-3.11.9-x64>" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock   # pinned numpy/pandas/dateutil/tzdata
.venv\Scripts\python.exe -m pip install --no-deps -e .         # makes `mtah` importable without PYTHONPATH
```

- `requirements-starter.txt` (kloppy, matplotlib, notebook) is for the `starter/` notebooks only:
  the pipeline parses the ATD itself, one line at a time, and never imports kloppy.
- `requirements.lock` records the exact four resolved runtime packages for the verified
  environment:

```text
numpy==2.4.6
pandas==3.1.1
python-dateutil==4.0.0
tzdata==2026.6
```

  A separate isolated venv created from `requirements.lock` produced an exact `pip freeze`
  match, a clean `pip check`, and passed the synthetic suite (141 tests at initial lock
  creation; 221 tests after S2, Data Engineering integration, and review hardening; **240 tests**
  after integrating Smart Tagging v0.2 in `8a093bb`). The
  lock is the reproducibility record for the verified Windows x64 / CPython 3.11 environment;
  other Python versions and platforms are not covered by that result.
- `scripts/setup.ps1` and `scripts/run_checks.ps1` are stricter than the package: they insist on
  CPython **3.11 x64 exactly** and refuse any other interpreter, and they expect a checkout-local
  `.venv\Scripts\python.exe` inside the repository root (`$repo\.venv`). In a primary checkout
  with `.venv` created via `scripts/setup.ps1`, the gate has been verified under CPython
  **3.11.9 x64 on Windows** (`scripts/setup.ps1` completed and `scripts/run_checks.ps1`
  `scripts/run_checks.ps1` passed). Separate checkouts without a local `.venv` invoke a shared `.venv`
  interpreter directly (`python -B -m unittest discover -s tests`, `pip check`, and
  `scripts/check_staged.py`). Reproducibility claims in this guide are scoped to that Windows
  x64 / CPython 3.11.9 environment; they do not by themselves extend to other versions or
  platforms.

Run the real-data pipeline end to end:

```powershell
# PowerShell, repository root; skip the PYTHONPATH line if you did `pip install --no-deps -e .`
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m mtah --source atd --input data --engine hybrid --out artifacts\run\submission.csv
.venv\Scripts\python.exe evaluation\validate_submission.py artifacts\run\submission.csv
```

```bash
# Git Bash / POSIX
PYTHONPATH=src python -m mtah --source atd --input data --engine hybrid --out artifacts/run/submission.csv
python evaluation/validate_submission.py artifacts/run/submission.csv
```

On success the CLI prints exactly one line and exits 0:

```
atd engine=hybrid rows=199319 validated=True sha256=<64 hex>
```

- `--input` is required with `--source atd` and rejected with `--source synthetic`.
- `--out` must be a new file (exit 1 if it exists); paths inside `evaluation/` are refused.
- Exit codes are 0 / 1 / 2 exactly as specified in
  `src/mtah/__main__.py` (`0` = success, `1` = output refused/exists/inside `evaluation/`, `2` = usage, input, or loader error). Every input failure is exit 2,
  printed to stderr, with no local path in the message and no CSV written.
- Keep run outputs under `artifacts/` — it is git-ignored (`.gitignore`).

Tests (synthetic/unit suite):

```bash
python -B -m unittest discover -s tests  # verified at 8a093bb / d3faa90: 240 tests, OK
```

## 2. Calling `prepare_match`

```python
from pathlib import Path
from mtah.pipeline import prepare_match

match = prepare_match(Path("data"))     # -> mtah.contracts.MatchInput
```

- Signature: `prepare_match(input_root: Path) -> MatchInput`. A `str` is accepted as well (it is
  converted with `Path(...)` on entry).
- Reads exactly the four tactical assets listed in section 3. It never opens the video, never
  touches the network, and prints nothing — stdout belongs to the CLI's single result line.
- Returns a contract v1 `MatchInput`: a frozen dataclass whose arrays are defensively copied and
  flagged read-only at construction, so a consumer cannot mutate them after the handoff.
- Failures raise `ValueError` or `TypeError` whose messages carry no local paths and no data
  excerpts (integration requirement 3), so the CLI reports exit 2 and the log stays shareable.
  Nothing is padded, trimmed, sorted or guessed to make counts fit: a malformed record, a frame
  mismatch, an unknown team or a missing asset stops preparation outright.

## 3. Why `input_root` is the data directory, not `data/tactical`

`games.asset_paths()` builds every asset path itself, and it appends the source folder:

```python
asset_paths(data_dir=root)["atd_raw"]   # -> <root>/tactical/BAR-ATH_tactical_atd.txt
```

`prepare_match` passes `input_root` through unchanged as `data_dir` to `games.require()` and to
the starter loaders (`starter.load_homography`, `starter.load_smart_tagging`), all of which use
that same helper. The argument is therefore the directory that *contains* `tactical/`:

```text
<input_root>/
├─ tactical/
│  ├─ BAR-ATH_tactical_atd.txt            tracking, one record per frame (players + ball)
│  ├─ BAR-ATH_tactical_atd.xml            metadata: teams, players, slot order
│  ├─ BAR-ATH_tactical_homography.csv     one 3x3 matrix per frame
│  └─ BAR-ATH_tactical_smart_tagging.xml  tactical phases and set pieces
└─ broadcast/                             not read by prepare_match
```

Pointing `input_root` at `data/tactical` makes the helper look for
`data/tactical/tactical/...` and fails with
`tactical match data not found under the provided input root` — exit 2, and the message names
the expected *relative* filenames instead of the local path.

`input_root` also takes precedence over `MTAH_DATA_DIR`: `games.py` only falls back to that
environment variable when `data_dir` is `None`, and `prepare_match` always passes it explicitly.
`MTAH_DATA_DIR` still applies to the rest of the tooling (`games.asset_paths()` with no
arguments, the notebooks, `starter/` helpers called directly).

## 4. `MatchInput` fields

Defined in `src/mtah/contracts.py`. Shapes below are for the competition bundle:
**N = 199319 frames**; **S** (slot count) is read dynamically from the ATD metadata, never hard-coded.

| Field | Type / shape | For BAR-ATH |
|---|---|---|
| `schema_version` | `int` | `1` |
| `match_id` | `str` | `"BAR-ATH"` |
| `source` | `str` | `"tactical"` (the only source contract v1 accepts) |
| `n_frames`, `fps` | `int` | `199319`, `25` |
| `pitch_length_m`, `pitch_width_m` | `float` | `105.3`, `68.0` |
| `coordinate_system` | `str` | `"pitch_normalized_0_1"` |
| `team_names` | mapping H/A → name | `{"H": "Barcelona", "A": "Atletic Club"}` |
| `segments` | tuple of half-open `(start, stop, half)` | `((11185, 90652, 1), (110808, 190198, 2))` |
| `frame` | `(N,) int64` | `arange(N)`, the canonical row key |
| `half` | `(N,) int8` | `0` outside the published blocks (40462 frames), `1` (79467), `2` (79390) |
| `slot_id` | `(S,) str` | `"1"` … `str(S)`, unique, non-empty |
| `slot_team` | `(S,) str` | `"H"` / `"A"` per slot from metadata teams |
| `player_xy` | `(N, S, 2) float64` | normalised x/y; absent pairs are both `NaN` |
| `player_observed` | `(N, S) bool` | True iff both coordinates are finite |
| `player_speed_mps` | `(N, S) float64` | third ATD value per slot; `NaN` where absent |
| `ball_xy` | `(N, 2) float64` | same convention; absent pairs are both `NaN` |
| `ball_observed` | `(N,) bool` | True iff both ball coordinates are finite |
| `ball_speed_mps` | `(N,) float64` | **all `NaN`** — see section 10 |
| `events` | `tuple[Event, ...]` | sorted Smart Tagging `Event` intervals (section 8) |
| `geometry` | `Geometry` or `None` | present for the tactical source (section 8) |

Derived, read-only properties of the record: `MatchInput` is frozen, every array is a copy with
`writeable = False`, `team_names` and `Event.tags` are `MappingProxyType`. Make a copy if you
need working memory.

## 5. Frame indexing conventions

| Where the index comes from | Basis | Conversion |
|---|---|---|
| ATD `frame_id` | **1-based**, `1 … 199319` | `video_frame = frame_id - 1` (`mtah.alignment.atd_to_video`) |
| `MatchInput.frame`, submission rows | **0-based**, `0 … 199318` | identity: `frame = arange(N)` |
| homography CSV `frame` column | **0-based** | already a submission row; only the count is checked |
| Smart Tagging `<start>` / `<end>` | seconds from video start | `start_frame = round(start_s * 25)` (starter loader) |
| published half blocks | inclusive last frame | converted to half-open `(start, end + 1, half)` |

Rules the loader enforces:

- **ATD records are exact and in order.** `_load_atd` walks the file line by line and raises if
  a record's id is not exactly its 1-based position, if a frame falls outside the timeline, or
  if the record count differs from `n_frames`. No padding, trimming or reordering.
- **Event intervals are half-open `[start, end)`.** `Event.contains(frame)` is
  `start_frame <= frame < end_frame`, and contract validation requires `0 <= start < end <= N`.
  `end_frame` is the first frame *after* the phase.
- **`segments` are half-open too**: `half[start:stop] = half`, so `stop` is the first frame after
  the block. `games.half_video_blocks()` publishes *inclusive* last frames (90651 and 190197);
  the pipeline adds 1 to get `90652` / `190198`.
- **Frames outside both segments** (before the match, halftime, after the final whistle) have
  `half == 0` and belong to no segment. Reset temporal state at block breaks; do not smooth
  across halftime.

## 6. Coordinates and pitch dimensions

- `coordinate_system = "pitch_normalized_0_1"`: **x along the pitch length, y along the width**,
  both nominally in `[0, 1]`. Fixed source orientation — no per-team mirroring, ever.
- Metres are `(x * pitch_length_m, y * pitch_width_m)` = `105.3 × 68.0` (the value `games.py`
  carries, measured from the corrected tracking; the ATD metadata reports 105 × 68 — see
  [DATA.md](../DATA.md), the 30 cm matters only if you compute distances in metres).
- The homography maps the **same 0–1 convention** to image pixels:
  `starter.load_homography.project(H[frame], x, y)`. Never feed metre values into it.
- Finite out-of-pitch coordinates are preserved, not clipped (a ball may be off the pitch);
  report them as quality information instead of moving them.
- Infinity is rejected outright; a partial pair (one coordinate finite, the other `NaN`) is a
  contract error. So `player_xy` / `ball_xy` entries are always both-finite or both-`NaN`, and
  `player_observed` / `ball_observed` mirror exactly that.

## 7. How player slots map to teams

- **The raw triple order comes from the EPTS `DataFormatSpecification`.**
  `_load_slot_teams` reads `PlayerChannelRef` entries in specification order and maps
  each through `PlayerChannels/PlayerChannel` to the `<Player playerId="...">` element, then
  from that element to its metadata team. The `//Players/Player` document order is retained only
  to build the player-to-team table and as a fallback for metadata with no
  `DataFormatSpecification`; it no longer defines slot order when a specification is present.
  Both nested (`<DataFormatSpecifications>/<DataFormatSpecification>`) and top-level
  `<DataFormatSpecification>` elements are inspected. Required and unique `id` attributes are
  enforced on `<Team>`, `<Player>`, and `<PlayerChannel>` entries; an unknown channel or a
  specification that does not cover every tracked player exactly once raises.
- Player IDs must still be the contiguous sequence `1 … S` and a non-contiguous or duplicate
  ID sequence raises. Team assignment for each slot derives from both the specification order and
  the metadata teams, never assumed document-order numbering.
- The vendor raw-file value order is **x, y, speed** for each player and the same three-value
  triple for the ball. `_load_atd` enforces exactly three fields per slot; it does not derive
  or accept other field orders. The EPTS specification governs slot order through
  `PlayerChannelRef`; its channel names are not used to reorder fields inside a player group.
- **The side comes from the metadata team name**, via
  `TEAM_SIDES = {"Barcelona": "H", "Atletic Club": "A"}`. An unknown team name raises
  `ATD metadata names an unknown team ...`; the mapping is never guessed from position.
- `_load_slot_teams` verifies metadata `<Player>` IDs cover the contiguous set `"1"` … `str(S)`
  and returns `slot_team` (`"H"` or `"A"`) in specification order; `prepare_match` sets
  `slot_id` to sequential 1-based slot index strings `"1"` … `str(S)`.
- Slot-mapping verification script: `scripts/verify_slot_mapping_local.py` (unit-tested with mock
  inputs in `tests/test_pipeline.py`). It checks element-by-element agreement between
  `_load_slot_teams` and the upstream `starter/load_atd.py` (`kloppy`) EPTS parser when run locally
  with optional starter dependencies installed, rejecting missing/blank or duplicate
  `player_channel_id` entries, missing `player` references, missing `split_register` blocks, and
  conflicting specifications without printing player identities or tracking excerpts.
- **A slot is a track, not a person.** There are no shirt numbers and no player identities;
  slots appear, vanish and swap identity as the tracker loses and re-acquires people (see
  [DATA.md](../DATA.md)). Any analysis has to be collective rather than per-player.

## 8. Smart Tagging events and homography

### `Event` records

| Field | Meaning |
|---|---|
| `event_id` | `smart-tagging-NNNN`, assigned in XML file order (1-based), so it is stable for a given bundle |
| `start_frame`, `end_frame` | half-open video-frame interval, validated `0 <= start < end <= N` |
| `code` | trimmed and upper-cased, e.g. `SET PIECES`, `BALL IN THE BOX` |
| `team` | `"H"`, `"A"` or `None` |
| `tags` | original key casing (`Team`, `Half`, `Type`, `Side`, `Direction of ball entry`, `Max Players in the box`) and original values, as a read-only mapping |

- Team mapping: `Barcelona` → `"H"`, `Atletic Club` → `"A"`, the literal `N/A`
  → `None` (including throw-ins). **Any other label raises** instead of being dropped
  or silently coerced.
- `events` is ordered by `(start_frame, end_frame, event_id)` as the contract requires;
  overlapping intervals are kept as-is.
- These are automatic tactical annotations, **not possession ground truth** — see section 10.

### `Geometry` record

| Field | Shape / dtype | Semantics |
|---|---|---|
| `matrices` | `(N, 3, 3) float64` | pitch→pixel transform per frame, row-major from the CSV |
| `valid` | `(N,) bool` | whether a usable matrix exists for that frame |
| `reprojection_error` | `(N,) float64` | lower is better, not a probability |

- Where `valid` is False the whole matrix row is `NaN`; where it is True the matrix is finite and
  well conditioned (condition number < 1e12), and the error is finite and non-negative.
- The pipeline checks only that the three arrays are length `N`;
  `Geometry.__post_init__` enforces matrix finiteness, conditioning, and non-negative reprojection error.
- Load and project through `starter.load_homography` (`load`, `project`, `unproject`); NaN in →
  NaN out, so an invalid frame gives obviously broken pixels rather than plausible wrong ones.
  Check `valid` first anyway.

## 9. Missing data and NaN handling

- An absent observation is **paired `NaN` plus `False` in the matching mask** — for players and
  ball alike. Nothing is zero-filled, interpolated or dropped.
- Player and ball tracks contain intermittent missing observations (dropouts); design for missing
  observations rather than assuming complete tracking coverage.
- Speeds are `NaN` wherever the source has no value; no speed is ever inferred as zero. The
  contract rejects infinite and negative speeds.
- `ball_speed_mps` is `NaN` on **every** frame by construction (section 10).
- A missing or unreadable asset, a malformed record or an unknown label raises — it never
  degrades into a partial `MatchInput`. Error messages name the expected relative filenames and
  point at DATA.md, with no local path in them.

### Error handling and diagnostic privacy

Parsing errors are sanitized before they reach the CLI. The loader reports safe diagnostic
context only, such as the record number, field category and field position:

- `invalid ATD frame id at record N`
- `invalid numeric ATD player coordinate at record N, field K`
- `invalid numeric ATD player speed at record N, field K`
- `invalid numeric ATD ball coordinate at record N, field K`
- `invalid numeric ATD ball speed at record N, field K`

Raw ATD field text, malformed-token text, private paths, unknown club strings, raw
homography/tagging parse values, unsupported XML encoding names, and invalid ATD frame IDs
are omitted or sanitized across `_load_slot_teams`, `_load_atd`, `_load_geometry`,
`_load_events`, and boundary-aware `_without_paths` so direct callers of `prepare_match` do not
receive paths or data excerpts. At the CLI boundary, `redact(str(exc), args.input)` strips
every spelling of `--input` (including unquoted paths), and `(ValueError, TypeError)` messages
additionally pass quoted fragments through `_loader_error_message` without letting English
contractions or possessives invert quote pairing. The loader’s exit code remains 2.

## 10. Known limitations

1. **Ball speed stays `NaN`.** The ball group's third value is documented as "speed" but its
   units and meaning are unverified, so it is not guessed (integration requirement 6.2). Consumers
   must derive ball speed from positions if they need it.
2. **Player speed units are an assumption.** `player_speed_mps` is passed through from the third
   ATD value per slot; [DATA.md](../DATA.md) documents the export as m/s, but this has not been
   verified against the video, and no engine consumes it today
   ([POSSESSION_ENGINE.md](POSSESSION_ENGINE.md) lists player speed as unused). Treat it as
   unverified until someone checks it.
3. **Smart Tagging is not possession ground truth.** It is Metrica's automatic inference from the
   same video and tracking, with false positives and false negatives; `Team` is `N/A` on every
   throw-in; the intervals are phases, not possessions. Evidence, never a label.
4. **Dataset assets are local and must never be committed.** `data/*` and `artifacts/` are
   git-ignored; `git ls-files data/` must stay `data/README.md`. `scripts/check_staged.py`
   rejects staged `_atd`, `_homography`, `_smart_tagging` and `possession_gt` files as likely
   competition assets. Do not redistribute the bundle — the terms you signed cover that.
5. **Tactical source only.** `MatchInput.source` must be `"tactical"`; the broadcast recording
   has its own timeline and no supplied mapping, and is out of contract v1.
6. **No scoring here.** `prepare_match` builds input only; ground truth is read exclusively by
   `evaluation/`, never by ingestion.
7. **Slot order depends on `DataFormatSpecification` when present.** The loader uses the
   specification's channel order. Metadata without a specification uses the
   `//Players/Player` document order as a fallback, so only that fallback remains an
   assumption for non-EPTS-shaped files.
8. **Multiple `DataFormatSpecification` sections must define the same slot order.**
    The delivered vendor metadata contains one specification for the whole timeline.
    If a metadata file contains multiple sections with differing channel orders,
    `_load_slot_teams` raises a fail-loud error; the current contract does not define
    a per-frame slot-order model.

## Handoff: loading `MatchInput` from another module

Consume the public source API — the same seam the CLI uses — instead of parsing raw files:

```python
from pathlib import Path

from mtah.sources import load_match      # dispatch: synthetic fixture or real-data loader
from mtah.engines import predict_with    # any consumer that only needs a MatchInput

match = load_match("atd", input_root=Path("data"))   # -> MatchInput, contract-validated
result = predict_with("hybrid", match)               # engines never see a path
```

- `load_match("atd", input_root=...)` resolves `mtah.pipeline.prepare_match` and type-checks its
  return (`TypeError` if it is not a `MatchInput`); an unknown source raises `ValueError`.
- `load_match("synthetic", frames=40)` returns an invented fixture and opens no assets at all —
  use it in tests, never the competition data.
- The CLI is a thin wrapper over exactly this call, so anything that works through
  `load_match` works through `python -m mtah --source atd --input ...`.
- For one-off inspection of raw assets, use the starter helpers (`starter.load_homography`,
  `starter.load_smart_tagging`, `starter/load_atd.py`) directly — but engine and pipeline code
  should go through `load_match` / `prepare_match` so validation happens in exactly one place.

## Verification performed for this document

### Python 3.11 clean-rebuild and verification summary

An earlier isolated rebuild (DE commit `81dd361`) installed exclusively from `requirements.lock`
produced the same four `pip freeze` lines, clean `pip check`, and **141 tests, OK**. Separate local
operational validation of the integrated data pipeline and exporter was subsequently performed
without publishing match-level artifacts or digests; competition accuracy and ground-truth scoring
have not been measured.

Integrated verification (`8a093bb` / `d3faa90`, combining DE `166f35f` and Smart Tagging v0.2 `d8a0dd0`; synthetic and mock fixtures only; competition bundle not
re-accessed, CPython 3.11.9 x64):

```text
python -B -m unittest discover -s tests      -> Ran 240 tests, OK
python -O -B -m unittest discover -s tests   -> Ran 240 tests, OK
python scripts/check_environment.py          -> environment matches requirements.lock
python -m pip check                          -> No broken requirements found
python scripts/check_staged.py               -> 0 paths staged, 0 rejected
python scripts/submission_preflight.py ...   -> PASS (synthetic and mock CLI outputs)
```
