# Smart Tagging v0.2 Implementation Report (`possession-rules-v0.2`)

Date: 2026-11-10
Milestone: Smart Tagging v0.2 implementation (`d8a0dd0`)
Environment: `<isolated-checkout>` (Python 3.11.9 x64)
Base commit: `5a4e9a4e5dcdaa24e607889fe883e15af92bfc9a` (decision record)
Implementation commit: `f5616c7287460ba91c67b7b4a1a1fff85edc064d` (`f5616c7`)

> **Compliance note (v0.2 implementation and synthetic-test phase):** During the v0.2 engine implementation and synthetic verification recorded in this milestone report, all runs used only invented synthetic fixtures (`tests/scenarios.py`, `mtah.synthetic`, `tests/mock_loader.py`) and public documentation/code; no competition match data, ground truth, or video was read by those runs, and no parameter was calibrated or tuned. Synthetic results verify rule logic and regression cases only and make no claim about competition accuracy.

> **Post-promotion status (`8a093bb` algorithm baseline / `d3faa90` reproduction baseline):** Integrated from `8a093bbf0270f35f3defe575220c5ff25b691baa`. Under CPython `3.11.9` (64-bit), `scripts/check_environment.py` verifies exact `3.11.9` and `requirements.lock`, and all `240` synthetic/mock tests pass in both normal (`-B`) and optimized (`-O -B`) runs. Separately from the synthetic test suite and from competition scoring, the non-GT Section 8 aggregate diagnostic noted as open in §9.1 was subsequently executed locally for operational verification without ground-truth scoring, accuracy measurement, or parameter selection/tuning. Historical milestone findings below are preserved unchanged.

---

## 1. Policy Implemented

Implements Option A (abstain) from `docs/ENGINE_DECISION_TAGGING_NO_ONSET.md`:

- **No confirmed movement evidence means no confirmed restart.**
- **Cluster formation (`deadball.cluster_set_pieces`):** Per segment, `SET PIECES` events whose `start_frame` lies inside `[seg_start, seg_stop)` (with `end_frame` clipped to `seg_stop`) are grouped transitively by interval overlap into `[S, E)` clusters. Abutting tags (`e0 == s0'`) stay separate clusters. A cluster's team is the common `"H"`/`"A"` team when all tagged members agree (`None`/`N/A` members do not vote), else `None`.
- **Candidates and superseded onsets:** Candidates are non-superseded movement evidence frames (`moving_inside` onsets with `t > seg_start`, or sparse-displacement `moved` hold releases) inside `[max(seg_start, S - w), E)`, where `w = set_piece_window_frames = 50`. An onset `h` followed strictly before `E` by at least `min_dead_frames` (25) consecutive observed rest frames (`motion_speed_mps < stationary_speed_mps`, 0.3 m/s) is superseded as repositioning; its motion span `[h, rest_start)` remains part of the stoppage (`superseded_motion`) and neither releases an active hold nor blocks backward set-piece lead-in (`live_evidence`).
- **Cluster state machine:**
  1. `early`: earliest candidate in `[S, S + w]` -> confirmed restart `R`, hold ends at `R`, lead-in runs back from `R`.
  2. `before`: nearest candidate in `[S - w, S)` -> confirmed restart `R`, hold ends at `R`, lead-in runs back from `R`.
  3. `late`: earliest candidate in `(S + w, E)` **only when a stoppage is in progress at `S`** (`reason[S - 1] in (DEAD_OUT, DEAD_HOLD)` or backward lead-in from `S` is `>= min_dead_frames`) -> confirmed restart `R`, hold covers `[S, R)`, lead-in runs back from `R`.
  4. `unconfirmed`: no candidate, **and stoppage in progress at `S`** -> `[S, E)` labelled `X` via `DEAD_UNCONFIRMED` (`reason = "abstain_unconfirmed_restart"`, `confidence = NaN`), hold suspended in `[S, E)` and ended at `E`, lead-in runs back from `S`. Post-`E` live stretch up to first confirmed control uses `"backfilled_unconfirmed_restart"` (`0.4`), else `"set_piece_team_prior"` (`0.4`) if cluster team is set, else `"abstain_no_evidence"` (`NaN`).
  5. `ignored`: no stoppage in progress at `S` (and no `early`/`before` candidate) -> no dead state change, no window, no team prior, and no entry in `restarts`.
- **Search boundary:** Onset searches never extend past `E`; dead-ball holds never extend past `E` for unconfirmed windows.

## 2. Relevant Design Decisions

- **Internal additions without public contract changes:**
  - `src/mtah/engine/config.py`: `ENGINE_VERSION` updated from `"possession-rules-v0.1"` to `"possession-rules-v0.2"` (automatically updates `config.digest()`). No config fields added or tuned.
  - `src/mtah/engine/deadball.py`: added `DEAD_UNCONFIRMED = 5`, `cluster_set_pieces`, shared hold-advancement helper `_advance_hold`, sparse-displacement `moved` detection/probing inside `[S, E)`, and `_superseded_rest_start`. Tentative `superseded_motion` marks added while probing/evaluating a cluster are reverted if the cluster resolves as `ignored` so false-positive tags over live play cannot suppress live evidence for later clusters.
  - `src/mtah/engine/temporal.py`: added `"abstain_unconfirmed_restart"` and `"backfilled_unconfirmed_restart"` to `REASONS` (13 codes total), mapped `DEAD_UNCONFIRMED` to `X` / `NaN` / `"abstain_unconfirmed_restart"`, and distinguished post-`DEAD_UNCONFIRMED` backfill (`0.4`, `"backfilled_unconfirmed_restart"`) from evidence-backed restart backfill (`0.5`, `"backfilled_restart"`). `restart_prior` keeps exact nearest-within-window semantics, preferring confirmed restarts over unconfirmed cluster ends on exact distance ties.
  - `src/mtah/engine/possession.py`: added `EngineTrace.restart_kinds: tuple = ()` recording one kind (`early`, `before`, `late`, `unconfirmed`, `ignored`) per cluster in segment/event order while keeping `EngineTrace.restarts` as 4-tuples `(restart_or_E, team, seg_start, seg_stop)` for non-ignored clusters.
- **Unchanged public surface:** `src/mtah/contracts.py`, `src/mtah/pipeline.py`, `src/mtah/export.py`, `src/mtah/__main__.py`, `evaluation/`, `starter/`, and `games.py` are untouched.

## 3. Test-First Evidence

All commands ran in an isolated checkout with `PYTHONPATH=<repo>\src` using Python 3.11.9 (`<repo>\.venv\Scripts\python.exe`).

1. **Baseline at `5a4e9a4` (before any edits):**
   - `unittest discover -s tests`: **185 tests OK (expected failures=3)** in 26.446s.
   - `unittest discover -s tests` with `-O`: **185 tests OK (expected failures=3)** in 24.404s.
2. **Test-first red run (after adding T4–T12 + 4 additional coverage tests and revising T3 expectation, before editing `src/mtah/engine/`):**
   - `test_engine_robustness.py`: **75 tests — 4 failures, 9 errors, 3 expected failures** (all 13 new tests failed/errored against v0.1).
   - `test_integration_harness.py`: **23 tests — 1 failure** (`ScorerEquivalenceTests.test_unconfirmed_window_is_dead_after_official_abstention_fill`).
3. **Post-implementation pre-expectation-update run (engine updated, before removing `@unittest.expectedFailure` and before updating §5f expectations):**
   - `test_engine*.py`: **108 tests — 7 failures, 1 error, 3 unexpected successes** (T1, T2, T3 all passed unexpectedly; all 13 new tests passed; only the §5f tests with v0.1 expectations failed/errored).
   - `test_integration_harness.py`: **23 tests — OK** (including T12).
4. **Final post-update run (all markers removed, §5f expectations updated):**
   - Complete suite, normal mode: **199 tests OK** (27.803s).
   - Complete suite, `-O` mode: **199 tests OK** (29.939s).

## 4. T1–T12 and Additional Regression Results

All T1–T12 tests and 4 supplementary coverage tests pass in both normal and `-O` runs:

| ID | Test | Outcome | Verified Behavior |
|---|---|---|---|
| T1 | `TaggingWindowTests.test_restart_late_inside_the_tag_window` | Pass (was `expectedFailure`) | `late`; `restart = 326`; `H*100 + D*226 + A*274` |
| T2 | `TaggingWindowTests.test_kick_inside_window_beats_a_placement_nudge_before_it` | Pass (was `expectedFailure`) | `early`; `restart in {290,291,292}`; `dead_reason[251:290] != LIVE` |
| T3 | `TaggingWindowTests.test_restart_just_after_the_tag_window` | Pass (was `expectedFailure`, revised) | `unconfirmed`; `restart = 375`; `H*100 + D*150 + X*125 + A*225`; `reason[250:375] == abstain_unconfirmed_restart`; `reason[375:395] == backfilled_unconfirmed_restart` (`0.4`); `reason[395] == controlled` |
| T4 | `TaggingWindowTests.test_no_onset_after_stoppage_abstains_in_window` | Pass | `unconfirmed`; `X*125` (`NaN`); `reason[375:] == set_piece_team_prior` (`0.4`) |
| T5 | `TaggingWindowTests.test_no_onset_window_is_bounded_by_tag_end` | Pass | `unconfirmed`; `dead_reason[375:] == LIVE`; `labels[250:375] == X*125`; `labels[375:475] == A*100` (`backfilled_unconfirmed_restart`, `0.4`) |
| T6 | `TaggingWindowTests.test_no_onset_without_stoppage_is_ignored` | Pass | `ignored`; `restarts == ()`; `dead_reason == LIVE`; `H*600` |
| T7 | `TaggingWindowTests.test_late_onset_without_stoppage_is_ignored` | Pass | `ignored`; `restarts == ()`; `dead_reason == LIVE`; `H*600` |
| T8 | `TaggingWindowTests.test_reposition_inside_window_is_not_the_restart` | Pass | `late`; `restart in {330,331,332}`; `dead_reason[200:330] != LIVE` including `260:270` |
| T9 | `TaggingWindowTests.test_reposition_without_kick_in_window_abstains` | Pass | `unconfirmed`; `labels[:250] == H*100+D*150`; `labels[250:375] == X*125`; `labels[375:] == A*175` |
| T10 | `TaggingWindowTests.test_slow_controlled_movement_after_stoppage_abstains` | Pass | `unconfirmed`; `labels[250:375] == X*125`; `labels[375:] == A*125` (`controlled`) |
| T11a | `TaggingWindowTests.test_overlapping_tags_share_one_confirmed_restart` | Pass | 1 cluster `early`; `restart = 256`; no `X`; `H*100 + D*156 + A*144` |
| T11b | `TaggingWindowTests.test_overlapping_tags_with_conflicting_teams_give_no_prior` | Pass | 1 cluster `unconfirmed` (`E = 330`); `H*100 + D*150 + X*150`; `reason[330:] == abstain_no_evidence` |
| T12 | `ScorerEquivalenceTests.test_unconfirmed_window_is_dead_after_official_abstention_fill` | Pass | Official `possession_metrics.fill_abstentions` maps `250:375` to `D` and gives identical `spell_scores["mean_f1"]` to replacing `250:375` with `D` |
| Extra | `test_before_tag_movement_onset_establishes_restart` | Pass | `before`; `restart = 236`; `H*100 + D*136 + A*364` |
| Extra | `test_overlapping_tag_with_untagged_member_keeps_agreeing_team` | Pass | `unconfirmed`; `None` + `"A"` -> team `"A"` prior after `330` |
| Extra | `test_abutting_tags_form_separate_clusters` | Pass | `[250, 280)` H and `[280, 310)` A stay 2 clusters; `A` prior at `310` |
| Extra | `test_sparse_displacement_hold_release_counts_as_movement_evidence` | Pass | Sparse `moved` at `322` releases hold as `late` restart |

**Spec-table notes verified during TDD:**
- In `T3`, `docs/ENGINE_DECISION_TAGGING_NO_ONSET.md` §5a listed `reason[375:396]` in the `S` column; in `stoppage_then(600, reappear=395)`, away control begins at frame `395` (`first.onset == 395`), so backfill spans `[375, 395)` (`reason[375:395] == "backfilled_unconfirmed_restart"`, `0.4`) and frame `395` itself is `"controlled"` (`0.9`).
- In `T5`, `stoppage_then(600, reappear=475)` establishes away control at `475`, so `[375, 475)` receives `"A"` via `"backfilled_unconfirmed_restart"` (`0.4`) rather than `"set_piece_team_prior"` (`0.4`); `"set_piece_team_prior"` is exercised in `T4` where no later control occurs.

## 5. Exact Files Changed

Implementation commit `f5616c7`:
- `src/mtah/engine/config.py`
- `src/mtah/engine/deadball.py`
- `src/mtah/engine/possession.py`
- `src/mtah/engine/temporal.py`
- `tests/test_engine.py`
- `tests/test_engine_robustness.py`
- `tests/test_integration_harness.py`
- `docs/POSSESSION_ENGINE.md`
- Internal pre-decision review note *(omitted from public snapshot; policy documented in `docs/ENGINE_DECISION_TAGGING_NO_ONSET.md`)*

Report commit (HEAD):
- `docs/ENGINE_V02_IMPLEMENTATION_REPORT.md`

## 6. Before / After Synthetic Behavior

### Updated existing §5f test expectations

| Test | v0.1 Behavior | v0.2 Behavior & Changed Frame Ranges / Reasons |
|---|---|---|
| `test_engine.DeadBallTests.test_set_piece_extends_dead_back_and_supplies_prior` | `H99 D151 A50` (`250:300` prior) | `H99 D151 X10 A40`: `250:260` `DEAD_UNCONFIRMED` -> `X` (`abstain_unconfirmed_restart`, `NaN`); `260:300` `A` (`set_piece_team_prior`, `0.4`) |
| `test_engine.DeadBallTests.test_set_piece_prior_can_be_disabled` | `250:300` `X*50` (`abstain_no_evidence`) | `250:300` `X*50` unchanged labels; `250:260` `abstain_unconfirmed_restart`, `260:300` `abstain_no_evidence` |
| `test_engine.DeadBallTests.test_throw_in_without_team_has_no_prior` | `250:300` `X*50` (`abstain_no_evidence`) | `250:300` `X*50` unchanged labels; `250:260` `abstain_unconfirmed_restart`, `260:300` `abstain_no_evidence` |
| `test_engine.DeadBallTests.test_set_piece_lead_in_stops_at_live_ball` | `restart=250`; `H199 D51 H50` | `unconfirmed`, `restart=260`; `H199 D51 X10 H40` (`250:260` `abstain_unconfirmed_restart`, `260:300` `set_piece_team_prior`) |
| `HalfTimeBoundaryTests.test_set_piece_lead_in_does_not_reach_previous_half` | `restart=131`; `H100 D31 H69` | `ignored` (lead-in before `102` is `2 < 25` frames); `restarts == ()`; `H200` (`100:131` no longer marked dead) |
| `S2GapEdgeTests.test_periodic_dropouts_do_not_turn_live_play_dead_before_a_tag` | `restart=330`; `H248 D82 H70` | `unconfirmed`, `restart=340`; `H248 D82 X10 H60` (`330:340` `abstain_unconfirmed_restart`, `340:400` `set_piece_team_prior`) |
| `SetPieceTransitionTests.test_back_to_back_set_pieces_use_the_restart_that_opens_play` | `H99 D201 A100` | `H99 D151 X10 D40 X10 A90` (`250:260` `DEAD_UNCONFIRMED`, `260:300` `DEAD_SET_PIECE`, `300:310` `DEAD_UNCONFIRMED`, `310:400` `A` prior) |
| `SetPieceTransitionTests.test_nearest_untagged_restart_gives_no_prior` | `H99 D201 X100` | `H99 D161 X10 D30 X100` (`260:270` `DEAD_UNCONFIRMED`, `270:300` `DEAD_SET_PIECE`, `300:310` `DEAD_UNCONFIRMED`, `310:400` `abstain_no_evidence`) |
| `SetPieceTransitionTests.test_overlapping_set_piece_events_are_deterministic` | `H99 D151 A50` | `H99 D151 X30 A20` (merged cluster `[250, 280)` `unconfirmed`; `250:280` `X`, `280:300` `A`) |
| `SetPieceTransitionTests.test_set_piece_at_segment_end` | `H149 D46 A5` | `H149 D46 X5` (`195:200` `DEAD_UNCONFIRMED`, clipped to segment end `200`) |
| `SetPieceTransitionTests.test_tagged_restart_ends_out_of_play_hold` | `restart=200`; `H60 D140 A100` | `unconfirmed`, `restart=210`; `H60 D140 X10 A90` (`200:210` `DEAD_UNCONFIRMED`, hold ends at `210`, `210:300` `A` prior) |

### Full-length synthetic & mock CLI outputs (unchanged vs v0.1 S2 closure)

Every mock tag in `tests/mock_loader.py` has continuous 1.5 m/s movement at `r + 1`, so all 78 clusters are `early` and CSV bytes are identical across v0.1 and v0.2:

- Synthetic baseline (`--engine baseline --frames 199319`): `sha256=13151c4c89668b48f718881fa9e921d4d2220154199407880a607a6fa8f630f0` (`H=99658 A=99660 D=1 X=0`)
- Synthetic hybrid (`--engine hybrid --frames 199319`, run twice): `sha256=a0f99a12195daac9f519041ad3990ecfb38e681d85de8aff653d8882345eb974` (byte-identical, `H=99659 A=99660 D=0 X=0`)
- Mock ATD hybrid (`tests/run_cli_with_mock_loader.py --source atd --engine hybrid`, run twice): `sha256=15f5a261c1983b1c58f7dc7ec66bab28c3b3491e62c919765e8e179c9db75cd9` (byte-identical, `H=79834 A=67212 D=11811 X=40462`)
- Mock ATD baseline (`tests/run_cli_with_mock_loader.py --source atd --engine baseline`): `sha256=1b97b56fd403a0bbdffe7f48f5986b694b20f6fe00560f6497fe7432f4def63a` (`H=68451 A=65485 D=65383 X=0`)

## 7. S2 Regression Results

All S2 tests remain green:
- `DerivedSpeedSymmetryTests`: 6/6 pass.
- `S2GapEdgeTests`: 6/6 pass (`test_periodic_dropouts_do_not_turn_live_play_dead_before_a_tag` updated per §5f for the `unconfirmed` `[330, 340)` tag).
- `LongMissingBallTests.test_sparse_ball_observations_still_give_control`: passes.
- `DeadBallTests` S2 gap-edge expectations (`H*99` lead-in stop): pass.

## 8. Normal, Optimized, Environment, and Validator Results

Standalone Smart Tagging v0.2 (`d8a0dd0`) results prior to integration with Data Engineering (`166f35f`):

| Check | Result |
|---|---|
| `python -B -m unittest discover -s tests` (Python 3.11.9) | **199 tests OK** (27.803s) |
| `python -B -O -m unittest discover -s tests` (Python 3.11.9) | **199 tests OK** (29.939s) |
| `python -B scripts/check_environment.py` | `environment matches requirements.lock` (Python 3.11.9, numpy 2.4.6, pandas 3.1.1, python-dateutil 4.0.0, tzdata 2026.6) |
| `python -m pip check` | `No broken requirements found.` |
| `git diff --check` | Clean (no whitespace/conflict errors) |
| `python -B scripts/check_staged.py` | `Staged safety: 9 paths checked; 0 rejected (metadata only)` |
| `scripts/submission_preflight.py` (all 4 outputs + repeat-run `--compare`) | `PASS` (`official validator: valid`, byte-identical across repeat runs) |
| `evaluation/validate_submission.py` (`synth-hybrid-1.csv`, `mock-hybrid-1.csv`) | `Valid` (exit 0; exact `frame,label` schema over `0..199318`) |

Combined baseline (`166f35f` + `d8a0dd0`, with `PYTHONPATH=<repo>/src` on CPython 3.11.9 x64):

- `python -B -m unittest discover -s tests`: **Ran 240 tests, OK**
- `python -O -B -m unittest discover -s tests`: **Ran 240 tests, OK**

## 9. Remaining Ambiguities & Unresolved Cases

Per `docs/ENGINE_DECISION_TAGGING_NO_ONSET.md` (§8–§9), no extra unapproved heuristics were invented; the following cases remain explicitly recorded:

1. **Non-GT aggregate diagnostics / cluster-kind sizing (§8):** Local aggregate counts from DE (`restart_kinds` breakdown, overlapping/conflicting cluster counts, tag duration quantiles, graded frames turning `X`) were open at the `d8a0dd0` handoff milestone; the non-GT Section 8 aggregate diagnostic was subsequently completed locally. `macroF1` and competition accuracy remain unmeasured, and ground-truth scoring has not been performed.
2. **Superseded-onset misfire (§9):** A true restart followed inside `[S, E)` by `>= min_dead_frames` (25) frames at rest `< 0.3 m/s` (e.g., player standing still on the ball) is treated as a reposition, yielding `X` or a later restart.
3. **Pre-window live onset before an out-of-play event:** Under the literal §2 rule, any onset `h` in `[S - w, E)` followed before `E` by `>= min_dead_frames` rest frames is superseded and marked `superseded_motion` from `h` to `rest_start`. If a live-play onset in `[S - w, S)` precedes a `DEAD_OUT` event that rests `< 0.3 m/s` for `>= 25` frames before `E`, `[h, rest_start)` is suppressed as `superseded_motion` and could be absorbed into `DEAD_SET_PIECE` lead-in.
4. **Post-`E` live stretch after `unconfirmed`:** If a true restart actually happened after `E` (a tag containment error), frames from `E` onward use live rules (`backfilled_unconfirmed_restart` or `set_piece_team_prior` at `0.4`).
5. **Overlapping retaken set pieces and segment-start straddling tags:** Overlapping tags merge into one cluster, and tags whose `start_frame` lies before a segment start remain dropped.

## 10. Compatibility with DE Integration Target (`166f35f`)

Verified by merging `d8a0dd0` onto DE target `166f35fdc3c7a83457f15d043cd76349be3eae51` relative to common base `7198196`:

- **Zero merge conflicts & disjoint file sets:** `166f35f` changes `docs/DATA_PIPELINE.md`, `scripts/check_environment.py`, `scripts/verify_slot_mapping_local.py`, `src/mtah/__main__.py`, `src/mtah/pipeline.py`, `tests/test_cli.py`, and `tests/test_pipeline.py`; Smart Tagging v0.2 changes 13 completely disjoint files (`src/mtah/engine/{config,deadball,possession,temporal}.py`, `tests/test_engine.py`, `tests/test_engine_robustness.py`, `tests/test_integration_harness.py`, and docs). Unlike earlier pre-fix DE candidates, `166f35f` does not modify `tests/test_integration_harness.py` (its diagnostic error test moved to `tests/test_cli.py`).
- **Seam & event invariants:** `pipeline.py::_load_events` emits half-open `[start_frame, end_frame)` `Event` tuples sorted by `(start_frame, end_frame, event_id)` with normalized upper-cased codes and `"H"`/`"A"`/`None` teams matching `deadball.cluster_set_pieces`. `src/mtah/contracts.py`, `src/mtah/engines.py`, `src/mtah/export.py`, and `src/mtah/__main__.py` remain unchanged and compatible.
- **Traceability:** `ENGINE_VERSION = "possession-rules-v0.2"` updates `config_digest`; `EngineTrace.restart_kinds` and `EngineTrace.reason` provide DE with local aggregate counts without changing `MatchInput`, `PredictionResult`, CLI flags, or CSV format.
