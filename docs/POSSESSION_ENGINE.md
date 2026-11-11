# Possession Intelligence Engine (`possession-rules-v0.2`)

Date: 2026-11-04 (v0.1 design); updated for Smart Tagging `v0.2` Option A (`possession-rules-v0.2`).

> **Release status (`possession-rules-v0.2`):** Smart Tagging `v0.2` Option A abstention (`possession-rules-v0.2`) and the real-data loader (`src/mtah/pipeline.py`) are integrated. `--engine hybrid` selects this engine in `src/mtah/__main__.py` (`baseline` remains the default CLI engine), and `scripts/check_environment.py` verifies the CPython `3.11.9` `requirements.lock` environment. Local non-ground-truth operational diagnostics verified CSV format and determinism on tactical data without measuring accuracy or competition scores. Historical design context below is preserved for traceability.

## Scope and safety

- The engine consumes contract v1 `MatchInput` and returns `PredictionResult`. `src/mtah/contracts.py`,
  `baseline.py`, `export.py`, the CLI and every official file are unchanged.
- No asset, public or private ground truth, video or score was opened, compared against or used to
  pick any value. Every parameter is a hand-set physical prior (see `EngineConfig`). Nothing was
  trained, tuned or calibrated.
- The engine reads no files, opens no network connection and uses no randomness. A test
  (`test_engine_source_is_isolated`) fails if engine source gains file, network or asset access.

## Integration

```python
from mtah.engine import predict, predict_with_trace, EngineConfig

result = predict(match)                          # PredictionResult, same call shape as mtah.baseline.predict
result, trace = predict_with_trace(match)        # plus local-only diagnostics
result = predict(match, EngineConfig(confirm_frames=12))
```

`result.engine_version` is `possession-rules-v0.2` (was `possession-rules-v0.1`). `result.config_digest` is the SHA-256 of the
canonical config JSON. `result.confidence` is filled in. `result.reason` is `None` (populated on `EngineTrace.reason`; omitted in CSV contract v1). `export_submission(result, match, path)` works unchanged. A test exports an engine
result through the official validator.

`mtah/__main__.py` integrates `--engine hybrid` alongside `--engine baseline` (`baseline` remains the default CLI engine; `--engine hybrid` executes `possession-rules-v0.2`).

## Pipeline

```text
MatchInput
  -> evidence.spatial_evidence   per-frame team-level control: H / A / CONTESTED / NONE (+ why)
  -> deadball.infer_dead         per-frame dead reason + set-piece restarts
  -> temporal.decide             labels H/A/D/X, confidence, engine reason
  -> PredictionResult (validated against the input)
```

### 1. Spatial possession evidence (`engine/evidence.py`)

- Distances are in metres: `(x * pitch_length_m, y * pitch_width_m)`.
- For each frame the engine takes the nearest observed home player and the nearest observed away
  player to the ball. It uses team-level minima only, so slot identity switches, fragmented tracks
  and duplicated tracks have no effect.
- Control goes to a team when its nearest player is within `control_radius_m` (1.5 m) and the
  frame is not contested.
- A frame is contested when both nearest players are inside the radius and their distances differ
  by less than `contest_margin_m` (0.5 m). An exact tie also counts as contested.
- No control is recorded when the ball is unobserved, when no player is observed, when the ball is
  travelling faster than `max_control_ball_speed_mps` (12 m/s), or when nobody is close.
- Ball speed is the source speed when it is finite. Otherwise it is derived from positions: the
  smaller of the backward and forward 5-frame displacement speeds, with a 1-frame fallback. This
  makes the frame where a ball is received or struck read as slow, and a ball in flight read as
  fast. The derivation never bridges a missing frame or a half boundary.

### 2. Missing-ball handling

- The ball is never interpolated or extrapolated. `trace.evidence.ball_xy_m` stays NaN wherever
  the input is NaN, and a test asserts this.
- A missing ball produces no control evidence. It is **never** dead-ball evidence by itself.
- Through a gap, possession stays with the last team that had it (CHALLENGE.md: possession stays
  with the last team while the ball is loose). The engine reason is `carried_ball_missing`, and
  confidence decays with a 2 s half-life down to 0.35.
- A change of possession hidden inside an occlusion is placed at the frame where the other team is
  first seen controlling the ball.

### 3. Dead-ball inference (`engine/deadball.py`)

The engine uses only documented signals:

| Signal | Rule | Engine reason |
|---|---|---|
| Ball observed beyond a line | Centre more than `out_margin_m` (1 m) outside the pitch. Finite out-of-pitch coordinates are preserved by the contract. | `dead_ball_out` |
| Stoppage persistence | After going out, the ball stays dead through missing or out frames until it is observed inside the pitch, moving at least `restart_speed_mps` (1 m/s), or until a set-piece restart. A placed corner, goal-kick or free-kick ball stays dead. | `dead_hold` |
| Smart Tagging `SET PIECES` | Overlapping tags in a segment merge into one `[S, E)` cluster (common team if all tagged members agree, else `None`). Candidates are non-superseded movement evidence frames (`moving_inside` onsets or sparse `moved` hold releases) in `[S − 50, E)`. Priority is `early` (`[S, S+50]`), else `before` (`[S−50, S)`), else `late` (`(S+50, E)` gated on a stoppage in progress at `S`). An onset followed before `E` by `>= min_dead_frames` at rest (`< stationary_speed_mps`) is superseded as repositioning and stays in the stoppage. With no candidate and a stoppage at `S`, the cluster is `unconfirmed`: `[S, E)` is labelled `X` (`abstain_unconfirmed_restart`, NaN) and any active hold ends at `E`. Without a stoppage, it is `ignored`. | `dead_set_piece_lead_in` / `abstain_unconfirmed_restart` |
| Unattended stationary ball | Ball observed slower than 0.3 m/s with nobody within the control radius for at least 3 s. | `dead_stationary_ball` |

Since P1-B.3:

- Motion for these rules is the larger of the reported and the position-derived speed.
- With sparse observations, a stoppage also ends when the ball's displacement between in-pitch
  observations at least 5 frames apart shows at least 1 m/s.
- A set-piece lead-in stops at controlled play: one team controls the ball and it is not
  stationary.

Verified by the engine robustness suite in `tests/test_engine_robustness.py` (with `v0.2` updates documented in [docs/ENGINE_V02_IMPLEMENTATION_REPORT.md](ENGINE_V02_IMPLEMENTATION_REPORT.md)).

Dead runs shorter than `min_dead_frames` (1 s) are dropped as line noise. Restart team ownership is
not predicted: a dead ball is always `D`.

### 4. Temporal possession reasoning (`engine/temporal.py`)

- **Control runs.** A team-level run is a sequence of frames with that team's control evidence.
  It tolerates up to `run_gap_frames` (2 s) without evidence, which covers passes between
  teammates and short occlusions. It ends at the first frame with opposing-team evidence.
- **Controlled possession.** A run counts as controlled when it has at least `confirm_frames`
  (10, which is 0.4 s) evidence frames. Shorter runs are touches or deflections and change nothing.
  Contested frames and aerial balls in flight never change possession.
- **Transition timing.** The engine runs offline in batch, so it can look ahead. A change is
  placed at the *onset* of the confirming run, not at the frame where the threshold is reached.
  This avoids a systematic 0.4 s lag (1 s late costs about 0.35 spellF1, per EVALUATION.md).
- **Transitions between Home, Away and Dead Ball:**
  - Home to Away, or Away to Home: at the onset of a confirmed opposing run.
  - Live to Dead: when a dead-ball rule fires. The possession ends there.
  - Dead to Live: the live stretch starts with the first team that establishes confirmed control.
    That team's label is back-filled to the restart frame (`backfilled_restart`, confidence 0.5 after
    an evidence-backed restart, or `backfilled_unconfirmed_restart`, confidence 0.4 after an
    unconfirmed `X` window ending at `E`), because CHALLENGE.md says every restart has a clear
    possessor.
  - If a live stretch has no confirmed control, the engine uses the `Team` tag of the set piece
    whose restart is nearest the stretch start, within ±2 s (`set_piece_team_prior`,
    confidence 0.4). If that nearest restart is tagged `N/A`, or there is none, the stretch is `X`
    (`abstain_no_evidence`). Tracking evidence always overrides the tag. (Fixed in P1-B.2; see
    `tests/test_engine_robustness.py`.)
- **Segments.** State resets at every published segment boundary, so nothing carries across
  halftime. Frames outside segments are `X` (`outside_segment`). DATA.md documents them as
  ungraded. The segment list is used only to decide where the half boundaries are. It is never
  used as a live/dead mask.

### 5. Output

- Labels are H, A, D or X on every frame.
- `confidence` is an ordinal, uncalibrated trust score: 0.9 when the frame itself has control, a
  decaying value when possession is carried, 0.5 for a back-filled restart, 0.4 for a tag prior or unconfirmed backfill,
  and 0.6 to 0.9 for dead reasons. X frames get NaN, as the contract requires. It is not a
  probability.
- `trace.reason` holds the engine's 13-code vocabulary (`temporal.REASONS`), and `trace.restart_kinds` records the per-cluster classification (`early`, `before`, `late`, `unconfirmed`, `ignored`).

## Assumptions

1. Team membership per slot (`slot_team`) is correct for the whole match. The engine never
   re-assigns teams.
2. Tracking coordinates are ball-centre and player-centre positions in the documented normalized
   frame. The out-of-pitch rule assumes the lines are at 0 and 1.
3. `ball_speed_mps` is the ball's ground-plane speed in m/s when present.
4. For Smart Tagging `SET PIECES`, the tagged interval `[start_frame, end_frame)` is a ~5 s window around the restart (containment semantics, `starter/load_smart_tagging.py`), and `Team` is the restarting team. Throw-ins are `N/A`, which is handled as no prior; overlapping tags with conflicting teams give no prior.
5. Players are not mirrored and attack direction is not used.
6. The control radius, contest margin, flight speed, confirmation length and dead-ball thresholds
   are physical priors from first principles. They are not measurements. Calibrating any of them
   needs licensed external data about other games, with its provenance recorded in `EngineConfig`.

## Known limitations

- **Loose-ball contests and aerial duels.** These depend on the 0.4 s confirmation rule. A
  confirmation that is too short over-splits possessions; one that is too long misses quick
  interceptions. The value has not been validated.
- **Uncontrolled first contact.** Back-dating to the run onset places a turnover at the first
  contact, even if that contact was not yet controlled.
- **Missing ball at the moment of going out.** Most balls that leave play may not be tracked
  beyond the line. Without the out-of-pitch signal, a stoppage is detected only through a
  set-piece tag or a stationary ball. Untagged restarts such as goal kicks or throw-ins with no
  tag are likely to be missed. Fouls inside the pitch, offsides and injuries have no geometric
  signature in v0.1.
- **Set-piece lead-in can over-extend.** It can mark up to 10 s of live but occluded play as dead
  when the ball is unobserved before a tagged restart.
- **Goalkeeper holding the ball.** A keeper standing still with the ball is control by proximity,
  which is correct. But "nobody near a stationary ball" could misfire if the keeper's track is lost.
- **No use of other Smart Tagging phase codes** (`BUILD UP`, `PROGRESSION`, transitions). The
  meaning of their `Team` tag (attacking or defending team) is unverified.
- **Player speed, homography and geometry are not used.**
- **Performance and determinism.** On a synthetic full-length input (199,319 frames, 46 slots, 43% ball missing),
  `predict` runs deterministically and reproduces identical bytes across repeat runs under CPython 3.11.9 x64 (`requirements.lock`).
- **Contract v1 scope.** Exported CSV submissions contain `frame,label` only, and `PredictionResult.reason` stays `None` (diagnostic codes in `temporal.REASONS` remain on `EngineTrace.reason`). Optional per-slot on-pitch masks and 3-D ball height are not part of `MatchInput` contract v1.

## Test coverage (`possession-rules-v0.2`)

`python -B -m unittest discover -s tests` (and `python -O -B -m unittest discover -s tests`) under CPython 3.11.9 x64 runs **240 synthetic/mock tests** covering ingestion, CLI, preflight, engine logic (`tests/test_engine.py`, `tests/test_engine_robustness.py`), and integration (`tests/test_integration_harness.py`), including:

- dribbling control
- a contested duel
- a pass in flight past a defender
- identity switch and duplicate track
- ball observed with no players
- speed derivation that does not bridge gaps
- short and long ball-missing gaps, with no invented positions and no D
- a turnover hidden in an occlusion
- handover at the onset of the receiving run
- a brief touch, with the confirmation threshold at exactly 9 and 10 frames
- passes between teammates
- segment reset and outside-segment X
- abstention when there is no evidence (`DEAD_UNCONFIRMED` / `abstain_unconfirmed_restart` in v0.2)
- ball out, hold through missing frames, a placed ball, and the restart
- a short out-of-pitch blip
- an unattended stationary ball
- set-piece lead-in, team prior, a disabled prior, a throw-in with N/A team, tracking overriding the tag, overlapping/abutting clusters, and superseded reposition onsets
- a lead-in that stops at a live ball
- determinism, read-only outputs and config-digest validation
- export through the official validator and `spellF1` abstention-fill equivalence
- engine source isolation

These synthetic and mock scenarios verify that the documented rules execute as specified; they do not measure competition accuracy. For detailed v0.2 rule tables and synthetic test matrices, see [docs/ENGINE_V02_IMPLEMENTATION_REPORT.md](ENGINE_V02_IMPLEMENTATION_REPORT.md).
