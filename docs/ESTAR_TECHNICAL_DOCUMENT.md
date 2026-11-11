# More than a Hack 2027 — Phase 1: Technical Document

> **Document status:** **Team compliance confirmed (11 November 2026); clean release snapshot published to `https://github.com/Fahim-Femure/mtah2027-estar` (`main`; currently private — verify public visibility, final CSV, and Form submission prior to deadline).**  
> **Intended final deliverable filename:** `Estar.pdf` (paired with submission file `Estar.csv` and repository URL `https://github.com/Fahim-Femure/mtah2027-estar` per `SUBMISSION.md`).  
> **Revision distinction:**  
> - **Algorithm baseline:** `8a093bbf0270f35f3defe575220c5ff25b691baa` (`possession-rules-v0.2`, integrating DE loader `166f35f` and Smart Tagging `v0.2` Option A `d8a0dd0`).  
> - **Code/reproduction baseline:** `d3faa906b41a6ed94a47fda07f3fcc30392fc183` (`README.md`, repository docs, `scripts/setup.ps1`, and `src/mtah/sources.py` module docstring updates over the algorithm baseline; no algorithm changes).  
> - **Document & publication snapshot scope:** The release repository (`https://github.com/Fahim-Femure/mtah2027-estar`, branch `main`) is a clean release snapshot derived from the reviewed `d3faa906b41a6ed94a47fda07f3fcc30392fc183` (`d3faa90`) code baseline, not a continuation of its original commit history. This technical document (`Estar.pdf` / `docs/ESTAR_TECHNICAL_DOCUMENT.md`) is the submission artifact prepared for that snapshot and is not itself contained in commit `d3faa90`.  
> - **Submission CSV & scoring status:** `Estar.csv` has not yet been generated or submitted; competition accuracy and scoring (`spellF1`, `macroF1`) remain unmeasured.

## Team Information

- **Team Name:** Estar
- **Team Members:**
  1. Mgs. M. Fahim Rahmatullah — Team Leader
  2. Khalil Amin Adillah — Member
  3. Raihan Daris Ramadhan — Member
- **Code Repository URL:** `https://github.com/Fahim-Femure/mtah2027-estar` (branch `main`; clean release snapshot of `d3faa906b41a6ed94a47fda07f3fcc30392fc183`; currently private — ensure public visibility before Form upload per `SUBMISSION.md`)

---

## TEAM COMPLIANCE CONFIRMATION

**Team-level confirmation recorded as of 11 November 2026:** The team owner has explicitly confirmed that all three members of team **Estar** — **Mgs. M. Fahim Rahmatullah — Team Leader**, **Khalil Amin Adillah — Member**, and **Raihan Daris Ramadhan — Member** — agree with all four competition compliance requirements under `RULES.md`, `SUBMISSION.md`, and `FAQ.md` (recorded strictly as a team-level confirmation by the team owner; no individual member signatures or personal signature declarations are attached):

1. **No comparing prediction outputs against match video** (*Confirmed at team level as of 11 November 2026*): No prediction outputs or model outputs were compared against the supplied competition video.
2. **No manual labeling of the provided match** (*Confirmed at team level as of 11 November 2026*): All submission labels produced by the pipeline come from deterministic code; no manual labels or frame/segment corrections were created on `BAR-ATH` for submission, training, or validation.
3. **No uploading provided competition data to cloud AI services** (*Confirmed at team level as of 11 November 2026*): No raw competition files, supplied videos, ground-truth files, tracking rows or coordinates, or non-sanitized match extracts were uploaded to cloud AI services. Only sanitized aggregate operational diagnostics were shared where permitted by competition guidance.
4. **No training, fitting, calibration, or tuning using provided video or ground truth** (*Confirmed at team level as of 11 November 2026*): Zero parameters were trained, fitted, calibrated, or tuned using the supplied competition video, public 5-minute ground truth, or private full-match ground truth.

### Remaining Pre-Submission Items

While team-level compliance is confirmed above as of 11 November 2026 and the clean release snapshot is staged at `https://github.com/Fahim-Femure/mtah2027-estar` (`main`), the following pre-submission actions remain open before final Phase 1 upload:

- **Repository visibility (`Pending`):** Confirm `https://github.com/Fahim-Femure/mtah2027-estar` is switched from **private** to **public** per `SUBMISSION.md` prior to Form submission so organizers can access and reproduce the snapshot without authentication.
- **Local `Estar.csv` generation, determinism check & validation (`Pending`):** On an authorized workstation holding `<data_root>`, generate `Estar.csv` and repeat-run `Estar_repeat.csv`, verify identity via `scripts/submission_preflight.py --compare`, and pass `evaluation/validate_submission.py`.
- **Google Form submission (`Pending`):** Confirm the official Google Form URL and team Google account, and submit `Estar.csv`, `Estar.pdf`, and `https://github.com/Fahim-Femure/mtah2027-estar` ahead of November 17, 2026.

---

## 1. Approach

Estar's system implements a deterministic, rule-based temporal possession inference pipeline built on normalized 2-D tracking coordinates and explicit abstention handling rather than learned weights or frame-by-frame classifiers.

The design aligns with the challenge structure documented in `EVALUATION.md` and `README.md`: submissions are ranked by **`spellF1`** (an interval-based metric that scores each true possession spell once regardless of duration, with `macroF1` reported alongside), while automatic tactical tracking (`BAR-ATH_tactical_atd.txt`) drops the ball on approximately 41% of graded frames, swaps player slot identities, and carries unverified raw ball speed values. Frame-wise nearest-player rules over-split possessions on brief deflections and tracking dropouts. Our `hybrid` engine (`mtah.engine`, `possession-rules-v0.2`) therefore separates contract-validated ingestion, per-frame spatial control evidence, stoppage/restart state inference, and offline batch confirmation of team control runs:

1. **Strict EPTS & Smart Tagging Ingestion (`src/mtah/pipeline.py`):**
   - Streams the raw tactical ATD text file line by line (`frame_id - 1` zero-based video-frame alignment via `mtah.alignment.atd_to_video`) without padding, trimming, reordering, or interpolating missing coordinates.
   - Resolves raw tracking slot order from EPTS metadata `<DataFormatSpecification>` (`PlayerChannelRef` -> `PlayerChannel` -> `<Player playerId>` -> `<Team>`), falling back to `<Player>` document order only if no specification exists, and rejecting conflicting specifications or non-contiguous slot IDs.
   - Enforces read-only contract guarantees in `MatchInput` (`src/mtah/contracts.py`, `schema_version=1`, `199,319` frames, `25` fps), rejecting partial coordinate pairs and non-finite values other than paired `NaN`.
   - Converts Smart Tagging timestamps (`round(seconds * 25)` via `mtah.alignment.tagging_to_video`) into half-open `[start_frame, end_frame)` `Event` records sorted deterministically by `(start_frame, end_frame, event_id)`.
2. **Spatial Control Evidence (`src/mtah/engine/evidence.py`):**
   - Converts normalized coordinates to metres using tactical pitch dimensions from `games.py` (`105.3 m × 68.0 m`).
   - Reduces player tracks to per-team (`H`/`A`) minimum distances to the ball (`d_home_m`, `d_away_m`), eliminating sensitivity to within-team slot identity switches and duplicate tracks.
   - Computes bilateral position-derived ball speed when source `ball_speed_mps` is `NaN` (`speed_lag_frames = 5`, lag `1` fallback, and farthest-observed bilateral window fallback across isolated dropouts; suppressing one-sided internal gap edges to `NaN` so tracking re-acquisitions do not trigger false high-speed flags).
3. **Dead-Ball & Temporal Possession Reasoning (`src/mtah/engine/deadball.py`, `temporal.py`, `possession.py`):**
   - Detects stoppages (`DEAD_OUT`, `DEAD_HOLD`, `DEAD_STATIONARY`) and evaluates clustered Smart Tagging `SET PIECES` windows using Option A abstention semantics (`DEAD_UNCONFIRMED` -> `X` when no movement restart evidence exists inside an ongoing stoppage).
   - Inside each live stretch of each published half segment, aggregates team control evidence using offline batch lookahead confirmation (`confirm_frames = 10`, `0.4 s`) to place possession transitions at confirmed run onsets and ignore isolated `< 10`-frame touches.
   - Exports a validated UTF-8 (no BOM), LF-only `frame,label` CSV atomically via `mtah.export.export_submission`, running `evaluation/possession_metrics.validate_submission` before renaming to destination.

---

## 2. Assets Used and Dropped (Software Pipeline Scope)

The classification below describes **strictly what the submitted software (`mtah.pipeline.prepare_match` and `mtah.engine`) opens and processes**. Team-level agreement by all three members with the four competition compliance requirements as of 11 November 2026 is recorded in the `TEAM COMPLIANCE CONFIRMATION` section above.

| Competition Asset / Field | Behavior in Submitted Code (`8a093bb` / `d3faa90`) | Software Status |
|---|---|---|
| `tactical/BAR-ATH_tactical_atd.txt` (positions) | Streams player `(x, y)` and ball `(x, y)` coordinates, finiteness masks, and contiguous `1..199319` frame IDs | **Used (positions & observation masks)** |
| `tactical/BAR-ATH_tactical_atd.xml` | Resolves `<Teams>`, `<Players>`, `<PlayerChannels>`, and `<DataFormatSpecification>` slot order to `"H"` / `"A"` (`Barcelona` / `Atletic Club`) | **Used (slot-to-team mapping)** |
| `tactical/BAR-ATH_tactical_smart_tagging.xml` (`SET PIECES`) | Uses upper-cased `"SET PIECES"` half-open `[start_frame, end_frame)` intervals and `"Team"` (`"H"`, `"A"`, or `N/A` -> `None`) for restart windows and live-stretch priors | **Used (`SET PIECES` only)** |
| Raw ATD ball 3rd column (`ball_speed`) | Explicitly dropped at ingestion (`ball_speed_mps` filled with all-`NaN` in `_load_atd`) because units and meaning are unverified; engine derives bilateral position speed instead | **Dropped at ingestion** |
| Raw ATD player 3rd column (`player_speed_mps`) | Parsed and validated as finite non-negative or `NaN` in `MatchInput`; not read by `mtah.engine` | **Loaded & validated only; unused by engine** |
| Non-`SET PIECES` Smart Tagging phases | Loaded into `MatchInput.events`; ignored by `mtah.engine` (`set_piece_codes = ("SET PIECES",)`) because phase team attribution semantics are unverified | **Loaded only; unused by engine** |
| `tactical/BAR-ATH_tactical_homography.csv` | Loaded and contract-validated in `Geometry` (`199,319` rows, finite well-conditioned `3×3` matrices where valid, non-negative reprojection errors); not read by `mtah.engine` | **Loaded & validated only; unused by engine** |
| `tactical/BAR-ATH_tactical.mp4` | Never opened or referenced by `prepare_match` or `mtah.engine` | **Not opened by submitted software** |
| `broadcast/*` (video, ATD, XML, homography) | Never opened (`MatchInput.source` is `"tactical"` only) | **Not opened by submitted software** |
| `evaluation/gt/*` & private ground truth | Never referenced by `src/mtah/` (`test_engine_source_is_isolated` verifies `mtah.engine` contains no file, evaluation, network, or GT access) | **Not accessed by submitted software** |

---

## 3. Training Data Provenance & Rule Parameterization

- **No machine-learning training and no labelled training datasets:** `mtah.engine` is purely rule-based with no training pipeline, learned weights, or statistical model fitting. No external labelled match datasets were used.
- **Explicit, hand-set rule parameters (`EngineConfig`):** Every threshold in `src/mtah/engine/config.py` (`ENGINE_VERSION = "possession-rules-v0.2"`) is an explicit, hand-set heuristic parameter rather than a fitted weight:
  - Spatial control evidence: `control_radius_m = 1.5`, `contest_margin_m = 0.5`, `max_control_ball_speed_mps = 12.0`, `speed_lag_frames = 5`
  - Temporal confirmation: `confirm_frames = 10` (`0.4 s` at `25` fps), `run_gap_frames = 50` (`2.0 s`)
  - Dead-ball & restart logic: `out_margin_m = 1.0`, `restart_speed_mps = 1.0`, `min_dead_frames = 25` (`1.0 s`), `stationary_speed_mps = 0.3`, `stationary_frames = 75` (`3.0 s`), `set_piece_codes = ("SET PIECES",)`, `set_piece_window_frames = 50` (`2.0 s`), `set_piece_max_lead_frames = 250` (`10.0 s`), `use_set_piece_team_prior = True`
- Repository source and implementation reports (`src/mtah/engine/config.py`, `docs/POSSESSION_ENGINE.md`, `docs/ENGINE_V02_IMPLEMENTATION_REPORT.md`) record that no parameter was fitted, calibrated, or tuned on competition data or ground truth; the team owner has confirmed as of 11 November 2026 that all three members agree with requirement 4 in the `TEAM COMPLIANCE CONFIRMATION` section above.
- **Automated test provenance & non-GT diagnostics:** Automated verification consists of `240` synthetic/mock tests passing in both normal (`python -B`) and optimized (`python -O -B`) runs using in-memory fixtures only (`mtah.synthetic`, `tests/scenarios.py`, `tests/mock_loader.py`). Separately from synthetic tests and from competition scoring, non-GT Section 8 operational diagnostics were executed locally by an authorized operator to inspect aggregate counts and CSV formatting; `macroF1`, `spellF1`, and competition accuracy remain unmeasured, and ground-truth scoring has not been performed.

---

## 4. Possession Rules Implemented (`possession-rules-v0.2`)

### A. Spatial Control, Loose Balls, and Contested Challenges (`src/mtah/engine/evidence.py`, `temporal.py`)
- **Single-Team Control (`HOME` / `AWAY`):** Assigned on frame `t` when the ball is observed, at least one player is observed, ball speed (where finite: source speed if finite, else bilateral derived speed) satisfies `<= 12.0 m/s`, the nearest observed player satisfies `min(d_home_m, d_away_m) <= 1.5 m`, and the frame is not contested.
- **Challenges / Contested (`CONTESTED`):** Assigned when both teams have at least one observed player within `1.5 m` of the ball and either `|d_home_m - d_away_m| < 0.5 m` or `d_home_m == d_away_m`. Contested frames cannot establish or transfer possession.
- **Loose, In-Flight, and Unobserved Balls (`NONE`):** Assigned when `ball_speed > 12.0 m/s` (`WHY_IN_FLIGHT`), nearest player `> 1.5 m` (`WHY_LOOSE`), no player observed (`WHY_NO_PLAYERS`), or ball unobserved (`WHY_BALL_MISSING`). Missing ball coordinates never count as dead-ball evidence by themselves.
- **Confirmation, Turnover Placement, and Carry:** Inside each live interval (`dead == LIVE`) within a segment, same-team control frames group into a `ControlRun`, bridging `NONE`/`CONTESTED` gaps up to `run_gap_frames = 50` frames (`2.0 s`) and breaking immediately on any opposing-team control frame. Runs with `count >= 10` evidence frames (`0.4 s`) confirm possession and place the possession change at `run.onset` via offline lookahead; shorter `< 10`-frame sequences are ignored as touches/deflections. Confirmed possession (`confidence = 0.9`, `"controlled"`) carries across live non-control frames until an opposing confirmed run or stoppage, decaying confidence via `0.35 + 0.55 * 0.5 ** (gap / 50.0)` (`"carried_ball_missing"` when missing, else `"carried"`).

### B. Dead-Ball Inference and Smart Tagging `v0.2` Restarts (`Option A`) (`src/mtah/engine/deadball.py`)
- **Out-of-Pitch & Stoppage Hold:** An observed ball beyond pitch bounds `[0, L] × [0, W]` by more than `out_margin_m = 1.0 m` triggers `DEAD_OUT` (`"dead_ball_out"`, `0.9`). Dead state persists (`DEAD_HOLD`, `"dead_hold"`, `0.7`) until a non-superseded in-pitch observation moving at `motion_speed_mps >= 1.0 m/s` (maximum of finite `ball_observed`-gated source speed and derived speed), a sparse in-pitch displacement `>= 5` frames apart showing `>= 1.0 m/s` (`_sparse_moved`), or a confirmed set-piece restart.
- **Stationary Unattended Ball:** An observed in-pitch ball moving `< 0.3 m/s` with no player within `1.5 m` for `>= 75` consecutive frames (`3.0 s`) marks `DEAD_STATIONARY` (`"dead_stationary_ball"`, `0.6`).
- **Smart Tagging `SET PIECES` Clustering & Option A Abstention (`v0.2`):**
  - Within each published half segment `[seg_start, seg_stop)`, `SET PIECES` events starting inside `[seg_start, seg_stop)` are clipped to `min(end_frame, seg_stop)` and merged transitively by overlap (`s0 < cur_e`); abutting events (`e0 == s0'`) remain separate clusters. A cluster takes team `"H"` or `"A"` iff all tagged members agree (`None`/`N/A` omitted).
  - Candidate restart evidence in `[max(seg_start, S - 50), E)` excludes superseded reposition movements (any movement onset followed strictly before `E` by `>= 25` consecutive rest frames `< 0.3 m/s`, whose pre-rest motion stays inside the stoppage).
  - First matching rule per cluster `[S, E)`:
    1. `early`: earliest candidate in `[S, S + 50]` -> confirmed restart `R` (ends active hold at `R`; backward lead-in runs from `R`).
    2. `before`: nearest candidate in `[max(seg_start, S - 50), S)` -> confirmed restart `R`.
    3. `late`: earliest candidate in `(S + 50, E)` **only if** a stoppage is already in progress at `S` -> confirmed restart `R` (prevents false-positive tags over live occlusions from turning play dead).
    4. `unconfirmed` (**Option A**): no candidate, **and** a stoppage is in progress at `S` -> cluster window `[S, E)` abstains as `X` (`DEAD_UNCONFIRMED`, `confidence = NaN`, `"abstain_unconfirmed_restart"`), active hold suspends across `[S, E)` and ends at `E` (never extending `D` past `E` or fabricating `restart_flag`), pre-`S` backward lead-in (`"dead_set_piece_lead_in"`, `0.6`, max `250` frames) remains anchored at `S`, and the live stretch starting at `E` receives `0.4` confidence (`"backfilled_unconfirmed_restart"` if later control confirms in the live stretch, or agreeing `"set_piece_team_prior"`, else `X`/`NaN` `"abstain_no_evidence"`).
    5. `ignored`: no candidate in `[S - 50, S + 50]` and no stoppage at `S` -> no dead state, abstention window, or team prior emitted.
- **Noise Cleanup & Segment Isolation:** Final non-live dead runs shorter than `25` frames (`1.0 s`) inside a segment revert to `LIVE`. State resets at segment boundaries; frames outside published segments (`half == 0`) are labelled `X` (`"outside_segment"`, `NaN`).

---

## 5. Limitations

1. **Hand-Set Heuristic Thresholds:** Radii (`1.5 m` / `0.5 m`), confirmation length (`10` frames / `0.4 s`), and speed/stillness limits are uncalibrated rule parameters; fast one-touch challenges or brief dribbles (`< 10` frames) can be missed or attributed late, while uncontrolled first contacts at a confirmed run's onset receive the turnover frame.
2. **Untagged In-Pitch Stoppages:** Stoppages occurring wholly inside the pitch without a `SET PIECES` tag or `>= 3 s` unattended stillness (such as fouls surrounded by players) lack a geometric detector in `v0.2`.
3. **Pre-Restart Lead-In Over-Extension:** Backward set-piece lead-in can label up to `250` frames (`10 s`) of missing pre-restart frames as `D`, potentially absorbing tail occlusions prior to a stoppage.
4. **Unused Signals & Contract v1 Metadata:** Raw ATD player speeds, non-`SET PIECES` tactical phases, homography matrices, 3-D ball height, tactical video frames, and broadcast feeds are unused by `mtah.engine`; contract v1 leaves `PredictionResult.reason = None` on exported CSVs (`EngineTrace` retains reason codes only locally).
5. **Smart Tagging `v0.2` Design Trade-Offs (recorded in `docs/ENGINE_V02_IMPLEMENTATION_REPORT.md` §9):**
   - *`macroF1` vs `spellF1` abstention cost:* In-segment `unconfirmed` `X` frames count as misses under `macroF1`, whereas `evaluation/possession_metrics.py::fill_abstentions` forward-fills `X` from the preceding `D` stoppage label under `spellF1`.
   - *Superseded-onset misfire:* A true kick inside `[S, E)` followed inside `[S, E)` by `>= 25` stationary frames (`< 0.3 m/s`) is classified as repositioning rather than a restart.
   - *Containment under-runs & tag edges:* Restarts occurring after `E` treat play from `E` onward as live (`0.4` backfill/prior); overlapping retaken set pieces merge into one cluster; events whose `start_frame` lies before a segment start are dropped.
6. **Fail-Loud Ingestion & Scope:** Partial coordinate pairs, conflicting `<DataFormatSpecification>` sections, and frame mismatches raise `ValueError` rather than guessing; runtime lock reproducibility has been verified specifically on CPython `3.11.9` (64-bit, Windows), and no competition accuracy has been measured.

---

## 6. Reproduction Instructions

From a clean checkout of `https://github.com/Fahim-Femure/mtah2027-estar` (branch `main`, clean release snapshot of code/reproduction baseline `d3faa906b41a6ed94a47fda07f3fcc30392fc183` over algorithm baseline `8a093bbf0270f35f3defe575220c5ff25b691baa`):

### Environment Setup (`CPython 3.11.9`, 64-bit, Windows + `requirements.lock`)
- Runtime pins in `requirements.lock`: `numpy==2.4.6`, `pandas==3.1.1`, `python-dateutil==4.0.0`, `tzdata==2026.6` (`pyproject.toml` declares `requires-python = ">=3.11"`; optional `requirements-starter.txt` dependencies `kloppy==3.18.0`, `matplotlib`, `notebook` are not required for submission generation).
- Create `.venv` with an explicit **CPython 3.11.9 (64-bit)** executable, install from `requirements.lock`, install `mtah-foundation` editable with `--no-deps` (preserving exact pins), and verify:

```powershell
# PowerShell (Windows, explicit CPython 3.11.9 64-bit executable)
& "<path-to-cpython-3.11.9-x64>" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -B scripts\check_environment.py
.\.venv\Scripts\python.exe -m pip check
```

- Alternatively, `.\scripts\setup.ps1 -Python311 <path-to-cpython-3.11.9-x64>` automates venv creation, `requirements.lock` + `--no-deps -e .` installation, and `pip check`; because `setup.ps1` only validates Python `3.11` and `64`-bit (`setup.ps1` does **not** verify micro-version `3.11.9` itself), supply CPython `3.11.9` (64-bit) explicitly and run `.\.venv\Scripts\python.exe -B scripts\check_environment.py` afterward to verify exact `3.11.9` and pinned dependency versions.
- If skipping `pip install --no-deps -e .` (or running from an alternate worktree whose shared `.venv` points elsewhere), set `PYTHONPATH` explicitly to this checkout's `src` directory (`$env:PYTHONPATH = "$PWD\src"` in PowerShell, or `PYTHONPATH=src` on POSIX). On POSIX platforms, invoke an explicit CPython `3.11.9` (64-bit) interpreter rather than assuming a generic `python3.11` binary is `3.11.9`.

### Expected Input Directory Layout
Pass `<data_root>` — the parent directory containing `tactical/` — to `--input` (do **not** pass `<data_root>/tactical`):

```text
<data_root>/
├─ tactical/
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/   (optional bundle folder; unused by mtah.pipeline.prepare_match and mtah.engine)
```

### Synthetic Verification vs Local Final CSV Generation (`Estar.csv`)
- **Step 1 — Synthetic/unit verification (`240` synthetic/mock tests; opens no competition match assets):**

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -O -B -m unittest discover -s tests
```

- **Step 2 — Local final CSV generation (`Estar.csv`):** On an authorized local workstation with unpacked competition data at `<data_root>`, generate `Estar.csv` and a repeat-run comparison file `Estar_repeat.csv` outside `evaluation/` (`artifacts\submission\` is gitignored and created automatically by `mtah.export`):

```powershell
# Canonical CLI invocation:
#   python -B -m mtah --source atd --input <data_root> --engine hybrid --out Estar.csv

.\.venv\Scripts\python.exe -B -m mtah --source atd --input <data_root> --engine hybrid --out artifacts\submission\Estar.csv
.\.venv\Scripts\python.exe -B -m mtah --source atd --input <data_root> --engine hybrid --out artifacts\submission\Estar_repeat.csv
.\.venv\Scripts\python.exe -B scripts\submission_preflight.py artifacts\submission\Estar.csv --compare artifacts\submission\Estar_repeat.csv
.\.venv\Scripts\python.exe -B evaluation\validate_submission.py artifacts\submission\Estar.csv
```
