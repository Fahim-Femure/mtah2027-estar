# Engine-owner decision: Smart Tagging restarts without movement evidence

Date: 2026-11-06.
Status: **decision recorded for joint review. No production code changed.**

| | |
|---|---|
| Base | Decision commit `4af56a6`; production `src/` is identical to pre-integration baseline `7198196` |
| Inputs | CHALLENGE.md, EVALUATION.md, `starter/load_smart_tagging.py`, `evaluation/possession_metrics.py` (read-only), engine source, DE's read-only audit, invented scenarios |
| Not used | Competition data, ground truth or video; nothing tuned or calibrated |
| Supersedes | Prior Option C restart proposal (§3, omitted from public snapshot), which recommended a D fallback |

## 1. Decision

**Option A, abstain, is adopted as the default. No restart is confirmed without movement
evidence.**

When a set-piece tag has no usable movement onset and a stoppage is already in progress, the
engine labels the tag window **X**. It does not label it D, and it does not force H or A. It then
treats the tag end as the latest possible restart.

Reasons, in order of weight:

1. **Contract meaning.** CHALLENGE.md defines `X` as "your system declines to guess". Inside the
   window we know a restart happened but not when, so every frame there could be D or live.
   That is exactly the uncertainty `X` exists for.
2. **The ranking metric gives the same result as D.**
   - `possession_metrics.fill_abstentions` forward-fills an `X` with the last non-`X` label in the
     run.
   - The policy writes `X` only when a stoppage is in progress at the tag start, so the frame
     before the window is `D`.
   - The X window is therefore scored by spellF1 exactly as a D window would be.
   - This is a property of the official code, checked by a planned test (T12), not an assumption.
3. **Assumed and evidence-backed restarts become distinguishable.** This was DE's finding.
   - Unconfirmed windows carry `abstain_unconfirmed_restart` and NaN confidence.
   - Possession inferred after an unconfirmed restart gets its own reason and lower confidence.

**Accepted cost.** macroF1 charges every `X` on a graded frame as a miss, whereas `D` would be
right on the pre-restart part of the window. The cost is bounded by the total window length of
no-onset clusters that have a stoppage in progress. DE's aggregate counts (§8) would size it;
choosing the option did not depend on them.

## 2. Definitions

| Term | Definition |
|---|---|
| Tag | A `SET PIECES` event `[s0, e0)` whose start lies in a segment; `e0` is clipped to the segment end |
| **Cluster** | The tags of one segment, grouped by overlap: tags whose intervals intersect, transitively, form one cluster with interval `[S, E) = [min s0, max e0)`. Abutting tags (`e0 == s0′`) are separate clusters |
| Cluster team | The common team when every tagged team in the cluster agrees, else `None` (no prior). Untagged (`N/A`) members do not vote |
| Movement evidence | A movement onset (`moving_inside` after a frame without it), or a frame where the existing sparse-displacement rule releases a hold (`moved`) |
| **Superseded onset** (repositioning) | Onset `h` after which, before `E`, the ball is observed at rest (motion speed `< stationary_speed_mps`) for at least `min_dead_frames` consecutive frames. Its motion frames, from `h` until the rest begins, count as part of the stoppage, not as live evidence |
| Stoppage in progress at `S` | Frame `S − 1` is dead from positive evidence (out of pitch, or hold), or the backward lead-in from `S` would survive `min_dead_frames` |

`w = set_piece_window_frames` (50) is unchanged.

**Reused thresholds; no new parameter.**
- `stationary_speed_mps` keeps its existing meaning, "ball at rest".
- `min_dead_frames` keeps its existing meaning, "shortest real stoppage". It is used here as the
  shortest rest that marks a reposition.
- Reusing them needs explicit approval (§9).

## 3. Exact state machine per cluster

Candidates are the non-superseded movement evidence frames in `[S − w, E)`. The first matching
row applies:

| # | Condition | Kind | Restart | Labels and reasons |
|---|---|---|---|---|
| 1 | A candidate inside `[S, S + w]` | `early` | earliest such frame `R` | Today's behaviour: the hold ends at `R`, the lead-in runs back from `R`, and existing live rules apply from `R` |
| 2 | Otherwise, a candidate in `[S − w, S)` | `before` | the nearest such frame `R` | as row 1 |
| 3 | Otherwise, a candidate inside `(S + w, E)` **and** a stoppage in progress at `S` | `late` | earliest such frame `R` | as row 1. The hold covers `[S, R)` as D, because there is positive stoppage evidence |
| 4 | Otherwise, a stoppage in progress at `S` | `unconfirmed` | none confirmed; `E` is the latest possible restart | `[S, E)` is labelled **X**, reason `abstain_unconfirmed_restart`, confidence NaN, including frames with ambiguous movement. The hold is **suspended** in the window and **ends at `E`**. The lead-in back from `S` is unchanged (D, `dead_set_piece_lead_in`). From `E`, existing live rules apply with distinguished reasons (below) |
| 5 | Otherwise | `ignored` | none | The cluster changes no dead state, adds no window and offers no team prior |

**After an `unconfirmed` cluster**, the live stretch starts at `E`. Up to the first confirmed
control, it is labelled as follows:

| Evidence | Label | Reason | Confidence |
|---|---|---|---|
| A later confirmed control run by team T | T, back-filled to `E` | **`backfilled_unconfirmed_restart`** (new) | **0.4**, versus 0.5 for an evidence-backed restart |
| No control, but the cluster team is set | that team | `set_piece_team_prior` | 0.4 |
| Neither | X | `abstain_no_evidence` | NaN |

**Never:**
- D held past `E` because of a tag without evidence;
- H or A on a window frame;
- a confirmed restart at `s0` without movement evidence.

These rules preserve the current behaviour where the old ±2 s rule found an onset, apart from
two changes: an onset inside the tag now beats one just before it, and a superseded reposition
never wins.

## 4. Review of onset selection

| Risk (from DE and the earlier review) | Handling |
|---|---|
| An onset more than 50 frames after `s0` but inside the tag is missed today | Row 3, gated on a stoppage. Without the gate, a false-positive tag over a live occlusion turned 122 live frames dead (prototype scenario N2b) |
| Onsets after the tag end | Not candidates. Allowing end + 50 turned 162 live frames dead (N2) and has no support in containment |
| An onset just before the window beats the kick inside it | The order is `early` > `before` > `late`. A kick inside `[S, S + w]` always wins. A pre-window onset can beat a `late` one only if it is **not** superseded, i.e. the ball kept moving and did not come back to rest, which makes it the likelier restart. A pre-window reposition followed by rest is superseded, and then the `late` kick wins (given a stoppage) |
| **Repositioning picked instead of the restart** (DE) | The superseded-onset rule: a move followed by at least 1 s at rest before `E` is not a restart, and its motion stays inside the stoppage. If no later candidate exists, the cluster is `unconfirmed`, so X, not a guessed restart |
| Slow controlled movement below the restart speed after a stoppage | Not movement evidence, so row 4: the whole window is X. The v2 prototype stopped the window at the first such frame and showed D afterwards; this decision replaces that with X |
| **Overlapping tags** (DE) | One cluster, one decision. Conflicting team tags give no prior. This removes the conflict where one member confirmed a restart while another member's unconfirmed window wrote X over live frames |
| Back-to-back (abutting) tags | Separate clusters, as today; a prior comes from the cluster nearest the live stretch |

## 5. Test plan (synthetic only)

All new tests go in `tests/test_engine_robustness.py::TaggingWindowTests` unless stated otherwise.
Tag `[250, 375)` unless stated. The "out" scenario is: home dribble 0–100, ball seen out at
100–150, then unobserved.

**Provenance of the expected values:**
- **M**: measured on the v2 prototype with Option A, which implements rows 1–5 without the cluster
  and superseded-onset rules.
- **S**: derived from this specification. These must be confirmed when each test goes red, before
  the implementation.

### 5a. The three existing expected failures, with markers removed

| ID | Test | Expected | Src |
|---|---|---|---|
| T1 | `test_restart_late_inside_the_tag_window` (seen moving at 325) | `late`; restart 326; `H*100 + D*226 + A*274` | M |
| T2 | `test_kick_inside_window_beats_a_placement_nudge_before_it` | restart 290–292; `dead_reason[251:290] != LIVE` | M |
| T3 | `test_restart_just_after_the_tag_window` (seen at 395), **expectation revised** | `unconfirmed`; `H*100 + D*150 + X*125 + A*225`; `reason[250:375] == abstain_unconfirmed_restart`; `reason[375:396] == backfilled_unconfirmed_restart` with confidence 0.4 | M (labels), S (new reason) |

### 5b. No-onset cases

| ID | Test | Scenario | Expected | Src |
|---|---|---|---|---|
| T4 | `test_no_onset_after_stoppage_abstains_in_window` | out, ball never seen again (n = 600), team A | `H*100 + D*150 + X*125 + A*225`; window confidence all NaN; `reason[375:] == set_piece_team_prior` with confidence 0.4. **Fails today** (`H*100 + D*150 + A*350`) | M |
| T5 | `test_no_onset_window_is_bounded_by_tag_end` | out; first movement at 475 | `dead_reason` has no hold after 374; `labels[375:475] == A*100` (prior); `labels[250:375] == X*125` | M |
| T6 | `test_no_onset_without_stoppage_is_ignored` | live occlusion 240–400 | `H*600`; no X, no D | M |
| T7 | `test_late_onset_without_stoppage_is_ignored` | live occlusion 240–360 (reappears inside the tag) | `H*600` | M |

### 5c. Ambiguous movement

| ID | Test | Scenario | Expected | Src |
|---|---|---|---|---|
| T8 | `test_reposition_inside_window_is_not_the_restart` | out; placed at 200; repositioned 1 m at 260–270 (2.5 m/s); at rest 270–330 with the taker beside it; kicked at 330 | restart ∈ {330, 331, 332}; `dead_reason[200:330] != LIVE`, **including** 260–270; `labels[340:] == A…` | S |
| T9 | `test_reposition_without_kick_in_window_abstains` | as T8, but at rest until 400 and kicked at 400 | `unconfirmed`; `labels[250:375] == X*125`; nothing before 250 changes. The post-`E` labels follow the live rules (taker control) and are a documented trade-off (§9) | S |
| T10 | `test_slow_controlled_movement_after_stoppage_abstains` | out, missing to 300, then 0.5 m/s controlled dribble | `labels[250:375] == X*125`; `labels[375:]` = A, `controlled` | S (the prototype gave X50 D75; this decision replaces that) |

### 5d. Overlapping tags

| ID | Test | Scenario | Expected | Src |
|---|---|---|---|---|
| T11a | `test_overlapping_tags_share_one_confirmed_restart` | out; `[250, 280)` A and `[250, 260)` A; seen moving at 255 (n = 400) | one cluster, `early`; restart 256; no X; `H*100 + D*156 + A*144` | S |
| T11b | `test_overlapping_tags_with_conflicting_teams_give_no_prior` | out; `[250, 300)` A and `[270, 330)` H; never seen again (n = 400) | cluster `[250, 330)`; `H*100 + D*150 + X*150`; `reason[250:330] == abstain_unconfirmed_restart`; `reason[330:] == abstain_no_evidence` | S |
| — | existing `test_overlapping_set_piece_events_are_deterministic` | `[250, 280)` and `[250, 260)`, both A, no onset | `H*99 + D*151 + X*30 + A*20`; determinism kept | M |

### 5e. Metric property

| ID | Test | Expected |
|---|---|---|
| T12 | `test_unconfirmed_window_is_dead_after_official_abstention_fill` (`tests/test_integration_harness.py`, which may load the official module read-only) | For T4's result: `pm.fill_abstentions(pm.label_codes(result.label), [(s, e) for s, e, _ in match.segments])[250:375]` all equal `pm.LABELS.index("D")`. This shows that spellF1 sees D in the window |

### 5f. Existing expectations that change (measured, Option A)

| Test | Old | New |
|---|---|---|
| `test_engine.DeadBallTests.test_set_piece_extends_dead_back_and_supplies_prior` | H99 D151 A50 | H99 D151 **X10** A40 |
| `…test_set_piece_lead_in_stops_at_live_ball` | restart 250; H199 D51 H50 | restart 260; H199 D51 **X10** H40 |
| `…test_set_piece_prior_can_be_disabled` | `[250:] == X*50` | unchanged labels; `reason[250:260] == abstain_unconfirmed_restart` |
| `…test_throw_in_without_team_has_no_prior` | `[250:] == X*50` | unchanged labels; reason as above |
| `HalfTimeBoundaryTests.test_set_piece_lead_in_does_not_reach_previous_half` | restart 131; H100 D31 H69 | `ignored` (no stoppage); H200 |
| `S2GapEdgeTests.test_periodic_dropouts_do_not_turn_live_play_dead_before_a_tag` | restart 330; H248 D82 H70 | H248 D82 **X10** H60 |
| `SetPieceTransitionTests.test_back_to_back_set_pieces_use_the_restart_that_opens_play` | H99 D201 A100 | H99 D151 X10 D40 X10 A90 |
| `…test_nearest_untagged_restart_gives_no_prior` | `[300:] == X*100` | H99 D161 X10 D30 X100 |
| `…test_overlapping_set_piece_events_are_deterministic` | H99 D151 A50 | H99 D151 X30 A20 |
| `…test_set_piece_at_segment_end` | H149 D46 A5 | H149 D46 X5 |
| `…test_tagged_restart_ends_out_of_play_hold` | restart 200; H60 D140 A100 | H60 D140 X10 A90 |

In every changed case, each X window sits directly after D.

**Unchanged:** all other tests. The synthetic hybrid (`a0f99a12…`) and mock (`15f5a261…`) CSV
hashes are unchanged, because their tags always have an onset (measured on the v2 prototype).

## 6. Contract and reason codes

**Contracts: no change.**
- `MatchInput` and `PredictionResult` are unchanged.
- `X` with NaN confidence is already required by `PredictionResult`.
- `PredictionResult.reason` stays `None`. The separate contract-vocabulary request still stands.

**Engine-internal changes:**

| Item | Change |
|---|---|
| `deadball` dead codes | Add `DEAD_UNCONFIRMED` |
| `temporal.REASONS` | Add `abstain_unconfirmed_restart` and `backfilled_unconfirmed_restart`. The reason array's string width grows to fit |
| `EngineTrace` | Add `restart_kinds`: one of early / before / late / unconfirmed / ignored per cluster, so DE's counts can come from the trace. `restarts` keeps its 4-tuple shape for existing callers |
| `ENGINE_VERSION` | `possession-rules-v0.2`. `config_digest` changes; config fields do not |

**Determinism:**
- Clusters are formed in event sort order.
- Every choice is earliest or nearest, with ties going to the earlier frame.
- It is still a pure function of `MatchInput` and config.

## 7. Affected files (when implemented)

| File | Change |
|---|---|
| `src/mtah/engine/deadball.py` | Clustering, candidate selection with superseded onsets and the stoppage gate, the unconfirmed window, hold suspension and release at `E` |
| `src/mtah/engine/temporal.py` | `DEAD_UNCONFIRMED` → X, the new reasons, prior from the cluster team at `E`, `backfilled_unconfirmed_restart` |
| `src/mtah/engine/possession.py` | `EngineTrace.restart_kinds` |
| `src/mtah/engine/config.py` | `ENGINE_VERSION` only |
| `tests/test_engine_robustness.py` | T1–T11, and the §5f expectation updates |
| `tests/test_engine.py` | §5f expectation updates (4 tests) |
| `tests/test_integration_harness.py` | T12 |
| `docs/POSSESSION_ENGINE.md` | Assumption 4 becomes containment; document the state machine |
| Historical pre-decision review/proposal notes *(omitted from public snapshot)* | Mark as superseded by this decision |

Not touched: `src/mtah/contracts.py`, `pipeline.py`, `export.py`, `__main__.py`, `evaluation/`,
`starter/`, `games.py`.

## 8. Requested from DE before implementation

Local run, aggregate counts only, no ground truth:
- clusters per kind (early / before / late / unconfirmed / ignored), as computed by the v0.2 rule;
- the number of overlapping clusters and of clusters whose teams conflict;
- the tag duration distribution (min / median / max, in frames);
- graded frames that would become X.

These counts size the macroF1 cost. They do not choose any parameter.

## 9. Approval requirements and unresolved trade-offs

**Approvals required:**
1. Engine owner and DE: the policy in §3 (Option A).
2. Engine owner and DE: reusing `stationary_speed_mps` and `min_dead_frames` in the
   superseded-onset rule.
3. Joint: `ENGINE_VERSION` v0.2 and the new engine-internal reason codes. No contract change is
   needed.
4. Integration: the implementation lands on a new branch, then goes through the usual suite,
   preflight and validator on Python 3.11.9 before final integration.

**Unresolved trade-offs:**

| Trade-off | Effect |
|---|---|
| macroF1 cost of X | Every window frame is a miss. The size is unknown until DE's counts arrive |
| After `E` with no movement seen | Containment says play is live. Possession then comes from later control (back-filled to `E`) or the tag team, labelled with distinct reasons and 0.4 confidence. If the true restart was after `E` (a tag error), those frames are wrong |
| The superseded-onset rule can misfire | A real restart followed by the ball resting at least 1 s inside the window (e.g. a player standing on the ball) would be treated as a reposition, giving X or a later restart rather than a wrong one |
| Cluster merging | Merging collapses a retaken set piece tagged with overlapping intervals into one decision |
| Out of scope | Tags straddling a segment start are still dropped, which has low impact |
