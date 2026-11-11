# Submission preflight

`scripts/submission_preflight.py` checks a `frame,label` CSV strictly before it is submitted. It
reads only the CSV files you name. It never opens match assets or ground truth, and it does not
score.

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -B scripts\submission_preflight.py <submission.csv>
& $py -B scripts\submission_preflight.py <run-1.csv> --compare <run-2.csv>     # reproducibility
& $py -B scripts\submission_preflight.py <file.csv> --expect-rows 40            # short synthetic file
```

It exits with 0 and prints `PASS`, or exits with 1 and prints `FAIL` followed by every problem it
found. It always prints the row count, label counts, SHA-256 and the official validator's verdict.

## What it checks

| Check | Rule | Failure text |
|---|---|---|
| Encoding | UTF-8, no BOM | `UTF-8 byte-order mark present ...`, `not valid UTF-8` |
| Line endings | LF only, with a final newline | `CR characters present ...`, `missing final newline` |
| Header | Exactly `frame,label` | `header must be exactly 'frame,label'` |
| Row count | Exactly 199,319 data rows (the tactical video) | `expected 199319 data rows, found N` |
| Row shape | `<digits>,<one capital letter>`, with no spaces or extra columns | `row k: malformed '...'` |
| Frame sequence | Row k carries frame k−1, so `0..199318` with no gaps, duplicates, reordering or 1-based IDs | `row k: expected frame k-1, found f` |
| Labels | `H`, `A`, `D` or `X` only | `row k: invalid label 'Z'` |
| Official validator | `evaluation/possession_metrics.validate_submission`, unchanged and imported read-only | `official validator: <reason>` |
| Reproducibility (`--compare`) | Byte-identical files | `not byte-identical to <name>` |

**What it does not check:**
- whether the file came from the real loader;
- whether the labels are any good.

A synthetic CSV of the right length passes. Provenance comes from the procedure below, not from
the file.

## Guarantees already enforced when the CSV is written (`python -m mtah`)

**Export path** (`mtah/export.py`, unchanged):
- The result is validated against its input `MatchInput`: same metadata and frames.
- The CSV is written to a temporary file in the destination directory and checked by the official
  validator.
- It is then published without overwrite (a Windows rename, or a POSIX exclusive hard link), and
  the temporary file is removed in every case.
- An existing destination is refused: exit 1, `error: output already exists; choose a new --out
  path`. The existing file is left byte-for-byte untouched.
- A destination inside `evaluation/` is refused: exit 1.
- Missing parent directories are created. Paths with spaces work.

**CLI failures** (`mtah/__main__.py`, `mtah/sources.py`):
- A missing, broken or wrong-typed loader exits with 2 and writes no CSV.
- Loader messages have the input path replaced by `<input>`.
- No placeholder or all-X file is ever written after a failure.

**Determinism:** with the same code, configuration and input, every run produces the same bytes.
This is shown by the SHA-256 printed by the CLI and by the preflight `--compare` check.

These guarantees are covered by `tests/test_cli.py`, `tests/test_integration_harness.py` and
`tests/test_submission_preflight.py`.

## Final procedure (local, authorised workstation, human-run)

1. Run `& $py scripts\check_environment.py`. It must print
   `environment matches requirements.lock`. If it doesn't, install from the lock with
   `& $py -m pip install -r requirements.lock`, then re-check.
2. Run both test suites. Both must report 0 failures:
   - `& $py -B -m unittest discover -s tests`
   - `& $py -B -O -m unittest discover -s tests`
3. Record `git rev-parse HEAD` and confirm `git status --short` is clean.
4. Generate the CSV twice, into fresh paths *(when running from a source checkout without `pip install --no-deps -e .`, or from another checkout without a local `.venv`, set `$env:PYTHONPATH = "$PWD\src"` in PowerShell — or `PYTHONPATH=src` on POSIX — before invoking `-m mtah`)*:

   ```powershell
   & $py -B -m mtah --source atd --input <local-asset-root> --engine hybrid --out artifacts\final\run-1.csv
   & $py -B -m mtah --source atd --input <local-asset-root> --engine hybrid --out artifacts\final\run-2.csv
   ```

5. Run `& $py -B scripts\submission_preflight.py artifacts\final\run-1.csv --compare artifacts\final\run-2.csv`.
   It must print `PASS` and `byte-identical`.
6. Run the official validator yourself as well:
   `& $py -B evaluation\validate_submission.py artifacts\final\run-1.csv`.
7. Keep the CSV, hashes and logs local. Submission is a separate, explicitly approved step.

## Reference results (synthetic, Python 3.11.9, 2026-11-05)

| Command | SHA-256 | Preflight |
|---|---|---|
| `python -m mtah --engine baseline --frames 199319` | `13151c4c89668b48f718881fa9e921d4d2220154199407880a607a6fa8f630f0` | PASS, H=99658 A=99660 D=1 X=0 |
| `python -m mtah --engine hybrid --frames 199319`, run twice | `a0f99a12195daac9f519041ad3990ecfb38e681d85de8aff653d8882345eb974` | PASS, byte-identical |
| `tests/run_cli_with_mock_loader.py --source atd --engine hybrid`, run twice (mock-loader fallback) | `15f5a261c1983b1c58f7dc7ec66bab28c3b3491e62c919765e8e179c9db75cd9` | PASS, H=79834 A=67212 D=11811 X=40462, byte-identical |
| `tests/run_cli_with_mock_loader.py --source atd --engine baseline` | `1b97b56fd403a0bbdffe7f48f5986b694b20f6fe00560f6497fe7432f4def63a` | PASS |
| `python -m mtah --source atd ...` with no loader installed | none | exit 2, no file |

The mock hybrid run has X=40,462, which is exactly the number of frames outside the published
segments. So the hybrid engine abstains only on ungraded frames of the mock match.
