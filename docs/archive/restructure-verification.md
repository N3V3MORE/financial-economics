# Restructuring verification — October 22, 2026

The structure-only refactor preserves the existing research features, commands,
configurations, calculations and result formats. Implementation is grouped by
responsibility; old module paths remain compatibility entry points. Developer-only
preservation checks were added. No research feature, model, study, or method was
added or removed.

## Backup and retained results

Starting source revision: `b78872e`.
Working branch: `restructure-project`.

The independent Desktop backup is:

`C:/Users/sushm/Desktop/financial-economics-before-restructure-20261022-123732`

Its immutable `baseline-manifest.json` records SHA-256 hashes and byte sizes for
528 files, including 360 files initially under `outputs/`. The backup also
contains source, tests, configurations, documentation, scripts, supplements and
local data snapshots. The initial copy was independently hash-checked.

Final preservation checks verify all 419 protected entries in both the backup
and current checkout: data, outputs, configurations, supplements and the
pre-existing untracked figure helper. Of the initial 360 output entries:

- 337 retain the same paths and exact bytes.
- 22 temporary GPU checkpoints were moved into the completed GRU stage by a
  pre-existing training job. Every completed checkpoint has the exact original
  byte hash. Their original copies remain in the Desktop backup.
- One pre-existing training log grew by appending; its original byte prefix is
  preserved in full and its original copy remains in the backup.

These 23 live transitions are individually declared and checked, rather than
ignored. The already-running job produced 90 additional output files; the
verifier reports those additions separately. Completed result files were not
rewritten by the refactor. The original untracked
`scripts/make_findings_figures.py` remains present with its baseline bytes and
has not been added to the commits.

## Independent numerical comparisons

Separate Python processes load the backed-up source and the restructured source
against identical, locally verified inputs. Comparison uses full ordered table
schemas, indexes, dates, dtypes and lossless float representations. No rounded
headline values or tolerance bands determine success.

| Saved experiment | Complete tables compared | Result |
|---|---:|---|
| `revised-risklab-20261022` | 199 | Exact match |
| `risklab-2022-2025-20261022` | 79 | Exact match |
| `trial-oxford-revised-20261022` | 199 | Exact match |
| `trial-risklab-2025-20261022` | 43 | Exact match |
| `trial-risklab-defensive-1996-20261022` | 79 | Exact match |
| `trial-risklab-defensive-2000-20261022` | 79 | Exact match |
| `trial-risklab-revised-20261022` | 199 | Exact match |
| **Saved-run total** | **877** | **Exact match** |

An additional 450 synthetic tables match exactly across daily, weekly and
monthly rebalancing; paper and annual risk conventions; four fee scenarios;
cash entry; benchmark costs; and final liquidation.

Coverage includes full portfolio return paths, every trading/fee-accounting
table, performance, conditional-return and Sharpe tests from the shared engine,
thresholds, static controls, decisions and decline tables. Source and input
identities remain unchanged throughout each comparison. Replay returns were
prepared locally with `load_returns(..., verified_monthly=True)`; licensed
observations are neither printed nor committed.

The replay does not refit GPU models, rerun every bootstrap, or regenerate every
historical report. Those paths remain implemented and covered by existing tests;
the saved artifacts are preserved independently. Study-specific extra test-family
assembly is covered by the original regression tests and source-equivalence
review, rather than claimed as part of the shared-engine replay.

## Other checks

- Complete test suite: **485 passed**, with warnings treated as errors; baseline
  was 420 passed. Original test files remain unchanged.
- Ruff: all checks passed.
- Nine original package/module command entry points: help text, standard error
  and exit status exactly match the baseline.
- All 23 original README fenced command blocks remain verbatim in
  [the complete command reference](../commands.md); local documentation links pass.
- Across 43 relocated implementations, 257 top-level function/class definitions
  are identical after import normalization. Eight changed definitions contain
  provenance/orchestration adjustments. No original definitions were removed.
- Built wheel: isolated installation imports, source identities, legacy imports,
  fresh `runpy`, reload and serialization checks pass.
- Committed source archive: exact LF source hashes match the reviewed relocation
  manifest independently of the working checkout.
- Old forecast compatibility still checks original source/input identities and
  saved file hashes. Unreviewed changes and retired GPU-math identities remain
  rejected. New runs record actual current source fingerprints.

## Scope ownership and adversarial review

Three Sol subagents at high reasoning wrote separate copied scopes:
11 datasets/settings modules, 16 models/portfolio/evaluation modules, and
16 studies/reporting modules. Their file ownership did not overlap. The lead
integrated the results in the checkout and owned compatibility/provenance.
The reviewer received independent copied snapshots and did not edit the checkout.

Adversarial rounds found and prompted fixes for imported and fresh `runpy`
execution, reload/pickle compatibility, incomplete backup inventory acceptance,
missing decline-table comparison, zero-column index hashing, and overly broad
provenance wording. The final code review reports no remaining concrete
preservation blocker. The distinction between whole-package and selected-module
provenance is documented in [the structure guide](../structure.md).

Major checkpoints are committed:

- `a596b7f`: baseline and preservation plan.
- `871dc1d`: grouped implementation and provenance compatibility.
- `6e6131b`: reload and serialization compatibility.
- `96653e3`: fresh `runpy` execution and caller-main preservation.
- `0686191`: documentation and full-table preservation verifier.

Machine-readable reports, test logs, replay requests, the wheel check and source
equivalence report remain local at:

`C:/Users/sushm/Desktop/financial-economics-restructure-verification-20261022`

The `*-comparison-final.json` files and `synthetic-comparison-final.json` are the
final numerical/preservation evidence. The backup and verification directory
include licensed inputs and stay local. Only aggregate verification facts are
recorded here.
