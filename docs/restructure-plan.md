# Structure-only restructuring

The starting revision is `ebedece` on `risklab-volatility-trial`.
Implementation work is on `restructure-project`.

## Preservation requirements

- Keep every existing research feature, configuration, CLI command, and module entry point.
- Preserve all data snapshots and existing output paths, file names, formats, and bytes.
- Preserve calculations, model training, random seeds, calendars, allocation rules, fees, tests, and statistical conventions.
- Keep historical provenance and forecast caches verifiable. A reviewed relocation may be recognized explicitly; scientific code changes must continue to invalidate incompatible caches.
- Add no research features or method changes. Developer-only verification checks support this migration.
- Leave the pre-existing untracked `scripts/make_findings_figures.py` untouched.

## Baseline

A separate local Desktop copy contains source, configurations, tests, documentation,
scripts, supplements, data, and outputs:

`C:/Users/sushm/Desktop/financial-economics-before-restructure-20261022-123732`

Its `baseline-manifest.json` records SHA-256 hashes and sizes for 528 files,
including 360 output files. Source and destination hashes agreed immediately
after copying. These local results and licensed snapshots remain excluded from Git.
The baseline suite passed 420 tests with warnings treated as errors.

## Architecture

Implementation modules move into `datasets`, `settings`, `models`, `portfolio`,
`evaluation`, `studies`, and `reporting`. The original flat module names remain
small compatibility entry points. Existing callers therefore keep working while
new readers can follow the implementation by responsibility. No legacy study is
retired. Configuration and output directories retain their original layout.

Three Sol writers at high reasoning work in distinct copied directories with
non-overlapping module ownership: datasets/settings; models/portfolio/evaluation;
and studies/reporting. They cannot edit the integration checkout under their task
instructions. The lead owns integration, CLI compatibility, provenance, tests,
documentation, and commits. Adversarial reviewers receive separate read-only
copies and report findings rather than editing the checkout.

## Verification and checkpoints

1. Commit the baseline and preservation plan.
2. Integrate relocated implementations and legacy entry points; verify import,
   monkeypatch, command, and provenance compatibility. Commit the verified structure.
3. Document the actual layout and existing commands; independently review changes.
4. Compare the complete retained output/data/configuration inventory with the
   Desktop baseline. Compare full numerical paths and inference under the baseline
   and restructured implementations, including saved-forecast replay when local
   source coverage permits. Run the complete suite and lint. Commit verification.
5. Address every material adversarial finding and repeat affected checks before
   declaring completion. Preserve a record of unavailable historical runs rather
   than claiming they were reproduced.

## Completion evidence

All implementation and verification steps are complete. See
[Restructuring verification](restructure-verification.md) for the protected-file
comparison, explicit live-job transitions, exact 877 saved-run and 450 synthetic
table comparisons, 485 passing tests, packaging/command checks, adversarial fixes
and coverage limits. No research feature or method changed.
