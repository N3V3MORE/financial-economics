# Project structure

The implementation is organized by responsibility under
`src/financial_economics/`. Every study and source tool runs through `fecon`
(or `python -m financial_economics`). The staged studies and source tools that were
once run as `python -m financial_economics.<module>` are `fecon` subcommands with
the same options, listed below. The former flat module names are no longer
importable; they survive only as provenance names in run records.

This is a move and import refactor. No methods, features or studies were removed.
Calculations, model settings, training rules, random seeds, calendars, allocation
rules, fees and statistical conventions retain their existing behavior. The
original phases, protection study, daily-return prediction experiment, paper
replications, extensions, methodology revision and Risk Lab work remain available.
The project does not introduce a new model or study in this restructuring.

## Responsibility groups

| Group | Responsibility |
|---|---|
| `datasets/` | Source adapters, immutable snapshots, source validation, return inputs and volatility measurements |
| `settings/` | Study and frozen experiment configuration loading and validation |
| `models/` | Forecast preparation, classical models, GRUs and GPU runtime checks |
| `portfolio/` | Allocation, holdings, fees, controls and decline comparisons |
| `evaluation/` | Metrics, inference, resampling, attribution and replication scoring |
| `studies/` | Experiment orchestration and staged workflows |
| `reporting/` | Tables, charts, reports and the original combined page |

`datasets` is separate from local downloaded snapshots in the repository's `data/`
directory. The CLI stays at the package root in `cli.py`; `__main__.py` supports
`python -m financial_economics`, and `fecon` is the installed command. Shared
provenance helpers remain at the root in `provenance.py`.

## Provenance after the move

Run records name modules by their original flat filenames, for example
`paper_gru.py`. `_layout.py` maps each name to its grouped file, and
`_relocation.json` records each file's exact hash just before and after the
reviewed move. Whole-package provenance records the hash of every package module
and the mapping file, so any code change requires a fresh run, as before the move.
Selected-module provenance (the reconstruction's saved forecasts) accepts a
pre-move digest while that module still matches its own post-move hash, so edits
to unrelated modules do not invalidate it. Retired GPU-math fingerprints remain
rejected. Saved metadata is never rewritten.

Python source files use LF line endings through `.gitattributes`, so these byte
identities survive Windows/Linux checkouts and wheel installation.

## Module map

Each former name below is relative to `financial_economics`, as is each module.
For example, the former `paper_revised` is now `studies.historical` and runs as
`fecon paper-revised` with its original options. The complete command reference
is in [Commands and study reference](commands.md).

| `fecon` subcommand | Module |
|---|---|
| `paper-reconstruction` | `studies.replication` |
| `paper-realistic` | `studies.realistic` |
| `paper-revised` | `studies.historical` |
| `proxy-study` | `studies.proxy` |
| `risklab-study` | `studies.recent` |
| `paper-figures` | `reporting.figures` |
| `paper-macro-sources` | `datasets.sources.macro` |
| `volatility-proxy` | `datasets.volatility` |
| `wrds-credentials` | `datasets.credentials` |

### datasets

| Former module | Module | Responsibility |
|---|---|---|
| `data` | [`datasets.public`](../src/financial_economics/datasets/public.py) | Public snapshot download, validation and calendar alignment |
| `return_data` | [`datasets.returns`](../src/financial_economics/datasets/returns.py) | Provider-independent returns and complete-month compounding |
| `wrds_credentials` | [`datasets.credentials`](../src/financial_economics/datasets/credentials.py) | Native Windows WRDS credential setup and retrieval |
| `wrds_source` | [`datasets.sources.crsp`](../src/financial_economics/datasets/sources/crsp.py) | Bounded CRSP extracts, security identities and monthly reconciliation |
| `paper_sources` | [`datasets.sources.paper`](../src/financial_economics/datasets/sources/paper.py) | Labelled public substitutes and paper-input preparation |
| `paper_macro_sources` | [`datasets.sources.macro`](../src/financial_economics/datasets/sources/macro.py) | WRDS/FRED yields and official EIA Brent snapshots |
| `paper_fx` | [`datasets.sources.fx`](../src/financial_economics/datasets/sources/fx.py) | Dated FX and local-currency foreign-bond valuation |
| `volatility_proxy` | [`datasets.volatility`](../src/financial_economics/datasets/volatility.py) | Risk Lab loaders and diagnostic Oxford-like measurement estimation |

### settings

| Former module | Module | Responsibility |
|---|---|---|
| `config` | [`settings.study`](../src/financial_economics/settings/study.py) | Study definition and validation |
| `prediction_config` | [`settings.prediction`](../src/financial_economics/settings/prediction.py) | Frozen daily-return prediction settings |
| `protection_config` | [`settings.protection`](../src/financial_economics/settings/protection.py) | Frozen ETF-versus-cash settings |

### models

| Former module | Module | Responsibility |
|---|---|---|
| `forecasting` | [`models.variance`](../src/financial_economics/models/variance.py) | Forecast calendars, inputs and EWMA variance forecasts |
| `gru_forecasting` | [`models.gpu`](../src/financial_economics/models/gpu.py) | Daily-return GRU forecasts, device checks and annual refits |
| `paper_gru` | [`models.paper_gru`](../src/financial_economics/models/paper_gru.py) | Two-layer GRU for the partial paper replication |
| `paper_models` | [`models.classical`](../src/financial_economics/models/classical.py) | ARFIMA, GARCH and GJR-GARCH forecasts |
| `paper_networks` | [`models.networks`](../src/financial_economics/models/networks.py) | Raw-BV GRUs for the full reconstruction |
| `paper_networks_log` | [`models.log_networks`](../src/financial_economics/models/log_networks.py) | Post hoc positive log-BV GRUs |

### portfolio

| Former module | Module | Responsibility |
|---|---|---|
| `costs` | [`portfolio.execution`](../src/financial_economics/portfolio/execution.py) | Self-financing ETF trading fees and holdings accounting |
| `comparison` | [`portfolio.controls`](../src/financial_economics/portfolio/controls.py) | Calibrated SPY/cash controls and comparisons |
| `stress` | [`portfolio.declines`](../src/financial_economics/portfolio/declines.py) | SPY decline windows and recovery comparisons |
| `fund_checks` | [`portfolio.fund_checks`](../src/financial_economics/portfolio/fund_checks.py) | Fund removals, additions, periods and cost scenarios |
| `prediction` | [`portfolio.prediction`](../src/financial_economics/portfolio/prediction.py) | Target-volatility portfolios from issued forecasts |

### evaluation

| Former module | Module | Responsibility |
|---|---|---|
| `analysis` | [`evaluation.metrics`](../src/financial_economics/evaluation/metrics.py) | Returns, risk, portfolio metrics and market exposure |
| `explain` | [`evaluation.attribution`](../src/financial_economics/evaluation/attribution.py) | Paired resampling and fund-level gain/loss attribution |
| `prediction_inference` | [`evaluation.prediction`](../src/financial_economics/evaluation/prediction.py) | Paired forecast and prediction-portfolio tests |
| `protection_inference` | [`evaluation.protection`](../src/financial_economics/evaluation/protection.py) | Protection HAC/Holm tests and paired bootstrap |
| `paper_evaluation` | [`evaluation.replication`](../src/financial_economics/evaluation/replication.py) | Paper-style model and asset scoring |

### studies

| Former module | Module | Responsibility |
|---|---|---|
| `paper` | [`studies.paper`](../src/financial_economics/studies/paper.py) | Partial GRU-UNIVAR replication on Oxford-Man BV |
| `paper_extension` | [`studies.extension`](../src/financial_economics/studies/extension.py) | Paper extension with trading delays, fees and cash income |
| `paper_reconstruction` | [`studies.replication`](../src/financial_economics/studies/replication.py) | Full reconstruction stages and saved-forecast reuse |
| `paper_realistic` | [`studies.realistic`](../src/financial_economics/studies/realistic.py) | Realistic trading tests using saved reconstruction forecasts |
| `paper_revised` | [`studies.historical`](../src/financial_economics/studies/historical.py) | Matched historical methodology revision and its stages |
| `proxy_study` | [`studies.proxy`](../src/financial_economics/studies/proxy.py) | Exploratory later-year study gated on an accepted estimated proxy |
| `risklab_study` | [`studies.recent`](../src/financial_economics/studies/recent.py) | Later-year Risk Lab study, including defensive-asset variants |
| `protection` | [`studies.protection`](../src/financial_economics/studies/protection.py) | Fixed ETF additions versus cash across costs and periods |

### reporting

| Former module | Module | Responsibility |
|---|---|---|
| `report` | [`reporting.baseline`](../src/financial_economics/reporting/baseline.py) | Phase 1 tables, charts and summary |
| `stress_report` | [`reporting.stress`](../src/financial_economics/reporting/stress.py) | Phase 2 decline report |
| `comparison_report` | [`reporting.comparison`](../src/financial_economics/reporting/comparison.py) | Phase 3 control-comparison report |
| `explain_report` | [`reporting.attribution`](../src/financial_economics/reporting/attribution.py) | Phase 4 attribution report |
| `prediction_report` | [`reporting.prediction`](../src/financial_economics/reporting/prediction.py) | Prediction tables, figures and run records |
| `protection_report` | [`reporting.protection`](../src/financial_economics/reporting/protection.py) | Protection tables, figures and run records |
| `paper_figures` | [`reporting.figures`](../src/financial_economics/reporting/figures.py) | Saved-run paper figure reconstruction |
| `page` | [`reporting.page`](../src/financial_economics/reporting/page.py) | Combined page from the original four phase outputs |

## Files around the package

| Location | Purpose |
|---|---|
| `configs/` | Existing study definitions and frozen experiment settings; paths and contents remain unchanged by the restructuring |
| `data/` | Local ignored source snapshots, retained in their existing paths and formats |
| `outputs/` | Local ignored results, checkpoints, figures and run records, retained in their existing paths and formats |
| `tests/` | Synthetic calculation, validation and command checks |
| `scripts/` | Independent protection verification, supplement refresh and GPU comparison tools |
| `supplement/` | Supplemental workbook and cash illustration |
| `docs/` | Research interpretation, methods, phase notes, command reference and this module map; `docs/archive/` keeps the restructure plan, verification record and cleanup log |
| `pyproject.toml`, `uv.lock` | Package entrypoints, dependencies, extras and locked environment |
| `.github/workflows/tests.yml` | Windows and Linux CI checks |

The module move does not relocate configurations, snapshots or saved results.
Run commands from the project root so their existing relative paths keep working.

## Where to start a change

Follow source transport or validation into `datasets`, configuration loading into
`settings`, forecasts into `models`, allocation and fees into `portfolio`, and
metrics or tests of economic claims into `evaluation`. Study sequencing belongs
in `studies`; presentation belongs in `reporting`. Keep the existing experiment
boundaries explicit when making changes: the paper's idealized reconstruction,
the realistic trading test and the matched revision answer different questions.

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the development workflow. Run
synthetic tests without contacting providers. For commands, source requirements,
GPU extras and dated result caveats, use [Commands and study reference](commands.md).
