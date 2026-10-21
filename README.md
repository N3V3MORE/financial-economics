# ETF diversification study

Do different ETFs help spread investment risk, especially when markets fall?

Phase 1 builds the baseline: historical returns, risk, correlations, a monthly equal-weight portfolio of the eight funds (the "mix"), and CAPM estimates with uncertainty. Phase 2 compares the mix with SPY in every SPY decline of at least 10%, including how long each took to recover. Phase 3 asks whether the mix did better than simply holding less stock, using a SPY + cash portfolio chosen from 2010–2017 and tested on 2018–2025. Phase 4 checks whether chance alone could explain the Phase 3 differences, and which funds produced the mix's gains and losses in each fall.

For a plain-language account of what the study has found, read [Results explained](docs/results-explained.md).

## Run the study

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the project folder:

```sh
uv sync --frozen
uv run fecon download
uv run fecon analyze
uv run fecon stress
uv run fecon compare
uv run fecon explain
```

Each analysis command writes a `summary.md` that links its charts and tables:

| Command | Opens | Question |
|---|---|---|
| `analyze` | `outputs/phase1/summary.md` | How did the funds and the mix perform over 2010–2025? |
| `stress` | `outputs/phase2/summary.md` | What happened in each SPY decline of 10% or more? |
| `compare` | `outputs/phase3/summary.md` | Did the mix beat a SPY + cash portfolio over 2018–2025? |
| `explain` | `outputs/phase4/summary.md` | Could chance explain that comparison, and which funds drove it? |

`download` saves prices from Yahoo Finance through yfinance and daily market and risk-free returns from Kenneth French. The other commands read that saved snapshot offline and check that its files and settings still match.

To refresh the data, use a new snapshot directory:

```sh
uv run fecon download --data-dir data/new-snapshot
uv run fecon analyze --data-dir data/new-snapshot --output-dir outputs/new-run
```

Existing snapshots are never silently replaced. Changing the ETFs or dates also requires a matching new snapshot. Incomplete or unavailable provider data produces an error.

`stress` and `compare` accept settings too, for example `--drawdown-threshold 0.10 --rolling-window 63` or `--evaluation-start 2018-01-01`. See [Phase 2](docs/phase-2.md) and [Phase 3](docs/phase-3.md).

`explain` rebuilds 2018–2025 10,000 times from runs of real trading days, 21 days long on average, keeping every portfolio on the same days, and reports how far the Phase 3 differences move. A fixed random seed makes reruns identical; `--resamples`, `--block-length`, and `--seed` change the settings. It also measures each fund's gain or loss in every SPY fall, adding up exactly to the mix's loss. See [Phase 4](docs/phase-4.md).

## Check which funds help

`fund-checks` removes each fund in turn and redistributes its weight equally
among the remaining funds. It keeps the original mix's SPY + cash comparison
and adds a separately calibrated SPY + cash comparison for each variant.
Each control uses only returns before its evaluation period.

```sh
uv run fecon fund-checks --data-dir data/public-review-20261019 --output-dir outputs/fund-checks-original
uv run fecon --config study-commodities.toml download --data-dir data/commodities-20261020
uv run fecon --config study-commodities.toml fund-checks --additions DBC --data-dir data/commodities-20261020 --output-dir outputs/fund-checks-commodities
```

The first command uses the recovered eight-fund snapshot. The other two use
a separate nine-fund snapshot, comparing the eight-fund baseline and its DBC
extension on exactly the same observations. Downloaded snapshots stay local;
use a new directory when fetching another data vintage.

Defaults cover 2016–2025, 2018–2025, 2020–2025, and four two-year periods
from 2018 through 2025. The longer windows overlap. Change them with
`--windows 2018-01-01:2025-12-31`, and costs with `--costs-bps 0 10 25`.
Ten basis points means $1 per $1,000 of ETF purchases or sales. Both sides of
a trade are charged; the RF cash proxy has no fee. These are cost scenarios,
not estimates of historical execution costs. Entry fees are included, exit
fees and taxes are excluded. Fees come out of portfolio capital before the
next return. Monthly targets stay fixed and holdings drift between resets.

Each output directory contains `performance.csv`, `differences.csv`,
`calibration.csv`, `daily_returns.csv`, and `run.json`. The JSON records
the source manifest and assumptions. Existing result directories are refused.
Differences are portfolio minus reference: higher growth and Sharpe are
favorable, lower volatility is favorable, and a positive drawdown difference
means a shallower loss. This is a historical check, not a fund-ranking rule.

## ETF additions versus cash: the protection study

The expanded study asks which funds help in stock-market declines **relative
to cash**, and what growth and risk come with that choice. It preserves the
original eight-fund work below. The frozen settings are in `protection.toml`
(14 funds, 2015–2025) and `protection-long.toml` (SPY/TLT/GLD/DBC, 2007–2025).
Read the measured findings in [results-explained.md](docs/results-explained.md#which-additions-earned-their-place-relative-to-cash).

The October 20, 2026 source decision was **WRDS / CRSP CIZ for both samples**.
All 1,848 main-sample and 912 longer-sample fund-months passed the daily/monthly
reconciliation threshold of one basis point; maximum differences were 0.0555
and 0.0433 basis points. Complete French trading calendars, historical PERMNO
coverage, ETF classifications, valid return flags, and duplicates were checked.
No provider splice, missing-return filling, or public-data fallback was needed.

```sh
uv sync --extra wrds
uv run --extra wrds fecon --config protection.toml download --source wrds --verify-monthly --data-dir data/protection-wrds-20261020
uv run --extra wrds fecon --config protection-long.toml download --source wrds --verify-monthly --data-dir data/protection-long-wrds-20261020
uv run fecon --config protection.toml protection --data-dir data/protection-wrds-20261020 --output-dir outputs/protection-main-final
uv run fecon --config protection-long.toml protection --data-dir data/protection-long-wrds-20261020 --output-dir outputs/protection-long-final
uv run python scripts/verify_protection.py --config protection.toml --data-dir data/protection-wrds-20261020 --output-dir outputs/protection-main-final
uv run python scripts/verify_protection.py --config protection-long.toml --data-dir data/protection-long-wrds-20261020 --output-dir outputs/protection-long-final
```

Skip downloads when those snapshots already exist. Downloads and protection
outputs refuse existing destinations; choose new names for a later vintage or
rerun. WRDS uses an existing local PostgreSQL password file and optional
`WRDS_USERNAME`; passwords and raw provider errors are never printed. There
is no automatic fallback. If verified WRDS coverage is unavailable, explicitly
use `download --source public` into a separately named directory and carry that
provider label through the entire run. Never combine the providers' returns.

`load_returns` supplies validated daily returns, aligned factors, an initial
valuation date, and provenance. CRSP returns are consumed directly. Its
normalized total-return index exists only to preserve legacy command support.
Version-1 public snapshots and earlier version-2 WRDS snapshots still load;
the protection command requires the extra monthly check for WRDS sources.

The main run contains 46 portfolio definitions across three cost scenarios:
39 candidate additions, three same-SPY-weight cash controls, SPY, and the
three fixed equal-weight fund lists. The longer run contains 13 definitions.
The primary statistical family is only the 13 fund-versus-cash comparisons
in 2015–2025. Allocation sizes, subperiods, and the longer sample are not
independent confirmations. All bootstrap jobs run sequentially on CPU in
batches of 250; no GPU training is used.

| Saved evidence in each output directory | Contents |
|---|---|
| `fund_conditions.csv`, `fund_monthly_returns.csv` | All, negative-SPY, and severe-SPY month descriptions and underlying local monthly series |
| `primary_tests.csv`, `bootstrap_intervals.csv` | Main sample only: HAC/Holm tests and 10,000-draw intervals at 6-, 3-, and 12-month blocks |
| `performance.csv`, `portfolio_conditions.csv` | All allocations, costs, fixed mixes, and descriptive periods |
| `portfolio_definitions.csv`, `target_weights.csv` | Explicit portfolio membership and weights |
| `daily_returns.csv`, `monthly_returns.csv`, `fees.csv` | Full local portfolio paths and fee/traded-notional accounting per initial dollar |
| `beta_calibration.csv`, `beta_performance.csv`, `beta_conditions.csv` | Training estimates, clipped control targets, realized exposures, and evaluation results |
| `spy_episodes.csv`, `episode_comparisons.csv`, `fund_episode_comparisons.csv` | Common SPY decline windows, losses, and recoveries |
| `protection.png`, `tradeoffs.png`, `declines.png` | The three main figures, each backed by a `figure_*.csv` table |
| `run.json` | Source manifest, frozen settings, units, code/config/output hashes, software versions, and run conventions |

The independently written verifier checks all main metrics, conditional means,
zero-cost monthly portfolio identities, and plotted numbers. It recomputes HAC
covariance and Holm adjustment using matrix algebra, without the inference
module. Synthetic tests cover resampling dependence and paired observations,
seed reproducibility, source failures, alignment, and fee accounting.

Raw licensed observations and generated detailed paths stay local in ignored
`data/` and `outputs/`. Do not redistribute them.
Repository documentation contains aggregate findings, not licensed observations.

## Predict volatility, then compare ETF protection with cash

`prediction.toml` extends the retained 2007–2025 SPY/TLT/GLD/DBC snapshot.
It adapts Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe
Haven Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).
Our forecast target is a future squared **daily** SPY return,
a noisy variance proxy under a zero-daily-mean approximation; it differs from
the paper's intraday-derived measure. This is an attributed adaptation.

Both models receive the previous 22 daily SPY returns and their squares.
EWMA selects one decay; the one-layer GRU selects hidden size and learning
rate from the frozen small grid and averages five fixed seeds. Training ends
in 2014, validation covers 2015–2016, and historical evaluation covers
2017–2025. Annual refits use available targets only. GRU epochs are frozen
after validation. All settings are explicit in `prediction.toml`.

```sh
uv run fecon --config prediction.toml predict --ewma-only --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-ewma
uv sync --frozen --extra prediction
uv run --extra prediction fecon --config prediction.toml predict --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-core
uv run fecon --config prediction.toml predict --forecast-run outputs/prediction-core --defensive-asset GLD --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-gold
uv run fecon --config prediction.toml predict --forecast-run outputs/prediction-core --defensive-asset DBC --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-commodities
```

Skip downloads when the retained snapshot exists. Default `predict` trains
on **CUDA:0**, verifies effective device placement, and stops if CUDA is
unavailable; it never retries training on CPU. `--ewma-only` and verified
`--forecast-run` reuse run without importing PyTorch. They are mutually
exclusive. Saved forecasts must match settings, snapshot provenance, and
file hashes. A saved baseline stays a baseline. Portfolio accounting and
bootstrap calculations are CPU-native; substantial runs stay sequential.

A/B use EWMA/GRU to divide SPY and cash. C/D keep the same respective SPY
targets and add up to 20% defensive ETF from available cash during predicted
stress. SPY targets a 10% annual volatility level without borrowing; adding
the ETF adds risk. Annual stress thresholds use preceding issued forecasts
from a common validation-start history, initially calibrated with the selected
models' archived validation forecasts. A forecast at Monday close trades
Tuesday close and earns Wednesday's daily return. Settings freeze at the
last 2016 trading close, so January 3, 2017 remains in cash and the first
forecast-directed return ends January 4.

Main results earn observed cash income and charge 10 bps per ETF dollar
bought or sold, including entry; 0/25 bps and zero cash income are separate
sensitivities. Exit fees and taxes are excluded. Output directories contain
return/allocation/fee tables, tuning evidence, model checkpoints when copied,
three PNG figures with their exact `figure_*.csv` sources, and hashed
`run.json` provenance. Existing destinations and paths inside source snapshots
or reused forecast runs are refused. No Markdown report is generated.

Only full TLT results have the three-test primary family: paired forecast
QLIKE, D−C protection, and D−B ETF-versus-cash protection. Bartlett HAC uses
20 daily or six monthly lags, small-sample correction, two-sided tests, and
one Holm adjustment. Monthly tests retain the full calendar with down/other
SPY indicators. Paired 10,000-draw stationary-bootstrap intervals use mean
blocks of six months, with three/twelve-month sensitivities. They condition
on fitted models and omit model-selection uncertainty. Secondary assets,
subperiods, and severe declines are descriptive. Better forecasts, better
portfolios, and an ETF improving on cash are **three separate claims**.
Previously inspected histories remain retrospective evidence, not unseen data.

## See the original phase results

After the four analysis commands, put the main findings on one page:

```sh
uv run fecon page
```

Open `outputs/page/index.html` in a browser. It is a single file with no outside links, so you can also send it to someone. Each chart sits next to a table of the exact numbers, and hovering over a growth chart shows the values for that week. The page reads only the saved outputs. It stops if they came from different snapshots or settings, or if Phases 3 and 4 used different test periods or SPY + cash splits.

## Find your way around

```text
study.toml                         Your ETFs, dates, and calculation settings
    |
    v
data/snapshot/                     Downloaded inputs, saved locally
    |
    v
src/financial_economics/           Python code
    |-- config.py                 Read the study settings
    |-- data.py                   Download, save, and validate the inputs
    |-- wrds_source.py            Private CRSP extracts and monthly reconciliation
    |-- return_data.py            Provider-independent returns and complete months
    |-- analysis.py               Calculate returns, risk, and market exposure
    |-- report.py                 Write Phase 1 tables, charts, and the summary
    |-- stress.py                 Find SPY declines and compare recoveries
    |-- stress_report.py          Write the Phase 2 report
    |-- comparison.py             Fit the SPY + cash portfolio and compare
    |-- comparison_report.py      Write the Phase 3 report
    |-- costs.py                  Deduct self-financing ETF trading fees
    |-- fund_checks.py            Compare removals, additions, periods, and costs
    |-- protection_config.py      Frozen protection-study settings
    |-- protection.py             Fixed additions, costs, periods, and beta controls
    |-- protection_inference.py   Thirteen HAC/Holm tests and paired bootstrap
    |-- protection_report.py      Protection tables, three figures, and run metadata
    |-- explain.py                Resample the comparison and split gains by fund
    |-- explain_report.py         Write the Phase 4 report
    |-- page.py                   Write the one-page summary of all four phases
    |-- cli.py                    Connect the commands
    |-- __main__.py               Support python -m financial_economics
    +-- __init__.py                Identify the Python package
    |
    v
outputs/phase1/ ... phase4/        Results, saved locally
    |
    v
outputs/page/index.html            One page with the main charts and numbers
```

`data/` and `outputs/` are created when you run the commands and are ignored by Git.

| Other files | Purpose |
|---|---|
| `tests/` | Check calculations, data validation, and the commands using synthetic examples |
| [docs/results-explained.md](docs/results-explained.md) | What the study has found, in plain language |
| [docs/methodology.md](docs/methodology.md) | Formulas, sources, assumptions, and interpretation |
| [docs/phase-1.md](docs/phase-1.md), [phase-2.md](docs/phase-2.md), [phase-3.md](docs/phase-3.md), [phase-4.md](docs/phase-4.md) | Each part's question, method, outputs, and checks |
| `pyproject.toml` | Python requirements, packages, and the `fecon` command |
| `uv.lock` | Exact package versions |
| `.github/workflows/tests.yml` | Run tests on Windows and Linux in GitHub Actions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaborator access, branches, and pull requests |

To change **what is studied**, edit `study.toml`. To change **how results are calculated**, start in `analysis.py`, `stress.py`, `comparison.py`, or `explain.py`. To change **how results are presented**, start in the matching `*_report.py` file.

## Current settings and assumptions

The study selects SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT for January 1, 2010 through December 31, 2025. The exact settings live in `study.toml`.

- CAGR measures compounded growth; volatility measures total return variability. Sharpe uses daily excess returns.
- The equal-weight portfolio resets monthly, with weights drifting between resets. Phases 1–4 omit costs and taxes; `fund-checks` includes explicit trading-cost scenarios and omits taxes.
- SPY is the portfolio benchmark. CAPM uses French's broader market factor; beta measures exposure to that factor. Phase 3 instead measures sensitivity to SPY's own excess returns.
- TLT is a risky bond fund. The risk-free return, and the "cash" in SPY + cash, come from French's factor file.
- Market falls are dated after the fact, using SPY's own peaks and troughs; they are not a trading rule.
- Phase 3 picks its SPY + cash split from earlier years, but the full dataset and funds had already been examined, so it is a historical check rather than a test on unseen data.
- Phase 4's rebuilt histories reuse the same 2018–2025 days. They show how much the comparison could move by chance, not new evidence; its ranges are percentile intervals rather than a formal significance test.
- These are retrospectively selected surviving funds. Full-period correlations do not establish protection during market declines, and historical alpha does not prove skill or mispricing.

## Development

The NumPy portfolio calculations and block bootstrap are CPU-native algorithms;
they run on CPU and do not select or fall back from a GPU backend.

```sh
uv run pytest -q
```

Tests use synthetic data and require no live finance service. GitHub Actions installs the locked environment and runs the tests on Windows and Linux.

This repository is private. Ask [@N3V3MORE](https://github.com/N3V3MORE) to invite your GitHub username before cloning. Follow [CONTRIBUTING.md](CONTRIBUTING.md) when making changes, and keep downloaded data and generated results out of Git.
