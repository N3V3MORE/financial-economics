# Commands and study reference

Run all commands below from the project root. Configuration, data and output paths
are relative to that root, even though this reference lives in `docs/`.

Every command runs through `fecon`. Staged studies once run as
`python -m financial_economics.<module>` are `fecon` subcommands with the same
arguments. Implementations live in responsibility groups; see
[Project structure](structure.md). No methods, features or studies were removed.

This page preserves the former README's command and reference content, including
its dated results and environment checks. Those records describe their stated
runs; they are not fresh validation of a later checkout or a claim that every
local snapshot is present. Use [the overview](../README.md),
[Final study](final-study.md) and [Handover](../handover.md) for context.

---

# ETF diversification study

Do different ETFs help spread investment risk, especially when markets fall?

Phase 1 builds the baseline: historical returns, risk, correlations, a monthly equal-weight portfolio of the eight funds (the "mix"), and CAPM estimates with uncertainty. Phase 2 compares the mix with SPY in every SPY decline of at least 10%, including how long each took to recover. Phase 3 asks whether the mix did better than simply holding less stock, using a SPY + cash portfolio chosen from 2010–2017 and tested on 2018–2025. Phase 4 checks whether chance alone could explain the Phase 3 differences, and which funds produced the mix's gains and losses in each fall.

For a plain-language account of what the study has found, read [Results explained](results-explained.md).

**Final research goal:** can a stock investor protect against market falls better than by simply holding less stock? The final stage adapts the target-volatility strategy of Kaczmarek et al. (2022) to answer it. Its question, methods, findings and limits are in [Final study](final-study.md).

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

`stress` and `compare` accept settings too, for example `--drawdown-threshold 0.10 --rolling-window 63` or `--evaluation-start 2018-01-01`. See [Phase 2](phase-2.md) and [Phase 3](phase-3.md).

`explain` rebuilds 2018–2025 10,000 times from runs of real trading days, 21 days long on average, keeping every portfolio on the same days, and reports how far the Phase 3 differences move. A fixed random seed makes reruns identical; `--resamples`, `--block-length`, and `--seed` change the settings. It also measures each fund's gain or loss in every SPY fall, adding up exactly to the mix's loss. See [Phase 4](phase-4.md).

## Check which funds help

`fund-checks` removes each fund in turn and redistributes its weight equally
among the remaining funds. It keeps the original mix's SPY + cash comparison
and adds a separately calibrated SPY + cash comparison for each variant.
Each control uses only returns before its evaluation period.

```sh
uv run fecon fund-checks --data-dir data/public-review-20261019 --output-dir outputs/fund-checks-original
uv run fecon --config configs/study-commodities.toml download --data-dir data/commodities-20261020
uv run fecon --config configs/study-commodities.toml fund-checks --additions DBC --data-dir data/commodities-20261020 --output-dir outputs/fund-checks-commodities
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
original eight-fund work below. The frozen settings are in `configs/protection.toml`
(14 funds, 2015–2025) and `configs/protection-long.toml` (SPY/TLT/GLD/DBC, 2007–2025).
Read the measured findings in [results-explained.md](results-explained.md#which-additions-earned-their-place-relative-to-cash).

The October 20, 2026 source decision was **WRDS / CRSP CIZ for both samples**.
All 1,848 main-sample and 912 longer-sample fund-months passed the daily/monthly
reconciliation threshold of one basis point; maximum differences were 0.0555
and 0.0433 basis points. Complete French trading calendars, historical PERMNO
coverage, ETF classifications, valid return flags, and duplicates were checked.
No provider splice, missing-return filling, or public-data fallback was needed.

```sh
uv sync --extra wrds
uv run --extra wrds fecon --config configs/protection.toml download --source wrds --verify-monthly --data-dir data/protection-wrds-20261020
uv run --extra wrds fecon --config configs/protection-long.toml download --source wrds --verify-monthly --data-dir data/protection-long-wrds-20261020
uv run fecon --config configs/protection.toml protection --data-dir data/protection-wrds-20261020 --output-dir outputs/protection-main-final
uv run fecon --config configs/protection-long.toml protection --data-dir data/protection-long-wrds-20261020 --output-dir outputs/protection-long-final
uv run python scripts/verify_protection.py --config configs/protection.toml --data-dir data/protection-wrds-20261020 --output-dir outputs/protection-main-final
uv run python scripts/verify_protection.py --config configs/protection-long.toml --data-dir data/protection-long-wrds-20261020 --output-dir outputs/protection-long-final
```

Skip downloads when those snapshots already exist. Downloads and protection
outputs refuse existing destinations; choose new names for a later vintage or
rerun. On Windows, save the WRDS username and password through the local
Windows sign-in dialog once:

```powershell
uv run --extra wrds fecon wrds-credentials
```

This stores a generic credential named `financial-economics:wrds` in Windows
Credential Manager for the current user on this computer. Subsequent WRDS
downloads read it directly into memory. Running the command again updates
the saved login. Passwords are never written to the repository or printed.
When no Windows credential exists, the existing PostgreSQL password file
and optional `WRDS_USERNAME` are used. If `WRDS_USERNAME` is set, it must
match the saved Windows login. Provider errors are sanitized.

[WRDS PostgreSQL access requires Duo Mobile push approval](https://wrds-www.wharton.upenn.edu/pages/about/log-in-to-wrds-using-two-factor-authentication/).
Connections allow up to two minutes for authentication and are not retried
automatically. A saved password does not bypass MFA.
There is no automatic data-provider fallback. If verified WRDS coverage is unavailable, explicitly
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

`configs/prediction.toml` extends the retained 2007–2025 SPY/TLT/GLD/DBC snapshot.
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
after validation. All settings are explicit in `configs/prediction.toml`.

```sh
uv run fecon --config configs/prediction.toml predict --ewma-only --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-ewma
uv sync --frozen --extra prediction
uv run --extra prediction fecon --config configs/prediction.toml predict --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-core
uv run fecon --config configs/prediction.toml predict --forecast-run outputs/prediction-core --defensive-asset GLD --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-gold
uv run fecon --config configs/prediction.toml predict --forecast-run outputs/prediction-core --defensive-asset DBC --data-dir data/protection-long-wrds-20261020 --output-dir outputs/prediction-commodities
```

Skip downloads when the retained snapshot exists. Default `predict` trains
on **cuda:0** through NVIDIA CUDA or AMD ROCm, verifies effective device
placement, and stops if a compatible GPU is unavailable; it never retries
training on CPU. PyTorch uses the same `cuda` device API for ROCm.
`--ewma-only` and verified
`--forecast-run` reuse run without importing PyTorch. They are mutually
exclusive. Saved forecasts must match settings, snapshot provenance, and
file hashes. A saved baseline stays a baseline. Portfolio accounting and
bootstrap calculations are CPU-native; substantial runs stay sequential.

For Windows x64 with Radeon 8060S (`gfx1151`), use the separate AMD extra:

```powershell
uv sync --frozen --extra prediction-rocm --extra reconstruction --extra wrds
uv run --frozen --extra prediction-rocm --extra reconstruction --extra wrds python -m pytest -q -W error --basetemp=.venv/pytest-amd-check
```

`prediction-rocm` pins PyTorch 2.11.0 + ROCm 7.14.1 and the matching device
runtime from AMD's official package index. The NVIDIA `prediction` and AMD
`prediction-rocm` extras are mutually exclusive. For AMD, replace
`--extra prediction` with `--extra prediction-rocm` in the training commands
below and retain the other extras needed by that command. This extra targets
Windows `gfx1151`; other AMD devices need their matching runtime packages.
See [AMD's compatibility matrix](https://rocm.docs.amd.com/en/docs-7.14.1/compatibility/compatibility-matrix.html)
and [PyTorch's HIP device semantics](https://docs.pytorch.org/docs/2.11/notes/hip.html).

An earlier October 21 local validation passed 335 tests, including seven GPU
tests, with warnings treated as errors. Separate model probes verified
forward/backward computation, Adam updates, exact same-seed repetition and
checkpoint reload on Radeon 8060S. ROCm GRUs train with MIOpen, about twice
as fast as native HIP. On Windows, MIOpen compiles its RNN kernels with the
C++ headers from Visual Studio 2022 Build Tools (C++ workload and Windows 11
SDK). Without them it fails with `'type_traits' file not found`; set
`FINANCIAL_ECONOMICS_NATIVE_RNN=1` to train with native HIP operations
instead. Run and checkpoint metadata record the actual runtime and HIP
version. Future runs also record NVIDIA cuDNN and AMD MIOpen availability,
enabled flag and version separately from observed training-forward
dispatch. Inference dispatch is not observed; unrecognized training evidence
is labelled unknown. Native GPU paths are identified separately, and
archived runs gain no invented metadata.
Deterministic training does not imply identical forecasts across different
GPU runtimes. GPU training runs in full float32 precision: NVIDIA's TF32
mode is off for cuDNN and matrix products, and run metadata records both
flags. Saved reconstruction GRU stages made before October 22, 2026 used
TF32 on NVIDIA and need a fresh run. To check how closely two GPUs agree,
run `scripts/compare_gpu_backends.py compare scripts/gpu-backend-reference/miopen`
on the second machine.

The current suite passes 392 tests with warnings treated as errors, including
the matched-model GPU checks on NVIDIA CUDA. The earlier 335-test result above is
historical; GPU tests skip when no compatible CUDA or ROCm device is available.

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

## Closer replication of the paper's simpler GRU

`configs/paper.toml` and `fecon paper` keep the existing `predict` experiment intact.
This is a **partial GRU-UNIVAR replication with ETF substitutes**, not a claim
to reproduce the paper's multivariate model or all thirteen assets.

Use the [archived Oxford-Man ZIP](https://web.archive.org/web/20220301022212id_/https://realized.oxford-man.ox.ac.uk/images/oxfordmanrealizedvolatilityindices.zip)
as `--oxford-archive`. The recovered SPX `bv` series has exactly 5,265
observations for 2000–2020. `22 * sqrt(bv)` gives mean 0.169511864 and sample
standard deviation 0.124568826, matching Table 1 of the
[original paper](https://doi.org/10.1016/j.ribaf.2021.101610).
This supports that interpretation of its unusual monthly scaling; it is
different from conventional square-root-of-time scaling and our 10% annual target.

The new model predicts next-session daily volatility from 22 previous observed
daily volatilities, each calculated as `sqrt(bv)`. This is our interpretation
of the paper's forecast units, supported by the Table 1 match. The model
uses two stacked GRU layers, MSE, dropout/L2 regularization, five expanding
chronological validation folds inside 2000–2009, and three fixed seeds.
Only two hidden sizes are compared. Validation freezes the training duration
before annual expanding refits for 2010–2020. This explicit small search is
our choice because the authors' exact tuning settings/code are unavailable.

```sh
uv run fecon --config configs/paper.toml paper --baseline-only --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --output-dir outputs/paper-baseline
uv run --extra prediction fecon --config configs/paper.toml paper --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --output-dir outputs/paper-gru
```

Training requires verified `cuda:0` through CUDA or ROCm; no CPU fallback.
This small GRU is built as two single-layer GRUs with dropout between them,
which equals one two-layer GRU with the same starting weights. cuDNN's own
two-layer dropout crashes Windows processes at shutdown, so this shape lets
NVIDIA GPUs use cuDNN (about five times faster; forecasts differ slightly
from the October 2026 native-kernel runs). ROCm uses MIOpen in the same
way. The EWMA baseline,
calendar preparation, and portfolio accounting are CPU-native. SPY, TLT,
and GLD total returns substitute for the paper's Datastream indices. Each
model runs a stock/cash portfolio and nine threshold/cap combinations per
defensive ETF: quantiles 90/95/99%, caps 10/20/30%, cash income zero, fees zero.
EWMA is an additional baseline, not the paper's ARFIMA/GARCH comparator.

To follow the paper closely, stress thresholds use the **full test-period**
forecast distribution. These thresholds are retrospective. Next-day returns
also assume idealized execution at the forecast issue close; this is not an
implementable trading-performance claim. Nineteen source BV dates are absent
over 2000–2020, including nine evaluation dates. Missing BV targets are never
filled or scored. Histories use the last 22 observed BVs; when a close has no
BV observation, no new allocation is issued and holdings drift. Every ETF
return day remains in the portfolio calculation.

New output directories contain CSV results, saved annual checkpoints, and
hashed `run.json` provenance. `paper_comparison.csv` places published
GRU-UNIVAR MSE/MAPE and clean Sharpe/Calmar beside this implementation;
`protection_by_quantile.csv` averages the three caps within each threshold.
The original scores use transformed monthly volatility; our training loss
differs only by the constant scaling factor. Differences in data, model selection, and execution limit any direct
numerical comparison. No statistical superiority claim or new report is generated.

## Extend the original volatility measure through 2021

`configs/paper-extension.toml` keeps Oxford SPX bipower variation and adds realistic
trading assumptions. Evaluation covers 2010–2021; outputs also separate
2010–2020, 2021, and the earlier experiment's overlapping 2017–2021 dates.
The archive ends in February 2022, so this version makes no full 2022–2025 claim.

```sh
uv run fecon --config configs/paper-extension.toml paper-extension --baseline-only --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --output-dir outputs/paper-extension-baseline
uv run --extra prediction fecon --config configs/paper-extension.toml paper-extension --selection-run outputs/paper-gru-20261020 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --output-dir outputs/paper-extension
```

`--selection-run` verifies the saved pre-2010 tuning evidence, then reuses its
32 hidden units and 17 epochs with three seeds. It performs no new search.
New annual models learn the two-session target: issue at Monday close, trade
Tuesday close, earn Wednesday's close-to-close return. Training stays on `cuda:0`
with native CUDA or HIP kernels. A reconstructed 2009 forecast history calibrates
the initial threshold; its fitted weights use pre-2009 labels, while the reused
architecture was selected through 2009. That calibration is known before
2010 evaluation and is not presented as a traded 2009 test. The first 2010
evaluation session stays in cash. Later annual thresholds use forecasts
issued strictly before the preceding annual close, with no test-period quantiles.

The main comparison retains the paper's scaling, earns observed cash interest,
and charges 10 bps per ETF dollar bought or sold. Fixed sensitivities use
0/25 bps, zero cash income at 10 bps, and the earlier 10% annual risk target.
Each model runs stock/cash plus separate TLT, GLD, and DBC variants at one
90th-percentile stress threshold and a 20% cap funded from available cash.
Entry costs are included; exit and taxes are excluded. Missing signals leave
holdings to drift; every ETF return date remains present.

Use `--forecast-run` instead of `--selection-run` to reuse a verified extension
run without importing PyTorch or retraining. Saved model settings, source
hashes, and forecast bytes must agree. The CSV outputs include period metrics,
turnover/fees, monthly returns, and defensive-asset comparisons in negative-SPY
months. `benchmark_performance.csv` gives SPY's gross buy-and-hold comparison.
Subperiod fee totals retain the original full-run capital denominator.
These are descriptive historical results; 2021 is an additional
historical year, not untouched evidence. The original `paper` command and
archived replication results remain available separately.

To compare daily and weekly SPY/cash trading with the same saved forecasts,
run each schedule into a new directory:

```sh
uv run fecon --config configs/paper-extension.toml paper-extension --forecast-run outputs/paper-extension-gru-20261020 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --cash-only --rebalance-frequency daily --output-dir outputs/paper-frequency-daily-20261021
uv run fecon --config configs/paper-extension.toml paper-extension --forecast-run outputs/paper-extension-gru-20261020 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --cash-only --rebalance-frequency weekly --output-dir outputs/paper-frequency-weekly-20261021
```

Daily remains the default; `[extension].rebalance_frequency` can also set it.
Weekly uses the first eligible execution close of each Monday–Sunday week,
then earns returns from the following session. If a signal is unavailable,
the next eligible close is used; weeks without signals retain holdings.
Holdings drift between trades, every daily return remains present, and costs
apply only when trading. Forecasts, cash interest, risk rules, cost scenarios
and evaluation dates stay the same. `decisions.csv` flags selected signals;
`target_weights.csv` carries the last selected target, not drifting holdings.
`run.json` records the schedule and source forecast hash. These commands use
CPU-native portfolio accounting and do not import PyTorch or retrain models.

To skip small target changes, use the fixed five-percentage-point experiment:

```sh
uv run fecon --config configs/paper-extension.toml paper-extension --forecast-run outputs/paper-extension-gru-20261020 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --data-dir data/protection-long-wrds-20261020 --cash-only --rebalance-frequency daily --min-allocation-change 0.05 --output-dir outputs/paper-allocation-change-5pp-20261021
```

After the first eligible entry, trade to the full new stock target only when
it differs from the last accepted target by at least five percentage points.
For example, 50% to 53% waits; a later 56% trades. Holdings drift does not
trigger trades. Decisions are separate for each model and risk rule and
continue across years. The same decisions apply to every fee/cash scenario.
`[extension].min_allocation_change` can also set the fractional threshold;
its default of zero preserves daily/weekly results. Positive values require
SPY/cash only. Combined with weekly, the filter checks only the first eligible
execution date each week. `decisions.csv` separates cadence candidates
(`scheduled`) from accepted resets (`rebalance`); metadata reports accepted
counts per strategy. The common `rebalance_signal_days` count is null when
filtering is enabled. Five points is an illustrative fixed choice, not an
optimized threshold; these previously examined years are not fresh validation.

The research settings are frozen at local tag `frozen-gru-study-20261021`.
The baseline is the explicit daily, zero-filter rule in `configs/paper-extension.toml`:
90th-percentile past-only stress threshold, 20% defensive cap and the existing
risk/cost/cash scenarios. Reuse `outputs/paper-extension-gru-20261020`, with
two GRU layers, 32 hidden units, 17 epochs and seeds 11/22/33. Its forecast
SHA-256 is `d59ed175760713b1ae7077df3f38cc894cab32bdbd46b56a67a5f765596aaac4`.
The saved weekly and five-point runs remain fixed sensitivity comparisons.
The final safe-haven check uses the existing `2010:2020` window, preserving
the model configuration's 2021 endpoint and all original forecasts.

## Reconstruct the paper's U.S. experiment

`configs/paper-reconstruction.toml` fixes the five published model families: ARFIMA,
GARCH, GJR-GARCH, GRU-UNIVAR and GRU-MULTIVAR. The separate reconstruction
uses 22-day histories, pre-2010 neural selection, annual GRU refits, daily
classical refits and the 2010–2020 test period. Each model has a stock/cash
strategy and 117 additions: 13 defensive assets, three stress thresholds and
three caps. Fees and cash interest are zero, as in the paper. Thresholds use
the full test period and therefore describe a retrospective comparison.

The GRUs share evaluation dates, but their training histories differ. The
one-input model uses BV history from 2000; the richer feature table starts
on April 11, 2002, with its first usable issue after the 22-observation
lookback. Their pre-2010 validation folds therefore cover different dates.
On NVIDIA GPUs these GRUs train with cuDNN, about five times faster than
native kernels; repeated runs match each other, but forecasts differ slightly
from the October 21, 2026 run, which used native kernels. Their two
single-layer GRUs keep dropout outside cuDNN, avoiding the shutdown crash
described above. ROCm trains them with MIOpen, about twice as fast as
native HIP; runs before October 22, 2026 used native HIP.
These results compare complete model configurations; isolating the value
of extra inputs requires a separate study with matched training histories
and validation folds. The existing settings and forecasts are preserved.

The original paper and appendix are available in the
[author's university dissertation](https://bip.ue.poznan.pl/download/attachment/1848/rozprawa-doktorska-mgr-t-kaczmarka.pdf#page=180).
The licensed Datastream inputs are unavailable to this account. The new
snapshot uses S&P price/total-return indices, public yield series, commodity
price proxies and raw bond-ETF prices. Its manifest records every substitution,
including Brent for WTI, AAA for AA three-year corporate yields and assumed
USD conversion of Swiss/German bonds. A separate Federal Reserve H.10 snapshot
supplies dated noon FX for those two valuations; its weekly publication makes
it unsuitable as an assumed same-day released forecasting feature. Futures price changes are not investable
futures total returns. International robustness is not reconstructed.

Run the stages sequentially in PowerShell. Use new snapshot/output directories
for a new data vintage; retain the existing Oxford archive described above.

```powershell
uv sync --frozen --extra prediction --extra reconstruction
uv run --extra prediction --extra reconstruction python -c "from financial_economics.datasets.sources.paper import download_sources; download_sources('data/paper-public-new')"
uv run --extra prediction --extra reconstruction python -c "from financial_economics.datasets.sources.fx import download_fx; download_fx('data/paper-fx-new')"
$reconstructionArgs = @('--data-dir', 'data/paper-public-new', '--fx-dir', 'data/paper-fx-new', '--oxford-archive', 'data/paper-source-20261020/oxfordman-20220301.zip', '--output-dir', 'outputs/paper-public-new')
foreach ($stage in @('classical', 'gru_univar', 'gru_multivar', 'evaluate')) {
    uv run --extra prediction --extra reconstruction fecon paper-reconstruction $stage @reconstructionArgs
    if ($LASTEXITCODE -ne 0) { break }
}
```

When FRED is unavailable, the same requested yield series can be retained
from the licensed WRDS `frb_all.rates_daily` table, with Brent spot prices
downloaded directly from EIA. Use a new macro snapshot and a new combined
source directory:

```powershell
uv run --frozen --extra prediction-rocm --extra reconstruction --extra wrds fecon paper-macro-sources --output-dir data/paper-macro-wrds-eia-20261021
uv run --frozen --extra prediction-rocm --extra reconstruction --extra wrds python -c "from financial_economics.datasets.sources.paper import download_sources; download_sources('data/paper-wrds-eia-20261021', macro_dir='data/paper-macro-wrds-eia-20261021')"
```

Pass the combined directory as `--data-dir` to every reconstruction stage.
Skip these download commands when those immutable directories already exist.
This retains WRDS DGS3/DAAA and EIA RBRTE as the counterparts of FRED
DGS3/DAAA/DCOILBRENTEU, with their actual provider identities and units in
the manifest. Raw responses, the WRDS query, and the upstream macro manifest
are copied and hashed. WRDS observations remain private and local.
The current FRED vintage cannot be compared during its outage; this is a
fresh source vintage, with the study's economic series definitions unchanged.

Classical fits run on CPU. Both neural stages require `cuda:0` through CUDA or ROCm and never fall
back to CPU. Evaluation reuses verified forecasts and does not retrain.
Each stage rejects existing output directories. Saved results include the
full strategy grid, forecasts, tuning, checkpoints, source/code hashes,
forecast comparisons, five decline windows and rounded published references.
Cache reuse verifies model inputs, settings, files and forecasting code. The
execution wrapper's original hash is retained separately when a reader-only
compatibility change permits reuse; calculation and valuation code is recorded
with the evaluation.

The paper omits its exact neural grid, seeds and several metric conventions;
the fixed replacements and estimator choices are recorded in `run.json`.
Results include conventional Sharpe and a separately labelled, inferred
CAGR/volatility calculation. Modified decline scores also show both arithmetic
and inferred geometric conventions. Do not treat either inference as a
confirmed published formula. Calendar dates alone do not verify that every
same-date external quote was available before the U.S. close.
PyTorch uses reset-after GRU gates; the paper's displayed equation uses
reset-before, while its executed Keras/TensorFlow settings are unpublished.

### Revised matched study after the methodology audit

`fecon paper-revised` corrects the practical limitations of the
historical reconstruction while retaining its original files and outputs.
The saved October 21 revision used identical 22-observation histories,
beginning October 3, 2000. Current runs again use the last 22 trading
observations, following the requested rollback of the calendar-day experiment.
The annual conversion remains sqrt(252), and forecasts still target t+2.
The four external features lag
one market session. Training, stopping and selection dates are identical;
early stopping and configuration scoring use separate, purged blocks.
Log-BV predictions are positive, with training-only smearing.

All six models (EWMA, ARFIMA, GARCH, GJR and both GRUs) forecast the session
actually earned: issue at close t, trade at close t+1, earn the t+2 return.
GARCH/GJR use analytic two-step variances; ARFIMA forecasts recursively;
EWMA holds its conditional variance level without future observations.

Since October 22, 2026 the volatility target is Risk Lab SPY 5-minute trade
realized variance (`trade_rv5`² / 252) from `--risklab-dir`; the archived
October 21 revision used Oxford-Man SPX bipower variation. A forecast is
issued only when its 22-session history is complete; dates Risk Lab lacks are
never filled, and portfolios hold their last target meanwhile. ARFIMA restarts
its fractional filter at every gap, so no lag spans a missing session. EWMA is
a truncated normalized average: the last 22 daily variances weighted 0.94^k
and divided by the weight sum, not the recursive RiskMetrics filter.
Forecast accuracy is scored on intraday and close-to-close targets with
variance MSE and QLIKE (robust to a noisy variance proxy) and volatility MSE
for reference. GARCH/GJR forecast close-to-close variance and the others
intraday variance, so each target rescales the other kind by a Hansen–Lunde
factor estimated on pre-2010 data only.
The equity risk budget is 10% annually, using sqrt(252), with past-only
stress cutoffs, observed cash income and verified ETF total returns.
Defensive additions retain the stock target to isolate their contribution;
the budget is not a guarantee about the complete portfolio's realized risk.

Costs are fixed at 0, 1, 2, 5, 10 and 25 bps per ETF dollar traded; 10 bps is
the main scenario. Every strategy, its static control and the benchmark start
in cash on the first evaluation day. The benchmark pays entry and final sale
costs too. Final liquidation uses actual drifted holdings. Negative-market
months are defined by gross stock returns, consistently across costs.
All 27 portfolio comparisons enter one Holm family and remain exploratory.

```powershell
$revisedArgs = @('--data-dir', 'data/paper-wrds-eia-20261021', '--returns-dir', 'data/protection-long-wrds-20261021', '--risklab-dir', 'data/risklab-spy-20261022', '--output-dir', 'outputs/revised-study-new')
foreach ($stage in @('freeze', 'classical', 'gru_univar_log', 'gru_multivar_log', 'evaluate')) {
    uv run --extra prediction --extra reconstruction fecon paper-revised $stage @revisedArgs
    if ($LASTEXITCODE -ne 0) { break }
}
```

Freeze writes source/config/code hashes before fitting. Each later stage
rejects changed inputs or calculation code, existing outputs, mixed protocols,
different target dates or different GRU fold dates. Ordinary GPU requirements
apply; classical fits and accounting are CPU-native and stages run sequentially.
See `outputs/revised-study-20261021/protocol.json` for the executed revision.
The archived reconstruction configurations retain their previous behavior.

The saved 22-observation revision covers every one of the 2,517 market days in 2011–2020.
At 10 bps, stock/cash Sharpe is 0.843 for EWMA, 0.827 for the one-input GRU
and 0.841 for the richer GRU. At zero fees those figures are 0.861, 0.885
and 0.892. The GRUs have lower measured volatility forecast error, but no
portfolio comparison survives the joint 27-test adjustment. Full results
and their limits are in `docs/final-study.md`; saved tables are under
`outputs/revised-study-20261021/evaluate`.

Remaining limits are explicit: the selected source inputs end in 2020;
historical macro vintages and exact publication times are unavailable;
intraday BV omits overnight risk; costs are assumptions; taxes are excluded.
The same history has been inspected repeatedly, so freezing this revision
does not turn it into an untouched confirmatory test. A current-data extension
requires a validated consistent volatility source, not an undocumented splice.

### Estimated Oxford-like measurements and later-year test

`fecon volatility-proxy` estimates same-day Oxford-like volatility
from public [Risk Lab](https://dachxiu.chicagobooth.edu/) intraday measures.
The current comparison adds IVV and VOO to SPY, flags gross trade/quote
disagreements and tests 78 mappings, including constrained combinations.
Coefficients use 2000–2013, and 2014–2016 selects the method before scoring
2017–February 25, 2022. These dates have already been inspected; successive
revisions are exploratory. The wider inputs and longer calibration history
change together, so this does not isolate the effect of adding another ETF.

The requested acceptance rule is now **at least 90% average accuracy and 85%
on every individual day**, with complete coverage of Oxford comparison dates,
including history offered for training. Accuracy means one minus absolute
percentage error in volatility levels. It does not measure future prediction
accuracy, raw-variance accuracy or the percentage of correct days.

| 2017–February 25, 2022; 1,285 days | Original substitute | Revised substitute |
|---|---:|---:|
| Average volatility accuracy | 93.3% | 93.7% |
| Days reaching 85% accuracy | 94.2% | 94.1% |
| Worst individual accuracy | 35.0% | 32.0% |
| February 2018 crash-week average error | 32.4% | 7.5% |

The revised comparison is saved in `outputs/volatility-proxy-final-20261022`.
It improves average error from 6.69% to 6.35%, but **fails the daily minimum**:
76 later-period dates fall below 85%, and some earlier errors exceed 100%.
All 5,552 Oxford dates remain in the comparison; failures are saved in
`failures.csv`. Raw BV average error in the later period is 12.8%.
Daily-price fallbacks remain explicitly flagged. The full diagnostic export
has 7,743 dates through October 9, 2026, including all 1,003 dates in 2022–2025.

**This substitute is not approved for a new training/backtest run.** The proxy
command returns a failure status when either accuracy requirement fails, and
`fecon proxy-study` rejects both this result and legacy runs lacking the new daily
check. Passing an older average-only gate is insufficient.

```powershell
uv run fecon volatility-proxy --data-dir data/volatility-proxy-20261022 --prices-dir data/volatility-proxy-prices-20261022 --ivv-dir data/volatility-proxy-ivv-20261022 --voo-dir data/volatility-proxy-voo-20261022 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --output-dir outputs/volatility-proxy-new
if ($LASTEXITCODE -eq 0) {
    $proxyArgs = @('--proxy-run', 'outputs/volatility-proxy-new', '--prices-dir', 'data/volatility-proxy-prices-20261022', '--returns-dir', 'data/protection-long-wrds-20261020', '--output-dir', 'outputs/proxy-study-new')
    foreach ($stage in @('freeze', 'forecast', 'evaluate')) {
        uv run --extra prediction --extra reconstruction fecon proxy-study $stage @proxyArgs
        if ($LASTEXITCODE -ne 0) { break }
    }
}
```

The archived later-year pilot used the original substitute, one log-BV GRU
and EWMA, 2021 calibration and
2022–2025 evaluation on verified WRDS/CRSP ETF total returns. The full
volatility history is estimated consistently; Oxford labels are not spliced
onto later proxy labels. The existing 32/32 architecture has no new width
search; pre-2021 folds select stopping epochs capped at 12, using seeds
11/22/33. It used 30-calendar-day histories, t+2 forecasts, 10% annual equity
risk, cash interest, entry/exit costs and the six fixed fee scenarios. Current
code has reverted to 22 trading observations; this pilot has not been rerun
with 22 because the substitute fails the stricter quality requirement.
The three Sharpe comparisons share one Holm adjustment. These are historical
research results: exact publication times and the ability to obtain these
measurements before trading remain unverified. The richer GRU is not included
because the retained external-input snapshot ends in 2020.

The completed pilot retains all 1,003 market days in 2022–2025. At 10 bps,
EWMA/GRU cash Sharpe is 0.335/0.340, versus about 0.433 for their static
controls. Before fees the timing figures are 0.358/0.420. Maximum drawdown
is 20.59%/21.27%, versus 23.17%/23.18% for the static controls. None of the
three comparisons survives Holm adjustment. These results are saved under
`outputs/proxy-study-20261022`; they do not establish that the GRU improves
investment performance. Five actual NVIDIA/cuDNN annual refits use 10 epochs
selected from pre-2021 stopping data. The 29 stage files/checkpoints match
their recorded hashes.

### Fixed GRUs and the realistic trading test

The raw-BV richer GRU predicted negative variance on 5.5% of days, almost all
in the very calm years 2017–2018. Two extra stages learn **log BV** instead, so
forecasts are always positive. Each keeps its raw counterpart's profiles,
seeds, fold dates and epoch rule, selects on log-scale pre-2010 MSE, and
restores the mean level with a training-only smearing factor. The fix was
added after the raw results were seen; it is labelled post hoc, and the raw
runs stay unchanged and reported.
`evaluate_fixed` scores all seven models paper-style in a separate directory.

```powershell
foreach ($stage in @('gru_univar_log', 'gru_multivar_log', 'evaluate_fixed')) {
    uv run --extra prediction --extra reconstruction fecon paper-reconstruction $stage @reconstructionArgs
    if ($LASTEXITCODE -ne 0) { break }
}
uv run fecon paper-realistic --data-dir data/protection-long-wrds-20261020 --forecast-run outputs/paper-public-new --output-dir outputs/paper-realistic-new
```

`configs/paper-realistic.toml` was fixed before the realistic run. It reuses the saved
forecasts with no refitting, on verified WRDS/CRSP SPY, TLT, GLD and DBC total
returns with French daily cash income. Reuse requires every selected forecast
stage to record matching, valid SHA-256 hashes for its source manifest,
Oxford archive and reconstruction config, in addition to verified forecast
bytes. The realistic run retains those input identities in its provenance.
A forecast issued at close t trades at close t+1 and earns the t+2 return;
fees are 10 bps per ETF dollar traded
(0 and 25 bps are sensitivities). Each year's 90th-percentile stress cutoff uses
only that model's earlier forecasts, so 2010 is calibration and trading is
scored over 2011–2020. A static SPY/cash control holds the model's mean 2010
stock target, reset monthly. This is a comparison with a static allocation
chosen before evaluation; it is not calibrated to match the timing strategy's
average 2011–2020 stock exposure or realized risk. Differences can therefore
reflect exposure as well as timing. The primary family has six tests for
ARFIMA and the simple GRU: timing minus static Sharpe (Ledoit–Wolf HAC),
and TLT and GLD minus cash in negative-SPY months (HAC), with one Holm
adjustment. Secondary models, DBC, subperiods, declines and bootstrap
intervals are descriptive.

The primary Holm adjustment retains all six prespecified tests when an
individual test is unavailable: it uses p=1 internally for that test and
keeps its reported p-value and adjusted p-value unavailable, with no
rejection. Sharpe HAC inference uses centered calculations and marks
constant excess return series unavailable. Identical or positively
proportional excess return series retain their Sharpe difference estimate
but have unavailable uncertainty and p-values when the sampling variance
is degenerate. If there are no months when the benchmark falls, the
conditional return difference estimate and test are unavailable.

On October 21, 2026 every stage was rerun from the saved data snapshots with
these inference fixes and cuDNN-trained GRUs (`outputs/rerun-20261021`). No
headline figure or Holm decision changed, so the documented findings stand.
A separate rebuild from fresh Yahoo/WRDS/EIA inputs on an AMD machine was
stopped on October 21 (`outputs/paper-wrds-eia-rocm-20261021`, with
`status.json` and `logs/` on that machine); that directory is absent on this
machine. The stopped run is not used for the reported results.

To compare trading schedules on these same saved forecasts, use separate runs:

```powershell
foreach ($frequency in @('daily', 'weekly', 'monthly')) {
    uv run fecon paper-realistic --data-dir data/protection-long-wrds-20261020 --forecast-run outputs/paper-reconstruction-20261021 --output-dir "outputs/paper-cadence-new-$frequency" --rebalance-frequency $frequency --skip-bootstrap
    if ($LASTEXITCODE -ne 0) { break }
}
```

Weekly and monthly schedules trade at the first eligible execution close of
each Monday–Sunday week or calendar month. The initial entry can belong to
the preceding period; it is retained. Missing signals postpone a trade to the
next eligible close, and periods without signals retain existing holdings.
All daily returns remain present and holdings drift between trades. The
2010 static-control calibration, annual stress cutoffs, monthly static
rebalancing, fees and cash income remain unchanged. `decisions.csv` marks
selected signals with `scheduled`; `run.json` records the effective schedule.

These are retrospective trading-frequency comparisons, not month-ahead
volatility forecasts. Weekly/monthly test results are descriptive and have
no primary significance family. The existing daily primary tests are retained.
`--skip-bootstrap` omits the optional descriptive bootstrap intervals and
records that omission; the default command still produces them. All schedule
comparisons run sequentially on CPU without refitting or importing PyTorch.

### Redraw the paper's figures

`fecon paper-figures` redraws the paper's Figures 3–7 from a saved reconstruction
run. It refits nothing, checks that the data snapshots and saved files are the
ones that run used, and writes each PNG beside the `figure*.csv` it is drawn from:

```powershell
uv run fecon paper-figures --run-dir outputs/paper-reconstruction-20261021 --data-dir data/paper-reconstruction-public-20261021 --fx-dir data/paper-reconstruction-fx-20261021 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --output-dir outputs/paper-figures-new
```

`figure6_paper_check.png` puts each proxy's correlation with the S&P 500 and
its daily volatility beside the paper's printed values (its Figure 6, to one
decimal, and Table A.2), so you can see which substitutes behave differently.
Figures 4 and 7 show raw GRU-MULTIVAR under the paper's idealized rules;
Figure 7 uses the paper's 90th-percentile, 30%-cap scenario.

### Post hoc checks after the figure comparison

Both checks were added after `figure6_paper_check.png` was seen. They are
labelled post hoc, add nothing to any primary test, and leave the original runs
unchanged beside them.

**Swiss and German bonds in local currency.** The paper's printed volatilities
and correlations for these two bonds match CHF/EUR returns, not USD-converted
ones. `evaluate_local` scores the five paper models again with only those two
proxies in local currency; it needs no FX snapshot. Add `--local-bonds` (and
drop `--fx-dir`) to draw the figures from it.

```powershell
uv run fecon paper-reconstruction evaluate_local --data-dir data/paper-reconstruction-public-20261021 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --output-dir outputs/paper-reconstruction-20261021
uv run fecon paper-figures --local-bonds --run-dir outputs/paper-reconstruction-20261021 --data-dir data/paper-reconstruction-public-20261021 --oxford-archive data/paper-source-20261020/oxfordman-20220301.zip --output-dir outputs/paper-figures-local-new
```

**IEF in the realistic test.** IEF, the 7–10 year Treasury ETF, was the only
proxy that helped all five models in the paper-style rebuild, but the frozen
realistic test held only TLT, GLD and DBC. `configs/paper-realistic-ief.toml` copies
`configs/paper-realistic.toml` and adds IEF as a descriptive secondary asset. Its WRDS
snapshot reproduces the existing SPY, TLT, GLD and DBC returns exactly, and
every existing portfolio and test in the new run matches the frozen run.

```powershell
uv run --extra wrds fecon --config configs/paper-realistic-ief.toml download --source wrds --verify-monthly --data-dir data/protection-long-ief-wrds-20261021
uv run fecon paper-realistic --config configs/paper-realistic-ief.toml --data-dir data/protection-long-ief-wrds-20261021 --forecast-run outputs/paper-reconstruction-20261021 --output-dir outputs/paper-realistic-ief-20261021
```

## See the original phase results

After the four analysis commands, put the main findings on one page:

```sh
uv run fecon page
```

Open `outputs/page/index.html` in a browser. It is a single file with no outside links, so you can also send it to someone. Each chart sits next to a table of the exact numbers, and hovering over a growth chart shows the values for that week. The page reads only the saved outputs. It stops if they came from different snapshots or settings, or if Phases 3 and 4 used different test periods or SPY + cash splits.

## Find your way around

```text
configs/study.toml                 Your ETFs, dates, and calculation settings
    |
    v
data/snapshot/                     Downloaded inputs, saved locally
    |
    v
src/financial_economics/           Python code
    |-- settings/                 Read study and frozen experiment settings
    |-- datasets/                 Download, save, and validate inputs: public and CRSP
    |                             snapshots, returns, WRDS login, paper sources, FX,
    |                             WRDS/EIA macro series, Risk Lab and proxy volatility
    |-- models/                   EWMA and classical forecasts, daily and paper GRUs
    |-- portfolio/                Fees, SPY/cash controls, declines, fund checks,
    |                             target-volatility portfolios
    |-- evaluation/               Metrics, attribution, paired tests, paper-style scores
    |-- studies/                  Paper, extension, rebuild, realistic, revised,
    |                             proxy, Risk Lab and protection studies
    |-- reporting/                Phase reports, prediction and protection outputs,
    |                             paper figures, and the one-page summary
    |
    |-- provenance.py             File hashes, git state, and timestamps for run records
    |-- _layout.py                Map provenance file names to grouped modules
    |-- cli.py                    Connect the fecon commands
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

The rebuild reuses saved forecasts only while `paper_sources.py`, `paper.py`,
`forecasting.py`, `paper_models.py`, `paper_networks.py`,
`paper_networks_log.py`, `paper_gru.py` and `gru_forecasting.py` keep the hashes
recorded with those forecasts. These are provenance names for files now in
`datasets/`, `studies/` and `models/`; [Project structure](structure.md) maps each. Editing any of them means retraining the GPU
stages, so they are kept as frozen and excluded from style-only lint rules.

| Other files | Purpose |
|---|---|
| `tests/` | Check calculations, data validation, and the commands using synthetic examples |
| `scripts/` | The independent protection verifier, the supplement workbook refresher and the GPU backend comparison |
| `supplement/` | The supplemental Excel workbook and the one-month T-bill cash illustration (R Markdown, its CSV, and a static HTML summary) |
| `configs/` | Frozen settings for each study; each command reads one through `--config` |
| [docs/results-explained.md](results-explained.md) | What the study has found, in plain language |
| [docs/methodology.md](methodology.md) | Formulas, sources, assumptions, and interpretation |
| [docs/phase-1.md](phase-1.md), [phase-2.md](phase-2.md), [phase-3.md](phase-3.md), [phase-4.md](phase-4.md) | Each part's question, method, outputs, and checks |
| `pyproject.toml` | Python requirements, packages, and the `fecon` command |
| `uv.lock` | Exact package versions |
| `.github/workflows/tests.yml` | Run tests on Windows and Linux in GitHub Actions |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Collaborator access, branches, and pull requests |

To change **what is studied**, edit `configs/study.toml`. To change **how results are calculated**, start in `evaluation/metrics.py`, `portfolio/declines.py`, `portfolio/controls.py`, or `evaluation/attribution.py`. To change **how results are presented**, start in the matching module under `reporting/`.

## Current settings and assumptions

The study selects SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT for January 1, 2010 through December 31, 2025. The exact settings live in `configs/study.toml`.

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
uv run ruff check
```

Tests use synthetic data and require no live finance service. Lint settings live in `pyproject.toml`. GitHub Actions installs the locked environment, runs the linter, and runs the tests on Windows and Linux with warnings treated as errors.

This repository is private. Ask [@N3V3MORE](https://github.com/N3V3MORE) to invite your GitHub username before cloning. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) when making changes, and keep downloaded data and generated results out of Git.
