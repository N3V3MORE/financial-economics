# Handover

A short record of what this project did, what it found, what was decided
and what is left. Full detail: [Final study](docs/final-study.md),
[Results explained](docs/results-explained.md), [README](README.md).

The methodology audit now has a separate matched implementation:
`financial_economics.paper_revised`, with frozen settings in
`outputs/revised-study-20261021/protocol.json`. It retains the original
reconstruction and labels the revised historical comparisons exploratory.
See the cross-reference table in `docs/final-study.md` and the runnable
commands in `README.md`.

Current code has returned to **22 trading observations**. The stricter
substitute check is saved in `outputs/volatility-proxy-final-20261022`:
added IVV/VOO inputs improve 2017–February 2022 average volatility accuracy
from 93.3% to 93.7%, but 76 of 1,285 dates miss the requested 85% individual
floor; the worst later date scores 32.0%. Earlier history has larger errors.
The candidate is diagnostic only and is rejected for new training/backtests.
The gate checks every available Oxford date, including training history.

The earlier October 22 experiment added an explicitly estimated volatility history in
`outputs/volatility-proxy-complete-20261022` and a separate later-year pilot
in `outputs/proxy-study-20261022`. Final overlap error is 6.7% on volatility
levels, 13.2% on raw BV; the missing 2018 crash week has 32.4% error under
the daily-range fallback. That archived pilot used 30-calendar-day histories;
its outputs stay unchanged, and it has not been rerun with 22. The October 21
numbers below retain their original 22-observation setup.

The later-year pilot completed all 1,003 trading days in 2022–2025. At 10 bps,
EWMA/one-input GRU cash Sharpe is 0.335/0.340, versus about 0.433 for their
2021-calibrated static mixes. Before fees it is 0.358/0.420. All three paired
tests are nonsignificant after Holm adjustment. This extends the historical
test with estimated volatility; it does not establish a profitable GRU edge.
The full table and the 2018 approximation weakness are in `docs/final-study.md`.

## The question

**Can a stock investor protect against market falls better than by simply
holding less stock?** The project started as an ETF diversification study,
then adapted Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe
Haven Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).
Kaczmarek and Grobelny (2023) later proposed a neural-network-driven
stock/Treasury allocation ([doi](https://doi.org/10.18559/ebr.2023.2.738)), so
dynamic Treasury allocation is not new; this project's addition is the
realistic-trading test.

## What was done, in order

| Stage | What it asked | Main output |
|---|---|---|
| Phases 1–4 | Does an eight-ETF mix beat SPY, and a SPY+cash mix with the same stock exposure? | `outputs/reproduced/phase1`–`phase4` |
| Fund checks | What happens when each fund is removed, or DBC added, across periods and fees? | `outputs/fund-checks-*` |
| Protection study | Which of 13 ETFs beat cash in months when SPY fell? | `outputs/protection-main-final`, `protection-long-final` |
| Early prediction | EWMA vs GRU on squared daily returns, 2017–2025 | `outputs/prediction-*` |
| Partial replication | One-input GRU on Oxford-Man BV, 2010–2020, paper's rules | `outputs/paper-gru-20261020` |
| Extension | Same with trading delay, fees, past-only cutoffs, 2010–2021 | `outputs/paper-extension-gru-20261020`, `paper-frequency-*`, `paper-allocation-change-5pp-*` |
| Full rebuild | All five paper models, 13 defensive assets, nine settings, 2010–2020 | `outputs/paper-reconstruction-20261021/evaluation` |
| Richer-GRU fix | Log-volatility version of both GRUs (post hoc) | `.../gru_*_log`, `.../evaluation_fixed` |
| Realistic test | Rebuilt forecasts traded realistically against a static mix, 2011–2020 | `outputs/paper-realistic-20261021` |
| Paper figures | Figures 3–7 redrawn, plus our data checked against the paper's printed statistics | `outputs/paper-figures-20261021`, `paper-figures-local-20261021` |
| Post hoc checks | Swiss/German bonds in local currency; IEF added to the realistic test | `.../evaluation_local`, `outputs/paper-realistic-ief-20261021` |
| Full verified rerun | Every stage above, from the saved data, on the final code | `outputs/rerun-20261021` |
| Matched methodology revision | Shared GRU histories, separate stopping/scoring, true t+2 forecasts, annual equity budget, full entry/exit accounting and 27 paired tests | `outputs/revised-study-20261021` |

## Datasets

| Folder in `data/` | Source | Used for |
|---|---|---|
| `public-review-20261019` | Yahoo Finance prices, Kenneth French daily factors, 2010–2025 | Phases 1–4, fund checks |
| `commodities-20261020` | Same sources, adds DBC | DBC addition check |
| `protection-wrds-20261020` | WRDS/CRSP total returns, 14 ETFs, 2015–2025 | Main protection study |
| `protection-long-wrds-20261020` | WRDS/CRSP SPY/TLT/GLD/DBC, 2007–2025, French RF | Long protection check, extension, realistic test |
| `paper-source-20261020` | Archived Oxford-Man realized volatility (to Feb 2022) | Volatility targets and inputs for every paper stage |
| `paper-reconstruction-public-20261021` | Yahoo and FRED: S&P 500, futures, bond ETFs, yields, Brent | Paper rebuild returns and richer-GRU inputs |
| `paper-reconstruction-fx-20261021` | Federal Reserve H.10 dated FX | Correct CHF/EUR conversion of Swiss and German bonds |
| `protection-long-ief-wrds-20261021` | WRDS/CRSP SPY/TLT/GLD/DBC/IEF, 2007–2025 (first four identical to the older snapshot) | IEF realistic check |

WRDS data is licensed. Raw data and outputs stay local and are ignored by Git.

## Key decisions

- **Rules before results.** Every configuration (`configs/paper-reconstruction.toml`,
  `configs/paper-realistic.toml` and the earlier ones) was fixed before its run.
  Model settings were chosen with pre-2010 data only.
- **Two versions of the paper.** The rebuild copies the paper's idealized
  rules: zero fees, zero cash interest, same-close trading, full-test stress
  cutoffs. The realistic test trades one day after each forecast, charges 10 bps
  per ETF dollar traded, earns French cash interest, and sets each year's
  cutoff from earlier forecasts only.
- **Pre-calibrated comparison.** Timing is compared with a static SPY/cash
  mix holding the model's mean **2010** stock target, reset monthly. It is
  not matched to 2011–2020 exposure. In practice timing held more stock
  than its mix (ARFIMA 75% against 56%; one-input GRU 79% against 59%) with
  similar volatility and lower beta, so its smaller maximum drawdown and
  COVID loss do not come from holding less stock. This exposure check was added afterwards and is
  descriptive; see the main result under Findings.
- **The paper's volatility scaling.** Monthly volatility is 22 × daily, as
  printed in the paper; textbook scaling uses √22. The 10% target is
  therefore about 7.2% a year. A conventional 10%-a-year target is reported
  beside it from the extension run (2010–2021, 0.1% fees): Sharpe 0.91
  against 0.87, worst fall −16.1% against −14.2%, less trading.
- **Fixed significance tests.** The realistic test has six primary tests
  (ARFIMA and the one-input GRU × timing vs static, TLT vs cash, GLD vs
  cash) under one Holm adjustment. Everything else is descriptive.
- **Richer-GRU fix labelled post hoc.** The raw version predicted negative
  variance on 5.5% of days (mostly in calm 2017–2018). The log version was
  added after seeing that, and is reported beside the raw version.
- **Two more post hoc checks.** The figure comparison showed the paper's
  Swiss/German bond statistics match local-currency returns, so
  `evaluate_local` rescored them in CHF/EUR. IEF was added to the realistic
  test as a descriptive asset. Both sit beside the original runs, which are unchanged.
- **No data splicing.** Providers are never mixed within one return series;
  missing values are never filled.
- **Scope.** An international robustness run was started, then removed as out of scope.
- **Vendor RNN kernels for GRU training.** GRUs train with the GPU vendor's
  RNN library: cuDNN on NVIDIA (about five times faster) and MIOpen on AMD
  ROCm (about two times faster). The one-input paper GRU is built as two single-layer
  GRUs, which is the same network, to avoid a Windows cuDNN shutdown crash.
  Forecasts differ slightly from the earlier native-kernel runs. Setting
  `FINANCIAL_ECONOMICS_NATIVE_RNN=1` forces PyTorch's native GPU RNN instead.
- **Full float32 precision (TF32 off).** From 2026-10-22, GPU training turns
  off NVIDIA's TF32 mode for cuDNN and matrix products; run metadata records
  both flags. On the RTX 5060 Ti benchmark this narrowed cuDNN-versus-native
  training gaps 10-1,000 fold at the same speed. AMD forecasts are unchanged
  to the last digit. NVIDIA GRU forecasts made earlier used TF32, so saved
  reconstruction GRU stages need a fresh run; the cache guard now refuses them.
- **Comparing results across GPUs.** Runs on different GPUs or libraries are
  not expected to match exactly. Treat one setup as the reference for headline
  results and the others as robustness checks. Judge agreement on what is
  reported (QLIKE, Holm decisions, portfolio figures), not raw forecast digits,
  against the spread between seeds. A training curve can briefly separate at a
  sharp turn even with TF32 off: the GPU-backend benchmark's five-input seed-33
  case split at steps 96-99 on both NVIDIA and AMD, then rejoined.

## Comparison with the paper (2010–2020, the paper's rules)

| Model | Paper forecast error (MSE) | Ours | Paper Sharpe | Ours |
|---|---:|---:|---:|---:|
| ARFIMA | 0.0034 | 0.0035 | 1.00 | 1.01 |
| GARCH | 0.0127 | 0.0102 | 0.95 | 1.01 |
| GJR-GARCH | 0.0146 | 0.0105 | 0.88 | 0.98 |
| One-input GRU | 0.0043 | 0.0034 | 0.95 | 1.03 |
| Richer GRU (raw / log fix) | 0.0031 | 0.0040 / 0.0035 | 0.97 | 0.98 / 1.01 |

**Where we agree with the paper:**
- ARFIMA matches closely.
- Better forecasts did not mean better portfolios.
- Most "safe" assets hurt.
- Long Treasuries were the best crash protector to 2020.

**Where we differ:**
- Our richer GRU configuration did not establish lower forecast error
  than ARFIMA or the one-input GRU. Its training span and validation folds
  differ, so this does not isolate the effect of additional inputs.
- Gold added the most Sharpe over the full period (+0.015) rather than
  Treasuries, though the paper's own gold figure is almost the same (+0.014);
  it rejected gold on Calmar and COVID, where ours also falls short.
- TLT lagged cash on average in falling-SPY months in 2022–2025.

## Findings

**October 21 matched revision.** The completed revised study retains all 2,517 market
days in 2011–2020. At 10 bps, EWMA/one-input GRU/richer GRU stock/cash Sharpe
is 0.843/0.827/0.841, versus 0.861/0.885/0.892 without fees. Both GRUs have
lower measured volatility forecast MSE than EWMA, but neither forecast
advantage survives adjustment; none of the 27 main portfolio tests is
significant after Holm adjustment. Timing has smaller maximum drawdown and
COVID loss than its static mix for all six models, but loses more in 2015–16
for all six. Realized volatility is about 12% for EWMA and the GRUs despite
the 10% equity budget. All 87 saved stage files match their hashes; all 369
tests pass, including GPU training. See `docs/final-study.md` for the table
and limits. The findings below describe the earlier experiments.

**Verified.** On October 21, 2026 every stage was rerun from the saved data
on the final code (`outputs/rerun-20261021`, 37 stages, about 20 minutes of
GPU time). The run includes the corrected inference (centered Sharpe tests,
six-test Holm family, forecast input checks) and GRUs retrained from scratch
with cuDNN. No headline figure and no Holm decision changed. Stages without
GRUs matched to rounding; the largest realistic-test Sharpe change was
0.00006. The original runs are kept unchanged beside the rerun.

**Statistically supported:**
- 11 of 13 ETFs lagged cash in falling months, 2015–2025 (Holm-adjusted).
- ARFIMA and the GRUs forecast far better than GARCH and GJR.

**Main result (descriptive, no uncertainty estimates):**
- Timing held more stock on average than its fixed mix (ARFIMA 75% against
  56%) with similar volatility, lower beta and smaller maximum drawdown and
  COVID loss. It stayed invested in calm periods and cut stock when forecast risk rose:
  effective risk allocation in this sample. This is not evidence that it
  predicted whether stocks would rise or fall, or that the advantage is
  statistically reliable or will persist.
  These two loss measures overlap. Across the other decline windows,
  timing lost more for seven of seven models in 2015–16, four in 2011 and
  three in 2018; ARFIMA's 2018 loss was 11.8% versus its fixed mix's 10.7%.
- The assumed trading costs erased the observed before-fee advantage
  (ARFIMA Sharpe +0.09 before fees, −0.04 at 10 bps). Fees are scenarios,
  and the before-fee advantage was not itself tested for significance.

**Consistent but not significant:**
- Timing cut the worst fall (maximum drawdown) for every model (about −11% to −16%, against −15% to −23% for the static mix); in the common COVID window ARFIMA lost 9.7% against 19.6%.
- Timing roughly halved the COVID loss, but lost more in the slow 2015–16 fall.
- After 10 bps fees, ARFIMA and one-input GRU timing had Sharpe estimates of 0.76–0.80 against 0.80 for the static mixes; their differences were not significant. This does not establish equivalence. The other models had estimates of 0.71–0.75. Before fees, ARFIMA had 0.89 against 0.80; at 25 bps it had 0.57.
- In falling months, TLT beat cash by +1.7 to +3.6 points per period up to 2021, then lost 4.3 points in 2022–2025.
- Gold was positive on average but inconsistent across models and crashes.

**Not shown:**
- The primary TLT and GLD tests did not establish improvement beyond cash.
- The primary timing tests did not establish a Sharpe improvement over the
  pre-calibrated static mixes.

**Thesis:** in these retrospective tests, timing reduced the worst
observed falls without an established Sharpe improvement over the
2010-calibrated static mixes after fees. Nonsignificance does not establish
equivalence. The tested defensive assets' contributions varied by model
and period, including a reversal for long Treasuries in 2022–2025; these
results do not establish a universally reliable haven or rule out benefits
in other settings.

## Limitations left

- **Substitute data.** Public prices and futures replace the paper's Datastream
  series; futures prices are not investable returns.
- **Bond prices.** The rebuild uses price-only bond ETFs, close to the paper's
  clean prices. The realistic test uses total returns.
- **Unpublished details.** The paper's code, seeds, tuning grid and some metric
  formulas are unpublished; our Sharpe formula and GRU details are inferred.
- **Different archived GRU histories.** The one-input model uses BV history from
  2000, while the richer feature table starts on April 11, 2002, before
  the 22-observation lookback. Training samples and validation-fold dates
  differ. The revised study matches both histories from October 3, 2000,
  with identical validation dates and a one-session lag on external inputs.
- **Validation reused in the original models.** GRU early stopping and configuration
  scoring share validation folds, so tuning scores can be optimistic. The
  revised study uses disjoint stopping and selection blocks.
- **Archived extension and realistic boundaries.** The extension refits a two-session
  target and assigned cutoffs by annual issue-date blocks; the realistic
  test delays saved one-session forecasts and assigned cutoffs by return
  year. The extension starts in cash, while an earlier signal can invest
  the realistic portfolio on its first evaluation day. These histories
  are retained as distinct setups. New extension runs share the earned-return-year
  cutoff rule. All revised models target t+2, and every comparator starts in cash.
- **ARFIMA bound.** ARFIMA's memory parameter hit its upper bound on 38% of days.
- **History already seen.** Forecasts are chronological out-of-sample
  predictions, but the study history was examined repeatedly. The revised
  calculation is exploratory rather than an untouched confirmatory test.
- **Small effects.** Safe-haven effects are a few hundredths of Sharpe, and none is significant.
- **Not reconstructed.** The paper's international markets (Nikkei, Nifty, STOXX 50).
- **No paper-style forecasts after 2020.** The Oxford-Man archive ends in
  February 2022 (the extension reaches 2021). The 2022–2025 Treasury
  reversal comes from fixed holdings, not the timing strategy, so it does not
  show that the paper's full strategy failed then. The early prediction
  study is the only, rougher, timing evidence for 2022–2025.
- **Exclusions.** Taxes remain excluded. Archived runs omit final sale fees;
  the matched revision includes liquidation costs and charges the benchmark too.

## Where things are

- **Branch:** `volatility-prediction`.
- **Frozen GRU study tag:** `frozen-gru-study-20261021`.
- **Fresh snapshot recorded on the AMD machine:** `data/protection-long-wrds-20261021` contained
  verified SPY/TLT/GLD/DBC returns for 4,780 sessions from January 3, 2007
  through December 31, 2025; all 912 fund-month comparisons passed.
  This directory is absent on this machine; these are the recorded AMD checks.
  A rebuild from these fresh inputs was stopped on the AMD machine; it is
  not needed for the reported results, which the full rerun verified.
- **Other recovered inputs recorded on the AMD machine:** `data/paper-oxford-20261021/oxfordman-20220301.zip`
  matched the recorded archive hash; `data/paper-fx-20261021` passed dated H.10
  coverage checks. FRED maintenance was resolved using WRDS DGS3/DAAA and
  direct EIA Brent in `data/paper-macro-wrds-eia-20261021`. The combined
  `data/paper-wrds-eia-20261021` snapshot had 19 series and 5,284 equity
  sessions; hashes, prior-only alignment, and official FX conversion passed.
  All 5,253 WRDS DGS3 observations matched the official Fed export exactly.
  The 4,197 overlapping AAA observations matched too; later WRDS AAA values
  lacked independent current-FRED comparison. Source identities and the
  fresh vintage were retained in the manifests. These four dated directories
  are absent on this machine; the recorded checks were not repeated here.
- **AMD training:** Radeon 8060S on Windows runs PyTorch 2.11.0 + ROCm 7.14.1.
  Native HIP GRU probes passed forward/backward, Adam updates, exact same-seed
  repetition and checkpoint reload. MIOpen's Windows RNN compiler first failed
  on a missing `type_traits` header: it borrows the C++ standard headers from
  Visual Studio. Installing Visual Studio 2022 Build Tools (C++ workload and
  Windows 11 SDK 10.0.26100) fixed it on 2026-10-22, so ROCm now trains with
  MIOpen 3.5.2 by default. Without those Build Tools, set
  `FINANCIAL_ECONOMICS_NATIVE_RNN=1` to keep the native HIP path. The first
  MIOpen fit compiles its kernels (about 15 seconds); later runs use the cache.
  With the project's deterministic settings, MIOpen repeated same-seed fits
  exactly, and its forecasts stayed within 2e-7 relative of native HIP with
  the same early-stopping epochs. Training steps were 1.9-2.5 times faster at
  hidden sizes 16 and 32. Runs before this change used native HIP.
  Run and checkpoint metadata identify ROCm and its HIP version. Future runs
  also record NVIDIA cuDNN and AMD MIOpen availability, enabled flag and
  version separately from observed training-forward dispatch; inference
  dispatch remains unobserved.
  Archived run metadata stays unchanged. Exact reviewed metadata-only code
  hash transitions preserve retained raw/log forecast reuse; arbitrary model
  changes still fail the cache guard. The MIOpen change to
  `gru_forecasting.py` was first reviewed as metadata-only for those caches,
  which were trained with NVIDIA cuDNN. Turning TF32 off then changed NVIDIA
  training math, so GRU stages cached before it need a fresh run.
  That AMD environment included the locked `prediction-rocm`, `reconstruction`
  and `wrds` extras; use `prediction-rocm` in AMD training commands. This
  machine's current validation uses NVIDIA CUDA.
- **Commands to rerun each stage:** README sections “Reconstruct the paper's
  U.S. experiment” and “Fixed GRUs and the realistic trading test”.
- **Stopped fresh-input rebuild (AMD machine):** `outputs/paper-wrds-eia-rocm-20261021/status.json`
  was the recorded status path; `logs/run.stdout.log` and
  `logs/run.stderr.log` were retained on that machine. Those paths are absent
  on this machine and cannot be checked locally here. The process tree was
  terminated during the first GRU stage because no
  meaningful results-preserving speedup had been validated. No stages were
  completed. Downloaded inputs and logs were retained there. Final output was
  planned
  at `outputs/paper-realistic-wrds-eia-rocm-20261021`.
- **Historical tests after the inference, provenance, Windows login, AMD and source fixes:**
  335 passed with warnings treated as errors in the main ROCm environment,
  including all seven actual GPU tests; none skipped.
- **Historical merged-suite result:** 345 passed after merging both machines'
  work, including GPU tests on NVIDIA.
- **Current tests and lint:** 392 passed with warnings treated as errors,
  including matched-model GPU checks on NVIDIA CUDA, using
  `.venv/Scripts/python.exe -m pytest -q -W error` with a workspace temporary
  directory. GPU tests skip without compatible CUDA or ROCm.
  `.venv/Scripts/ruff.exe check` passed.
- **Write-up draft:** [Google Doc](https://docs.google.com/document/d/1aUVYdh1XzAlTyHhGPN4SLtOsYPSjuieeSwIQuM8PsnE/edit?usp=sharing).
