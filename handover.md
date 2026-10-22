# Handover

A short record of what this project did, what it found, what was decided
and what is left. Full detail: [Final study](docs/final-study.md),
[Results explained](docs/results-explained.md), [README](README.md).

## The question

**Can a stock investor protect against market falls better than by simply
holding less stock?** The project started as an ETF diversification study,
then adapted Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe
Haven Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).

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

- **Rules before results.** Every configuration (`paper-reconstruction.toml`,
  `paper-realistic.toml` and the earlier ones) was fixed before its run.
  Model settings were chosen with pre-2010 data only.
- **Two versions of the paper.** The rebuild copies the paper's idealized
  rules: zero fees, zero cash interest, same-close trading, full-test stress
  cutoffs. The realistic test trades one day after each forecast, charges 10 bps
  per ETF dollar traded, earns French cash interest, and sets each year's
  cutoff from earlier forecasts only.
- **Pre-calibrated comparison.** Timing is compared with a static SPY/cash
  mix holding the model's mean 2010 stock target, reset monthly. It does
  not ensure matching average 2011–2020 stock exposure or realized risk,
  so differences can reflect exposure as well as timing.
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

The following numbers and significance statements are retained from the
archived runs. The primary Holm correction now retains all six planned
tests when one is unavailable, using p=1 internally while leaving its
reported p-values unavailable and its rejection flag false. Sharpe HAC
inference uses centered calculations and marks constant excess returns
unavailable; identical or positively proportional excess returns retain
their difference estimate with unavailable uncertainty and p-values when
sampling variance is degenerate. Conditional return comparisons are
unavailable when there are no months when stocks fall. Saved forecast stages
must have matching, valid hashes for their source manifest, Oxford archive
and reconstruction config, retained in the realistic run's provenance.
The original archived inputs and model forecasts remain unavailable on the
AMD machine. They are present on the original machine
(`outputs/paper-reconstruction-20261021`), where the evaluation and realistic
test were rerun from them before these fixes and matched the saved results.
Fresh WRDS returns and Yahoo/WRDS/EIA reconstruction inputs are verified.
The AMD reconstruction was stopped at the user's request on October 21; model forecasts are not yet
complete and historical results have not been regenerated with these fixes.

**Statistically supported:**
- 11 of 13 ETFs lagged cash in falling months, 2015–2025 (Holm-adjusted).
- ARFIMA and the GRUs forecast far better than GARCH and GJR.

**Consistent but not significant:**
- Timing cut the worst fall for every model (about −11% to −16%, against −15% to −23% for the static mix).
- Timing roughly halved the COVID loss, but lost more in the slow 2015–16 fall.
- After 10 bps fees, ARFIMA and one-input GRU timing had Sharpe estimates of 0.76–0.80 against 0.80 for the static mixes; their differences were not significant. This does not establish equivalence. The other models had estimates of 0.71–0.75. Before fees, ARFIMA had 0.89 against 0.80; at 25 bps it had 0.57.
- In falling months, TLT beat cash by +1.7 to +3.6 points per period up to 2021, then lost 4.3 points in 2022–2025.
- Gold was positive on average but inconsistent across models and crashes.

**Not shown:**
- The archived primary TLT and GLD tests did not establish improvement
  beyond cash.
- The archived primary timing tests did not establish a Sharpe improvement
  over the pre-calibrated static mixes.

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
- **Different GRU histories.** The one-input model uses BV history from
  2000, while the richer feature table starts on April 11, 2002, before
  the 22-observation lookback. Training samples and validation-fold dates
  differ. A matched-history study is needed to attribute differences to
  the additional inputs.
- **ARFIMA bound.** ARFIMA's memory parameter hit its upper bound on 38% of days.
- **History already seen.** 2010–2020 was examined repeatedly, so results are
  retrospective, not out of sample.
- **Small effects.** Safe-haven effects are a few hundredths of Sharpe, and none is significant.
- **Not reconstructed.** The paper's international markets (Nikkei, Nifty, STOXX 50).
- **No volatility forecasts after 2020.** The Oxford-Man archive ends in
  February 2022; 2022–2025 is covered only by the fixed-holdings protection study.
- **Exclusions.** Taxes and exit fees are excluded throughout.

## Where things are

- **Branch:** `volatility-prediction`.
- **Frozen GRU study tag:** `frozen-gru-study-20261021`.
- **Fresh local snapshot:** `data/protection-long-wrds-20261021` contains
  verified SPY/TLT/GLD/DBC returns for 4,780 sessions from January 3, 2007
  through December 31, 2025; all 912 fund-month comparisons passed.
  Model forecast reconstruction was stopped at the user's request; new results are not yet complete.
- **Other recovered inputs:** `data/paper-oxford-20261021/oxfordman-20220301.zip`
  matches the recorded archive hash; `data/paper-fx-20261021` passes dated H.10
  coverage checks. FRED maintenance was resolved using WRDS DGS3/DAAA and
  direct EIA Brent in `data/paper-macro-wrds-eia-20261021`. The combined
  `data/paper-wrds-eia-20261021` snapshot has 19 series and 5,284 equity
  sessions; hashes, prior-only alignment, and official FX conversion passed.
  All 5,253 WRDS DGS3 observations match the official Fed export exactly.
  The 4,197 overlapping AAA observations match too; later WRDS AAA values
  lack independent current-FRED comparison. All actual source identities
  and the fresh vintage are retained in the manifests.
- **AMD training:** Radeon 8060S on Windows runs PyTorch 2.11.0 + ROCm 7.14.1.
  Native HIP GRU probes passed forward/backward, Adam updates, exact same-seed
  repetition and checkpoint reload. MIOpen's Windows RNN compiler failed on
  a missing `type_traits` header, so ROCm uses the native GPU RNN path.
  Run and checkpoint metadata now identify ROCm and its HIP version.
  The main `.venv` now includes the locked `prediction-rocm`, `reconstruction`
  and `wrds` extras; use `prediction-rocm` in AMD training commands.
- **Commands to rerun each stage:** README sections “Reconstruct the paper's
  U.S. experiment” and “Fixed GRUs and the realistic trading test”.
- **Stopped full rebuild:** `outputs/paper-wrds-eia-rocm-20261021/status.json`
  records the interrupted stage and original process ID; `logs/run.stdout.log` and
  `logs/run.stderr.log` retain progress and errors. The process tree was terminated during the first GRU stage because no
  meaningful results-preserving speedup had been validated. No stages were
  completed. Downloaded inputs and logs are retained. Final output was planned
  at `outputs/paper-realistic-wrds-eia-rocm-20261021`.
- **Tests after the inference, provenance, Windows login, AMD and source fixes:**
  335 passed with warnings treated as errors in the main ROCm environment,
  including all seven actual GPU tests; none skipped.
- **Tests and lint:** `uv run pytest -q` (345 pass after merging both machines' work, including GPU tests on the NVIDIA machine; GPU tests skip without CUDA or ROCm)
  and `uv run ruff check`.
- **Write-up draft:** [Google Doc](https://docs.google.com/document/d/1aUVYdh1XzAlTyHhGPN4SLtOsYPSjuieeSwIQuM8PsnE/edit?usp=sharing).
