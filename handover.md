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
- **The right comparison.** Timing is compared with a static SPY/cash mix
  holding the model's average 2010 stock share, not only with SPY.
- **Fixed significance tests.** The realistic test has six primary tests
  (ARFIMA and the one-input GRU × timing vs static, TLT vs cash, GLD vs
  cash) under one Holm adjustment. Everything else is descriptive.
- **Richer-GRU fix labelled post hoc.** The raw version predicted negative
  variance on 5.5% of days (mostly in calm 2017–2018). The log version was
  added after seeing that, and is reported beside the raw version.
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
- Our richer GRU did not beat the simpler models.
- Gold added the most Sharpe over the full period.
- Treasuries stopped protecting in 2022–2025.

## Findings

**Statistically supported:**
- 11 of 13 ETFs lagged cash in falling months, 2015–2025 (Holm-adjusted).
- ARFIMA and the GRUs forecast far better than GARCH and GJR.

**Consistent but not significant:**
- Timing cut the worst fall for every model (about −11% to −16%, against −15% to −23% for the static mix).
- Timing roughly halved the COVID loss, but lost more in the slow 2015–16 fall.
- After 10 bps fees, ARFIMA and one-input GRU timing tied the static mix on Sharpe (0.76–0.80 against 0.80); the other models trailed (0.71–0.75). Before fees, ARFIMA beat its mix (0.89 against 0.80); at 25 bps it lost (0.57).
- In falling months, TLT beat cash by +1.7 to +3.6 points per period up to 2021, then lost 4.3 points in 2022–2025.
- Gold was positive on average but inconsistent across models and crashes.

**Not shown:**
- No safe haven significantly improved the strategy.
- No timing strategy significantly beat the static mix.

**Thesis:** volatility timing protects against big, fast crashes but does not
earn more per unit of risk than holding less stock, and no safe haven,
including the paper's long Treasuries, stayed reliable over time.

## Limitations left

- **Substitute data.** Public prices and futures replace the paper's Datastream
  series; futures prices are not investable returns.
- **Bond prices.** The rebuild uses price-only bond ETFs, close to the paper's
  clean prices. The realistic test uses total returns.
- **Unpublished details.** The paper's code, seeds, tuning grid and some metric
  formulas are unpublished; our Sharpe formula and GRU details are inferred.
- **ARFIMA bound.** ARFIMA's memory parameter hit its upper bound on 38% of days.
- **History already seen.** 2010–2020 was examined repeatedly, so results are
  retrospective, not out of sample.
- **Small effects.** Safe-haven effects are a few hundredths of Sharpe, and none is significant.
- **Not reconstructed.** The paper's international markets (Nikkei, Nifty, STOXX 50).
- **No volatility forecasts after 2020.** The Oxford-Man archive ends in
  February 2022; 2022–2025 is covered only by the fixed-holdings protection study.
- **Exclusions.** Taxes and exit fees are excluded throughout.

## Where things are

- **Branch:** `volatility-prediction`. It is not pushed; `main` is unchanged.
- **Frozen GRU study tag:** `frozen-gru-study-20261021`.
- **Commands to rerun each stage:** README sections “Reconstruct the paper's
  U.S. experiment” and “Fixed GRUs and the realistic trading test”.
- **Tests:** `uv run pytest -q` (226 pass; GPU tests skip without CUDA).
