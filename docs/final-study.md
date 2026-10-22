# Final study: does volatility timing with "safe" assets protect stock investors?

This page states the project's final research goal and what the evidence
shows. It ties the original ETF study to the paper we adapted:
Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe Haven
Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).

## The research goal

**Can a U.S. stock investor protect against market falls better than by
simply holding less stock?** We test the paper's answer: forecast stock
volatility, hold less stock when forecast risk is high, and in the most
stressed periods move some cash into a "safe haven" asset.

Three questions, each answered with rules fixed before its results were seen:

1. **Method.** Can we rebuild the paper's forecasting models and strategy on
   public data, and get similar numbers?
2. **Timing.** Once trading is realistic, does volatility timing beat a fixed
   SPY/cash mix calibrated to the model's mean 2010 stock target?
3. **Safe havens.** Do Treasuries or gold add protection beyond cash, and does
   that protection last, including 2022–2025?

This follows directly from the original project. Phase 3 showed that an
eight-ETF mix did not clearly beat SPY plus cash, and the protection study
did not establish that any ETF beat cash in falling months. The paper offers an
alternative to fixed holdings; this study checks whether it changes that answer.

**Related work.** Two of the paper's authors later proposed a dynamic
stock/Treasury allocation driven by neural-network volatility forecasts
(Kaczmarek and Grobelny, 2023, “How to fly to safety without overpaying for the
ticket”, *Economics and Business Review* 9(2),
[doi](https://doi.org/10.18559/ebr.2023.2.738)). Dynamic Treasury allocation is
therefore not new. Our addition is the realistic-trading test: a one-day
delay, stated fee scenarios, interest on cash, past-only stress cutoffs and a
fixed mix chosen before evaluation.

## What we did

| Step | Data | Rules | Where |
|---|---|---|---|
| Rebuild the paper (2010–2020) | Oxford-Man S&P 500 bipower variation; public price proxies for all 13 defensive assets | Paper's own idealized rules: zero fees, zero cash interest, full-test stress cutoffs | `outputs/paper-reconstruction-20261021/evaluation` |
| Fix the richer GRU | Same | Learns log volatility so forecasts are always positive; added after results were seen, labelled post hoc | `.../gru_univar_log`, `.../gru_multivar_log`, `.../evaluation_fixed` |
| Realistic test (2011–2020) | Same saved forecasts; WRDS/CRSP SPY, TLT, GLD, DBC total returns; French cash rate | Trade one day after each forecast, 0.1% fees, cash earns interest, cutoffs from past forecasts only, six pre-chosen tests | `outputs/paper-realistic-20261021` |
| 2022–2025 check | WRDS/CRSP ETF returns, 2007–2025 | Fixed holdings versus cash in months when SPY fell | `outputs/protection-long-final` |

The five forecasting models are the paper's: ARFIMA, GARCH, GJR-GARCH, a
one-input GRU and a richer GRU with gold, oil and yield inputs. Every model's
settings were chosen with pre-2010 data only. The one-input GRU uses BV
history from 2000; the richer feature table starts on April 11, 2002,
before applying the 22-observation lookback. Their training histories
and validation-fold dates differ, although their evaluation dates match.
This compares complete model configurations; isolating the effect of
extra inputs requires matched training histories and validation folds.
The strategy holds
`min(10% / predicted volatility, 100%)` in stocks and the rest in cash. When
predicted volatility is above its 90th percentile, up to 20% moves from cash
into the defensive asset.

**What "10% volatility" means here.** Following the paper's printed
conversion (its Eq. 24), predicted monthly volatility is 22 × predicted daily
volatility. Textbook scaling multiplies by √22 instead, so the paper's 10%
monthly target equals about **7.2% a year** (0.10 / 22 × √252). We keep the
paper's convention because it reproduces the paper's results and its frequent
weight changes; with textbook scaling a 10% monthly target would rarely cut
stocks at all. A conventional 10%-a-year target is reported alongside in
section 2. Adding a defensive asset keeps the stock weight, so it changes total
portfolio risk.

## What we found

**These results are verified.** On October 21, 2026 every stage was rerun
from the saved data snapshots on the final code (`outputs/rerun-20261021`).
That code includes the corrected inference: centered Sharpe tests, a
six-test Holm family that keeps unavailable tests, and input checks on saved
forecasts. All five GRUs were retrained from scratch with cuDNN. **No headline
figure and no Holm decision changed.** The largest Sharpe change in the
realistic test was 0.00006; one primary p-value moved from 0.9781 to 0.9783.
GRU forecasts differ slightly from the earlier native-kernel runs, which are
kept unchanged. A separate rebuild from fresh Yahoo/WRDS/EIA inputs on the AMD
machine was stopped on October 21 and is not used here
(`outputs/paper-wrds-eia-rocm-20261021/status.json`).

### 1. The rebuild approximates several reported metrics

| Model | Paper forecast error (MSE) | Ours | Paper Sharpe | Ours |
|---|---:|---:|---:|---:|
| ARFIMA | 0.0034 | 0.0035 | 1.00 | 1.01 |
| GARCH | 0.0127 | 0.0102 | 0.95 | 1.01 |
| GJR-GARCH | 0.0146 | 0.0105 | 0.88 | 0.98 |
| One-input GRU | 0.0043 | 0.0034 | 0.95 | 1.03 |
| Richer GRU | 0.0031 | 0.0040 | 0.97 | 0.98 |

ARFIMA's rounded metrics are close to the paper's reported values. The
one-input GRU has a similar forecast-error estimate; the test did
not detect a difference, which does not establish equivalence. Our raw
richer GRU had higher forecast error than the paper's and produced negative
forecasts on 5.5% of days, almost all in the very calm years 2017–2018.

**The fix.** Learning log volatility removes every negative forecast. Both
fixed GRUs keep the same settings menu, seeds and pre-2010 selection, and both
chose the same profile (two layers of 32 units).

| Model | Forecast error (MSE) | Percentage error (MAPE) | Negative forecasts | Sharpe |
|---|---:|---:|---:|---:|
| Richer GRU, original | 0.0040 | 36.0% | 5.5% of days | 0.98 |
| Richer GRU, fixed | 0.0035 | 28.6% | none | 1.01 |
| One-input GRU, fixed | 0.0033 | 29.2% | none | 1.01 |
| Paper's richer GRU | 0.0031 | 26.2% | not reported | 0.97 |

The fixed richer GRU has a forecast-error estimate close to ARFIMA's and
the lowest percentage-error estimate of all seven models, bringing its
reported metrics closer to the paper. The tests did not establish
lower forecast error than ARFIMA or the one-input GRU configurations. Because
training histories and validation folds differ, this does not isolate whether extra
inputs help. The fix is reported beside the original because it was added
after the original results were seen.

As in the paper, **a better forecast did not mean a better portfolio**: GARCH's
forecast error was three times ARFIMA's, yet their Sharpe ratios were almost equal.

### 2. Timing reduced observed crash losses without an established Sharpe improvement

Realistic test, 2011–2020, 0.1% fees, cash earning interest:

| Portfolio | Annual growth | Sharpe | Worst fall | COVID fall | 2015–16 fall |
|---|---:|---:|---:|---:|---:|
| SPY | 13.8% | 0.80 | −33.7% | −33.7% | −13.0% |
| ARFIMA timing | 7.4% | 0.76 | −14.5% | −9.7% | −11.6% |
| Fixed mix, 2010-calibrated stock target (56%) | 8.0% | 0.80 | −19.6% | −19.6% | −7.3% |
| One-input GRU timing | 8.1% | 0.80 | −14.5% | −10.2% | −11.8% |
| Fixed mix, 2010-calibrated stock target (59%) | 8.4% | 0.80 | −20.5% | −20.5% | −7.7% |

**The fixed mix uses the 2010 average stock share.** Each fixed mix holds
its model's mean stock target over calibration year 2010, reset monthly. It
was set before 2011 and is not matched to 2011–2020 exposure. Because 2010
was volatile, timing in fact held *more* stock afterwards, yet with less risk:

| 2011–2020, 0.1% fees | Average stock share | Volatility | Beta to SPY |
|---|---:|---:|---:|
| ARFIMA timing | 75% | 9.2% | 0.46 |
| ARFIMA fixed mix | 56% | 9.4% | 0.55 |
| One-input GRU timing | 79% | 9.6% | 0.48 |
| One-input GRU fixed mix | 59% | 9.9% | 0.58 |

So timing's smaller crash losses did not come from holding less stock on
average; it cut stock when forecast risk rose. Every model shows the same
pattern. This exposure check was added after the results were seen and is
descriptive.

- **No established Sharpe improvement.** Timing minus the fixed mix is
  −0.04 (ARFIMA) and −0.005 (GRU); neither difference was significant. This
  does not establish equal performance.
- **Fees change the historical comparison.** Timing trades about 116 times
  its own value over ten years. ARFIMA's Sharpe was 0.89 against 0.80 before
  fees, 0.76 against 0.80 at 0.1%, and 0.57 against 0.80 at 0.25%.
- **Better forecasts did not rescue it.** The fixed GRUs forecast best but
  traded even more (about 137 times their value); after 0.1% fees their Sharpe
  was 0.75, below their fixed mixes' 0.80.
- **Fast crashes versus slow falls.** Timing roughly halved the COVID loss
  but lost more in the slow 2015–16 decline.
- **It depended on the period.** Timing beat the fixed mix in 2016–2020
  (Sharpe 1.06 against 0.78) and lost in 2011–2015 (0.47 against 0.83).

**A conventional 10%-a-year target.** The extension run
(`paper-extension-gru`: one-input GRU retrained for the delayed trade,
2010–2021, 0.1% fees, cash earning interest) scored both conventions on the
same forecasts. It is descriptive and is a different setup from the
realistic test above:

| 2010–2021, 0.1% fees | Average stock share | Volatility | Annual growth | Sharpe | Worst fall | Trading (× value) |
|---|---:|---:|---:|---:|---:|---:|
| Paper's target (≈7.2% a year) | 77% | 9.6% | 8.7% | 0.87 | −14.2% | 114 |
| 10% a year | 90% | 12.0% | 11.3% | 0.91 | −16.1% | 74 |
| SPY, before any fees | 100% | 16.9% | 15.1% | 0.89 | −33.7% | — |

The 10%-a-year target holds more stock and trades less, so fees cost it
less. The pattern is the same: much smaller worst falls than SPY, with a
Sharpe ratio close to SPY's rather than clearly above it.

### 3. Safe havens: Treasury protection varied by period

**Paper-style rebuild (2010–2020, 13 assets):**
- Most "safe" assets made things worse: oil, silver, platinum, copper, cocoa,
  corn and wheat lowered Sharpe for every model.
- **Swiss and German bonds** were roughly neutral (about −0.004 Sharpe), as in
  the paper, once valued in their own currencies. Our first run converted them
  to dollars, which made them look worse (−0.02 to −0.06). The paper's printed
  volatilities and correlations match the local-currency version, so it is the
  closer copy. It was added after we saw this and is labelled post hoc; the
  dollar run is kept.
- **Gold** had the largest average gain (+0.015 Sharpe), helping with four of
  five models (+0.01 to +0.03; GARCH −0.01). It slightly worsened the COVID fall.
- **U.S. Treasuries** added a little. The 10-year proxy (IEF) was the only
  asset that helped with all five models (+0.01 to +0.02); the 30-year proxy
  (TLT) helped with four. Long Treasuries were the best crash protector:
  +1.4 points against cash in COVID and +1.5 in 2011.

**Realistic test (2011–2020):** adding TLT or GLD in stress raised Sharpe by
about 0.02–0.04, and TLT cut the COVID loss from 9.7% to 7.6%. None of the
four safe-haven tests survived the multiple-testing adjustment: the effects
are too small to separate from luck over ten years.

**IEF check (descriptive, added later):** the 7–10 year Treasury ETF, which
helped every model in the rebuild, added only about a third of TLT's gain
under realistic trading: +0.005 to +0.014 Sharpe for six of seven models
(TLT: +0.007 to +0.058 for all seven). It cut the ARFIMA COVID loss from 9.7%
to 8.9% (TLT: 7.6%). In falling months it beat cash by 0.03–0.06 points a month,
far from significant. Shorter bonds give a smaller, not a more reliable, cushion.

**2022–2025: average return minus cash in months when SPY fell (fixed holdings):**

| Period | Falling months | TLT | GLD |
|---|---:|---:|---:|
| 2010–2014 | 19 | +3.64 pp | −0.18 pp |
| 2015–2019 | 15 | +2.10 pp | +2.26 pp |
| 2020–2021 | 8 | +1.71 pp | −1.10 pp |
| 2022–2025 | 17 | **−4.31 pp** | +0.67 pp |

The paper's favorable result for long Treasuries came from an earlier
period. In this fixed-holdings check, their average return relative to cash
in falling-stock months changed sign in 2022–2025. This describes the
examined periods rather than a permanent property of Treasuries.

**Keep this separate from the timing strategy.** These rows are fixed
holdings in months when SPY fell, not the paper's forecast-driven strategy.
The volatility archive ends in February 2022, so the full strategy is not
tested after 2020 (2021 in the extension). We can say TLT's relative
performance in falling-stock months reversed; we cannot say the paper's
complete strategy failed in 2022–2025. The early prediction study
(squared daily returns, 2017–2025) is the only timing evidence after 2021,
and its setup is rougher.

## The answer

**In these retrospective tests, volatility timing reduced the worst
observed falls but did not establish a Sharpe improvement over the
2010-calibrated static mixes after 0.1% fees.** The nonsignificant tests
do not establish equivalence, and the controls are not calibrated to match
evaluation exposure or risk. Defensive-asset effects varied by model and
period:
the primary TLT and GLD tests did not establish improvement beyond cash,
and the fixed-holdings Treasury result reversed in 2022–2025. These
findings support caution about treating the tested assets as dependable
protection; they do not rule out benefits in other settings.

## How sure are we

- **Small effects.** The safe-haven gains are a few hundredths of Sharpe, and
  no primary test was significant after Holm adjustment, in the original run
  or the verified rerun.
- **History already seen.** 2010–2020 was examined in earlier runs, so this is
  a careful historical check, not a test on untouched data.
- **Substitute data.** Public prices replace the paper's Datastream series;
  commodity futures prices are not investable returns. The rebuild uses
  price-only bond ETFs, close to the paper's clean prices; the realistic test
  uses total returns, which include coupons.
- **Idealized versus realistic.** The paper-style rebuild uses its zero-fee,
  same-close, full-test-cutoff rules. Only the realistic test describes
  delayed trading with specified fee and cash assumptions; point-in-time
  source vintages and actual execution costs remain unverified.
- **The richer GRU fix was added after we saw the results.** It is reported
  separately and never replaces the original run.
- **Not covered.** The paper's international markets, and forecasts after
  2020 (the volatility archive ends in February 2022).
