# Final study: does volatility timing with "safe" assets protect stock investors?

The project's final research goal, and what the evidence shows. It links the
original ETF study to the paper we adapted:
Kaczmarek, Będowska-Sójka, Grobelny and Perez (2022), “False Safe Haven
Assets: Evidence From the Target Volatility Strategy Based on Recurrent
Neural Network” ([paper](https://doi.org/10.1016/j.ribaf.2021.101610)).

Sections 1–3 report the original reconstruction and realistic test. A later
methodology audit led to the matched revision (`fecon paper-revised`), a new
exploratory calculation on history that had already been inspected. Its results
follow under "Changes after cross-checking the paper".

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

**Verification.** On October 21, 2026 every stage was rerun
from the saved data snapshots on the final code (`outputs/rerun-20261021`).
That code includes the corrected inference: centered Sharpe tests, a
six-test Holm family that keeps unavailable tests, and input checks on saved
forecasts. All five GRUs were retrained from scratch with cuDNN. **No headline
figure and no Holm decision changed.** The largest Sharpe change in the
realistic test was 0.00006, and one primary p-value moved from 0.9781 to 0.9783.
GRU forecasts differ slightly from the earlier native-kernel runs. Both sets are
saved. A rebuild from fresh Yahoo/WRDS/EIA inputs on an AMD machine
was stopped on October 21 and is not used here. Its status file,
`outputs/paper-wrds-eia-rocm-20261021/status.json`, is on that machine only.

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

### 2. Timing reduced maximum drawdown and COVID loss without an established Sharpe improvement

Realistic test, 2011–2020, 0.1% fees, cash earning interest:

| Portfolio | Annual growth | Sharpe | Worst fall | COVID fall | 2015–16 fall |
|---|---:|---:|---:|---:|---:|
| SPY | 13.8% | 0.80 | −33.7% | −33.7% | −13.0% |
| ARFIMA timing | 7.4% | 0.76 | −14.5% | −9.7% | −11.6% |
| Fixed mix, 2010-calibrated stock target (56%) | 8.0% | 0.80 | −19.6% | −19.6% | −7.3% |
| One-input GRU timing | 8.1% | 0.80 | −14.5% | −10.2% | −11.8% |
| Fixed mix, 2010-calibrated stock target (59%) | 8.4% | 0.80 | −20.5% | −20.5% | −7.7% |

Worst fall is each portfolio's maximum drawdown over 2011–2020; COVID fall
is the common February 19 – March 23, 2020 window. The two timing portfolios
shown reached their worst drawdown from December 5, 2014 to February 11,
2016; their fixed controls reached theirs during COVID. These comparisons
overlap and are not independent decline episodes.

**Main result: timing held more stock, with similar or lower risk.** Each
fixed mix holds its model's mean stock target over calibration year 2010,
reset monthly. It was set before 2011 and is not matched to 2011–2020
exposure. Because 2010 was volatile, timing in fact held *more* stock
afterwards:

| 2011–2020, 0.1% fees | Average stock share | Volatility | Beta to SPY |
|---|---:|---:|---:|
| ARFIMA timing | 75% | 9.2% | 0.46 |
| ARFIMA fixed mix | 56% | 9.4% | 0.55 |
| One-input GRU timing | 79% | 9.6% | 0.48 |
| One-input GRU fixed mix | 59% | 9.9% | 0.58 |

Timing held substantially more stock on average, with similar volatility,
lower beta and smaller maximum drawdown and COVID loss. All seven models
show those comparisons. Average stock share counts calm and turbulent days
equally, while risk comes mostly from days
with large market moves. The results are consistent with holding more stock
in calm periods and less when forecast risk rose.

Protection varied across the other declines. Timing lost more than its
fixed mix for all seven models in 2015–16, four in 2011 and three in 2018.
For example, ARFIMA lost 11.8% in 2018 against its fixed mix's 10.7%.

This is evidence of effective risk allocation in this sample. It is not
evidence that the forecasts predicted whether stocks would rise or fall
(they forecast volatility, not direction), and these figures carry no
uncertainty estimates, so they do not show the advantage is statistically
reliable or will persist. The exposure check was added after the results
were seen and is descriptive.

- **No established Sharpe improvement.** Timing minus the fixed mix is
  −0.04 (ARFIMA) and −0.005 (GRU); neither difference was significant. This
  does not establish equal performance.
- **The assumed trading costs erased the observed advantage.** Timing trades
  about 116 times its own value over ten years. ARFIMA's Sharpe was 0.89
  against 0.80 before fees, 0.76 against 0.80 at 0.1%, and 0.57 against 0.80
  at 0.25%. The fees are stated scenarios, not measured costs, and the
  before-fee advantage was not itself tested for significance.
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
realistic test above. The extension refits for a two-session target; the
realistic test delays trades on saved one-session forecasts. The extension
assigned annual cutoffs by issue-date blocks starting at the preceding
year-end close, while the realistic test assigned them by return year.
The extension starts in cash; the realistic test can use an earlier signal
on its first evaluation day. The reported histories include these boundary
differences. New extension runs share the realistic module's
earned-return-year cutoff rule; the matched revision also standardizes
the initial cash day for strategies, static controls and benchmark:

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

**2007–2025: average return minus cash in months when SPY fell (fixed holdings):**

| Period | Falling months | TLT | GLD |
|---|---:|---:|---:|
| 2007–2009 | 17 | +0.64 pp | +1.41 pp |
| 2010–2014 | 19 | +3.64 pp | −0.18 pp |
| 2015–2019 | 15 | +2.10 pp | +2.26 pp |
| 2020–2021 | 8 | +1.71 pp | −1.10 pp |
| 2022–2025 | 17 | **−4.31 pp** | +0.67 pp |

The paper's favorable result for long Treasuries came from an earlier
period. In this fixed-holdings check, Treasuries beat cash on average in
each shown pre-2022 subperiod, with a smaller +0.64-point cushion in
2007–2009. Their average return relative to cash in falling-stock months
changed sign in 2022–2025. Gold's result also varied by period. These are
descriptions of the examined periods, not permanent properties of either asset.

**Keep this separate from the timing strategy.** These rows are fixed
holdings in months when SPY fell, not the paper's forecast-driven strategy.
The Oxford-Man archive ends in February 2022, so the Oxford-based strategy is
not tested after 2020 (2021 in the extension). Later timing evidence comes from
two other studies. The Risk Lab 2022–2025 study runs the full strategy on Risk
Lab volatility ([findings](findings-risklab.md)): timing had a smaller worst
fall than its fixed mix, long Treasuries were the worst storm-day asset, no
portfolio test was significant, and the sign of timing's Sharpe advantage
depends on how missing volatility dates are handled. Under the strict rule a
data gap left timing in cash through January 2022, when SPY fell 5.9%; that
month explains most of its Sharpe lead. The early prediction
study (squared daily returns, 2017–2025) also covers these years, with a
rougher setup. We can say TLT's relative performance in falling-stock months
reversed; neither study shows that the paper's complete strategy reliably
failed or succeeded in 2022–2025.

## Changes after cross-checking the paper

The matched revision is recorded in
`outputs/revised-study-20261021/protocol.json`, alongside the original caches,
configurations and results. It uses the same Oxford BV definition and verified
ETF total returns, and it does not splice another volatility measure onto the
archive.

| Issue | Treatment in the revised calculation |
|---|---|
| Paper's 22-times volatility conversion | Conventional 10% annual equity risk budget, using sqrt(252) |
| Negative GRU variance predictions | Both GRUs learn log BV and produce positive forecasts with train-only smearing |
| Different training histories | Both use identical dated histories beginning October 3, 2000; no artificial 2002 cutoff |
| Same validation used for stopping and scoring | Separate chronological stopping and selection blocks, purged by label availability |
| Delayed one-step predictions | Every model targets the actual t+2 earned return; GRUs train directly for that horizon |
| Missing simple comparator | EWMA plus ARFIMA, GARCH and GJR on identical forecast dates |
| Assumed same-day external input release | Four external features lag one market session; historical vintages remain unverified |
| Future-informed stress thresholds | Past-only annual cutoffs, shared with the realistic module and assigned by earned-return year |
| Different initial trading conventions | Every portfolio starts in cash on the first evaluation day |
| Uncharged final sale or benchmark costs | Final liquidation uses actual drifted holdings; the buy-and-hold comparator also pays entry and exit costs |
| One assumed execution price | Fixed 0/1/2/5/10/25-bps scenarios, with 10 bps as the main scenario |
| Changes in down-month classification with costs | Falling months are defined by gross underlying stock returns |
| Multiple inspected variants | One Holm family for 27 paired portfolio comparisons; all results labelled exploratory |

Holding the stock target fixed when adding TLT, GLD or DBC is deliberate:
it isolates the contribution of replacing cash. The 10% budget therefore
describes the equity allocation, not a guaranteed total-portfolio volatility.
Realized volatility and its ratio to that budget are reported for every path.
Static stock/cash controls hold each model's mean calibration-year target.

The frozen run records source/configuration/code hashes and rejects changed
inputs, mixed stage identities, overwritten outputs and unequal GRU folds.
It cannot repair repeated inspection of the same history. Taxes, actual
execution spreads, publication-time vintages and a current validated volatility
source remain unresolved. A 2022–2025 timing conclusion still cannot be drawn
from the fixed-holdings check above.

### Results of the matched revision

The completed run covers all 2,517 market days in 2011–2020. Both GRUs share
5,074 dated input windows, identical pre-2010 selection folds and the same
annual refit dates. Every model forecasts t+2 directly or recursively;
missing signals retain holdings rather than remove return days. Results
below are for stock/cash timing and each model's monthly-reset static mix.
Ten bps means 0.1% of each ETF dollar bought or sold, including entry and
final liquidation.

| Model | Timing Sharpe, no fees | Timing Sharpe, 10 bps | Static Sharpe, 10 bps | Timing maximum drawdown, 10 bps | Static maximum drawdown, 10 bps |
|---|---:|---:|---:|---:|---:|
| EWMA | 0.861 | 0.843 | 0.796 | −16.04% | −27.18% |
| ARFIMA | 0.883 | 0.831 | 0.795 | −15.47% | −25.33% |
| GARCH | 0.857 | 0.756 | 0.795 | −14.14% | −21.02% |
| GJR | 0.855 | 0.748 | 0.795 | −14.74% | −22.23% |
| One-input log GRU | 0.885 | 0.827 | 0.796 | −15.66% | −27.41% |
| Richer log GRU | 0.892 | 0.841 | 0.796 | −15.09% | −27.37% |

The richer GRU leads before fees, but EWMA narrowly leads at 10 bps.
EWMA trades about 20.6 times portfolio value over the decade, compared with
68.0 for the one-input GRU and 59.5 for the richer GRU. These are sums of
purchases and sales relative to each day's wealth, including liquidation.
At 25 bps their respective Sharpes are 0.817, 0.740 and 0.766. Costs can
therefore change the ordering even when the GRUs forecast better.

Measured annual-volatility MSE is 30.5% lower for the one-input GRU and
25.4% lower for the richer GRU than for EWMA, on 2,499 common observed labels.
Neither GRU's forecast advantage over EWMA survives the forecast comparison
adjustment. No main portfolio comparison survives the joint 27-test Holm
adjustment either. All 27 are available; none establishes an improvement.
These findings do not establish equivalence. The annual-volatility errors
also have different units and a different horizon from the paper-style
monthly errors above, so their numerical levels should not be compared.

After 10-bps costs, annual returns are 10.39%, 10.05% and 10.29% for EWMA,
the one-input GRU and the richer GRU; realized annual volatility is 11.89%,
11.73% and 11.80%. The nominal 10% equity risk budget is therefore not a
realized-risk guarantee, even before adding a defensive asset. SPY earns
13.64% annually with 17.19% volatility, Sharpe 0.797 and a 33.70% maximum
drawdown under the same entry/exit convention.

Every timing model has a smaller maximum drawdown and COVID loss than its
static mix. Timing nevertheless loses more in 2015–16 for all six models,
in 2011 for four and in 2018 for three. Adding gold to EWMA raises full-period
Sharpe from 0.843 to 0.893 while worsening maximum drawdown from 16.04% to
16.67%: average risk-adjusted performance and the single worst peak-to-trough
loss measure different things. This is not evidence of reliable crash
protection.

Saved tables, forecasts, decisions and model checkpoints are under
`outputs/revised-study-20261021`. At verification, all 87 saved stage files
matched their recorded hashes and the calculation code matched the frozen
protocol. That revision passed all 369 tests, including actual GPU training.
The October 22 calendar changes require new runs and do not replace these results.
These are revised exploratory results on previously examined history. The
original results below are a separate experiment.

### Archived Oxford-like estimates and calendar-month pilot, October 22

A separate approximation uses public
SPY intraday volatility measures, with coefficients fitted on 2000–2009 and
method selection on 2010–2016. A daily SPX range model, selected on the same
earlier dates, supplies explicitly flagged estimates when the intraday source
is unavailable. It does not fill raw prices or create closed-market days.
This gap handling was added after the initial source inspection; the complete
comparison is exploratory, although no final-period labels fit its parameters.

| Comparison period | Days scored | Mean volatility error | Median volatility error |
|---|---:|---:|---:|
| Final period, 2017–February 25, 2022 | 1,285 | 6.7% | 5.8% |
| 2021–February 25, 2022 | 287 | 7.6% | 6.9% |
| COVID decline, February 19–March 23, 2020 | 24 | 7.5% | 6.1% |
| February 5–9, 2018, daily-range fallback | 5 | 32.4% | 34.8% |

The user's 90% criterion is applied to volatility levels as 100% minus mean
absolute percentage error: the final score is 93.3%. It is not a percentage
of correct days: 77.9% fall within 10%, and the 95th-percentile error is 15.5%.
Average raw BV error is 13.2%, so the raw-variance field does not reach 90%
under that definition. The high-stress subset has 8.5% mean volatility error,
but the weak 2018 fallback prevents a claim of uniformly accurate crash data.

The full estimated series has 7,744 observations through October 9, 2026;
33 use the daily-price fallback. All 1,003 market dates in 2022–2025 have
estimates, with 15 fallback dates. Outputs and fitted coefficients are under
`outputs/volatility-proxy-complete-20261022`. They remain labelled estimates.
Accuracy after the last Oxford observation is unobservable without another
reference source, and actual historical publication timing is unverified.

The archived later-year pilot used observations dated from t−29 through t,
a 30-calendar-day window. Internal padding was ignored in scaling and both
GRU layers. At that stage, 384 tests passed with warnings treated as errors,
including actual packed GRU training on NVIDIA. The calendar-day change has
since been reverted: current runs again use 22 trading observations. The
annual conversion remains sqrt(252), and the target session still ends t+2.

The separate pilot in `outputs/proxy-study-20261022` trains the one-input
GRU on the estimated history, with five annual refits from 2021 through 2025.
One previously selected 32/32 profile (no new width search) and three seeds are fixed; pre-2021 validation
selects 10 epochs within a maximum of 12. Calendar year 2021 calibrates the
stock/cash controls, and investment performance is scored on actual verified
ETF total returns for all 1,003 market days in 2022–2025. The richer GRU is
excluded because its external-input snapshot ends in 2020.

| Stock/cash strategy, 2022–2025 | Annual return, 10 bps | Annual volatility, 10 bps | Sharpe, 10 bps | Maximum drawdown, 10 bps |
|---|---:|---:|---:|---:|
| SPY benchmark | 10.84% | 18.01% | 0.434 | −24.57% |
| EWMA timing | 7.87% | 12.84% | 0.335 | −20.59% |
| EWMA static control | 10.50% | 16.90% | 0.433 | −23.17% |
| One-input GRU timing | 7.89% | 12.62% | 0.340 | −21.27% |
| GRU static control | 10.50% | 16.90% | 0.433 | −23.18% |

Before fees, EWMA and GRU timing Sharpe is 0.358 and 0.420; at 25 bps it
is 0.302 and 0.220. The GRU trades about 40.1 times portfolio value over the
four years, versus EWMA's 11.5. The three paired Sharpe tests share a Holm
adjustment and establish no improvement. Their adjusted p-values are all 1.
The 2021-calibrated static controls hold about 94% stock. Timing has smaller
maximum drawdown, but does not beat either static control on Sharpe; its
realized annual volatility also exceeds the nominal 10% equity budget.

The 2022-only annualized returns are −18.79% for the benchmark, −17.89% for
EWMA timing and −18.29% for GRU timing, versus about −17.58% for either static
control. Over 2023–2025, timing Sharpe is 1.020/1.055, versus about 1.095 for
the static controls. These slices describe the continuously held paths;
final liquidation is charged only at the last 2025 close.

Both archived models forecast the actual t+2 earned return's volatility. The 30-day
histories in these saved forecasts contain 18–22 actual observations, with
no padded value entering either GRU layer. All forecasts are positive, all
training labels precede their refit dates, and the recorded backend is
NVIDIA CUDA/cuDNN. At completion, all 29 stage files/checkpoints matched their
hashes and the calculation code matched the frozen protocol. These historical comparisons
remain exploratory. The 93.3% source score does not describe forecast accuracy
or prove that the measurement was available before a historical trade.

### Requested 22-session rollback and stronger approximation check

The active revised study and proxy study now use the last **22 trading
observations**. The calendar-window helper and its tests have been removed.
Saved experiments keep their recorded settings. The 2022–2025 pilot
above used 30 calendar days and has not been rerun with 22 sessions while the
substitute fails the new quality rule.

The new approximation adds public Risk Lab IVV and VOO measurements to SPY.
Each follows the S&P 500 but remains an ETF substitute for the index.
An ETF's inputs are flagged unavailable when its five-minute trade/quote
volatility ratio is outside 1/1.5 to 1.5. The date stays in the evaluation,
using another estimator or an explicitly flagged daily-price fallback.
The 78 candidates include log-linear/ridge mappings, quadratic mappings,
positive-weight combinations and medians. Calibration ends in 2013 and
selection uses 2014–2016; earlier history must also reach 90% average
accuracy before a candidate enters the ranking. Selection minimizes the
worst observed error across those two earlier periods. The selected model
is a ridge mapping using all three ETFs and daily price features.

The source snapshots, protocol, coefficients, per-date estimates and failures
are saved in `outputs/volatility-proxy-final-20261022`. Two earlier failed
trials and the intermediate robust result remain separate. Those trials
informed quality fixes, so the final comparison is exploratory. No
2017–2022 labels fit coefficients or determine the recorded selection rank,
but this is not an untouched test. The longer calibration and additional
inputs change together; their separate contributions are unmeasured.

| 2017–February 25, 2022; 1,285 days | Original substitute | Revised substitute |
|---|---:|---:|
| Average volatility accuracy, 1 minus mean percentage error | 93.31% | 93.65% |
| Mean volatility error | 6.69% | 6.35% |
| Dates reaching 85% individual accuracy | 1,210 / 1,285 | 1,209 / 1,285 |
| Worst individual accuracy | 35.04% | 31.98% |
| February 2018 crash-week mean error | 32.41% | 7.51% |
| COVID decline mean error | 7.49% | 5.93% |

The new candidate improves average error and the two displayed crash slices,
but does **not** improve the number of dates meeting the 85% floor or the
worst later date. The remaining worst later date is November 23, 2018;
December 27, 2021 still needs a daily-price fallback. Some 2000–2013 errors
exceed 100%, making the complete training history unsuitable under the
requested floor too. All 5,552 Oxford dates remain in the comparison.
The later-period median error is 5.11%, 80.16% of dates are within 10%,
and raw BV mean error is 12.83%. The full estimated export has 7,743 dates,
including all 1,003 dates in 2022–2025; all fields remain labelled estimates.

The requested gate requires **90% average and 85% individual accuracy**, with
complete coverage, separately for calibration, selection and the final
period. The final candidate fails. Its command returns a failure status,
and the study loader rejects it and legacy average-only approvals. It has
not been used in a new training or investment test. This threshold concerns
historical measurement agreement, not future forecast success. Direct
post-February-2022 agreement with Oxford remains unobservable. Improving
the worst dates further likely requires a sufficiently complete intraday
price history for the index and a matched reconstruction of its measure;
the summary inputs tested here have not met that standard.

Verification: 392 tests passed with warnings treated as errors, including
actual matched GRU training on NVIDIA CUDA; lint passed. Regression checks
cover the 22-session window, both acceptance thresholds, earlier-history
failures, source identities and diagnostic-only exports.

## The answer from the original realistic test

**In these retrospective tests, volatility timing reduced maximum drawdown
and COVID loss but did not establish a Sharpe improvement over the
2010-calibrated static mixes after 0.1% fees.** The nonsignificant tests
do not establish equivalence. Timing held more stock on average than its
fixed mix (75% against 56% for ARFIMA) with similar volatility, lower beta
and smaller maximum drawdown and COVID loss: effective risk allocation in
this sample, whose before-fee advantage the assumed trading costs erased.
Defensive-asset effects varied by model and period:
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
- **Validation reused in the original models.** GRU early stopping and configuration
  scoring use the same validation folds. These scores can be optimistic,
  and the effect on model rankings has not been measured with separate
  selection data in those archived results. The matched revision separates
  stopping and selection blocks. Evaluation dates remain separate from tuning.
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
