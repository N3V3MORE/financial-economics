# Phase 2: diversification during market declines

## Objective

Ask whether the existing equal-weight mix held up better than SPY during major declines, and whether gold and long-duration bonds moved differently from equities as market conditions changed.

## Scope

Use the retained Phase 1 daily returns and the same monthly portfolio. Select every non-overlapping SPY episode reaching a 10% decline from a previous peak. Compare losses over identical dates, measure recovery from the common SPY trough, and plot correlations using 63 trailing daily observations.

There is no new fund selection, optimized weighting, transaction-cost model, or out-of-sample trading test in this phase. The selected funds, dates, zero-cost assumption, and monthly rebalancing rule remain the Phase 1 baseline.

## Method

1. Compound SPY's daily returns, including the actual pre-sample closing date as the initial capital anchor. Track each peak until the first close that regains it. Keep an episode if its lowest value is at least 10% below that peak. A final episode without recovery ends at the sample's last close. Smaller declines within the same unfinished episode are not counted again.
2. Use the SPY peak and trough as common dates for SPY and the equal-weight mix. Report the compounded return between those closes and the lowest value reached within that window, including the starting capital. These are losses from a common starting date, not independently timed maximum drawdowns.
3. Starting at the SPY trough, find each portfolio's first close at or above its own wealth on the common SPY peak date. Report elapsed trading observations and calendar days. If it is already at or above that value at the trough, elapsed time is zero. If it has not recovered by the sample end, its recovery date and duration are missing; follow-up time is recorded separately. Recovery equality permits a 1e-12 relative tolerance in wealth levels to avoid delays caused only by numerical rounding.
4. Calculate each non-SPY asset's Pearson correlation with SPY over the latest 63 daily returns, including the current date. Early incomplete windows are missing. Undefined correlations, such as a window with constant returns, remain missing.

The equal-weight mix uses its continuous full-sample history. It is not reset to equal weights at an episode peak. Rebasing a chart to a common starting value changes the display, not the portfolio.

The 10% threshold defines the historical sample; it is not an entry signal. Peaks, troughs, and recovery dates use later observations and are known only afterward. Sixty-three daily observations approximate a quarter; they are not a fixed calendar quarter. Overlapping windows are descriptive rather than independent statistical evidence.

## Milestones

1. **Episode calculations:** test selection, common-window returns, recovery dates, and unfinished episodes on small synthetic paths.
2. **Outputs:** add an offline command, inspectable CSVs, same-date loss and recovery figures, rolling correlations, and a plain-language report with recorded settings and definitions.
3. **Verification:** run the retained real snapshot, inspect figures, independently check selected episodes, and repeat the analysis offline to check reproducibility.

## Done when

- One documented command reads the hash-checked snapshot and writes `outputs/phase2/summary.md` without contacting data providers.
- Every qualifying benchmark episode is included once, and the loss comparisons use common dates.
- Missing recovery dates cannot be confused with immediate recovery.
- Tests cover threshold selection, compounding, the initial anchor, recovery timing, unfinished episodes, and trailing-window alignment.
- The saved report states what diversification did during these historical episodes without claiming future protection, causation, or an investable strategy.
- Generated figures are inspected and saved numerical outputs reproduce in the locked environment.

## Run

Generate the Phase 1 snapshot first, if needed. Then:

```sh
uv run fecon stress
```

The defaults are equivalent to:

```sh
uv run fecon stress --data-dir data/snapshot --output-dir outputs/phase2 --drawdown-threshold 0.10 --rolling-window 63
```

To inspect a different descriptive threshold or window, write to a separate output directory and report that choice. Do not choose parameters only because they make the mix look better.

Outputs are `episodes.csv`, `episode_comparison.csv`, `rolling_correlations.csv`, `summary.md`, and `run.json`, plus three figures when qualifying episodes exist. If no episode reaches the threshold, the report says so and omits the two episode figures. The correlation figure is still produced. Full definitions and sample bounds are recorded in `run.json`.

## Status

Implemented and locally verified on October 15, 2026:

- All 63 synthetic and workflow tests passed with warnings treated as errors.
- The retained, hash-checked snapshot produced eight qualifying SPY episodes and sixteen comparison rows across 4,024 daily observations.
- An independent reconstruction from the retained adjusted prices matched the monthly portfolio, all episode dates and losses, recovery dates and durations, and censoring. Checks using `numpy.corrcoef` agreed with 109 selected trailing windows across all eight correlation series to within 6e-13.
- An offline repeat reproduced all three CSVs, run metadata, and the summary byte for byte in the locked environment.
- All three figures were inspected for readable dates, units, legends, and clipping.

The mix lost less over all eight common peak-to-trough windows. It regained the common starting value sooner in seven episodes and on the same date in the 2022 episode. These results do not isolate diversification from differences in market exposure, and do not estimate future protection.

Verification was local on Windows. The branch has not been pushed, so hosted checks have not run for these changes.

## After Phase 2

[Phase 3](phase-3.md) compares the existing mix with a SPY/cash portfolio whose target is calibrated using 2010–2017 and frozen for evaluation over 2018–2025. It reuses this phase's 10% SPY episode and recovery definitions. Because the dataset and funds had already been inspected, the comparison is a chronological historical evaluation. Trading costs and taxes remain zero for this controlled comparison; optimization and cost modelling remain later work.
