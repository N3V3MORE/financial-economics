# What the project has found

The eight-fund mix was calmer than SPY over the historical sample. Phase 3 shows
why that alone is not enough to call it a better portfolio: holding some cash
also reduces stock-market risk, and that simpler alternative performed well.

These numbers describe the retained snapshot. Reproduce the supporting reports
with `fecon analyze`, `fecon stress`, and `fecon compare`.

## How the three phases connect

Phase 1 compared eight ETFs and a portfolio holding equal amounts of each,
reset monthly. Over 2010–2025, the mix grew by 11.62% a year, compared with
14.05% for SPY. Its annual volatility was lower: 13.05% versus 17.21%.
Volatility measures how much returns vary, so the mix offered a smoother path
with less growth in this sample.

Phase 2 looked at market declines. Across all eight qualifying SPY downturns,
the mix lost less over the same dates. It recovered sooner in seven and on
the same date in one. This strengthens the description of a calmer portfolio,
but leaves an important question: how much came from diversification, and how
much came from simply holding less stock-market risk?

## What Phase 3 changes

We added a simple comparison portfolio holding SPY and cash. We used only
2010–2017 to estimate how strongly the mix moved with SPY after subtracting
the risk-free return. A regression with an intercept gave a stock sensitivity,
or beta, of about 0.69. Roughly, a 1% move in SPY's excess return was associated
with a 0.69% move in the mix's excess return, on average in that period.

That set the comparison target at **69.19% SPY and 30.81% cash**. Cash earns
Kenneth French's daily risk-free return; it represents a cash proxy rather than
a specific traded fund. Both portfolios reset to their targets monthly, with
holdings drifting between resets. We fixed the control's target before
evaluating 2018–2025.

| 2018–2025 | Eight-fund mix | SPY + cash | SPY |
|---|---:|---:|---:|
| $1,000 grows to | $2,380 | $2,286 | $2,895 |
| Compounded annual growth | 11.48% | 10.92% | 14.25% |
| Annual volatility | 14.93% | 13.31% | 19.46% |
| Sharpe ratio | 0.619 | 0.639 | 0.641 |
| Largest fall from a prior peak | −27.90% | −24.02% | −33.72% |

The mix earned 0.56 percentage points more per year than SPY plus cash, but
its returns varied more and its largest fall was deeper. Its Sharpe ratio,
which measures average excess return relative to excess-return variability,
was slightly lower. We have not tested whether these differences are
statistically significant.

The downturn comparison tells a similar story. The mix lost more than SPY
plus cash in all five evaluation-period downturns. For example, from the
February 19, 2020 SPY peak to the March 23 trough, the mix lost 27.85% and
SPY plus cash lost 24.02%. The mix subsequently regained its starting value
six trading days sooner. Across the five episodes it recovered sooner in
four, but took 104 more trading days in the 2022 episode. Recovery is measured
from the common SPY trough to each portfolio's own value at the common SPY peak.

## What this means for the project

Our evidence supports a narrower conclusion: **this mix reduced risk relative
to SPY, but did not clearly improve on the simpler SPY/cash comparison.** Its
extra growth came with extra risk. A smaller loss than SPY does not by itself
show that eight funds are worth the added complexity.

The comparison is approximate. During evaluation, the mix's realized SPY beta
was 0.73, compared with 0.68 for the control. Their stock sensitivity did not
stay matched, and beta does not measure every risk carried by bonds, gold,
real estate, or sector funds. We therefore cannot attribute the remaining
performance difference entirely to diversification, or conclude that
diversification generally fails.

We already inspected the full dataset and chose surviving funds. This is a
chronological historical evaluation, not a test on untouched future data.
Trading costs and taxes are zero. Phase 1 and Phase 3 also cover different
dates, so their annual growth rates should not be compared as if they were
alternative estimates for the same period.

## How sure are we, and why? (Phase 4)

Eight years is one history, so we rebuilt it 10,000 times from month-long
runs of real trading days, keeping all three portfolios on the same days.
The mix's Sharpe ratio was 0.020 lower than SPY plus cash's. Across the
rebuilt histories, the middle 95% of that gap ran from −0.215 to +0.171,
and the mix came out ahead in 41% of them. The growth gap, +0.56 points a
year, ranged from −2.57 to +3.91. These eight years cannot tell the two
portfolios apart on risk-adjusted return. The mix's higher volatility
showed up in every rebuilt history.

We also measured each fund's gain or loss in every market fall, per $1 in
the mix at the SPY peak. The amounts add up to the mix's loss. Two funds
swapped roles. In the 2020 fall, long-term Treasuries (TLT) added 1.6
points and energy (XLE) lost 7.5, the largest loss of any fund. In the 2022
fall, TLT lost 3.9 points and XLE gained 5.0, the only fund to gain. The
SPY/TLT daily correlation was negative every year from 2010 to 2021 and
slightly positive from 2022 to 2025. Research on stock–bond correlations
links this to high inflation and tight monetary policy. The bond cushion
that helped in 2020 was gone in 2022, which helps explain the mix's slower
2022 recovery.

## What to do next

Phase 4 explained the contributions; see [Phase 4](phase-4.md).

Next, check whether the conclusion holds up under reasonable changes:
other training and evaluation years, dropping one fund at a time, and a
simple allowance for trading costs. These are checks on data we have
already looked at, not fresh proof.

Only after that is it worth changing the weights. Optimizing now would
likely find weights that fit this particular history. First decide what
new weights are meant to fix and how they will be tested.

The film follows the same story. The numbers, definitions, and source
records are in the four generated reports and the phase documents.
