# What the project has found

The project's final question and answer, which adapt Kaczmarek et al.
(2022), are in [Final study](final-study.md). This page keeps the earlier
ETF findings that motivate it.

The expanded study is now complete on verified WRDS/CRSP data. Its main
answer is that **none of the 13 additions demonstrated a reliably positive
average contribution versus cash in negative-SPY months after the prescribed
multiple-testing adjustment**. Gold had the most promising observed result,
but substantial uncertainty remains. Bonds' protection changed sharply by
period; growth funds added growth rather than protection on this measure.
The [expanded findings below](#which-additions-earned-their-place-relative-to-cash)
use 2015–2025 and a separate 2007–2025 check. The original 2010–2025 public-data
results remain the motivation and are preserved in the following sections.

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

Eight years is one history, so we rebuilt it 10,000 times from runs of
real trading days averaging 21 days, keeping all three portfolios on the same days.
The mix's Sharpe ratio was 0.020 lower than SPY plus cash's. Across the
rebuilt histories, the middle 95% of that gap ran from −0.215 to +0.171,
and the mix came out ahead in 41% of them. The growth gap, +0.56 points a
year, ranged from −2.57 to +3.91. These eight years cannot tell the two
portfolios apart on risk-adjusted return. That is not the same as showing
they are equal: both ranges include sizeable differences either way. The
mix's higher volatility showed up in every rebuilt history.

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

## What the fund-removal checks add

On October 20, 2026 we recovered the public-data snapshot saved on October 19
and reran all four phases. It reproduces the headline figures above at their
displayed precision, including the bootstrap ranges. This establishes numerical
reproduction from the recovered snapshot; it does not establish that these are
the exact source bytes used for the original write-up.

We then removed each fund in turn, splitting the remaining allocation equally
and keeping monthly rebalancing. The comparison includes the original SPY + cash
portfolio and a separate control fitted to each variant's earlier stock
sensitivity. We tested seven periods and costs of 0, 10, and 25 basis points
per ETF dollar bought or sold. These costs are assumptions. Entry is charged,
exit is not, and taxes are excluded. Cash is still French's RF proxy.

At 10 basis points, the 2018–2025 results are:

| Portfolio | Annual growth | Annual volatility | Largest fall |
|---|---:|---:|---:|
| Original eight funds | 11.43% | 14.93% | −27.91% |
| SPY + cash | 10.90% | 13.31% | −24.02% |
| Without SPY | 11.01% | 14.46% | −27.18% |
| Without QQQ | 10.22% | 14.19% | −28.00% |
| Without VNQ | 12.33% | 14.59% | −26.10% |
| Without GLD | 10.58% | 16.67% | −31.09% |
| Without XLE | 11.48% | 14.34% | −26.72% |
| Without XLY | 11.12% | 14.15% | −27.08% |
| Without VIG | 11.30% | 14.82% | −27.88% |
| Without TLT | 13.23% | 17.26% | −33.06% |

VNQ is the clearest candidate for further investigation: removing it increased
growth and reduced both volatility and the largest loss in all three longer
windows, at all three cost levels. It did not improve every shorter period.
For example, removing it slightly reduced growth and increased volatility in
2018–2019. Its removal also left a deeper worst loss than its own SPY/cash
control, despite a higher historical Sharpe ratio.

TLT illustrates why growth alone is a poor removal rule. Removing it raised
2018–2025 growth by 1.80 percentage points, but increased volatility by 2.33
points and deepened the largest loss by 5.16 points. Gold had a useful role
over the longer windows: removing it reduced growth and increased risk. XLE's
role changed with the period. Removing it helped in 2020–2021 but reduced
annual growth by 3.94 points and deepened the largest loss by 6.87 points in
2022–2023, at the same 10-basis-point cost assumption.

The original mix still had a lower Sharpe ratio and a deeper worst loss than
SPY + cash in each of the 2016–2025, 2018–2025, and 2020–2025 windows, at all
three costs. These windows overlap and are not three independent tests.
The four shorter windows cover 2018–2019, 2020–2021, 2022–2023, and 2024–2025.
The mix's growth advantage over SPY + cash changed sign across those periods.

## Does the commodities addition help?

Before running the extension, we chose one question: does adding DBC improve
protection, especially in 2022–2023, enough to justify its effect on longer-term
growth? DBC provides exposure to commodity futures; see the
[issuer's fund description](https://www.invesco.com/us/en/financial-products/etfs/invesco-db-commodity-index-tracking-fund.html).
The test gives each of the nine funds a one-ninth target weight. This changes
every original fund's weight, so the result describes that allocation change.

At 10 basis points, adding DBC reduced the largest 2022–2023 loss from 19.92%
to 18.16% and raised annual growth from 1.40% to 2.09%. Over 2018–2025,
however, it reduced annual growth from 11.43% to 10.95%. Volatility fell from
14.93% to 14.27%, while Sharpe fell from 0.616 to 0.607. Its largest loss,
27.63%, was still deeper than the common SPY/cash control's 24.02%.
DBC helped in a particular difficult period; this test does not establish a
general improvement over the simpler comparison.

DBC and the original eight funds were downloaded together on October 20, so
both sides of this comparison use the same prices and dates. The common-fund
metrics differ from the recovered snapshot by less than 0.001 percentage
points. Source manifests, full daily returns, and all comparison rows are
retained in `outputs/fund-checks-original` and `outputs/fund-checks-commodities`.
The original reruns are in `outputs/reproduced/phase1` through `phase4`.

## Earlier removal-study follow-up

Keep the original mix, the mix without VNQ, and their SPY/cash controls as a
small research shortlist. Decide in advance how much extra loss would be
acceptable for extra growth, then evaluate later data without retuning the
fund selection. The VNQ result is a hypothesis worth testing, not a confirmed
improvement. All these funds and histories were selected retrospectively;
the removal checks have no multiple-testing adjustment.

`uv run fecon page` puts the main charts and numbers on one page. The full
numbers, definitions, and source records are in the four generated reports
and the phase documents.

## Which additions earned their place relative to cash?

The main study asks what each dollar moved from cash into another ETF
contributed when SPY fell. It uses all 14 funds on the same trading dates,
January 2015–December 2025. There were 132 months, including 40 negative-SPY
months and 12 months in which SPY lost at least 5%. These are historical
descriptions of continuously held portfolios, not signals available before
a decline. All figures below come from the October 20, 2026 WRDS/CRSP snapshot;
cash is compounded French daily RF.

| Candidate | Mean return minus cash in negative-SPY months (percentage points) | Frequency beating cash | Holm result at 5% |
|---|---:|---:|---|
| QQQ | -4.32 | 12.5% | Negative effect |
| IWF | -4.15 | 10.0% | Negative effect |
| SCHG | -4.24 | 10.0% | Negative effect |
| VUG | -4.36 | 10.0% | Negative effect |
| XLK | -4.24 | 10.0% | Negative effect |
| ARKK | -7.56 | 15.0% | Negative effect |
| XLE | -4.97 | 27.5% | Negative effect |
| XLY | -4.56 | 12.5% | Negative effect |
| VNQ | -3.80 | 22.5% | Negative effect |
| VIG | -3.54 | 10.0% | Negative effect |
| TLT | -0.70 | 37.5% | Inconclusive |
| GLD | +0.91 | 45.0% | Inconclusive |
| DBC | -2.43 | 30.0% | Negative effect |

GLD was the only positive observed mean, but its pointwise 95% stationary-
bootstrap interval runs from **-0.46 to +2.33 percentage points**. Its
Holm-adjusted p-value is 0.376. It beat cash in fewer than half the negative
months: positive average contribution does not imply frequent protection.
TLT's interval is -2.99 to +1.75 points, with adjusted p-value 0.522. DBC's
interval is -4.30 to -0.72 points, with adjusted p-value 0.031. The eleven
negative effects surviving Holm adjustment are evidence that those funds
lagged cash on this particular endpoint, not proof that they lack other uses.

The intervals use 10,000 paired draws and a six-month mean block length;
three- and twelve-month sensitivities are saved alongside them. They are
pointwise intervals, not simultaneous protection guarantees. The formal
decision uses the separate full-calendar HAC tests and Holm adjustment.
No candidate has a significantly positive result under that decision rule.

## When protection changed

| Negative-SPY months in each period | Count | TLT minus cash | GLD minus cash | DBC minus cash |
|---|---:|---:|---:|---:|
| 2015–2019 | 15 | +2.10 pp | +2.26 pp | -3.11 pp |
| 2020–2021 | 8 | +1.71 pp | -1.10 pp | -5.01 pp |
| 2022–2025 | 17 | -4.31 pp | +0.67 pp | -0.61 pp |

These smaller samples are descriptive. TLT's sharp change is consistent with
protection depending on market conditions, but this comparison does not prove
that inflation or interest rates caused it. XLE and DBC helped over the fixed
2022 SPY peak-to-trough episode, while performing poorly in many other stock
declines. A favorable episode is not the same as a positive average across
all negative months. The episode heatmap includes every candidate and every
qualifying SPY drawdown, without selecting only helpful episodes.

The separate 2007–2025 sample contains 228 months, with 76 negative-SPY months
and 24 severe months. Its negative-month means are **+0.68 pp for TLT,
+0.75 pp for GLD, and -2.76 pp for DBC**. In severe months the corresponding
means are +1.65, +1.51, and -5.27 pp. This longer sample includes the financial
crisis and changes the description of TLT. It is not part of the 13-test
primary family and is not directly comparable to a different fund's shorter
sample as though the dates were the same.

## What growth and risk came with the additions?

Portfolio weights are fixed at 90/10, 80/20, and 60/40 and reset monthly.
Before costs, the monthly advantage over the matching SPY/cash control is
the candidate weight times its return minus cash. Thus GLD's +0.91 pp
negative-month mean becomes about +0.09, +0.18, or +0.36 pp at the three
weights. The larger effects are allocation scaling, not three discoveries.

At the **10-basis-point trading-cost scenario**, 90% SPY / 10% GLD grew by
13.50% annually versus 12.36% for 90% SPY / 10% cash. Its volatility was
16.08% versus 15.95%, and its maximum drawdown was slightly deeper:
30.93% versus 30.62%. At 60% SPY / 40% GLD, growth was 13.40% versus 9.02%
for the matching cash control, but volatility rose from 10.53% to 12.30%
and the worst loss deepened from 20.99% to 22.74%. Gold's growth benefit
in this sample did not mean it beat cash on every risk measure.

The growth/technology additions also raised long-term growth relative to
their matching cash controls, together with greater variability. Their poor
negative-month contribution does not make them automatic failures: it means
their role here was growth exposure rather than protection relative to cash.

The three supporting fixed mixes show the same distinction. All use the
same 2015–2025 dates, monthly rebalancing, and 10-basis-point cost scenario:

| Fixed portfolio | CAGR | Annual volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| 100% SPY reference | 13.45% | 17.79% | 0.686 | -33.70% |
| Original eight, equal weight | 10.51% | 13.56% | 0.658 | -27.90% |
| Jake's twelve, equal weight | 13.77% | 17.77% | 0.703 | -32.42% |
| Combined fourteen, equal weight | 12.67% | 16.10% | 0.697 | -29.46% |

Jake's expanded mix had more growth than the original mix, alongside more
variability and a deeper worst loss. These fixed-list comparisons do not
establish that every addition independently improved the portfolio. Adding
more tickers also changes every equal weight and can add overlapping equity
exposure. Full results at zero and 25 basis points remain in the tables.

The supplementary stock-risk check trained beta on 2015–2019 and evaluated
2020–2025. **24 of 42 candidate/fixed-mix controls required clipping** to the
allowed 0–100% SPY range. Training betas ranged from 0.464 to 1.105; realized
portfolio betas in evaluation ranged from 0.547 to 1.257. Consequently these
controls do not all match the portfolios' realized stock exposure, let alone
every form of risk. Their estimates, targets, realized exposures, and results
are retained rather than concealed behind a generic “risk matched” label.

## Evidence, limits, and the next decision

Both samples passed exact daily coverage and CRSP daily/monthly reconciliation.
The maximum differences were 0.0555 basis points for the main sample and
0.0433 for the longer sample, below the one-basis-point threshold. No public
fallback or provider splice was used. Historical identifiers, ETF classifications,
duplicates, flags, units, and source hashes were checked. Two existing
alignment bugs were repaired: cash dates must match portfolio dates, and
contribution weights are aligned by both date and asset.

The independent verifier recomputes all saved main performance metrics,
conditional means, zero-cost identities, and figure values. Its separate
matrix implementation agrees with the HAC covariance and Holm correction.
Tests also cover bootstrap pairing/dependence and fixed-seed reproducibility,
source failures, no partial months, undefined estimates, and fee accounting.
See `scripts/verify_protection.py` and the commands in the README.

The main evidence is in `outputs/protection-main-final`, with the separate
longer check in `outputs/protection-long-final`. Each contains three PNG
figures, their source tables, the full comparison tables, and `run.json`.
Source extracts and detailed paths stay private and outside Git. The earlier
original-eight outputs remain in `outputs/reproduced` and are reproducible
with the original `configs/study.toml` and its saved public snapshot.

For the paper and professor discussion, lead with the distinction between
**growth, protection relative to cash, and protection that changes by period**.
Use the full 13-fund table and uncertainty figure, then the trade-off and
decline figures. Do not present gold as a proven winner or a larger bootstrap
count as new historical evidence. The next prospective step is to keep these
rules fixed and evaluate later observations without changing the fund list
to fit them. Optimization, annual winner-picking, individual stocks, and
causal macroeconomic modeling remain outside this course-project scope.
