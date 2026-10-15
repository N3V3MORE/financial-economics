# Phase 4: how sure are we, and why did it happen?

## Question

Phase 3 found no clear advantage for the eight-fund mix over a SPY/cash control during 2018–2025. Phase 4 asks two follow-up questions about the same dates and portfolios:

1. **How sure are we?** Could the observed differences easily have come out the other way?
2. **Why did it happen?** Which funds produced the mix's gains and losses, especially in market falls?

Nothing from Phases 1–3 changes. The snapshot, the mix, the frozen 69.19% SPY target, and the evaluation dates are reused exactly. This phase adds no optimization, fund changes, cost model, or new data.

## Uncertainty ranges

Eight years is one history. To judge how much the comparison depends on its particular sequence of days, rebuild that history many times from blocks of real evaluation days, then recompute the comparison each time.

The method is the stationary bootstrap of Politis and Romano (1994):

- Each rebuilt history has the same length as the evaluation period, 2,011 days by default.
- It starts on a random day, then follows actual consecutive days. Each day there is a 1-in-21 chance of jumping to a new random day instead, so blocks average about one month. Blocks wrap from the last day to the first.
- Every portfolio and the risk-free rate use the same drawn days. Shared market moves therefore stay paired, and the comparison is like-for-like.
- Blocks keep short-run dependence and volatility clustering that resampling single days would destroy.

For each rebuilt history, calculate the mix-minus-reference difference in Sharpe ratio, CAGR, and annualized volatility, using the Phase 1 formulas. Report the observed difference, the 2.5th–97.5th percentile range over 10,000 rebuilt histories, and the share of histories in which the mix came out better. Lower volatility counts as better.

The range is a percentile bootstrap interval, not a formal studentized test such as Ledoit and Wolf (2008). It is meant to show the size of plausible chance variation. The share favoring the mix describes the rebuilt histories and is not a probability about future returns. Maximum drawdown and recovery dates depend on the exact order of days, so they are not resampled.

The default random seed is fixed, so a repeated run reproduces every number. `--resamples`, `--block-length`, and `--seed` change the settings; changing the seed should move the range endpoints only slightly.

## Fund contributions

The mix holds drifting weights between monthly resets. Its beginning-of-day weights come from the same calculation as Phase 1, over the full history, so they match the holdings entering every window.

For a window from one close to a later close, each fund's contribution is

\[
c_i=\sum_{t} V_{t-1}\,w_{i,t}\,r_{i,t},
\]

where \(V_{t-1}\) is the mix's value before day \(t\), starting at $1, \(w_{i,t}\) is the fund's beginning-of-day weight, and \(r_{i,t}\) is its return. This is simply the dollars each fund gained or lost, per $1 in the mix at the start. The contributions add up exactly to the mix's compounded window return.

Windows are the five Phase 3 SPY declines, from the common SPY peak close to the common trough close, plus the whole evaluation period. Contributions describe where gains and losses arose. They do not show what a different mix would have earned, because removing a fund would also change the other weights.

## Bond hedge check

Also report the correlation between daily SPY and TLT returns in each calendar year from 2010. A negative value means the bond fund tended to rise on days stocks fell. `--bond-fund` chooses another study asset.

## Run and outputs

After Phase 1's snapshot exists:

```sh
uv run fecon explain
```

Defaults: `data/snapshot`, the 2018-01-01 split, `outputs/phase4`, 10,000 resamples, an average block of 21 days, seed 20260101, and TLT. The command works offline and takes about 20 seconds.

| Output | Contents |
|---|---|
| `uncertainty.csv` | Observed differences, 95% ranges, share favoring the mix, settings |
| `contributions.csv` | Each fund's contribution in each window |
| `bond_correlation.csv` | Yearly SPY/TLT daily-return correlation |
| `uncertainty.png` | Sharpe and CAGR differences with their ranges |
| `contributions.png` | Fund contributions in each Phase 3 decline, on a shared scale |
| `run.json` | Settings, Phase 3 calibration, source hashes, versions, definitions, limitations |
| `summary.md` | Plain-language interpretation with tables and figures |

## Results on the retained snapshot

- **The Sharpe gap is within chance variation.** Mix minus SPY + cash: −0.020, with a 95% range of −0.215 to +0.171. The mix came out ahead in 41% of rebuilt histories.
- **So is the growth gap.** Mix minus SPY + cash CAGR: +0.56 percentage points a year, with a range of −2.57 to +3.91.
- **The volatility difference is not.** The mix was 1.62 percentage points more volatile than SPY + cash, with a range of +1.11 to +2.11, and 4.53 points less volatile than SPY. Volatility is estimated much more precisely than average returns.
- **Bonds and energy swapped roles.** In the 2020 fall, TLT added +1.6 points and XLE lost −7.5, the largest single loss. In the 2022 fall, TLT lost −3.9 and XLE added +5.0, the only fund to gain.
- **The stock–bond hedge weakened.** The SPY/TLT correlation was negative every year from 2010 to 2021, between −0.71 and −0.14. It was slightly positive every year from 2022 to 2025, between +0.06 and +0.13. Published research connects positive stock–bond correlations with high inflation and restrictive monetary policy.
- **Over 2018–2025 as a whole,** QQQ (+27.9 points) and GLD (+27.2) contributed most of the mix's +138% growth; TLT subtracted 3.6.

## Interpretation limits

Resampling reuses only the days this sample contains. It cannot add new evidence, and the funds and dates had already been inspected. A range that includes zero means these data cannot distinguish the portfolios; it does not prove they are equal. Contributions are descriptive, not causal. Trading costs and taxes remain zero.

## Verification

- Synthetic tests check bootstrap reproducibility and block structure, zero differences for identical portfolios, agreement with the Phase 1 formulas, exact contribution sums across a monthly reset, a hand-calculated contribution example, invalid windows and settings, and the offline command.
- All Phase 1–3 outputs, including figures, were regenerated and matched the previous files byte for byte.
- A repeated `fecon explain` run reproduced all seven Phase 4 outputs byte for byte.
- Contribution sums match the Phase 3 common-window losses: −27.85% in 2020 and −18.82% in 2022.

## References

- Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Ledoit, O., and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850–859.
- Molenaar, R., Senechal, E., Swinkels, L., and Wang, Z. (2023). Empirical evidence on the stock–bond correlation.
