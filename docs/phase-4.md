# Phase 4: how sure are we, and why did it happen?

## Question

Phase 3 found no clear advantage for the eight-fund mix over a SPY + cash portfolio during 2018–2025. Phase 4 asks two follow-up questions about the same dates and portfolios. Could chance alone explain the differences? And which funds produced the mix's gains and losses in each market fall?

Nothing from Phases 1–3 changes. The snapshot, the mix, the 69.19% SPY target, and the evaluation dates are reused as they are.

## Could chance explain the differences?

Eight years is one history. To see how much the comparison depends on the particular order of days, we rebuild those years 10,000 times and recalculate the comparison each time.

Each rebuilt history is 2,011 days long, like the original period, and is drawn in runs of consecutive real days. Some days appear more than once and others not at all. A run starts on a random day, and each day has a 1-in-21 chance of jumping to a new random day, so run lengths vary and average 21 trading days, roughly a month. This is the stationary bootstrap of Politis and Romano (1994). Runs keep calm and stormy stretches together, which drawing single days would break apart. Every portfolio and the risk-free rate use the same drawn days, so the comparison stays like for like.

For each rebuilt history we calculate the mix minus the other portfolio for Sharpe ratio, yearly growth (CAGR), and yearly fluctuation (volatility), with the Phase 1 formulas. The report gives the observed difference, the middle 95% of the rebuilt differences, and how often the mix came out better. For fluctuation, lower counts as better.

This range shows how much the result could move by chance. It is not a formal significance test such as Ledoit and Wolf (2008), and the share of histories favouring the mix is not a probability about the future. Worst declines and recoveries depend on the exact order of days, so they are not resampled.

The random seed is fixed, so reruns give identical numbers. `--resamples`, `--block-length`, and `--seed` change the settings.

## Which funds drove the falls?

The mix holds drifting weights between monthly resets. For a window from one close to a later close, each fund's contribution is

\[
c_i=\sum_{t} V_{t-1}\,w_{i,t}\,r_{i,t},
\]

where \(V_{t-1}\) is the mix's value before day \(t\), starting at $1, \(w_{i,t}\) is the fund's weight at the start of the day, and \(r_{i,t}\) is its return. In other words, it is the dollars each fund gained or lost per $1 in the mix at the start. The contributions add up exactly to the mix's return over the window.

The windows are the five Phase 3 SPY falls, from the SPY peak to the trough, plus the whole evaluation period. The window must start and end on trading days. Contributions show where gains and losses came from. They don't show what a different mix would have earned, because removing a fund would change the other weights too.

The report also gives the correlation between daily SPY and TLT returns in each year since 2010. A negative value means the bond fund tended to rise on days stocks fell. `--bond-fund` picks another study fund.

## Run it

```sh
uv run fecon explain
```

This uses `data/snapshot`, the 2018-01-01 split, and `outputs/phase4`. It runs offline and takes about 20 seconds. It writes `uncertainty.csv`, `contributions.csv`, and `bond_correlation.csv`, two charts, `run.json` with the settings and source hashes, and `summary.md`.

## Results

Against SPY + cash, the mix's Sharpe ratio was 0.020 lower. The 95% range ran from −0.215 to +0.171, and the mix came out ahead in 41% of rebuilt histories. Its yearly growth was 0.56 points higher, with a range of −2.57 to +3.91. These eight years cannot tell the two apart. The mix's higher fluctuation, 1.62 points (range +1.11 to +2.11), showed up in every rebuilt history.

Bonds and energy swapped roles. In the 2020 fall, TLT added 1.6 points and XLE lost 7.5, the largest loss of any fund. In the 2022 fall, TLT lost 3.9 points and XLE gained 5.0, the only fund to gain. The SPY/TLT correlation was negative every year from 2010 to 2021, between −0.71 and −0.14, and slightly positive from 2022 to 2025. Research on stock–bond correlations links positive values to high inflation and tight monetary policy (Molenaar, Senechal, Swinkels and Wang, 2023).

Over 2018–2025 as a whole, QQQ (+27.9 points) and GLD (+27.2) contributed the most to the mix's +138% growth. TLT took away 3.6.

## Limits

The rebuilt histories reuse the days this sample contains, and the funds and dates had already been examined, so this is not new evidence. A range that includes zero means the data cannot separate the portfolios; it does not prove they are equal. The rebuilds reshuffle the finished portfolio returns with the 69.19% SPY split held fixed. They do not redo the choice of funds or the fitting of that split, so they leave out the uncertainty from those research choices.

The match between the two portfolios was also only approximate. Over 2018–2025 the mix moved 0.73% for each 1% move in SPY, and SPY + cash moved 0.68%. Part of the difference between them comes from that gap in stock exposure, not only from diversification. Trading costs and taxes are still zero.

## References

- Politis, D. N., and Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Ledoit, O., and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850–859.
- Molenaar, R., Senechal, E., Swinkels, L., and Wang, Z. (2023). Empirical evidence on the stock–bond correlation.
