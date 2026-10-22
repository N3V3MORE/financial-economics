# Methodology

## Scope and data

The study uses the fixed instruments and dates in `configs/study.toml`. These are retrospective choices of surviving instruments, not a rule for selecting funds using information available at the start of 2010. The study does not generalize automatically to all ETFs.

Yahoo Finance's `Adj Close`, retrieved through yfinance with `auto_adjust=False`, supplies the adjusted price series. This is a provider snapshot, not unmodified exchange data. Distributions and splits are reflected in its adjustments; do not add dividends again. Provider corrections may change a later download.

Download includes seven calendar days before the requested start. The last pre-sample price must match the immediately preceding trading date in French's retained calendar, so the first return cannot silently span multiple sessions. yfinance's end date is exclusive; the CLI requests the day after the configured inclusive end. [yfinance download documentation](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).

The French daily factor ZIP supplies `Mkt-RF` and `RF`. Its percent returns are divided by 100 exactly once. RF is already a daily return, not an annual yield needing division by 252. Mkt-RF is already excess return. [French factor definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).

All internal and CSV returns are decimal fractions: 0.01 means 1%. Price dates and French trading dates must match exactly in the requested sample. Missing, duplicate, nonfinite, or nonpositive selected prices cause errors. No forward filling, backward filling, zero filling, or silent date dropping is used.

## Returns and annualization

For adjusted price P, daily simple return is `r[t] = P[t] / P[t-1] - 1`. Wealth starts at 1 and compounds as `W[t] = product(1 + r[1:t])`.

- Cumulative return: `W[T] - 1`.
- CAGR: `W[T] ** (252 / T) - 1`, using T trading observations and the configured trading-days-per-year value (252 by default).
- Annualized volatility: `sqrt(252) * sample_std(r)`, with denominator T - 1.
- Annualized Sharpe: `sqrt(252) * mean(r - RF) / sample_std(r - RF)`. A zero denominator produces an undefined value, not infinity.
- Drawdown: `W[t] / max(1, W[1], ..., W[t]) - 1`. Maximum drawdown is the most negative value, including a loss on the first day.

CAGR is compounded growth; arithmetic mean times 252 is a different quantity. Square-root-of-time volatility and Sharpe scaling are conventions assuming sufficiently weak serial dependence; they do not measure every tail risk. Annual-return rows compound daily returns within calendar years. Partial configured years remain partial, not annualized.

## Equal-weight baseline

Each selected instrument starts with the same portfolio weight. At each month's first return, weights reset to equal using the preceding close. Otherwise, holdings drift with returns:

`portfolio_return[t] = sum(start_weight[i,t] * return[i,t])`

`end_weight[i,t] = start_weight[i,t] * (1 + return[i,t]) / (1 + portfolio_return[t])`

The baseline is fully invested and long only, with zero trading costs and taxes. Fund operating expenses are reflected in observed fund performance and are not subtracted again. This is a transparent comparison rule, not an optimized or recommended allocation.

## CAPM and uncertainty

Each fund and the equal-weight portfolio use:

`r[i,t] - RF[t] = alpha[i] + beta[i] * (Mkt-RF[t]) + error[i,t]`

An intercept is always included. OLS coefficients use Newey-West HAC uncertainty with Bartlett weights, five lags by default, a finite-sample covariance correction, and Student t reference probabilities. The lag count is recorded in the study. [statsmodels robust covariance options](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).

`capm.csv` includes daily alpha, beta, their standard errors, 95% intervals, two-sided zero-coefficient p-values, R-squared, and observation counts. Alpha is a decimal return per day; beta is dimensionless. Testing beta = 0 differs from testing beta = 1. A nonsignificant alpha does not prove a model correct.

French's market factor covers a broader U.S. equity universe than SPY. SPY is only the accessible portfolio-comparison benchmark. CAPM regressions on GLD and TLT are descriptive equity-market exposure regressions, not a complete pricing model for gold or bonds. Constant market returns, constant excess responses, and numerically perfect fits cannot provide ordinary residual-based inference and are rejected clearly.

## Correlation and covariance

Correlation and sample covariance use matched daily simple returns. Annualized covariance multiplies sample daily covariance by 252. Co-movement does not identify its economic cause, and full-period correlation does not establish crisis protection.

## Reproduction

`download` writes a new snapshot only after validation. Its manifest retains source URLs, UTC retrieval time, settings, provider-library versions, observation bounds, and SHA-256 file hashes. Downloaded prices are retained locally; French's source ZIP is retained alongside the converted factors.

`analyze` checks settings and hashes before reading the saved sample. It makes no finance-provider requests. It writes CSVs, PNGs, a Markdown summary, and `run.json` with the consumed snapshot hashes. New downloads may differ; an unchanged saved snapshot is the reproducible unit. Numerical CSVs and run metadata should repeat exactly within the locked environment; PNG metadata and floating-point results across other platforms are not promised byte-identical.

Raw snapshots and generated outputs are ignored by Git. The tests use synthetic data, including controlled failures. Live provider availability is verified separately and is not a requirement for CI tests.

## Expanded protection study: frozen design

`configs/protection.toml` freezes the union of the original eight funds and Jake's
twelve: SPY plus QQQ, IWF, SCHG, VUG, XLK, ARKK, XLE, XLY, VNQ, VIG, TLT,
GLD, and DBC. January 2015–December 2025 is the common comparison; the
2007–2025 SPY/TLT/GLD/DBC sample is separate and descriptive. The equal-weight
eight-, twelve-, and fourteen-fund mixes use fixed membership, with no search
for favorable historical weights. Growth/technology, sectors/real estate,
dividend growth, and other assets are discussion categories, not independent
sources of diversification.

For candidate j, let `x[j,m] = return[j,m] - RF[m]` and let `D[m]` indicate a
negative SPY month. The primary effect is `mean(x[j,m] | D[m] = 1)`; the win
frequency is the fraction of those months with `x[j,m] > 0`. Returns and RF
are compounded from complete daily observations, with no skipped first return
or partial month. Negative and severe months (SPY <= -5%) are identified after
the portfolios have been held. This is not a strategy that predicts declines.

Before costs, the monthly difference between `(1-q) SPY + q fund` and
`(1-q) SPY + q cash` is exactly `q * (fund monthly return - RF monthly return)`.
The implementation and independent verifier check this in every month.
The 10%, 20%, and 40% additions illustrate allocation consequences; they do
not create three independent statistical confirmations. Over several months,
compounding means cumulative differences need not scale linearly in q.

Daily portfolio paths reuse the self-financing monthly cost engine: drift
within months, target resets before the next month's first return, and a fee
of 0, 10, or 25 basis points per ETF dollar bought or sold. Fees reduce capital
before the next return. Entry is charged, exit and taxes are excluded, cash
is frictionless, and operating expenses are already reflected in fund returns.
These are scenarios, not historical estimates of execution costs.
`fees.csv` records actual fee and traded notional per initial portfolio dollar;
monthly sums use each day's pre-trade portfolio wealth, rather than adding
percentages with different denominators.

Performance conventions remain 252-day CAGR, sample daily volatility, daily
excess-return Sharpe, and maximum drawdown including the initial wealth anchor.
The full-sample path is sliced for descriptive subperiods; entry costs are not
recharged on each slice. Subperiod drawdowns use a fresh local wealth anchor.
For the supplementary beta check, daily gross excess-return OLS with an
intercept is trained on 2015–2019. The slope determines a frozen SPY/cash
target clipped to 0–100% stocks; both evaluated portfolios start at targets
in January 2020 and pay their applicable entry costs. Raw slopes, clipping,
realized portfolio beta, and realized control beta are reported. A clipped
control does not match an exposure above one, and beta matching never
equalizes all sources of risk.

## Primary uncertainty and multiple comparisons

Monthly fund-minus-RF returns are regressed on two columns, `D` and `1-D`,
without an intercept. The coefficient on `D` is the negative-month mean.
All months remain in the regression so the HAC calculation respects their
calendar spacing. Standard errors use Bartlett HAC with six monthly lags and
the `n/(n-k)` small-sample covariance correction; two-sided p-values use a
Student t reference with `n-k` degrees of freedom. Holm correction applies
across all 13 candidates at 5%, following the supported
[statsmodels implementation](https://www.statsmodels.org/v0.14.4/generated/statsmodels.stats.multitest.multipletests.html).
Unavailable tests keep their place in the family but are reported as NA and
never rejected. A significant negative effect is evidence against protection,
not a successful hedge.

Pointwise 95% percentile intervals use 10,000 paired stationary-bootstrap
draws, seed 20260101, and a six-month mean block length, with three- and
twelve-month sensitivities. Every draw resamples the full monthly sequence
jointly, preserving the fund/SPY/RF observations together and geometrically
distributed runs of adjacent months, wrapping at the sample end. Negative-SPY
months are reidentified within each draw. Draws with no such months are
explicitly unavailable. These intervals are not simultaneous family-wise
intervals; inferential rejection comes from the separate Holm-adjusted tests.
See [Politis and Romano's stationary bootstrap](https://doi.org/10.1080/01621459.1994.10476870).

Subperiod, severe-month, and 2007–2025 comparisons are descriptive with
observation counts. No p-value search, winner selection, or portfolio
optimization is performed. Multiple-testing adjustment cannot remove
hindsight or survivorship, and changing providers does not make previously
examined history unseen data. This limitation follows the concern motivating
[Harvey, Liu, and Zhu](https://www.nber.org/papers/w20592).

## CRSP source acceptance and cash

The private adapter queries `crsp.stksecurityinfohist` for dated PERMNO
identities, `crsp.dsf_v2.dlyret` for daily total returns, and
`crsp.msf_v2.mthret` for monthly checks. Identity histories must cover the seed
and every trading day exactly once, map each ticker to a distinct PERMNO,
and classify each security as FUND/ETF. Daily missing-return flags must be
`NA` (the CRSP flag code, not a missing CSV cell). Monthly return flags must
be `CR`: CRSP's `metaflaginfo` dictionary, flag type AR, defines this as a
full-period compounded return without missing daily returns. The definition
was verified against WRDS metadata on October 20, 2026.

Daily returns must be finite, greater than -100%, duplicate-free, and present
on exactly the requested French trading calendar. Monthly reconciliation
compounds daily returns and rejects absolute differences over 0.0001, one
basis point, from CRSP's monthly series. It also rejects missing fund-months
and duplicate monthly observations. Queries, parameters, original extracts,
source definitions, retrieval times, SHA-256 hashes, and coverage summaries
are retained privately. Direct returns feed the protection engine; normalized
indices are only a compatibility representation for earlier commands.

CIZ compounds daily returns with dividends reinvested on ex-dates, unlike the
legacy monthly construction. See [CRSP's methodology summary](https://www.crsp.org/wp-content/uploads/guides/CRSP_US_Stock_%26_Indexes_Databases_Summary_of_CIZ_Differences_to_Legacy_Files.pdf)
and [French's transition explanation](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library.html).
French daily RF is the cash proxy in every calculation, compounded on matching
dates. Its underlying source changes from Ibbotson through May 2024 to the ICE
BofA US 1-Month Treasury Bill Index in June 2024; see the
[factor definition](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html).
It is a conceptual cash proxy, not a tradable fund or the supplemental
yield-divided-by-twelve approximation.

Time-varying stock–bond relationships motivate the period checks, rather than
a claim that this comparison identifies inflation's causal effect. See
[Campbell, Sunderam, and Viceira](https://campbell.scholars.harvard.edu/sites/g/files/omnuum5881/files/campbell/files/campbellsunderamviceira_20160523.pdf)
and [BIS](https://www.bis.org/publications/correlation-equity-and-bond-returns).
Separating all months, negative months, and severe falls follows the distinction
between ordinary-market hedging and crash protection in
[Baur and Lucey](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6288.2010.00244.x).
This is a focused course-project comparison of actual ETFs against cash;
journal-level novelty remains unestablished.
