# Methodology

## Scope and data

The study uses the fixed instruments and dates in `study.toml`. These are retrospective choices of surviving instruments, not a rule for selecting funds using information available at the start of 2010. The study does not generalize automatically to all ETFs.

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
