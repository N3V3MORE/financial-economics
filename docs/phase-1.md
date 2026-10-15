# Phase 1: ETF analysis baseline

## Objective

Build a reproducible baseline for the ETF study proposed in Draft V1. Establish what the selected funds earned, how their returns moved together, and their exposure to U.S. equity-market movements.

Phase 1 describes historical returns. It does not establish that an optimized strategy works on unseen data or that an asset is intrinsically mispriced.

## Study choices

- Eight unleveraged, U.S.-listed instruments: SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT.
- Requested return period: January 1, 2010 through December 31, 2025, inclusive.
- Dividend- and split-adjusted prices from Yahoo Finance through yfinance. Keep a pre-start closing price to calculate the first sample return.
- Daily risk-free and market excess returns from Kenneth French's U.S. daily factor file.
- SPY is the portfolio-comparison benchmark. French's broad U.S. market factor is the CAPM regressor. These are different series.
- An equal-weight portfolio resets to 1/8 per instrument at the start of each month and lets weights drift within the month. Trading costs and taxes are zero in this baseline.
- Prices, returns, and factors are matched strictly. Missing observations cause an error; they are not filled or silently dropped.

The universe is a retrospective selection of surviving funds. Results apply to these instruments and dates, not to all ETFs or a historically investable fund-selection rule.

## Milestones

1. **Data and setup:** locked Python environment, readable study settings, separate download/offline-analysis commands, and a hashed source snapshot.
2. **Calculations and outputs:** returns, CAGR, volatility, Sharpe, drawdowns, correlations, CAPM with uncertainty, and the equal-weight baseline; produce CSVs, static figures, and a concise Markdown summary.
3. **Verification and collaboration:** focused synthetic regression tests, one real-data run, an offline repeat, documentation, and continuous integration that does not depend on live finance services.

## Done when

- A collaborator can install the locked environment and run the documented commands.
- Downloaded source values, settings, retrieval time, and hashes are recorded locally.
- Offline analysis checks snapshot integrity and reproduces the calculations.
- Tests protect return arithmetic, date alignment, percentage conversion, drawdowns, monthly rebalancing, and regression results.
- The generated figures are readable, and documentation distinguishes historical descriptions from predictive or causal claims.

## Later phases

Efficient-frontier optimization and CAL construction; chronological out-of-sample evaluation and transaction costs; stress-period studies; optional factor-model extensions. Crypto, leveraged/volatility products, DCF, and dividend-discount valuation remain separate extensions.

## Status

Implemented and locally verified on October 15, 2026:

- 37 synthetic tests passed with warnings treated as errors.
- Live download and analysis succeeded for 4,024 daily observations, January 4, 2010 through December 31, 2025, across all eight instruments.
- An offline repeat reproduced all six CSVs, run metadata, and the Markdown summary byte for byte in the locked environment.
- Independent price-ratio, monthly-portfolio, geometric-growth, and CAPM coefficient checks agreed with the saved outputs. Direct HAC covariance checks also agreed for SPY and GLD.
- All four rendered figures were inspected for labels, units, clipping, and readability.
- GitHub Actions is configured for Windows and Linux; its execution status is reported on the pull request.

Downloaded snapshots and generated outputs remain local and ignored by Git. Each collaborator can reproduce the study using the README commands.
