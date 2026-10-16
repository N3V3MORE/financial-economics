# Phase 3: a fairer diversification comparison

## Question

Did the eight-fund mix offer useful benefits beyond holding less stock-market risk? Compare it with a SPY/cash control and SPY alone over 2018–2025. Set the control's allocation using 2010–2017 only.

The existing funds, source snapshot, and monthly equal-weight mix remain unchanged. This phase adds no optimization, new fund selection, trading-cost model, or tax model.

## Calibration

First calculate the mix's monthly equal-weight holdings over the full return history. Slice those returns into training and evaluation afterward. This preserves the holdings entering evaluation, including when a custom split falls inside a month.

For training observations, estimate the intercept regression

\[
r_{mix,t}-r_{f,t}=a+b(r_{SPY,t}-r_{f,t})+e_t.
\]

The regressor is SPY's excess return, not French's broad-market Mkt-RF factor used by the Phase 1 CAPM. The intercept prevents the slope from being forced through the origin. Variable daily RF is subtracted from both returns.

Set the stock target to `min(1, max(0, b))` and the cash target to `1 - stock target`. A slope below zero produces all cash; a slope above one produces all SPY. Save the raw slope and whether clipping occurred, because an unlevered long-only control cannot match an exposure outside that range.

Training observations precede the requested evaluation start; evaluation includes that date and all later observations in the configured snapshot. The default cutoff is 2018-01-01. On the retained snapshot, the actual training dates are 2010-01-04 through 2017-12-29, and evaluation begins on 2018-01-02. Each part needs at least three observations and varying SPY excess returns for its exposure regression. The data must have matching return and RF dates, with no missing observations.

Changing evaluation-period returns cannot change the fitted allocation. No evaluation performance, episode dates, or future returns enter calibration.

## Holdings and cash

Initialize the SPY/cash control at its fitted target before the first evaluation return. Reset to those same targets before the first return of each subsequent calendar month. Between resets, hold units fixed and let weights drift:

\[
r_{p,t}=\sum_i w_{i,t}r_{i,t},\qquad
w_{i,t+1}=\frac{w_{i,t}(1+r_{i,t})}{1+r_{p,t}}.
\]

The next month's first return uses the frozen targets instead of the preceding drifted weights. The mix retains its existing monthly equal-weight rule. SPY alone is a buy-and-hold reference.

Cash earns French's daily risk-free return in decimal units. This is a conceptual cash-return proxy, not a traded ETF. Trading costs and taxes are zero. Observed ETF returns already reflect fund operating expenses.

Report realized SPY exposure separately for all three portfolios during both training and evaluation. The control's target only approximates the mix's sensitivity: monthly drift changes its realized beta, and the mix's beta may change after training. The control's training returns are a hypothetical diagnostic generated after fitting the target; that target was not known in 2010.

## Evaluation and downturns

Compare compounded growth, CAGR, annualized total-return volatility, daily-excess-return Sharpe, and maximum drawdown using the existing Phase 1 formulas and the same evaluation dates. Growth begins with $1,000 at the preceding observed close, 2017-12-29 in the default snapshot. Maximum drawdown includes that initial capital in the running peak.

Reuse Phase 2's 10% SPY rule within the evaluation period. Keep each non-overlapping SPY peak-to-first-recovery episode whose decline reaches 10%; retain an unfinished final episode through evaluation end. All three portfolios use identical SPY peak and trough closes.

Recovery means first regaining each portfolio's own wealth at the common SPY peak, searching on or after the common SPY trough. Record trading observations and calendar days from that trough. A portfolio already at or above its starting value has zero elapsed days. A 1e-12 relative tolerance handles equality. Unobserved dates and durations are missing, never zero, and follow-up time is saved separately. Common-date episode losses differ from each portfolio's independently timed maximum drawdown.

## Run and outputs

After obtaining the Phase 1 snapshot:

```sh
uv run fecon compare
```

The defaults use `data/snapshot`, the 2018-01-01 cutoff, and `outputs/phase3`. The command verifies the retained snapshot and works offline; only `download` contacts data providers. The output directory must be outside the source snapshot.

| Output | Contents |
|---|---|
| `calibration.json` | Requested and actual split dates, sample sizes, fitted intercept and slope, frozen weights, clipping flag, evaluation anchor |
| `evaluation_returns.csv` | Evaluated daily returns for SPY, the mix, and SPY/cash |
| `performance.csv` | Growth, CAGR, volatility, Sharpe, maximum drawdown, observation counts |
| `stock_exposure.csv` | Realized SPY slopes and daily intercepts in both periods |
| `episodes.csv` | Evaluation-period SPY episode dates and recovery status |
| `episode_comparison.csv` | Same-date returns, window minima, recovery dates and durations, censoring |
| `growth.png` | Growth of the same $1,000 over the evaluation dates |
| `drawdowns.png` | Declines from each portfolio's own peak |
| `episode_losses.png` | All three portfolios' returns over identical SPY decline dates |
| `run.json` | Source hashes, versions, settings, formulas, assumptions, limitations |
| `summary.md` | Plain-language interpretation with saved tables and figures |

All three figures are produced even when no episode qualifies; the episode figure then says that no SPY decline reached the threshold. Numerical outputs use consistent date formatting and precision, and metadata adds no new run timestamp. Data and generated results remain ignored by Git.

## Interpretation limits

This is a chronological historical evaluation. The allocation is fitted on earlier returns, but the complete dataset and surviving funds had already been inspected. It is not an untouched out-of-sample strategy test. The selected fund universe also limits how far the results can be generalized.

The control makes the lower-market-risk explanation easier to assess. A performance difference can still reflect changing stock sensitivity, bond duration, sector exposure, or other risks that SPY beta does not capture. The report therefore shows realized betas and does not claim causal attribution or statistical superiority. Benchmark peaks, troughs, and recovery dates are selected retrospectively.

Optimization, cost modelling, and a genuinely new evaluation sample remain separate work.

## Verification

Verification covers training-only calibration; split boundaries; insufficient or constant training data; clipped weights; monthly drift and resets; all-SPY/all-cash endpoints; invalid weights; preserved equal-weight results; and the existing Phase 1–2 tests. A retained-snapshot run is checked independently for calibration, selected daily returns, and recovery dates, then repeated offline to compare saved numerical outputs. Each figure is inspected for readable labels and consistent dates.

Implemented and locally verified on October 15, 2026:

- All 101 synthetic and workflow tests passed with warnings treated as errors, including all Phase 1–2 tests.
- The retained, hash-checked snapshot supplied 2,013 training and 2,011 evaluation observations. Its fitted target was 69.189920% SPY and 30.810080% cash; no clipping was needed.
- An independent reconstruction used the retained price and RF files directly, without project calculation helpers. It checked all 6,033 evaluation return values, calibration, both-period exposures, all performance measures, five episode calendars, and all fifteen recovery dates, durations, minima dates, and censoring fields. Dates and statuses matched exactly; the largest numerical difference was 4.63e-12, consistent with CSV rounding.
- An offline repeat reproduced all five CSVs, calibration, run metadata, and summary byte for byte. Rerunning Phases 1–2 also reproduced every saved CSV, metadata file, and summary byte for byte.
- All three final figures were inspected for readable labels, legends, units, clipping, and consistent evaluation dates.

Over 2018–2025, the mix's CAGR was 11.48%, versus 10.92% for SPY/cash. Its annual volatility was higher (14.93% versus 13.31%), its maximum drawdown deeper (-27.90% versus -24.02%), and its Sharpe lower (0.619 versus 0.639). The mix lost more over all five common downturn windows, but recovered sooner in four; its 2022 recovery was later. Evaluation betas also differed: 0.7274 for the mix and 0.6838 for the control. These findings do not establish a superior portfolio or isolate a diversification effect.

Verification was local on Windows. No changes were pushed, so hosted checks have not run for this phase. The independent audit and repeat-check evidence remain local in `outputs/phase3-verification/`; the generated report is `outputs/phase3/summary.md`.
