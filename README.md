# Financial Economics

An ETF study connecting portfolio theory to actual returns. Phase 1 establishes return and risk comparisons, co-movement, a monthly equal-weight portfolio, and CAPM estimates with uncertainty. Phase 2 compares the same portfolios during benchmark declines. Phase 3 asks whether the mix offers benefits beyond simply holding less stock-market risk.

The question is whether apparently different ETF exposures provide useful diversification, including during market declines. Phase 3 uses earlier returns to set a SPY/cash comparison, then evaluates later returns. The dataset and fund selection had already been inspected, so this remains a chronological historical evaluation.

## Current study

Eight unleveraged instruments: SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT. The requested return sample is 2010-01-01 through 2025-12-31. Exact choices live in [study.toml](study.toml).

Prices come from Yahoo Finance through yfinance; daily risk-free and broad-market excess returns come from Kenneth French. Downloaded data stays local. A manifest records its origin, settings, retrieval time, and hashes so the same snapshot can be analyzed again offline.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/N3V3MORE/financial-economics.git
cd financial-economics
uv sync --frozen
uv run fecon download
uv run fecon analyze
```

Open `outputs/phase1/summary.md`. It links four figures and summarizes the saved CSV results. Only `download` contacts the finance data providers. `analyze` uses the retained, hash-checked snapshot.

To compare declines using that same snapshot:

```sh
uv run fecon stress
```

Open `outputs/phase2/summary.md`. By default, the report includes every non-overlapping SPY episode reaching a 10% drawdown, compares losses over the same SPY peak-to-trough dates, and measures recovery from the common trough. It also plots correlations over 63 trailing daily observations. Unfinished recoveries are marked missing, never zero.

The settings can be made explicit:

```sh
uv run fecon stress --data-dir data/snapshot --output-dir outputs/phase2 --drawdown-threshold 0.10 --rolling-window 63
```

`stress` runs offline. The existing monthly portfolio continues through each episode; chart rebasing does not reset its weights. Episode dates are selected retrospectively and do not define a trading rule.

To compare the mix with a simple portfolio holding less SPY:

```sh
uv run fecon compare
```

Open `outputs/phase3/summary.md`. The command uses only 2010–2017 to estimate the mix's sensitivity to SPY excess returns, including an intercept. It bounds that estimate between 0% and 100% to set a frozen SPY target, with the remainder in French's daily risk-free return as a cash proxy. The target is calculated from the snapshot, not hard-coded.

The three portfolios are evaluated over 2018–2025: the existing eight-fund mix, SPY plus cash, and SPY alone. Monthly targets stay fixed while holdings drift between resets. The report saves performance, training and evaluation exposure estimates, identical-date downturn and recovery comparisons, calibration details, source hashes, and three figures. `compare` also runs offline. See [Phase 3](docs/phase-3.md) for the exact split and assumptions.

A snapshot is never silently replaced. To refresh data, choose a new directory:

```sh
uv run fecon download --data-dir data/new-snapshot
uv run fecon analyze --data-dir data/new-snapshot --output-dir outputs/new-run
```

Changing the assets or dates in `study.toml` requires a matching new snapshot. If a provider is unavailable or returns incomplete data, downloading fails visibly; there is no synthetic-data fallback.

## What the results mean

- CAGR measures compounded historical growth; volatility measures total return variability.
- Sharpe uses daily excess returns. Phase 1 CAPM beta measures exposure to French's broad-market factor; Phase 3 beta measures sensitivity to SPY excess returns. Neither measures total risk.
- The equal-weight portfolio resets monthly, with weights drifting between resets. Trading costs and taxes are zero.
- SPY is the portfolio benchmark. TLT is a risky bond fund, not the risk-free asset.
- A whole-period correlation matrix does not establish protection in a market decline. Historical alpha is not proof of skill or mispricing.

See [Phase 1](docs/phase-1.md), [Phase 2](docs/phase-2.md), [Phase 3](docs/phase-3.md), and the [baseline methodology](docs/methodology.md) for scope, formulas, assumptions, and sources.

## Watch the results

The unified film follows the question through all three phases: the mix's lower
risk, its behavior during market declines, and the comparison with SPY plus cash.
For a plain-language explanation and the next research step, read
[Results explained](docs/results-explained.md).

Generate all three reports, then render the captioned Manim film:

```sh
uv run fecon analyze
uv run fecon stress
uv run fecon compare
uv sync --frozen --group animation
uv run --frozen --group animation manim -qh --fps 30 --media_dir outputs/unified --progress_bar none -v WARNING animations/project_results.py ProjectResults
```

The rendered file is
`outputs/unified/videos/project_results/1080p30/ProjectResults.mp4`;
the delivery copy is `outputs/unified/project-results.mp4`.
The film distinguishes the full 2010–2025 history from the 2018–2025 comparison.
It shows why a mix can lose less than SPY yet fail to improve on a simpler
SPY/cash portfolio. Growth, risk, downturn losses, and recovery each answer a
different question; CAPM details stay in the written reports.

The film is silent, with on-screen captions, and needs no LaTeX installation.
It uses Segoe UI on Windows; Manim's text renderer substitutes an available font
on other systems. All figures come from the retained local analysis outputs.
Rendering stops if the three reports disagree with the verified source snapshot.
Growth paths use weekly endpoints; the 2020 detail and numerical estimates use
daily data. The time cursor advances all portfolios together, and rebasing the
2020 display does not reset their holdings. `outputs/unified/evidence.json`
records the input files and chapter times.
For a faster draft, replace `-qh --fps 30` with `-ql --fps 15`.
No provider data is downloaded during rendering. Generated videos and rendering caches
stay inside ignored `outputs/`.

The approved Phase 1 film remains available at
`outputs/animation/phase1-results-final.mp4`. Its source is
`animations/phase1_results.py`; the unified film does not replace that source.

## Develop and collaborate

```sh
uv run pytest -q
```

Tests use small synthetic examples and require no live finance service. GitHub Actions runs the locked environment and tests on Windows and Linux.

This repository is private. Ask [@N3V3MORE](https://github.com/N3V3MORE) to invite your GitHub username and accept the invitation before cloning. Work on a branch and open a pull request; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

| Path | Purpose |
|---|---|
| `study.toml` | Universe, dates, and calculation settings |
| `src/financial_economics/` | Data, calculations, CLI, and outputs |
| `tests/` | Synthetic numerical and workflow checks |
| `animations/` | Manim presentations of the saved study results |
| `docs/` | Phase plan and methodology |
| `data/` | Ignored local source snapshots |
| `outputs/` | Ignored reproducible tables and figures |
