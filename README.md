# Financial Economics

An ETF study connecting portfolio theory to actual returns. Phase 1 establishes a reproducible baseline: return and risk comparisons, co-movement, a monthly equal-weight portfolio, and CAPM estimates with uncertainty.

The longer-term question is whether apparently different ETF exposures provide useful diversification, including during market declines. Phase 1 supplies the data and calculations; it does not yet answer the stress-period or out-of-sample questions.

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

A snapshot is never silently replaced. To refresh data, choose a new directory:

```sh
uv run fecon download --data-dir data/new-snapshot
uv run fecon analyze --data-dir data/new-snapshot --output-dir outputs/new-run
```

Changing the assets or dates in `study.toml` requires a matching new snapshot. If a provider is unavailable or returns incomplete data, downloading fails visibly; there is no synthetic-data fallback.

## What the results mean

- CAGR measures compounded historical growth; volatility measures total return variability.
- Sharpe uses daily excess returns. Beta measures exposure to French's market factor, not total risk.
- The equal-weight portfolio resets monthly, with weights drifting between resets. Trading costs and taxes are zero.
- SPY is the portfolio benchmark. TLT is a risky bond fund, not the risk-free asset.
- A whole-period correlation matrix does not establish protection in a market decline. Historical alpha is not proof of skill or mispricing.

See [the Phase 1 plan](docs/phase-1.md) and [methodology](docs/methodology.md) for scope, formulas, assumptions, and sources.

## Watch the results

After generating the analysis, render the captioned Manim film:

```sh
uv sync --frozen --group animation
uv run --frozen --group animation manim -qh --fps 30 --media_dir outputs/animation --progress_bar none -v WARNING animations/phase1_results.py PhaseOneResults
```

The 1080p video is `outputs/animation/videos/phase1_results/1080p30/PhaseOneResults.mp4`.
It follows two portfolios through the same dates, then zooms into the 2020 decline
with both rebased to $100 at the February 19 close. The existing monthly portfolio
continues unchanged; rebasing the display does not add a rebalance.
The 2020 example is illustrative, not evidence of protection in every downturn.
Its losses from the common starting date differ from full-sample maximum drawdowns.
The film ends with full-period growth and volatility; CAPM stays in the written report.
Every estimate comes from the local analysis outputs; inconsistent return statistics stop the render.
The film is silent, with on-screen captions, and needs no LaTeX installation. It uses Segoe UI
on Windows; Manim's text renderer substitutes an available font on other systems.
For a faster draft, replace `-qh --fps 30` with `-ql --fps 15`.
Set `FECON_RESULTS_DIR` to show another analysis of the same eight-fund SPY universe.
Growth curves and their markers advance under one shared time cursor. Full-period
paths use weekly points; the 2020 detail and numerical statistics use daily data.
No provider data is downloaded during rendering. Generated videos and rendering caches
stay inside ignored `outputs/`.

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
| `animations/` | Manim presentation of the saved Phase 1 results |
| `docs/` | Phase plan and methodology |
| `data/` | Ignored local source snapshots |
| `outputs/` | Ignored reproducible tables and figures |
