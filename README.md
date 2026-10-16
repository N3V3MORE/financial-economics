# ETF diversification study

Do different ETFs help spread investment risk, especially when markets fall?

The current code builds the baseline: historical returns, risk, correlations, a monthly equal-weight portfolio, and CAPM estimates with uncertainty. Stress-period analysis and testing on unseen data are planned extensions.

## Run the study

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the project folder:

```sh
uv sync --frozen
uv run fecon download
uv run fecon analyze
```

Open `outputs/phase1/summary.md` for the results. It links the charts and accompanies six CSV tables.

`download` saves prices from Yahoo Finance through yfinance and daily market and risk-free returns from Kenneth French. `analyze` reads that saved snapshot offline and checks that its files and settings still match.

To refresh the data, use a new snapshot directory:

```sh
uv run fecon download --data-dir data/new-snapshot
uv run fecon analyze --data-dir data/new-snapshot --output-dir outputs/new-run
```

Existing snapshots are never silently replaced. Changing the ETFs or dates also requires a matching new snapshot. Incomplete or unavailable provider data produces an error.

## Find your way around

```text
study.toml                         Your ETFs, dates, and calculation settings
    |
    v
data/snapshot/                     Downloaded inputs, saved locally
    |
    v
src/financial_economics/           Python code
    |-- config.py                 Read the study settings
    |-- data.py                   Download, save, and validate the inputs
    |-- analysis.py               Calculate returns, risk, and market exposure
    |-- report.py                 Write tables, charts, and the summary
    |-- cli.py                    Connect the download and analyze commands
    |-- __main__.py               Support python -m financial_economics
    +-- __init__.py                Identify the Python package
    |
    v
outputs/phase1/                    Results, saved locally
```

`data/` and `outputs/` are created when you run the commands and are ignored by Git.

| Other files | Purpose |
|---|---|
| `tests/` | Check calculations, data validation, and the command workflow using synthetic examples |
| [docs/methodology.md](docs/methodology.md) | Formulas, sources, assumptions, and interpretation |
| [docs/phase-1.md](docs/phase-1.md) | Research scope, completed baseline work, and later extensions |
| `pyproject.toml` | Python requirements, packages, and the `fecon` command |
| `uv.lock` | Exact package versions |
| `.github/workflows/tests.yml` | Run tests on Windows and Linux in GitHub Actions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaborator access, branches, and pull requests |

To change **what is studied**, edit `study.toml`. To change **how results are calculated**, start in `analysis.py`. To change **how results are presented**, start in `report.py`.

## Current settings and assumptions

The study selects SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT for January 1, 2010 through December 31, 2025. The exact settings live in `study.toml`.

- CAGR measures compounded growth; volatility measures total return variability. Sharpe uses daily excess returns.
- The equal-weight portfolio resets monthly, with weights drifting between resets. Trading costs and taxes are zero.
- SPY is the portfolio benchmark. CAPM uses French's broader market factor; beta measures exposure to that factor.
- TLT is a risky bond fund. The risk-free return comes from French's factor file.
- These are retrospectively selected surviving funds. Full-period correlations do not establish protection during market declines, and historical alpha does not prove skill or mispricing.

## Development

```sh
uv run pytest -q
```

Tests use synthetic data and require no live finance service. GitHub Actions installs the locked environment and runs the tests on Windows and Linux.

This repository is private. Ask [@N3V3MORE](https://github.com/N3V3MORE) to invite your GitHub username before cloning. Follow [CONTRIBUTING.md](CONTRIBUTING.md) when making changes, and keep downloaded data and generated results out of Git.
