# ETF diversification study

Do different ETFs help spread investment risk, especially when markets fall?

Phase 1 builds the baseline: historical returns, risk, correlations, a monthly equal-weight portfolio of the eight funds (the "mix"), and CAPM estimates with uncertainty. Phase 2 compares the mix with SPY in every SPY decline of at least 10%, including how long each took to recover. Phase 3 asks whether the mix did better than simply holding less stock, using a SPY + cash portfolio chosen from 2010–2017 and tested on 2018–2025. Phase 4 checks whether chance alone could explain the Phase 3 differences, and which funds produced the mix's gains and losses in each fall.

For a plain-language account of what the study has found, read [Results explained](docs/results-explained.md).

## Run the study

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from the project folder:

```sh
uv sync --frozen
uv run fecon download
uv run fecon analyze
uv run fecon stress
uv run fecon compare
uv run fecon explain
```

Each analysis command writes a `summary.md` that links its charts and tables:

| Command | Opens | Question |
|---|---|---|
| `analyze` | `outputs/phase1/summary.md` | How did the funds and the mix perform over 2010–2025? |
| `stress` | `outputs/phase2/summary.md` | What happened in each SPY decline of 10% or more? |
| `compare` | `outputs/phase3/summary.md` | Did the mix beat a SPY + cash portfolio over 2018–2025? |
| `explain` | `outputs/phase4/summary.md` | Could chance explain that comparison, and which funds drove it? |

`download` saves prices from Yahoo Finance through yfinance and daily market and risk-free returns from Kenneth French. The other commands read that saved snapshot offline and check that its files and settings still match.

To refresh the data, use a new snapshot directory:

```sh
uv run fecon download --data-dir data/new-snapshot
uv run fecon analyze --data-dir data/new-snapshot --output-dir outputs/new-run
```

Existing snapshots are never silently replaced. Changing the ETFs or dates also requires a matching new snapshot. Incomplete or unavailable provider data produces an error.

`stress` and `compare` accept settings too, for example `--drawdown-threshold 0.10 --rolling-window 63` or `--evaluation-start 2018-01-01`. See [Phase 2](docs/phase-2.md) and [Phase 3](docs/phase-3.md).

`explain` rebuilds 2018–2025 10,000 times from month-long blocks of real trading days, keeping every portfolio on the same days, and reports how far the Phase 3 differences move. A fixed random seed makes reruns identical; `--resamples`, `--block-length`, and `--seed` change the settings. It also measures each fund's gain or loss in every SPY fall, adding up exactly to the mix's loss. See [Phase 4](docs/phase-4.md).

## Watch the results

A short captioned film follows the question through all four phases. Run the four analysis commands first, then:

```sh
uv sync --frozen --group animation
uv run --frozen --group animation manim -qh --fps 30 --disable_caching --media_dir outputs/unified --progress_bar none -v WARNING animations/project_results.py ProjectResults
uv run --frozen --group animation python scripts/verify_render.py
```

The last command checks the video's size, frame rate, and length, saves the finished film as `outputs/unified/project-results.mp4`, and builds `outputs/unified/final-contact.png`, one page of review frames from every chapter. Before drawing anything, the film recalculates every number it shows from the saved snapshot, and it stops if a report disagrees; repeating the Part 4 resampling adds about 20 seconds. `--disable_caching` keeps the chapter times recorded in `outputs/unified/evidence.json` exact. For a quicker draft, replace `-qh --fps 30` with `-ql --fps 15` and `outputs/unified` with `outputs/unified-draft`, then run `scripts/verify_render.py --draft`. The film needs no LaTeX and uses Segoe UI on Windows; other systems substitute an available font.

The earlier Phase 1 film, `animations/phase1_results.py`, remains available.

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
    |-- report.py                 Write Phase 1 tables, charts, and the summary
    |-- stress.py                 Find SPY declines and compare recoveries
    |-- stress_report.py          Write the Phase 2 report
    |-- comparison.py             Fit the SPY + cash portfolio and compare
    |-- comparison_report.py      Write the Phase 3 report
    |-- explain.py                Resample the comparison and split gains by fund
    |-- explain_report.py         Write the Phase 4 report
    |-- presentation.py           Check Phase 1 results for the films
    |-- unified_presentation.py   Check all saved results for the film
    |-- cli.py                    Connect the commands
    |-- __main__.py               Support python -m financial_economics
    +-- __init__.py                Identify the Python package
    |
    v
outputs/phase1/ ... phase4/        Results, saved locally
    |
    v
animations/project_results.py      The film, built from those results
scripts/verify_render.py           Check the rendered film and save review frames
```

`data/` and `outputs/` are created when you run the commands and are ignored by Git.

| Other files | Purpose |
|---|---|
| `tests/` | Check calculations, data validation, the commands, and the film's data checks using synthetic examples |
| [docs/results-explained.md](docs/results-explained.md) | What the study has found, in plain language |
| [docs/methodology.md](docs/methodology.md) | Formulas, sources, assumptions, and interpretation |
| [docs/phase-1.md](docs/phase-1.md), [phase-2.md](docs/phase-2.md), [phase-3.md](docs/phase-3.md), [phase-4.md](docs/phase-4.md) | Each part's question, method, outputs, and checks |
| `pyproject.toml` | Python requirements, packages, and the `fecon` command |
| `uv.lock` | Exact package versions |
| `.github/workflows/tests.yml` | Run tests on Windows and Linux in GitHub Actions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Collaborator access, branches, and pull requests |

To change **what is studied**, edit `study.toml`. To change **how results are calculated**, start in `analysis.py`, `stress.py`, `comparison.py`, or `explain.py`. To change **how results are presented**, start in the matching `*_report.py` file or in `animations/`.

## Current settings and assumptions

The study selects SPY, QQQ, VNQ, GLD, XLE, XLY, VIG, and TLT for January 1, 2010 through December 31, 2025. The exact settings live in `study.toml`.

- CAGR measures compounded growth; volatility measures total return variability. Sharpe uses daily excess returns.
- The equal-weight portfolio resets monthly, with weights drifting between resets. Trading costs and taxes are zero.
- SPY is the portfolio benchmark. CAPM uses French's broader market factor; beta measures exposure to that factor. Phase 3 instead measures sensitivity to SPY's own excess returns.
- TLT is a risky bond fund. The risk-free return, and the "cash" in SPY + cash, come from French's factor file.
- Market falls are dated after the fact, using SPY's own peaks and troughs; they are not a trading rule.
- Phase 3 picks its SPY + cash split from earlier years, but the full dataset and funds had already been examined, so it is a historical check rather than a test on unseen data.
- Phase 4's rebuilt histories reuse the same 2018–2025 days. They show how much the comparison could move by chance, not new evidence; its ranges are percentile intervals rather than a formal significance test.
- These are retrospectively selected surviving funds. Full-period correlations do not establish protection during market declines, and historical alpha does not prove skill or mispricing.

## Development

```sh
uv run pytest -q
```

Tests use synthetic data and require no live finance service. GitHub Actions installs the locked environment and runs the tests on Windows and Linux.

This repository is private. Ask [@N3V3MORE](https://github.com/N3V3MORE) to invite your GitHub username before cloning. Follow [CONTRIBUTING.md](CONTRIBUTING.md) when making changes, and keep downloaded data and generated results out of Git.
