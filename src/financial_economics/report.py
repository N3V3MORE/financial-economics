"""Export inspectable tables and standalone research figures."""

import json
from importlib.metadata import version
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .analysis import capm_summary, drawdowns, monthly_equal_weight, performance_summary
from .config import Study


def _save(figure, destination: Path) -> None:
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _figures(returns: pd.DataFrame, metrics: pd.DataFrame, benchmark: str, output: Path) -> None:
    comparison = returns[[benchmark, "Equal weight"]]
    colors = ["#2563a6", "#b36b00"]
    with plt.rc_context({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"}):
        wealth = (1 + comparison).cumprod()
        anchor = pd.DataFrame(1.0, index=[wealth.index[0] - pd.Timedelta(days=1)],
                              columns=wealth.columns)
        figure, axis = plt.subplots(figsize=(10, 5))
        growth = pd.concat([anchor, wealth])
        for i, name in enumerate(comparison):
            axis.plot(growth.index.to_pydatetime(), growth[name].to_numpy(),
                      color=colors[i], linestyle=["-", "--"][i], label=name)
        axis.set(title="Growth of an initial $1", ylabel="Portfolio value ($)", xlabel="")
        axis.grid(axis="y", alpha=.2)
        axis.legend(title=None)
        _save(figure, output / "growth.png")

        figure, axis = plt.subplots(figsize=(10, 5))
        declines = drawdowns(comparison)
        for i, name in enumerate(comparison):
            axis.plot(declines.index.to_pydatetime(), declines[name].to_numpy(),
                      color=colors[i], linestyle=["-", "--"][i], label=name)
        axis.set(title="Declines from previous portfolio peaks", ylabel="Drawdown", xlabel="")
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=.2)
        axis.legend(title=None)
        _save(figure, output / "drawdowns.png")

        correlation = returns.drop(columns="Equal weight").corr()
        figure, axis = plt.subplots(figsize=(8, 7))
        cells = axis.imshow(correlation, cmap="RdBu", vmin=-1, vmax=1)
        axis.set_xticks(range(len(correlation)), correlation.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(correlation)), correlation.index)
        for i in range(len(correlation)):
            for j in range(len(correlation)):
                value = correlation.iloc[i, j]
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10,
                          color="white" if abs(value) > .65 else "#202020")
        axis.set_title("Daily adjusted-price return correlations", pad=15)
        figure.colorbar(cells, ax=axis, fraction=.046, pad=.04, label="Correlation")
        _save(figure, output / "correlations.png")

        figure, axis = plt.subplots(figsize=(9, 6))
        for i, (name, row) in enumerate(metrics.iterrows()):
            portfolio = name == "Equal weight"
            axis.scatter(row.annualized_volatility * 100, row.cagr * 100,
                         marker="*" if portfolio else "o", s=160 if portfolio else 45,
                         color=colors[1] if portfolio else colors[0])
            axis.annotate(name, (row.annualized_volatility * 100, row.cagr * 100),
                          xytext=(6, 7 if i % 2 else -13), textcoords="offset points", fontsize=10)
        axis.set(title="Realized return and volatility", xlabel="Annualized volatility (%)",
                 ylabel="CAGR (%)")
        axis.margins(.17)
        axis.grid(alpha=.2)
        _save(figure, output / "risk_return.png")


def write_outputs(study: Study, returns: pd.DataFrame, factors: pd.DataFrame,
                  manifest: dict, output: Path) -> None:
    combined = returns.copy()
    combined["Equal weight"] = monthly_equal_weight(returns)
    metrics = performance_summary(combined, factors["RF"], study.trading_days)
    capm = capm_summary(combined, factors, study.hac_lags)
    correlation = returns.corr()
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "daily_returns": combined,
        "performance": metrics,
        "capm": capm,
        "correlation": correlation,
        "covariance_annualized": returns.cov() * study.trading_days,
        "annual_returns": ((1 + combined).groupby(combined.index.year).prod() - 1).rename_axis("year"),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", float_format="%.12g", na_rep="NA")
    _figures(combined, metrics, study.benchmark, output)
    run = {
        "study": study.settings(), "observations": len(returns),
        "actual_start": str(returns.index[0].date()),
        "actual_end": str(returns.index[-1].date()),
        "snapshot_files": manifest["files"],
        "software_versions": {"python": platform.python_version(), **{
            name: version(name) for name in ["financial-economics", "numpy", "pandas", "statsmodels", "matplotlib"]
        }},
        "portfolio": {"rule": "monthly equal weight; drift within month",
                      "trading_costs": 0, "taxes": 0},
        "capm": {"market": "French Mkt-RF", "covariance": "HAC",
                 "lags": study.hac_lags, "kernel": "Bartlett",
                 "finite_sample_correction": True, "inference": "Student t reference"},
    }
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    lines = ["# Phase 1: ETF baseline", "",
             f"{run['actual_start']} to {run['actual_end']}; {len(returns):,} daily observations.", "",
             "Historical descriptions of a fixed, retrospectively selected ETF universe. "
             "No optimized strategy or out-of-sample investment result is claimed.", "",
             "## Return and risk", "",
             "| Asset | CAGR | Annual volatility | Sharpe | Maximum drawdown |",
             "|---|---:|---:|---:|---:|"]
    for name, row in metrics.iterrows():
        sharpe = f"{row.sharpe:.2f}" if np.isfinite(row.sharpe) else "undefined"
        lines.append(f"| {name} | {row.cagr:.2%} | {row.annualized_volatility:.2%} | "
                     f"{sharpe} | {row.max_drawdown:.2%} |")
    lines += ["", f"Annualization uses {study.trading_days} trading days. CAGR compounds returns; volatility "
              "and Sharpe use square-root-of-time scaling. Maximum drawdown includes initial capital.",
              "", "![Growth of $1](growth.png)", "", "![Drawdowns](drawdowns.png)",
              "", "![Realized return and volatility](risk_return.png)", "",
              "## Market exposure", "",
              f"CAPM uses French's broad U.S. market excess return. {study.benchmark} is the portfolio benchmark, "
              "not that factor. These regressions describe equity-market exposure; they do not establish "
              "a complete pricing model for gold or long-duration bonds.", "",
              "| Asset | Daily alpha | Alpha p-value | Beta | Beta 95% interval | R-squared |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, row in capm.iterrows():
        p = "<0.001" if row.alpha_pvalue < .001 else f"{row.alpha_pvalue:.3f}"
        lines.append(f"| {name} | {row.alpha_daily:.4%} | {p} | {row.beta:.3f} | "
                     f"[{row.beta_ci_low:.3f}, {row.beta_ci_high:.3f}] | {row.r_squared:.3f} |")
    lines += ["", f"Inference uses Newey-West HAC errors with {study.hac_lags} lags, Bartlett weights, "
              "a finite-sample covariance correction, and a Student t reference distribution. "
              "The CSV includes alpha and beta standard errors, confidence intervals, and p-values.",
              "", "## Co-movement", "", "![Correlations](correlations.png)", "",
              "Correlation describes co-movement, not its cause. A whole-period matrix does not "
              "establish how diversification behaves in market declines.", "",
              "## Assumptions and limits", "",
              "- Adjusted-price returns reflect the provider's distribution and split adjustments; dividends are not added twice.",
              "- The equal-weight portfolio holds all selected funds, resets monthly, and drifts between resets.",
              "- Trading costs and taxes are zero. Observed fund returns already reflect fund operating expenses.",
              "- TLT is a risky long-duration bond fund, not the risk-free asset.",
              "- Historical mean returns and CAPM alpha are estimates, not guaranteed future premiums or proof of skill.",
              "- No missing prices or unmatched factor dates were filled or silently removed.", "",
              "## Sources and reproduction", "",
              "ETF provider snapshot: Yahoo Finance via yfinance. Factors: "
              "[Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).",
              "", "The source manifest records settings, retrieval time, versions, and file hashes. "
              "run.json records the snapshot hashes and conventions used for this analysis. "
              "Downloaded values are kept locally rather than redistributed in Git.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
