"""Test each fund's place in a fixed monthly mix, using CPU-native calculations."""

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from financial_economics.evaluation.metrics import (_check_returns, monthly_equal_weight,
                       monthly_equal_weight_holdings, performance_summary)
from financial_economics.portfolio.controls import stock_exposure
from financial_economics.portfolio.execution import monthly_costed_path


DEFAULT_WINDOWS = (
    "2016-01-01:2025-12-31", "2018-01-01:2025-12-31", "2020-01-01:2025-12-31",
    "2018-01-01:2019-12-31", "2020-01-01:2021-12-31",
    "2022-01-01:2023-12-31", "2024-01-01:2025-12-31",
)
MEASURES = ["cagr", "annualized_volatility", "sharpe", "max_drawdown"]


@dataclass(frozen=True)
class FundChecks:
    performance: pd.DataFrame
    differences: pd.DataFrame
    calibration: pd.DataFrame
    daily_returns: pd.DataFrame
    portfolios: dict[str, list[str]]


def check_funds(returns: pd.DataFrame, risk_free: pd.Series, base_assets: list[str],
                additions: tuple[str, ...] = (), benchmark: str = "SPY",
                windows: tuple[str, ...] = DEFAULT_WINDOWS,
                costs_bps: tuple[float, ...] = (0, 10, 25), trading_days: int = 252
                ) -> FundChecks:
    """Compare fixed portfolios, never selecting weights from evaluation results.

    Keep the original mix's SPY/cash control as a common reference. Also fit a
    separate control for each variant, using only dates before each test window.
    Returns include a fresh entry fee for every portfolio at each window's seed
    close. Mixes enter at their existing monthly weights, not a midmonth reset.
    """
    _check_returns(returns)
    _check_returns(risk_free.to_frame("RF"))
    if not returns.index.equals(risk_free.index):
        raise ValueError("Risk-free dates must match return dates exactly.")
    selected = [*base_assets, *additions]
    if (len(base_assets) < 2 or len(set(selected)) != len(selected)
            or not set(selected).issubset(returns.columns) or benchmark not in base_assets
            or "RF" in returns):
        raise ValueError("Choose distinct available base funds and additions, including the benchmark.")
    if not windows or not costs_bps or len(set(costs_bps)) != len(costs_bps):
        raise ValueError("Choose test windows and distinct cost scenarios.")
    portfolios = {"Original mix": list(base_assets)}
    portfolios.update({f"Without {fund}": [asset for asset in base_assets if asset != fund]
                       for fund in base_assets})
    portfolios.update({f"With {fund}": [*base_assets, fund] for fund in additions})
    gross = {name: monthly_equal_weight(returns[assets]) for name, assets in portfolios.items()}
    holdings = {name: monthly_equal_weight_holdings(returns[assets])
                for name, assets in portfolios.items()}
    performance, differences, calibrations, daily_tables = [], [], [], []
    seen = set()
    common_control = f"{benchmark} + cash"
    for window in windows:
        start, end = (pd.Timestamp(value) for value in window.split(":"))
        if any(pd.isna(day) or day.tz is not None or day != day.normalize() for day in (start, end)):
            raise ValueError("Window bounds must be timezone-naive daily dates.")
        training = returns.index[returns.index < start]
        dates = returns.index[(returns.index >= start) & (returns.index <= end)]
        # Bounds may include a holiday, but not an unobserved calendar month.
        if (start > end or end.to_period("M") > returns.index[-1].to_period("M")
                or len(training) < 3 or len(dates) < 3):
            raise ValueError("Each window needs at least three earlier and three evaluation observations.")
        actual = (dates[0], dates[-1])
        if actual in seen:
            raise ValueError("Test windows must select distinct actual periods.")
        seen.add(actual)
        context = {"window": window, "start": str(dates[0].date()), "end": str(dates[-1].date())}
        fits = {}
        for name, series in gross.items():
            frame = pd.DataFrame({benchmark: returns[benchmark], "Mix": series}).loc[training]
            raw_beta = float(stock_exposure(frame, risk_free.loc[training], benchmark).loc["Mix", "beta"])
            fits[name] = float(np.clip(raw_beta, 0, 1))
            calibrations.append({**context, "portfolio": name, "training_start": str(training[0].date()),
                                 "training_end": str(training[-1].date()), "training_n": len(training),
                                 "raw_beta": raw_beta, "stock_weight": fits[name],
                                 "weight_clipped": raw_beta != fits[name]})
        for cost in costs_bps:
            paths, accounting = {}, {}
            for name, assets in portfolios.items():
                inputs = returns.loc[dates, assets].assign(RF=risk_free.loc[dates])
                targets = {**dict.fromkeys(assets, 1 / len(assets)), "RF": 0.0}
                initial = {**holdings[name].loc[dates[0]].to_dict(), "RF": 0.0}
                path = monthly_costed_path(inputs, targets, cost, initial)
                paths[name], accounting[name] = path["return"], path
                control_name = common_control if name == "Original mix" else f"Control: {name}"
                control_inputs = returns.loc[dates, [benchmark]].assign(RF=risk_free.loc[dates])
                control = monthly_costed_path(
                    control_inputs, {benchmark: fits[name], "RF": 1 - fits[name]}, cost)
                paths[control_name], accounting[control_name] = control["return"], control
            spy = monthly_costed_path(returns.loc[dates, [benchmark]].assign(RF=risk_free.loc[dates]),
                                      {benchmark: 1, "RF": 0}, cost)
            paths[benchmark], accounting[benchmark] = spy["return"], spy
            frame = pd.DataFrame(paths)
            metrics = performance_summary(frame, risk_free.loc[dates], trading_days)
            # Descriptive beta uses gross SPY as the market, not its entry-fee return.
            exposure_input = frame.drop(columns=benchmark).assign(**{benchmark: returns.loc[dates, benchmark]})
            exposure = stock_exposure(exposure_input, risk_free.loc[dates], benchmark)
            for name, row in metrics.iterrows():
                performance.append({**context, "cost_bps": cost, "portfolio": name, **row.to_dict(),
                                    "realized_beta": float(exposure.loc[name, "beta"]),
                                    "annual_traded_fraction": accounting[name].traded_fraction.sum() * trading_days / len(dates),
                                    "sum_fee_fractions": accounting[name].fee_fraction.sum()})
            for name in portfolios:
                own_control = common_control if name == "Original mix" else f"Control: {name}"
                for reference in dict.fromkeys(["Original mix", common_control, own_control]):
                    if name != reference:
                        differences.append({**context, "cost_bps": cost, "portfolio": name,
                                            "reference": reference,
                                            **{measure: float(metrics.loc[name, measure] - metrics.loc[reference, measure])
                                               for measure in MEASURES}})
            frame.index.name = "Date"
            daily_tables.append(frame.reset_index().assign(window=window, cost_bps=cost))
    return FundChecks(pd.DataFrame(performance), pd.DataFrame(differences),
                      pd.DataFrame(calibrations), pd.concat(daily_tables, ignore_index=True), portfolios)


def write_fund_checks(result: FundChecks, study, manifest: dict, output: Path,
                      windows: tuple[str, ...], costs_bps: tuple[float, ...]) -> None:
    """Retain numeric evidence and settings; refuse to replace earlier results."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("performance", "differences", "calibration", "daily_returns"):
        getattr(result, name).to_csv(output / f"{name}.csv", index=False,
                                    float_format="%.12g", date_format="%Y-%m-%d", na_rep="NA")
    run = {"study": study.settings(), "source_manifest": manifest,
           "portfolios": result.portfolios, "windows": list(windows), "costs_bps": list(costs_bps),
           "backend": {"classification": "CPU-native", "library": "NumPy", "device": "cpu"},
           "versions": {name: version(name) for name in ("numpy", "pandas", "financial-economics")},
           "method": {"rebalance": "monthly before first return; weights drift within month",
                      "controls": "pre-window gross beta with intercept, clipped to [0,1]; common and variant-specific",
                      "fees": "per absolute ETF dollar bought or sold; self-financing; entry charged, no exit charge",
                      "cash": "French RF proxy, no transaction fee", "taxes": "excluded",
                      "initial_mix": "weights carried from full-history monthly portfolio; entry charged at window seed close"},
           "limitations": ["Retrospective surviving funds; previously examined history.",
                           "Overlapping windows are not independent evidence.",
                           "Costs are scenarios, not measured historical execution costs.",
                           "No multiple-testing adjustment or proof of future superiority.",
                           "Matched training beta need not stay matched in evaluation."]}
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
