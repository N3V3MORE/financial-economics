"""Fixed ETF additions versus same-stock-weight cash controls; no weight search."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from financial_economics.evaluation.metrics import performance_summary
from financial_economics.portfolio.execution import monthly_costed_path
from financial_economics.evaluation.protection import describe_funds, infer_protection
from financial_economics.datasets.returns import compound_months
from financial_economics.portfolio.declines import compare_episodes, drawdown_episodes


@dataclass(frozen=True)
class ProtectionResult:
    tables: dict[str, pd.DataFrame]
    metadata: dict


def _beta(portfolio, market, rf):
    x, y = (market - rf).to_numpy(), (portfolio - rf).to_numpy()
    variance = np.var(x, ddof=1)
    return float(np.cov(x, y, ddof=1)[0, 1] / variance) if len(x) > 2 and variance > 0 else np.nan


def _summarize(paths, cash, monthly, monthly_market, comparisons, window, cost, trading_days, threshold):
    metrics = performance_summary(paths, cash, trading_days).reset_index(names="portfolio")
    metrics["window"], metrics["cost_bps"] = window, cost
    rows = []
    for portfolio, control in comparisons.items():
        for condition, mask in [("negative_spy", monthly_market < 0),
                                ("severe_spy", monthly_market <= threshold)]:
            gap = (monthly[portfolio] - monthly[control]).loc[mask]
            rows.append({"portfolio": portfolio, "control": control, "window": window,
                         "cost_bps": cost, "condition": condition, "months": len(gap),
                         "mean_advantage": gap.mean(),
                         "beat_control_frequency": (gap > 0).mean() if len(gap) else np.nan,
                         "status": "available" if len(gap) else "no_matching_months"})
    return metrics, pd.DataFrame(rows)


def run_protection(data, study, settings):
    settings.validate(study)
    if not data.daily.index.equals(data.factors.index):
        raise ValueError("Cash-return dates must exactly match ETF dates.")
    daily = data.daily.reindex(columns=study.tickers).copy()
    daily["RF"] = data.factors.RF
    monthly = compound_months(daily, daily.index, study.start, study.end)
    benchmark = study.benchmark
    candidates = [fund for fund in study.tickers if fund != benchmark]
    definitions, comparisons, roles = {}, {}, []

    def add(name, values, kind, fund="", allocation=np.nan):
        weights = {asset: values.get(asset, 0.0) for asset in daily.columns}
        if name in definitions:
            raise ValueError("Portfolio names must be distinct.")
        definitions[name] = weights
        roles.append({"portfolio": name, "kind": kind, "fund": fund, "candidate_allocation": allocation})

    add(benchmark, {benchmark: 1}, "reference")
    for allocation in settings.allocations:
        control = f"{1-allocation:.0%} {benchmark} + {allocation:.0%} cash"
        add(control, {benchmark: 1-allocation, "RF": allocation}, "cash_control", allocation=allocation)
        for fund in candidates:
            name = f"{1-allocation:.0%} {benchmark} + {allocation:.0%} {fund}"
            add(name, {benchmark: 1-allocation, fund: allocation}, "candidate", fund, allocation)
            comparisons[name] = control
    for name, funds in settings.fixed_portfolios.items():
        add(name, {fund: 1/len(funds) for fund in funds}, "fixed_equal_weight")

    full = f"{study.start[:4]}:{study.end[:4]}"
    windows = list(dict.fromkeys([full, *settings.windows]))
    fund_rows = []
    for window in windows:
        start, end = window.split(":")
        desc = describe_funds(monthly.loc[start:end], benchmark, settings.severe_threshold)
        desc["window"] = window
        fund_rows.append(desc)
    tables = {"fund_monthly_returns": monthly, "fund_conditions": pd.concat(fund_rows, ignore_index=True),
              "portfolio_definitions": pd.DataFrame(roles),
              "target_weights": pd.DataFrame(definitions).T.rename_axis("portfolio")}
    if settings.primary_inference:
        tests, intervals = infer_protection(monthly, settings, benchmark)
        tables.update(primary_tests=tests, bootstrap_intervals=intervals)

    performance, advantages, fees, stored_daily, stored_monthly, episode_rows = [], [], [], [], [], []
    episodes = drawdown_episodes(daily[benchmark], data.initial_date, threshold=.10)
    tables["spy_episodes"] = episodes
    tables["fund_episode_comparisons"] = compare_episodes(daily, episodes, data.initial_date)
    gross_paths = None
    for cost in settings.costs_bps:
        paths = {}
        for name, targets in definitions.items():
            path = monthly_costed_path(daily, targets, cost)
            paths[name] = path["return"]
            wealth_before = (1 + path["return"]).cumprod().shift(1, fill_value=1)
            accounting = pd.DataFrame({"fees_per_initial_dollar": wealth_before * path.fee_fraction,
                                       "traded_per_initial_dollar": wealth_before * path.traded_fraction})
            accounting = accounting.groupby(accounting.index.to_period("M")).sum().reset_index(names="month")
            accounting["portfolio"], accounting["cost_bps"] = name, cost
            fees.append(accounting)
        paths = pd.DataFrame(paths)
        if cost == 0:
            gross_paths = paths
        months = compound_months(paths, daily.index, study.start, study.end)
        # This is an accounting identity, checked for every fund, weight, and month.
        if cost == 0:
            for row in roles:
                if row["kind"] == "candidate":
                    actual = months[row["portfolio"]] - months[comparisons[row["portfolio"]]]
                    expected = row["candidate_allocation"] * (monthly[row["fund"]] - monthly.RF)
                    if not np.allclose(actual, expected, rtol=0, atol=5e-13):
                        raise ArithmeticError("Zero-cost allocation identity failed.")
        for window in windows:
            start, end = window.split(":")
            metrics, gaps = _summarize(paths.loc[start:end], daily.RF.loc[start:end], months.loc[start:end],
                                      monthly[benchmark].loc[start:end], comparisons, window, cost,
                                      study.trading_days, settings.severe_threshold)
            performance.append(metrics)
            advantages.append(gaps)
        for frame, stored in ((paths, stored_daily), (months, stored_monthly)):
            copy = frame.copy()
            copy.columns = [f"{name} | {cost:g}bps" for name in copy.columns]
            stored.append(copy)
        episode_table = compare_episodes(paths, episodes, data.initial_date)
        episode_table["cost_bps"] = cost
        episode_rows.append(episode_table)

    # Freeze gross training betas, then start both evaluated portfolios and controls at the
    # evaluation seed with entry costs. No evaluation observations enter weight selection.
    training = gross_paths.loc[settings.training_start:settings.training_end]
    evaluation = daily.loc[settings.evaluation_start:]
    calibration, beta_metrics, beta_gaps = [], [], []
    for row in roles:
        if row["kind"] not in ("candidate", "fixed_equal_weight"):
            continue
        name = row["portfolio"]
        raw = _beta(training[name], daily.loc[training.index, benchmark], daily.RF.loc[training.index])
        if not np.isfinite(raw):
            calibration.append({"portfolio": name, "status": "undefined_training_beta"})
            continue
        weight = float(np.clip(raw, 0, 1))
        control_weights = dict.fromkeys(daily.columns, 0.)
        control_weights.update({benchmark: weight, "RF": 1-weight})
        for cost in settings.costs_bps:
            pair = pd.DataFrame({name: monthly_costed_path(evaluation, definitions[name], cost)["return"],
                                 "Frozen SPY/cash control": monthly_costed_path(evaluation, control_weights, cost)["return"]})
            calibration.append({"portfolio": name, "cost_bps": cost, "training_beta": raw,
                                "control_spy_target": weight, "clipped": raw != weight,
                                "realized_portfolio_beta": _beta(pair[name], evaluation[benchmark], evaluation.RF),
                                "realized_control_beta": _beta(pair.iloc[:, 1], evaluation[benchmark], evaluation.RF),
                                "training_days": len(training), "evaluation_days": len(evaluation), "status": "available"})
            pair_months = compound_months(pair, evaluation.index, settings.evaluation_start, study.end)
            metrics, gaps = _summarize(pair, evaluation.RF, pair_months,
                                      monthly.loc[settings.evaluation_start:, benchmark],
                                      {name: "Frozen SPY/cash control"},
                                      f"{settings.evaluation_start[:4]}:{study.end[:4]}", cost,
                                      study.trading_days, settings.severe_threshold)
            metrics["matched_portfolio"] = name
            beta_metrics.append(metrics)
            beta_gaps.append(gaps)
    tables.update(performance=pd.concat(performance, ignore_index=True),
                  portfolio_conditions=pd.concat(advantages, ignore_index=True),
                  fees=pd.concat(fees, ignore_index=True), daily_returns=pd.concat(stored_daily, axis=1),
                  monthly_returns=pd.concat(stored_monthly, axis=1),
                  episode_comparisons=pd.concat(episode_rows, ignore_index=True),
                  beta_calibration=pd.DataFrame(calibration),
                  beta_performance=pd.concat(beta_metrics, ignore_index=True) if beta_metrics else pd.DataFrame(),
                  beta_conditions=pd.concat(beta_gaps, ignore_index=True) if beta_gaps else pd.DataFrame())
    return ProtectionResult(tables, {"initial_valuation_date": str(data.initial_date.date()),
                                    "daily_observations": len(daily), "monthly_observations": len(monthly),
                                    "candidate_count": len(candidates), "portfolio_count": len(definitions),
                                    "zero_cost_allocation_identity": "passed", "backend": "CPU-native NumPy",
                                    "subperiod_costs": "slice continuously held full-sample paths; no repeated entry fee",
                                    "beta_costs": "fresh evaluation-period entry for portfolio and frozen control"})
