"""Daily target-volatility portfolios using previously issued variance forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analysis import _check_frame, _check_returns, performance_summary
from .costs import dynamic_costed_path
from .prediction_inference import infer_prediction
from .return_data import compound_months
from .stress import compare_episodes, drawdown_episodes


@dataclass(frozen=True)
class PredictionResult:
    tables: dict[str, pd.DataFrame]
    metadata: dict


def _validated_forecasts(forecasts, daily, settings):
    required = {"trade_date", "return_date", "squared_return", "ewma_variance"}
    if not isinstance(forecasts, pd.DataFrame) or not required.issubset(forecasts.columns):
        raise ValueError("Forecasts need trade/return dates, squared returns, and EWMA variance.")
    _check_frame(forecasts[["squared_return", "ewma_variance"]], "Forecasts")
    rows = forecasts.copy()
    if (rows.squared_return < 0).any() or (rows.ewma_variance <= 0).any():
        raise ValueError("Forecasts require nonnegative labels and positive variance.")
    calendar = daily.index
    positions = calendar.get_indexer(rows.index)
    if (positions < 0).any() or (positions + 2 >= len(calendar)).any():
        raise ValueError("Forecast dates need two subsequent trading-calendar observations.")
    for name, offset in (("trade_date", 1), ("return_date", 2)):
        dates = pd.DatetimeIndex(pd.to_datetime(rows[name]))
        if not dates.equals(calendar[positions + offset]):
            raise ValueError(f"Forecast {name} must follow the trading calendar at t+{offset}.")
        rows[name] = dates
    expected_labels = daily.SPY.loc[rows.return_date].to_numpy(dtype=float) ** 2
    if not np.allclose(rows.squared_return, expected_labels, atol=1e-18, rtol=1e-12):
        raise ValueError("Forecast squared-return labels must match their earned return dates.")
    models = ["ewma"]
    if "gru_variance" in rows:
        if (not pd.api.types.is_numeric_dtype(rows.gru_variance.dtype)
                or pd.api.types.is_bool_dtype(rows.gru_variance.dtype)):
            raise ValueError("GRU variances must be numeric.")
        values = rows.gru_variance.to_numpy(dtype=float)
        if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
            raise ValueError("Available GRU variance forecasts must be finite and positive.")
        models.append("gru")
    preceding = calendar[calendar <= pd.Timestamp(settings.validation_end)]
    if not len(preceding):
        raise ValueError("Validation needs an observed closing date before evaluation.")
    freeze = preceding[-1]
    freeze_position = calendar.get_loc(freeze)
    if freeze_position + 2 >= len(calendar):
        raise ValueError("Evaluation needs two trading dates after settings freeze.")
    evaluation = calendar[(calendar >= pd.Timestamp(settings.evaluation_start))]
    if not len(evaluation):
        raise ValueError("No evaluation trading dates are available.")
    expected = evaluation[evaluation >= calendar[freeze_position + 2]]
    eligible = rows.loc[(rows.index >= freeze) & (rows.return_date >= evaluation[0])].copy()
    if not pd.DatetimeIndex(eligible.return_date).equals(expected):
        raise ValueError("Evaluation forecasts must map every eligible return date exactly once.")
    for model in models:
        if not np.isfinite(eligible[f"{model}_variance"].to_numpy(dtype=float)).all():
            raise ValueError(f"Every evaluation-use {model.upper()} forecast must be available.")
    return rows, eligible, evaluation, freeze, models


def _allocations(rows, eligible, evaluation, calendar, settings, trading_days, defensive_asset, models):
    years = sorted(set(evaluation.year))
    asofs = pd.DatetimeIndex([calendar[calendar < pd.Timestamp(year=year, month=1, day=1)][-1]
                             for year in years])
    blocks = np.searchsorted(asofs.to_numpy(), eligible.index.to_numpy(), side="right") - 1
    if (blocks < 0).any():
        raise ValueError("Forecasts cannot precede their first annual settings freeze.")
    definitions, decision_rows, thresholds = {}, [], []
    for model in models:
        variance = eligible[f"{model}_variance"].to_numpy(dtype=float)
        volatility = np.sqrt(trading_days * variance)
        stock = np.clip(settings.volatility_target / volatility, 0, 1)
        cutoffs = []
        for year, asof in zip(years, asofs):
            calibration = ((rows.index >= pd.Timestamp(settings.validation_start))
                           & (rows.index < asof))
            previous = rows.loc[calibration, f"{model}_variance"].dropna().to_numpy(dtype=float)
            if not len(previous):
                raise ValueError(f"{model.upper()} needs preceding issued forecasts for annual stress thresholds.")
            cutoff = float(np.quantile(np.sqrt(trading_days * previous), settings.stress_quantile))
            cutoffs.append(cutoff)
            thresholds.append({"model": model, "evaluation_year": year, "refit_asof": asof,
                               "stress_volatility_threshold": cutoff,
                               "calibration_forecasts": len(previous)})
        cutoff = np.asarray(cutoffs)[blocks]
        stress = volatility > cutoff
        bonds = np.where(stress, np.minimum(settings.defensive_cap, 1 - stock), 0)
        for name, allocation in (("A" if model == "ewma" else "B", np.zeros(len(stock))),
                                 ("C" if model == "ewma" else "D", bonds)):
            target = pd.DataFrame({"SPY": 0., defensive_asset: 0., "RF": 1.}, index=evaluation)
            target.loc[eligible.return_date, "SPY"] = stock
            target.loc[eligible.return_date, defensive_asset] = allocation
            target.loc[eligible.return_date, "RF"] = 1 - stock - allocation
            definitions[name] = target
        decision = eligible[["trade_date", "return_date"]].reset_index(names="forecast_date")
        if model == "gru" and "gru_refit_asof" in eligible:
            decision["gru_refit_asof"] = eligible.gru_refit_asof.to_numpy()
        decision["model"] = model
        decision["refit_asof"] = asofs[blocks]
        decision["predicted_variance"] = variance
        decision["predicted_annualized_volatility"] = volatility
        decision["stress_volatility_threshold"] = cutoff
        decision["stress"] = stress
        decision["spy_target"] = stock
        decision["defensive_target"] = bonds
        decision_rows.append(decision)
    return definitions, pd.concat(decision_rows, ignore_index=True), pd.DataFrame(thresholds)


def run_prediction(data, study, settings, forecasts, defensive_asset="TLT"):
    """Run A/C for EWMA alone or A/B/C/D when saved GRU forecasts are supplied.

    Targets are indexed by earned-return dates. Before the first executable
    frozen-settings forecast, the portfolio stays in cash, retaining complete
    monthly evaluation calendars. No model is trained by this function.
    """
    settings.validate(study)
    _check_returns(data.daily)
    if not data.daily.index.equals(data.factors.index):
        raise ValueError("Cash-return dates must exactly match ETF dates.")
    if defensive_asset in ("SPY", "RF") or defensive_asset not in data.daily:
        raise ValueError("The defensive ETF must be an available non-SPY return column.")
    daily = data.daily[["SPY", defensive_asset]].copy()
    daily["RF"] = data.factors.RF
    _check_returns(daily)
    if daily.index[0] < pd.Timestamp(study.start) or daily.index[-1] > pd.Timestamp(study.end):
        raise ValueError("Prediction returns must lie inside the declared study dates.")
    rows, eligible, evaluation, freeze, models = _validated_forecasts(forecasts, daily, settings)
    definitions, decisions, thresholds = _allocations(rows, eligible, evaluation, daily.index,
                                                      settings, study.trading_days, defensive_asset, models)
    definitions = {name: definitions[name] for name in sorted(definitions)}
    evaluated = daily.loc[evaluation]
    benchmark_months = compound_months(evaluated[["SPY"]], daily.index,
                                       settings.evaluation_start, study.end).SPY
    weights = pd.concat([target.reset_index(names="Date").assign(portfolio=name)
                         for name, target in definitions.items()], ignore_index=True)
    full = f"{settings.evaluation_start[:4]}:{study.end[:4]}"
    windows = list(dict.fromkeys([full, *settings.windows]))
    scenarios = [(f"cash_{cost:g}bps", float(cost), True) for cost in settings.costs_bps]
    scenarios.append(("zero_cash_10bps", 10., False))
    metrics, conditions, accounting, stored_daily, stored_monthly = [], [], [], [], []
    main_paths = main_months = None
    pairs = [("C", "A")]
    if "gru" in models:
        pairs += [("B", "A"), ("D", "C"), ("D", "B")]
    for scenario, cost, cash_income in scenarios:
        inputs = evaluated.copy()
        if not cash_income:
            inputs["RF"] = 0.
        paths, fees_by_portfolio = {}, {}
        for name, target in definitions.items():
            path = dynamic_costed_path(inputs, target, cost)
            paths[name] = path["return"]
            wealth_before = (1 + path["return"]).cumprod().shift(1, fill_value=1.)
            fees = path.copy()
            fees["fees_per_initial_dollar"] = wealth_before * path.fee_fraction
            fees["traded_per_initial_dollar"] = wealth_before * path.traded_fraction
            fees_by_portfolio[name] = fees
            accounting.append(fees.reset_index(names="Date").assign(
                portfolio=name, scenario=scenario, cost_bps=cost, cash_income=cash_income))
        paths = pd.DataFrame(paths)
        months = compound_months(paths, daily.index, settings.evaluation_start, study.end)
        if cash_income and cost == 10:
            main_paths, main_months = paths, months
        for window in windows:
            first, last = window.split(":")
            selected = paths.loc[first:last]
            if len(selected) < 2:
                continue
            # Sharpe always uses observed RF as the opportunity-cost benchmark,
            # including when the portfolio's cash-income sensitivity earns zero.
            summary = performance_summary(selected, evaluated.RF.loc[selected.index],
                                          study.trading_days).reset_index(names="portfolio")
            summary["window"] = window
            summary["scenario"], summary["cost_bps"], summary["cash_income"] = scenario, cost, cash_income
            summary["turnover"] = [fees_by_portfolio[name].loc[first:last, "traded_fraction"].sum()
                                    for name in summary.portfolio]
            summary["traded_per_initial_dollar"] = [fees_by_portfolio[name].loc[first:last, "traded_per_initial_dollar"].sum()
                                                     for name in summary.portfolio]
            summary["fees_per_initial_dollar"] = [fees_by_portfolio[name].loc[first:last, "fees_per_initial_dollar"].sum()
                                                  for name in summary.portfolio]
            metrics.append(summary)
            market = benchmark_months.loc[first:last]
            for portfolio, control in pairs:
                gap = (months[portfolio] - months[control]).loc[first:last]
                for condition, mask in (("negative_spy", market < 0), ("severe_spy", market <= -.05)):
                    values = gap.loc[mask]
                    conditions.append({"portfolio": portfolio, "control": control, "window": window,
                                       "scenario": scenario, "cost_bps": cost, "cash_income": cash_income,
                                       "condition": condition, "months": len(values),
                                       "mean_advantage": values.mean(),
                                       "beat_control_frequency": (values > 0).mean() if len(values) else np.nan,
                                       "status": "available" if len(values) else "no_matching_months"})
        stored_daily.append(paths.add_suffix(f" | {scenario}"))
        stored_monthly.append(months.add_suffix(f" | {scenario}"))
    tables = {"forecasts": rows, "decisions": decisions, "annual_thresholds": thresholds,
              "target_weights": weights, "performance": pd.concat(metrics, ignore_index=True),
              "portfolio_conditions": pd.DataFrame(conditions), "fees": pd.concat(accounting, ignore_index=True),
              "daily_returns": pd.concat(stored_daily, axis=1),
              "monthly_returns": pd.concat(stored_monthly, axis=1),
              "benchmark_monthly_returns": benchmark_months.to_frame(),
              "scenarios": pd.DataFrame(scenarios, columns=["scenario", "cost_bps", "cash_income"])}
    primary = "gru" in models and defensive_asset == "TLT" and main_months is not None
    if primary:
        tests, intervals = infer_prediction(eligible, main_months.assign(SPY=benchmark_months), settings)
        tables.update(primary_tests=tests, bootstrap_intervals=intervals)
    if main_paths is not None:
        seed = daily.index[daily.index < evaluation[0]][-1]
        episodes = drawdown_episodes(evaluated.SPY, seed)
        tables["spy_episodes"] = episodes
        tables["episode_comparisons"] = compare_episodes(main_paths.assign(SPY=evaluated.SPY), episodes, seed)
    return PredictionResult(tables, {
        "stage": "full" if "gru" in models else "baseline", "models": models,
        "defensive_asset": defensive_asset, "freeze_asof": str(freeze.date()),
        "evaluation_start": str(evaluation[0].date()), "evaluation_end": str(evaluation[-1].date()),
        "daily_observations": len(evaluation), "monthly_observations": len(benchmark_months),
        "forecast_observations": len(eligible), "warmup_cash_days": len(evaluation) - len(eligible),
        "primary_inference": primary, "backend": "CPU-native NumPy accounting and bootstrap",
        "execution": "forecast t close; trade t+1 close; earn daily return ending t+2",
        "annual_thresholds": "common validation-start history; issue dates strictly before preceding year-end trading close",
        "cash_sensitivity_sharpe": "observed RF remains the excess-return benchmark",
        "fees": "ETF purchases plus sales; entry included; exit and taxes excluded",
        "subperiod_costs": "slices of continuously held evaluation paths; no repeated entry",
        "bootstrap_scope": "conditional on fitted forecasts; excludes model-selection uncertainty"})
