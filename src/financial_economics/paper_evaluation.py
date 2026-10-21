"""Paper-style U.S. comparisons with explicit metric and data limitations.

Portfolio accounting and HAC inference are CPU-native. No parameters are fitted
to portfolio returns. Full-test stress quantiles are deliberately retrospective.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .analysis import _check_returns, performance_summary
from .costs import dynamic_costed_path
from .paper_sources import ASSETS
from .prediction_inference import _hac_test


MODELS = ("arfima", "garch", "gjr", "gru_univar", "gru_multivar")
# Post hoc log-BV GRU fixes are compared with their raw-BV model's published row.
FIXED_MODELS = (*MODELS, "gru_univar_log", "gru_multivar_log")
DECLINES = (("COVID", "2020-02-19", "2020-03-23"),
            ("2018", "2018-09-20", "2018-12-24"),
            ("2011", "2011-04-29", "2011-10-03"),
            ("2010", "2010-04-23", "2010-07-02"),
            ("2015_2016", "2015-07-20", "2016-02-11"))
# Main Tables 3 and 4, model order above; ratios are rounded published values.
PUBLISHED = ((.0034, .3197, 1., .80), (.0127, .7920, .95, .71),
             (.0146, .7320, .88, .71), (.0043, .3046, .95, .70),
             (.0031, .2618, .97, .75))


def _metrics(paths, trading_days):
    result = performance_summary(paths, pd.Series(0., index=paths.index), trading_days)
    volatility = result.annualized_volatility.replace(0, np.nan)
    result["paper_inferred_sharpe"] = result.cagr / volatility
    result["calmar"] = result.cagr / (-result.max_drawdown).replace(0, np.nan)
    # Israelsen's negative-return adjustment; geometric annual numerator is inferred.
    result["paper_inferred_modified_sharpe"] = result.cagr / volatility.pow(np.sign(result.cagr))
    annual_mean = paths.mean() * trading_days
    result["arithmetic_modified_sharpe"] = annual_mean / volatility.pow(np.sign(annual_mean))
    return result.rename_axis("portfolio")


def _forecast_scores(forecasts, settings, models=MODELS):
    labeled = forecasts.realized_bv.notna()
    actual_bv = forecasts.loc[labeled, "realized_bv"].to_numpy(dtype=float)
    if not len(actual_bv) or not np.isfinite(actual_bv).all() or (actual_bv <= 0).any():
        raise ValueError("Forecast scoring requires positive observed BV labels.")
    actual = settings["monthly_multiplier"] * np.sqrt(actual_bv)
    scores, losses = [], {}
    for model in models:
        predicted_bv = forecasts.loc[labeled, f"{model}_bv"].to_numpy(dtype=float)
        predicted = settings["monthly_multiplier"] * np.sqrt(predicted_bv)
        losses[model] = {"monthly_mse": (actual - predicted) ** 2,
                         "monthly_mape": np.abs(actual - predicted) / actual}
        scores.append({"model": model, "n_obs": len(actual),
                       "forecast_observations": len(forecasts),
                       "floor_forecasts": int((forecasts[f"{model}_bv"] <= 1e-12).sum()),
                       "floor_fraction": float((forecasts[f"{model}_bv"] <= 1e-12).mean()),
                       "bv_mse": float(np.mean((actual_bv - predicted_bv) ** 2)),
                       **{name: float(values.mean()) for name, values in losses[model].items()}})
    tests = []
    for row, column in combinations(models, 2):
        for measure in ("monthly_mse", "monthly_mape"):
            difference = losses[column][measure] - losses[row][measure]
            inference = _hac_test(difference, np.ones((len(difference), 1)),
                                  0, settings["dm_hac_lags"], .05)
            tests.append({"row_model": row, "column_model": column, "measure": measure,
                          "loss_column_minus_row": float(difference.mean()),
                          "dm_statistic": (float(difference.mean() / inference["hac_se"])
                                           if inference["status"] == "available" else np.nan),
                          "n_obs": len(difference), **inference})
    tests = pd.DataFrame(tests)
    valid = tests.p_value.notna()
    tests["holm_p_value"] = np.where(valid, multipletests(tests.p_value.fillna(1), method="holm")[1], np.nan)
    return pd.DataFrame(scores).set_index("model"), tests


def evaluate_paper(returns, forecasts, settings, models=MODELS):
    """Evaluate each model and all 13 x 3 x 3 defensive additions."""
    _check_returns(returns, minimum=2)
    if set(returns) != {"EQUITY", *ASSETS}:
        raise ValueError("Evaluation requires the equity index and exactly 13 defensive proxies.")
    daily = returns.loc[settings["evaluation_start"]:settings["evaluation_end"]].copy()
    eligible = forecasts.loc[forecasts.return_date.isin(daily.index)].copy()
    if (eligible.empty or not eligible.index.is_unique or not eligible.index.is_monotonic_increasing
            or eligible.return_date.duplicated().any()
            or not eligible.return_date.is_monotonic_increasing
            or not (eligible.index < pd.DatetimeIndex(eligible.return_date)).all()):
        raise ValueError("Evaluation requires ordered, unique preceding-close forecasts.")
    dates = pd.DatetimeIndex(eligible.return_date)
    if dates[0] != daily.index[0]:
        raise ValueError("A forecast must fund the first evaluation return.")
    positions = daily.index.get_indexer(dates)
    later = positions > 0
    if not eligible.index[later].equals(daily.index[positions[later] - 1]):
        raise ValueError("Each forecast must target the next observed equity session.")
    for model in models:
        values = eligible[f"{model}_bv"].to_numpy(dtype=float)
        asof = eligible[f"{model}_refit_asof"]
        if (not np.isfinite(values).all() or (values <= 0).any()
                or asof.isna().any() or (asof > eligible.index).any()):
            raise ValueError(f"Invalid or future-fitted {model} forecasts.")
    scores, forecast_tests = _forecast_scores(eligible, settings, models)
    paths, definitions, thresholds, targets = {"EQUITY": daily.EQUITY}, [], [], []
    for model in models:
        volatility = pd.Series(settings["monthly_multiplier"] * np.sqrt(eligible[f"{model}_bv"].to_numpy()), index=dates)
        stock = (settings["monthly_target"] / volatility).clip(upper=1)
        weights = pd.DataFrame({"EQUITY": stock, "DEFENSIVE": 0., "RF": 1 - stock}, index=dates)
        proxy_returns = pd.DataFrame({"EQUITY": daily.EQUITY, "DEFENSIVE": 0., "RF": 0.}, index=daily.index)
        clean_name = f"{model}_clean"
        paths[clean_name] = dynamic_costed_path(proxy_returns, weights.reindex(daily.index).ffill(),
                                               rebalance_dates=dates)["return"]
        definitions.append({"portfolio": clean_name, "model": model, "asset": "cash", "quantile": np.nan, "cap": 0.})
        allocation = pd.DataFrame({"model": model, "stock_target": stock, "monthly_volatility": volatility}, index=dates)
        for quantile in settings["quantiles"]:
            threshold = float(volatility.quantile(quantile))
            thresholds.append({"model": model, "quantile": quantile, "threshold": threshold})
            for cap in settings["caps"]:
                defensive = np.minimum(1 - stock, cap).where(volatility >= threshold, 0.)
                allocation[f"defensive_q{quantile:g}_cap{cap:g}"] = defensive
                weights["DEFENSIVE"] = defensive
                weights["RF"] = 1 - weights.EQUITY - defensive
                dated_weights = weights.reindex(daily.index).ffill()
                for asset in ASSETS:
                    name = f"{model}_{asset}_q{quantile:g}_cap{cap:g}"
                    proxy_returns["DEFENSIVE"] = daily[asset]
                    paths[name] = dynamic_costed_path(proxy_returns, dated_weights, rebalance_dates=dates)["return"]
                    definitions.append({"portfolio": name, "model": model, "asset": asset, "quantile": quantile, "cap": cap})
        targets.append(allocation.rename_axis("return_date").reset_index())
    paths = pd.DataFrame(paths, index=daily.index)
    metrics = _metrics(paths, settings["trading_days"])
    details = pd.DataFrame(definitions).set_index("portfolio").join(metrics)
    additions = details.loc[details.asset.ne("cash")].copy()
    measures = ("sharpe", "paper_inferred_sharpe", "calmar")
    for measure in measures:
        additions[f"{measure}_minus_clean"] = [row[measure] - metrics.loc[f"{row.model}_clean", measure]
                                                 for _, row in additions.iterrows()]
    mean_columns = [*measures, *(f"{measure}_minus_clean" for measure in measures)]
    asset_means = additions.groupby(["model", "asset"])[mean_columns].mean()
    by_quantile = additions.groupby(["model", "asset", "quantile"])[mean_columns].mean()
    decline_rows = []
    for label, peak, trough in DECLINES:
        window = paths.loc[(paths.index > peak) & (paths.index <= trough)]
        if len(window) < 2:
            continue
        measures_in_decline = _metrics(window, settings["trading_days"])
        measures_in_decline["decline"] = label
        measures_in_decline["peak"], measures_in_decline["trough"] = peak, trough
        decline_rows.append(measures_in_decline.reset_index())
    declines = pd.concat(decline_rows, ignore_index=True) if decline_rows else pd.DataFrame()
    decline_means = pd.DataFrame()
    if not declines.empty:
        described = declines.merge(details[["model", "asset"]], left_on="portfolio", right_index=True, how="left")
        benchmark = described.loc[described.portfolio.eq("EQUITY")].assign(asset="EQUITY")
        described = pd.concat([described.loc[~described.portfolio.eq("EQUITY")],
                               *(benchmark.assign(model=model) for model in models)], ignore_index=True)
        decline_means = described.groupby(["decline", "model", "asset"])[
            ["cumulative_return", "paper_inferred_modified_sharpe", "arithmetic_modified_sharpe"]].mean()
        for measure in ("cumulative_return", "paper_inferred_modified_sharpe", "arithmetic_modified_sharpe"):
            decline_means[f"{measure}_minus_clean"] = [
                row[measure] - decline_means.loc[(decline, model, "cash"), measure]
                for (decline, model, asset), row in decline_means.iterrows()]
    reference = dict(zip(MODELS, PUBLISHED, strict=True))
    published = pd.DataFrame([reference[model.removesuffix("_log")] for model in models],
                             index=pd.Index(models, name="model"),
                             columns=["published_monthly_mse", "published_monthly_mape",
                                      "published_clean_sharpe", "published_clean_calmar"])
    comparison = published.join(scores)
    for measure in ("cagr", "sharpe", "paper_inferred_sharpe", "calmar"):
        comparison[f"reconstruction_clean_{measure}"] = [metrics.loc[f"{model}_clean", measure] for model in models]
    return {"daily_returns": paths, "performance": metrics, "portfolio_definitions": details,
            "model_target_weights": pd.concat(targets, ignore_index=True), "thresholds": pd.DataFrame(thresholds),
            "forecast_scores": scores, "forecast_tests": forecast_tests, "asset_means": asset_means,
            "asset_by_quantile": by_quantile, "decline_performance": declines,
            "decline_asset_means": decline_means, "published_comparison": comparison}, {
        "scope": "U.S. primary experiment only; international robustness not reconstructed",
        "portfolios": len(definitions), "models": list(models), "defensive_assets": len(ASSETS),
        "variants_per_asset": len(settings["quantiles"]) * len(settings["caps"]),
        "forecast_scoring": "common observed BV labels; monthly volatility = 22 * sqrt(raw BV); MAPE is a decimal ratio",
        "variance_floor": "forecasts below 1e-12 floored after seed averaging; floor counts include all eligible issues and imply 100% stocks",
        "missing_signals": "retain every equity return date; weights drift until next available issue",
        "execution": "idealized close-t information for t+1 return; same-date release timing not verified",
        "quantiles": "each model's full-test predicted volatility, including forecasts with missing labels; retrospective",
        "allocation": "stock=min(target/volatility,1); defensive=stress*min(1-stock,cap); residual cash",
        "fees_and_cash": "zero trading fees and zero cash interest, matching the paper",
        "averaging": "arithmetic mean of nine individually calculated ratios, then subtract clean ratio",
        "standard_sharpe": "arithmetic daily mean / sample daily standard deviation * sqrt(252); zero risk-free rate",
        "paper_inferred_sharpe": "CAGR / annual sample volatility; inferred from published benchmark, formula not supplied",
        "calmar": "CAGR / absolute maximum drawdown, including initial wealth 1",
        "declines": "five published fixed windows; peak-day return excluded, trough included",
        "paper_inferred_modified_sharpe": "CAGR / annual_volatility**sign(CAGR); geometric annual numerator inferred, not verified",
        "arithmetic_modified_sharpe": "annual arithmetic mean / annual_volatility**sign(mean); alternative convention provided",
        "dm_tests": f"paired loss differential with Bartlett HAC, {settings['dm_hac_lags']} lags, t inference; positive favors row model; bandwidth absent from paper",
        "multiple_testing": f"raw p-values plus Holm adjustment across {len(forecast_tests)} forecast comparisons; no portfolio-significance claim",
        "log_bv_fixes": "models ending _log learn log BV after raw-BV results were seen; compared with the raw model's published row",
        "classical_window": "fixed rolling length matching observed 2006-2009 data; paper also inconsistently mentions expanding",
        "gru_gate": "PyTorch reset-after GRU; paper Eq.17 shows reset-before, but its executed Keras/TensorFlow settings are unpublished",
        "published_reference": "Kaczmarek et al. (2022), Tables 3-6; https://doi.org/10.1016/j.ribaf.2021.101610",
        "unresolved": "original data, exact hyperparameter grid, seeds, normalization, GRU output/target details, estimators, calendar and metric conventions differ or are unpublished",
    }
