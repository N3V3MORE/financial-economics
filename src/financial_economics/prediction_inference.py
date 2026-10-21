"""Prespecified paired forecast and portfolio inference; CPU-native bootstrap."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .analysis import _check_returns
from .explain import resample_rows
from .forecasting import qlike


def _hac_test(values, design, coefficient, lags, alpha):
    result = {"hac_se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
              "p_value": np.nan, "status": "insufficient_observations"}
    if len(values) <= max(lags + 1, design.shape[1]):
        return result
    if np.linalg.matrix_rank(design) < design.shape[1]:
        result["status"] = "missing_condition"
        return result
    fit = sm.OLS(values, design).fit(
        cov_type="HAC", use_t=True,
        cov_kwds={"maxlags": lags, "kernel": "bartlett", "use_correction": True})
    if not np.isfinite(fit.bse[coefficient]) or fit.bse[coefficient] <= 1e-14:
        result["status"] = "degenerate_variance"
        return result
    bounds = np.asarray(fit.conf_int(alpha=alpha))[coefficient]
    return {"hac_se": float(fit.bse[coefficient]), "ci_low": float(bounds[0]),
            "ci_high": float(bounds[1]), "p_value": float(fit.pvalues[coefficient]),
            "status": "available"}


def infer_prediction(forecasts, monthly, settings):
    """Three two-sided HAC tests, one Holm family, and paired monthly intervals.

    Forecast effects are GRU minus EWMA QLIKE (negative favors GRU). Portfolio
    effects are down-SPY-month return differences (positive favors D).
    Bootstrap intervals condition on the saved fitted-model returns.
    """
    _check_returns(monthly)
    if not {"SPY", "B", "C", "D"}.issubset(monthly.columns):
        raise ValueError("Prediction inference requires SPY and portfolios B, C, D.")
    observed = forecasts.squared_return.to_numpy(dtype=float)
    loss = (qlike(observed, forecasts.gru_variance.to_numpy(dtype=float))
            - qlike(observed, forecasts.ewma_variance.to_numpy(dtype=float)))
    records = [{"test": "forecast_gru_minus_ewma", "measure": "mean_qlike_difference",
                "effect": float(loss.mean()), "n_obs": len(loss), "negative_months": np.nan,
                **_hac_test(loss, np.ones((len(loss), 1)), 0,
                            settings.forecast_hac_lags, settings.alpha)}]
    down = monthly.SPY.to_numpy() < 0
    design = np.column_stack([down, ~down]).astype(float)
    gaps = np.column_stack([(monthly.D - monthly.C).to_numpy(),
                            (monthly.D - monthly.B).to_numpy()])
    names = ("D_minus_C", "D_minus_B")
    for column, name in enumerate(names):
        values = gaps[:, column]
        records.append({"test": name, "measure": "negative_spy_month_mean_return_difference",
                        "effect": float(values[down].mean()) if down.any() else np.nan,
                        "n_obs": len(monthly), "negative_months": int(down.sum()),
                        **_hac_test(values, design, 0, settings.monthly_hac_lags, settings.alpha)})
    tests = pd.DataFrame(records)
    valid = tests.p_value.notna()
    rejected, adjusted, _, _ = multipletests(tests.p_value.fillna(1),
                                            alpha=settings.alpha, method="holm")
    tests["holm_p_value"] = np.where(valid, adjusted, np.nan)
    tests["holm_reject"] = rejected & valid
    tests["family_size"] = 3

    n = len(monthly)
    if max(settings.block_lengths) > n:
        raise ValueError("Bootstrap block lengths must fit inside the monthly sample.")
    # Gaps and SPY use identical indices; each draw identifies its own down months.
    values = np.column_stack([gaps, monthly.SPY.to_numpy()])
    intervals = []
    for block in settings.block_lengths:
        rng = np.random.default_rng(settings.seed)
        batches = []
        for done in range(0, settings.resamples, 250):
            indices = resample_rows(n, min(250, settings.resamples - done), block, rng)
            sample = values[indices]
            mask = sample[:, :, 2] < 0
            counts = mask.sum(axis=1)
            sums = (sample[:, :, :2] * mask[:, :, None]).sum(axis=1)
            means = np.full(sums.shape, np.nan)
            np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
            batches.append(means)
        draws = np.concatenate(batches)
        for column, name in enumerate(names):
            finite = draws[np.isfinite(draws[:, column]), column]
            bounds = (np.quantile(finite, [settings.alpha / 2, 1 - settings.alpha / 2])
                      if len(finite) else [np.nan, np.nan])
            intervals.append({"test": name, "mean_block_months": block,
                              "effect": tests.loc[tests.test.eq(name), "effect"].iloc[0],
                              "ci_low": bounds[0], "ci_high": bounds[1],
                              "months": n, "negative_months": int(down.sum()),
                              "valid_draws": len(finite),
                              "unavailable_draws": settings.resamples - len(finite),
                              "resamples": settings.resamples, "seed": settings.seed,
                              "interval_type": "pointwise stationary-bootstrap percentile",
                              "status": "available" if len(finite) else "no_negative_months"})
    return tests, pd.DataFrame(intervals)
