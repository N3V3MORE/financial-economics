"""Conditional means, full-calendar HAC, and paired stationary bootstrap on CPU."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .analysis import _check_returns
from .explain import resample_rows


def describe_funds(monthly, benchmark="SPY", threshold=-.05):
    _check_returns(monthly)
    rows = []
    for condition, mask in [("all", np.ones(len(monthly), dtype=bool)),
                            ("negative_spy", monthly[benchmark] < 0),
                            ("severe_spy", monthly[benchmark] <= threshold)]:
        for fund in monthly.columns.difference([benchmark, "RF"], sort=False):
            excess = (monthly[fund] - monthly.RF).loc[mask]
            rows.append({"fund": fund, "condition": condition, "months": len(excess),
                         "mean_minus_cash": excess.mean(),
                         "beat_cash_frequency": (excess > 0).mean() if len(excess) else np.nan,
                         "status": "available" if len(excess) else "no_matching_months"})
    return pd.DataFrame(rows)


def infer_protection(monthly, settings, benchmark="SPY"):
    """All candidate tests are one family; allocation and period checks are not extra tests."""
    _check_returns(monthly)
    funds = monthly.columns.difference([benchmark, "RF"], sort=False).tolist()
    n = len(monthly)
    if not funds or max(settings.block_lengths) > n:
        raise ValueError("Inference needs candidates and block lengths within the sample.")
    down = (monthly[benchmark] < 0).to_numpy()
    values = monthly[funds].to_numpy() - monthly[["RF"]].to_numpy()
    design = np.column_stack([down, ~down]).astype(float)
    rows = []
    for column, fund in enumerate(funds):
        y = values[:, column]
        record = {"fund": fund, "months": n, "negative_months": int(down.sum()),
                  "mean_minus_cash": float(y[down].mean()) if down.any() else np.nan,
                  "hac_se": np.nan, "p_value": np.nan, "inference_status": "unavailable"}
        if down.any() and (~down).any() and n > max(2, settings.hac_lags + 1):
            fit = sm.OLS(y, design).fit(cov_type="HAC", use_t=True,
                                       cov_kwds={"maxlags": settings.hac_lags,
                                                 "kernel": "bartlett", "use_correction": True})
            if np.isfinite(fit.bse[0]) and fit.bse[0] > 1e-14:
                record.update(hac_se=float(fit.bse[0]), p_value=float(fit.pvalues[0]),
                              inference_status="available")
            else:
                record["inference_status"] = "degenerate_variance"
        rows.append(record)
    tests = pd.DataFrame(rows)
    valid = tests.p_value.notna()
    # Unavailable tests retain a place in the frozen family, but are never called significant.
    rejected, adjusted, _, _ = multipletests(tests.p_value.fillna(1), alpha=settings.alpha, method="holm")
    tests["holm_p_value"] = np.where(valid, adjusted, np.nan)
    tests["holm_reject"] = rejected & valid
    tests["family_size"] = len(funds)
    intervals = []
    for block in settings.block_lengths:
        rng = np.random.default_rng(settings.seed)
        batches = []
        for done in range(0, settings.resamples, 250):
            indices = resample_rows(n, min(250, settings.resamples - done), block, rng)
            mask = down[indices]
            counts = mask.sum(axis=1)
            sums = (values[indices] * mask[:, :, None]).sum(axis=1)
            means = np.full(sums.shape, np.nan)
            np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
            batches.append(means)
        draws = np.concatenate(batches)
        for column, fund in enumerate(funds):
            finite = draws[np.isfinite(draws[:, column]), column]
            bounds = np.quantile(finite, [settings.alpha / 2, 1 - settings.alpha / 2]) if len(finite) else [np.nan, np.nan]
            intervals.append({"fund": fund, "mean_block_months": block,
                              "ci_low": bounds[0], "ci_high": bounds[1],
                              "valid_draws": len(finite), "unavailable_draws": settings.resamples - len(finite),
                              "resamples": settings.resamples, "seed": settings.seed,
                              "interval_type": "pointwise stationary-bootstrap percentile",
                              "status": "available" if len(finite) else "no_negative_months"})
    return tests, pd.DataFrame(intervals)
