"""CPU-native daily classical forecasts; estimators reconstruct, not duplicate, Ox.

ARFIMA(0,d,0) fits raw BV with a conditional Gaussian likelihood, a profiled
constant mean, and stationary d in [-.49, .49]. Finite-history fractional
filtering assumes pre-sample deviations from the mean are zero. This is not
integer ARIMA or the original proprietary ARFIMA estimator. Only the final
raw-BV forecast is floored for portfolio use; its unfloored value is retained.

GARCH/GJR use arch's Normal constant-mean models on 100 times decimal log-price
returns. Their analytic one-step innovation variances are divided by 10,000.
All fits are sequential CPU algorithms, independent of the neural GPU backend.
"""

from importlib.metadata import version

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve


def _fractional_weights(d, size):
    """Coefficients of (1-L)**d, retaining every available historical lag."""
    weights = np.ones(size)
    lags = np.arange(1, size)
    weights[1:] = np.cumprod((lags - 1 - d) / lags)
    return weights


def _fractional_prediction(values, d, mean):
    weights = _fractional_weights(d, len(values) + 1)
    return float(mean - weights[1:] @ (np.asarray(values)[::-1] - mean))


def _arfima_forecast(values, variance_floor=1e-12):
    values = np.asarray(values, dtype=float)
    scale = float(values.mean())
    normalized = values / scale
    if np.ptp(normalized) <= 1e-12:
        raise RuntimeError("ARFIMA requires variation to identify fractional memory.")

    def residuals(d):
        weights = _fractional_weights(d, len(values))
        filtered = fftconvolve(normalized, weights)[:len(values)]
        constant = weights.cumsum()
        mean = float(filtered @ constant / (constant @ constant))
        errors = filtered - mean * constant
        return errors, mean

    def objective(d):
        errors, _ = residuals(d)
        return float(np.log(max(float(errors @ errors / len(values)), np.finfo(float).tiny)))

    fitted = minimize_scalar(objective, bounds=(-.49, .49), method="bounded",
                             options={"xatol": 1e-7, "maxiter": 200})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"ARFIMA optimizer failed: {fitted.message}")
    errors, mean = residuals(float(fitted.x))
    raw_forecast = _fractional_prediction(values, float(fitted.x), mean * scale)
    if not np.isfinite(raw_forecast):
        raise RuntimeError("ARFIMA produced a nonfinite raw-BV forecast.")
    return max(raw_forecast, variance_floor), {
        "fit_status": 0, "message": str(fitted.message), "d": float(fitted.x),
        "mean_bv": mean * scale, "innovation_bv_variance": float(np.mean(errors ** 2) * scale ** 2),
        "raw_forecast_bv": raw_forecast, "forecast_floored": raw_forecast < variance_floor,
        "objective_evaluations": int(fitted.nfev),
    }


def _garch_forecast(values, asymmetric=False, starting_values=None):
    from arch import arch_model

    model = arch_model(100 * np.asarray(values, dtype=float), mean="Constant", vol="GARCH",
                       p=1, o=int(asymmetric), q=1, power=2, dist="normal", rescale=False)
    fitted = model.fit(disp="off", show_warning=False, starting_values=starting_values,
                       options={"maxiter": 1000, "ftol": 1e-8})
    attempts = 1
    if fitted.convergence_flag != 0 or not np.isfinite(fitted.params).all():
        # Retry the same estimator from its own defaults, never another model.
        fitted = model.fit(disp="off", show_warning=False, starting_values=None,
                           options={"maxiter": 3000, "ftol": 1e-8})
        attempts = 2
    status = int(fitted.convergence_flag)
    if status != 0 or not np.isfinite(fitted.params).all():
        raise RuntimeError(f"GARCH optimizer failed (status {status}): {fitted.optimization_result.message}")
    forecast = float(fitted.forecast(horizon=1, method="analytic", reindex=False)
                     .residual_variance.iloc[-1, 0] / 10000)
    if not np.isfinite(forecast) or forecast <= 0:
        raise RuntimeError("GARCH produced a nonpositive or nonfinite variance forecast.")
    parameters = fitted.params
    return forecast, {
        "fit_status": status, "fit_attempts": attempts,
        "message": str(fitted.optimization_result.message),
        "mean_return": float(parameters["mu"] / 100),
        "omega_bv": float(parameters["omega"] / 10000),
        "alpha": float(parameters["alpha[1]"]), "beta": float(parameters["beta[1]"]),
        "gamma": float(parameters.get("gamma[1]", 0.)),
        "last_residual_return": float(np.asarray(fitted.resid)[-1] / 100),
        "last_conditional_bv": float(np.asarray(fitted.conditional_volatility)[-1] ** 2 / 10000),
        "raw_forecast_bv": forecast, "forecast_floored": False,
        "objective_evaluations": int(fitted.optimization_result.nfev),
    }, parameters.to_numpy()


def _validate_series(series, name, positive=False):
    if (not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex)
            or series.index.empty or series.index.hasnans or series.index.has_duplicates
            or not series.index.is_monotonic_increasing or series.index.tz is not None
            or not series.index.equals(series.index.normalize())
            or not np.isfinite(series.to_numpy(dtype=float)).all()
            or (positive and (series <= 0).any())):
        raise ValueError(f"{name} requires sorted unique daily dates and finite {'positive ' if positive else ''}values.")


def classical_forecasts(rows, bipower, price_returns, *, window="rolling",
                        window_start="2006-01-01", window_end="2009-12-31",
                        min_observations=60, variance_floor=1e-12, progress=print):
    """Fit each issue close for caller-provided one-session-ahead target dates.

    `rows` is the one-step prepared forecast table indexed by forecast_date;
    return_date is retained, but realized target columns are never read. Prices
    must be supplied as decimal daily log-price returns, not total portfolio returns.
    Missing observations are not filled. Both sources must contain every issue.

    Rolling lengths count each source's observed window_start:window_end rows,
    then take that many observations through each issue. Expanding samples begin
    at window_start. Source-specific counts and actual training endpoints are
    saved. No estimator failures are replaced with another model's forecasts.

    Return (rows_with_forecasts, daily_fit_records, estimator_metadata).
    """
    _validate_series(bipower, "BV", positive=True)
    _validate_series(price_returns, "Price returns")
    if (not isinstance(rows, pd.DataFrame) or rows.empty or "return_date" not in rows
            or not isinstance(rows.index, pd.DatetimeIndex) or rows.index.hasnans
            or rows.index.has_duplicates or not rows.index.is_monotonic_increasing
            or rows.index.tz is not None or not rows.index.equals(rows.index.normalize())):
        raise ValueError("Classical forecasts require ordered unique daily issue dates and return_date.")
    targets = pd.DatetimeIndex(rows.return_date)
    if (targets.hasnans or targets.tz is not None or targets.has_duplicates
            or not targets.is_monotonic_increasing or not targets.equals(targets.normalize())
            or not (targets > rows.index).all() or "trade_date" in rows):
        raise ValueError("Classical models require one-step prepared targets; delayed trade_date tables are unsupported.")
    if (window not in ("rolling", "expanding") or type(min_observations) is not int
            or min_observations < 10 or isinstance(variance_floor, bool)
            or not np.isfinite(variance_floor) or variance_floor <= 0):
        raise ValueError("Choose rolling/expanding, at least ten observations, and a positive finite variance floor.")
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
            or start != start.normalize() or end != end.normalize()
            or not start < end <= rows.index[0]):
        raise ValueError("The initial estimation window must finish by the first forecast issue.")
    sources = {"arfima": bipower, "garch": price_returns, "gjr": price_returns}
    lengths = {name: len(series.loc[start:end]) for name, series in sources.items()}
    if any(size < min_observations for size in lengths.values()):
        raise ValueError("The initial estimation window has too few observations for a classical model.")
    if any(not rows.index.isin(series.index).all() for series in sources.values()):
        raise ValueError("Every classical forecast issue needs observed BV and price returns on that date.")
    result, records, starts = rows.copy(), [], {}
    for name in sources:
        result[f"{name}_bv"] = np.nan
        result[f"{name}_refit_asof"] = pd.NaT
        result[f"{name}_fit_status"] = 0
    for issue, row in rows.iterrows():
        if progress is not None and (not records or issue.month != records[-1]["forecast_date"].month):
            progress(f"Classical daily fits: {issue.date()}, {window} samples")
        for name, source in sources.items():
            history = source.loc[start:issue]
            if window == "rolling":
                history = history.iloc[-lengths[name]:]
            if len(history) < min_observations:
                raise ValueError(f"{name} has insufficient history at {issue.date()}.")
            try:
                if name == "arfima":
                    prediction, details = _arfima_forecast(history.to_numpy(), variance_floor)
                else:
                    prediction, details, starts[name] = _garch_forecast(
                        history.to_numpy(), asymmetric=name == "gjr", starting_values=starts.get(name))
            except (RuntimeError, ValueError, FloatingPointError) as error:
                raise RuntimeError(f"{name} fit failed at {issue.date()}: {error}") from error
            result.loc[issue, f"{name}_bv"] = prediction
            result.loc[issue, f"{name}_refit_asof"] = issue
            records.append({"forecast_date": issue, "return_date": row.return_date, "model": name,
                            "refit_asof": issue, "train_first": history.index[0],
                            "train_last": history.index[-1], "training_observations": len(history), **details})
    metadata = {
        "backend": "CPU-native; sequential SciPy/arch fits", "device": "cpu",
        "estimator_status": "transparent reconstruction; not the original proprietary Ox estimators",
        "window": window, "window_start": str(start.date()), "window_end": str(end.date()),
        "rolling_observations_by_model": lengths, "min_observations": min_observations,
        "refit_frequency": "every forecast issue close; history ends at issue",
        "forecast_horizon": 1, "output_units": "raw decimal-return variance / daily BV",
        "arfima": {"order": [0, "d", 0], "target": "raw BV; literal published RBV specification",
                   "estimator": "conditional Gaussian; profiled constant mean; FFT fractional filtering",
                   "d_bounds": [-.49, .49], "presample": "deviations from fitted mean set to zero",
                   "variance_floor": variance_floor, "floor_rule": "final forecast only; unfloored value saved"},
        "garch": {"order": [1, 1], "gjr_asymmetric_order": 1, "distribution": "Normal",
                  "mean": "Constant", "input": "100 * supplied decimal log-price returns",
                  "variance_conversion": "analytic one-step residual variance / 10000",
                  "warm_start": "preceding successful fit; arch may reject infeasible starting values",
                  "fit_failure": "one same-model retry without warm start and maxiter 3000; then raise"},
        "library_versions": {name: version(name) for name in ("arch", "numpy", "scipy")},
    }
    return result, pd.DataFrame(records), metadata
