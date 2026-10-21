"""Fractional memory, fitted variance units, dated histories, and explicit failures."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.signal import fftconvolve

pytest.importorskip("arch", reason="Classical reconstruction tests require the reconstruction extra.")

from financial_economics.paper_models import (
    _arfima_forecast, _fractional_prediction, _fractional_weights, _garch_forecast,
    classical_forecasts,
)


def _history():
    dates = pd.bdate_range("2008-01-01", periods=260)
    generator = np.random.default_rng(218)
    shocks = generator.standard_normal(len(dates))
    variance = np.empty(len(dates))
    returns = np.empty(len(dates))
    variance[0] = .0002
    for position in range(len(dates)):
        if position:
            variance[position] = .00001 + .10 * returns[position - 1] ** 2 + .85 * variance[position - 1]
        returns[position] = np.sqrt(variance[position]) * shocks[position]
    innovations = generator.standard_normal(len(dates))
    memory = fftconvolve(innovations, _fractional_weights(-.35, len(dates)))[:len(dates)]
    bv = pd.Series(.0002 * (2 + .15 * memory), index=dates)
    price_returns = pd.Series(returns, index=dates)
    issues = dates[220:223]
    rows = pd.DataFrame({"return_date": dates[221:224], "realized_bv": bv.iloc[221:224].to_numpy()},
                        index=issues.rename("forecast_date"))
    options = {"window_start": str(dates[0].date()), "window_end": str(dates[219].date()),
               "min_observations": 60, "progress": None}
    return bv, price_returns, rows, options


def test_fractional_coefficients_and_prediction_retain_distant_lags():
    np.testing.assert_allclose(_fractional_weights(.4, 5), [1, -.4, -.12, -.064, -.0416])
    np.testing.assert_array_equal(_fractional_weights(0, 5), [1, 0, 0, 0, 0])
    values = np.array([1., 2., 3., 4.])
    assert _fractional_prediction(values, 0, 2.) == 2.
    baseline = _fractional_prediction(values, .4, 2.)
    changed = values.copy()
    changed[0] += 1
    assert _fractional_prediction(changed, .4, 2.) - baseline == pytest.approx(.0416)


def test_arfima_fits_fractional_memory_and_keeps_raw_bv_units():
    bv, _, _, _ = _history()
    prediction, details = _arfima_forecast(bv.to_numpy())
    assert .05 < details["d"] < .49
    assert prediction == pytest.approx(_fractional_prediction(bv, details["d"], details["mean_bv"]))
    larger, larger_details = _arfima_forecast(10 * bv.to_numpy())
    assert larger == pytest.approx(10 * prediction, rel=1e-7)
    assert larger_details["d"] == pytest.approx(details["d"], abs=1e-7)
    assert details["fit_status"] == 0 and not details["forecast_floored"]


@pytest.mark.parametrize("asymmetric", [False, True])
def test_garch_analytic_recursion_uses_decimal_variance_units(asymmetric):
    _, price_returns, _, _ = _history()
    prediction, details, _ = _garch_forecast(price_returns.to_numpy(), asymmetric)
    residual = details["last_residual_return"]
    expected = (details["omega_bv"] + details["alpha"] * residual ** 2
                + details["beta"] * details["last_conditional_bv"]
                + details["gamma"] * (residual < 0) * residual ** 2)
    assert prediction == pytest.approx(expected, rel=1e-12)
    assert 1e-7 < prediction < .01 and details["fit_status"] == 0
    if not asymmetric:
        assert details["gamma"] == 0


@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_garch_retries_same_model_once_then_fails_explicitly(monkeypatch, retry_succeeds):
    calls = []
    parameters = pd.Series({"mu": .01, "omega": .02, "alpha[1]": .05, "beta[1]": .9})

    def fit(**options):
        calls.append(options)
        return SimpleNamespace(
            convergence_flag=0 if len(calls) == 2 and retry_succeeds else 8,
            params=parameters, resid=np.array([1.]), conditional_volatility=np.array([2.]),
            optimization_result=SimpleNamespace(message="synthetic optimizer status", nfev=4),
            forecast=lambda **kwargs: SimpleNamespace(residual_variance=pd.DataFrame([[3.]])),
        )

    monkeypatch.setattr("arch.arch_model", lambda *args, **kwargs: SimpleNamespace(fit=fit))
    if retry_succeeds:
        prediction, details, _ = _garch_forecast(np.array([.01, -.01]))
        assert prediction == pytest.approx(.0003) and details["fit_attempts"] == 2
    else:
        with pytest.raises(RuntimeError, match="status 8"):
            _garch_forecast(np.array([.01, -.01]))
    assert len(calls) == 2
    assert calls[1]["starting_values"] is None and calls[1]["options"]["maxiter"] == 3000


def test_classical_forecasts_use_only_available_history_and_record_daily_windows():
    bv, price_returns, rows, options = _history()
    original, fits, metadata = classical_forecasts(rows, bv, price_returns, **options)
    assert metadata["backend"].startswith("CPU-native")
    assert metadata["rolling_observations_by_model"] == dict.fromkeys(("arfima", "garch", "gjr"), 220)
    assert fits.training_observations.eq(220).all()
    assert (fits.train_last == fits.forecast_date).all()
    assert (fits.refit_asof < fits.return_date).all()
    for name in ("arfima", "garch", "gjr"):
        assert original[f"{name}_refit_asof"].tolist() == rows.index.tolist()
        assert original[f"{name}_bv"].gt(0).all()
        assert original[f"{name}_fit_status"].eq(0).all()
    future = rows.index[-1]
    changed_bv, changed_returns, changed_rows = bv.copy(), price_returns.copy(), rows.copy()
    changed_bv.loc[future:] *= 3
    changed_returns.loc[future:] *= 2
    changed_rows["realized_bv"] = 1000
    altered, _, _ = classical_forecasts(changed_rows, changed_bv, changed_returns, **options)
    columns = [f"{name}_bv" for name in ("arfima", "garch", "gjr")]
    pd.testing.assert_frame_equal(original.iloc[:-1][columns], altered.iloc[:-1][columns])
    _, expanding, _ = classical_forecasts(rows.iloc[:2], bv, price_returns, window="expanding", **options)
    assert expanding.training_observations.tolist() == [221] * 3 + [222] * 3


def test_invalid_dates_windows_and_failed_fits_are_explicit(monkeypatch):
    bv, price_returns, rows, options = _history()
    delayed = rows.assign(trade_date=rows.index)
    with pytest.raises(ValueError, match="one-step"):
        classical_forecasts(delayed, bv, price_returns, **options)
    with pytest.raises(ValueError, match="finish by"):
        classical_forecasts(rows, bv, price_returns, **{**options, "window_end": "2010-01-01"})
    with pytest.raises(ValueError, match="observed BV and price returns"):
        classical_forecasts(rows, bv, price_returns.drop(rows.index[0]), **options)
    monkeypatch.setattr("financial_economics.paper_models.minimize_scalar", lambda *args, **kwargs:
                        SimpleNamespace(success=False, message="forced failure"))
    with pytest.raises(RuntimeError, match=f"arfima fit failed at {rows.index[0].date()}.*forced failure"):
        classical_forecasts(rows, bv, price_returns, **options)
