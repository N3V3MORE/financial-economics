from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_economics.config import load_study
from financial_economics.forecasting import ewma_forecasts, prepare_forecasts, qlike
from financial_economics.prediction_config import PredictionSettings, load_prediction


def test_trading_dates_and_inputs_cross_year_without_using_future_returns():
    dates = pd.to_datetime(["2014-12-29", "2014-12-30", "2014-12-31", "2015-01-02",
                            "2015-01-05", "2015-01-06", "2015-01-07"])
    returns = pd.Series(np.arange(1, 8) / 100, index=dates)
    prepared = prepare_forecasts(returns, dates, lookback=3)
    assert prepared.rows.index[0] == pd.Timestamp("2014-12-31")
    assert prepared.rows.trade_date.iloc[0] == pd.Timestamp("2015-01-02")
    assert prepared.rows.return_date.iloc[0] == pd.Timestamp("2015-01-05")
    assert prepared.rows.squared_return.iloc[0] == pytest.approx(.05 ** 2)
    np.testing.assert_allclose(prepared.inputs[0, :, 0], [.01, .02, .03])
    np.testing.assert_allclose(prepared.inputs[0, :, 1], np.array([.01, .02, .03]) ** 2)
    with pytest.raises(ValueError, match="calendar"):
        prepare_forecasts(returns.drop(dates[3]), dates, lookback=3)
    with pytest.raises(ValueError, match="increasing"):
        prepare_forecasts(returns.iloc[::-1], dates[::-1], lookback=3)


def test_ewma_uses_exact_history_and_only_available_validation_targets():
    dates = pd.bdate_range("2014-11-01", "2017-02-01")
    returns = pd.Series(.005 + .002 * np.sin(np.arange(len(dates)) / 7), index=dates)
    settings = PredictionSettings()
    prepared = prepare_forecasts(returns, dates)
    forecasts, tuning = ewma_forecasts(prepared, settings)
    validation = ((prepared.rows.index >= settings.validation_start)
                  & (prepared.rows.return_date <= settings.validation_end))
    expected_losses = []
    for row in tuning.itertuples():
        weights = row.decay ** np.arange(21, -1, -1)
        weights /= weights.sum()
        expected = prepared.inputs[:, :, 1] @ weights
        expected_losses.append(qlike(prepared.rows.squared_return.to_numpy()[validation],
                                    expected[validation]).mean())
        assert row.validation_observations == validation.sum()
        assert row.qlike == pytest.approx(expected_losses[-1])
        if row.selected:
            np.testing.assert_allclose(forecasts.ewma_variance, expected)
    assert tuning.selected.sum() == 1
    assert tuning.selected.iloc[int(np.argmin(expected_losses))]
    late_december = prepared.rows.loc["2016-12-30"]
    assert late_december.return_date.year == 2017
    assert not validation.loc["2016-12-30"]


def test_future_evaluation_returns_cannot_change_earlier_forecasts_or_selection():
    dates = pd.bdate_range("2014-11-01", "2017-03-01")
    returns = pd.Series(.001 + .01 * np.sin(np.arange(len(dates))), index=dates)
    settings = PredictionSettings()
    prepared = prepare_forecasts(returns, dates)
    original, tuning = ewma_forecasts(prepared, settings)
    changed = returns.copy()
    changed.loc["2017-02-01":] = -.2
    changed_prepared = prepare_forecasts(changed, dates)
    altered, altered_tuning = ewma_forecasts(changed_prepared, settings)
    earlier = prepared.rows.index < pd.Timestamp("2017-02-01")
    np.testing.assert_array_equal(prepared.inputs[earlier], changed_prepared.inputs[earlier])
    pd.testing.assert_frame_equal(tuning, altered_tuning)
    pd.testing.assert_series_equal(original.ewma_variance.loc[:"2017-01-31"],
                                   altered.ewma_variance.loc[:"2017-01-31"])


def test_prediction_config_preserves_snapshot_identity_and_freezes_valid_settings():
    path = Path(__file__).resolve().parents[1] / "configs" / "prediction.toml"
    study = load_study(path)
    original = load_study(path.with_name("protection-long.toml"))
    assert study.settings() == original.settings()
    settings = load_prediction(path, study)
    assert settings == PredictionSettings()
    assert settings.settings()["seeds"] == (11, 22, 33, 44, 55)
    with pytest.raises(ValueError, match="ordered"):
        replace(settings, training_end="2015-01-02").validate(study)
    with pytest.raises(ValueError, match="ewma_decays"):
        replace(settings, ewma_decays=(1,)).validate(study)


def test_qlike_accepts_zero_observation_and_rejects_nonpositive_predictions():
    np.testing.assert_allclose(qlike([0, .04], [.01, .02]),
                               np.log([.01, .02]) + np.array([0, 2]))
    with pytest.raises(ValueError, match="positive variances"):
        qlike([0], [0])
