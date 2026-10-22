from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_economics.paper_gru import _cv_folds, _require_cuda, _training_scale, paper_gru_forecasts


def _prepared():
    calendar = pd.bdate_range("2009-07-01", "2011-01-10")
    values = .0001 * (2 + np.sin(np.arange(len(calendar)) / 7))
    inputs = np.lib.stride_tricks.sliding_window_view(values, 3)[:-1, :, None]
    rows = pd.DataFrame({"return_date": calendar[3:], "realized_bv": values[3:]},
                        index=calendar[2:-1].rename("forecast_date"))
    return SimpleNamespace(rows=rows, inputs=inputs), calendar


def _settings():
    return SimpleNamespace(training_end="2009-12-31", evaluation_start="2010-01-01",
                           hidden_sizes=(2,), seeds=(11,), folds=2, batch_size=256,
                           max_epochs=1, patience=1, learning_rate=.001, dropout=.1,
                           weight_decay=.0001, lookback=3)


def test_expanding_validation_purges_unavailable_labels_and_preserves_training_scaling():
    prepared, _ = _prepared()
    training = (prepared.rows.return_date <= "2009-12-31").to_numpy()
    folds = _cv_folds(prepared.rows, training, 5)
    previous = 0
    for train, validation in folds:
        assert train.sum() > previous
        assert not (train & validation).any()
        assert prepared.rows.loc[train, "return_date"].max() <= prepared.rows.index[validation][0]
        assert prepared.rows.loc[validation, "return_date"].max() <= pd.Timestamp("2009-12-31")
        original = _training_scale(prepared, train)
        assert original["target_scale"] == pytest.approx(
            np.sqrt(prepared.rows.loc[train, "realized_bv"]).mean())
        np.testing.assert_allclose(original["feature_mean"], np.sqrt(prepared.inputs[train]).mean(axis=(0, 1)))
        altered = SimpleNamespace(rows=prepared.rows.copy(), inputs=prepared.inputs.copy())
        altered.inputs[~train] = 100
        altered.rows.loc[~train, "realized_bv"] = 100
        assert _training_scale(altered, train) == original
        previous = train.sum()


def test_tiny_cuda_refits_save_checkpoints_and_ignore_future_targets(tmp_path):
    try:
        torch, _ = _require_cuda()
    except RuntimeError as error:
        if "no compatible GPU" in str(error) or "prediction extra" in str(error):
            pytest.skip(str(error))
        raise
    prepared, calendar = _prepared()
    settings = _settings()
    # Forecasting needs no realized label on the date being forecast.
    prepared.rows.loc[prepared.rows.index[-1], "realized_bv"] = np.nan
    forecasts, tuning, metadata = paper_gru_forecasts(
        prepared, settings, calendar, tmp_path / "original", progress=lambda message: None)
    altered = SimpleNamespace(rows=prepared.rows.copy(), inputs=prepared.inputs.copy())
    later = altered.rows.index >= "2010-06-01"
    altered.rows.loc[later, "realized_bv"] *= 5
    altered.inputs[later] *= 5
    changed, changed_tuning, _ = paper_gru_forecasts(
        altered, settings, calendar, tmp_path / "altered", progress=lambda message: None)
    earlier = forecasts.index < "2010-06-01"
    np.testing.assert_array_equal(forecasts.loc[earlier, "gru_bv"], changed.loc[earlier, "gru_bv"])
    pd.testing.assert_frame_equal(tuning, changed_tuning)
    assert forecasts.loc["2009-12-31", "gru_refit_asof"] == pd.Timestamp("2009-12-31")
    assert np.isfinite(forecasts.iloc[-1].gru_bv)
    assert metadata["device"] == "cuda:0" and metadata["refit_epochs"] == 1
    assert metadata["backend"] == ("ROCm-required PyTorch" if torch.version.hip else "CUDA-required PyTorch")
    assert metadata["hip_version"] == torch.version.hip
    assert len(metadata["checkpoints"]) == 2
    for relative_path in metadata["checkpoints"]:
        saved = torch.load(tmp_path / "original" / relative_path,
                           map_location="cuda:0", weights_only=True)
        assert saved["num_layers"] == 2
        assert saved["input_units"] == "sqrt(raw daily BV)"
        assert saved["target_units"] == "sqrt(raw daily BV)"
        expected_backend = ("native ROCm/HIP; MIOpen disabled" if torch.version.hip
                            else "cuDNN (observed training forward)")
        assert saved.get("rnn_backend") == expected_backend
        assert saved["cudnn_used"] is (not bool(torch.version.hip))
        assert saved["cudnn_version"] == (None if torch.version.hip else torch.backends.cudnn.version())
        assert saved["rnn_autograd_nodes"] == metadata["rnn_autograd_nodes"]
        assert saved["hip_version"] == torch.version.hip
        assert saved["train_last_target"] <= saved["refit_asof"]
        assert all(tensor.device == torch.device("cuda:0") for tensor in saved["state_dict"].values())
    with pytest.raises(ValueError, match="never replaced"):
        paper_gru_forecasts(prepared, settings, calendar, tmp_path / "original")


def test_frozen_two_session_cuda_refits_skip_tuning_and_preserve_causality(tmp_path, monkeypatch):
    from financial_economics.paper import prepare_paper_forecasts

    try:
        torch, _ = _require_cuda()
    except RuntimeError as error:
        if "no compatible GPU" in str(error) or "prediction extra" in str(error):
            pytest.skip(str(error))
        raise
    _, calendar = _prepared()
    bv = pd.Series(.0001 * (2 + np.sin(np.arange(len(calendar)) / 7)), index=calendar)
    prepared = prepare_paper_forecasts(bv, calendar, lookback=3, horizon=2)
    settings, frozen = _settings(), {"hidden_size": 2, "epochs": 1}
    monkeypatch.setattr("financial_economics.paper_gru._cv_folds",
                        lambda *args: pytest.fail("Frozen selection must not run cross-validation"))
    forecasts, tuning, metadata = paper_gru_forecasts(
        prepared, settings, calendar, tmp_path / "frozen", progress=lambda message: None,
        frozen_selection=frozen)
    changed = SimpleNamespace(rows=prepared.rows.copy(), inputs=prepared.inputs.copy())
    later = changed.rows.index >= "2010-06-01"
    changed.rows.loc[later, "realized_bv"] *= 5
    changed.inputs[later] *= 5
    altered, _, _ = paper_gru_forecasts(
        changed, settings, calendar, tmp_path / "changed", progress=lambda message: None,
        frozen_selection=frozen)
    earlier = forecasts.index < "2010-06-01"
    np.testing.assert_array_equal(forecasts.loc[earlier, "gru_bv"], altered.loc[earlier, "gru_bv"])
    assert tuning.empty and metadata["folds"] == []
    assert metadata["validation_mse"] is None
    assert metadata["frozen_selection"] == frozen
    assert metadata["refit_epochs"] == 1 and metadata["device"] == "cuda:0"
    assert len(metadata["checkpoints"]) == 2
    for relative_path in metadata["checkpoints"]:
        saved = torch.load(tmp_path / "frozen" / relative_path,
                           map_location="cuda:0", weights_only=True)
        assert saved["train_last_target"] <= saved["refit_asof"]
        assert saved["epochs"] == 1 and saved["hidden_size"] == 2
        assert all(value.device == torch.device("cuda:0") for value in saved["state_dict"].values())
    for invalid in ({}, {"hidden_size": 3, "epochs": 1}, {"hidden_size": 2, "epochs": True},
                    {"hidden_size": 2, "epochs": 2}):
        with pytest.raises(ValueError, match="Frozen GRU selection"):
            paper_gru_forecasts(prepared, settings, calendar, tmp_path / "invalid",
                                frozen_selection=invalid)
