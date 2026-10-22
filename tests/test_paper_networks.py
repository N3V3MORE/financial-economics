from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_economics.forecasting import ForecastData
from financial_economics.paper_networks import _require_cuda, _training_scale, reconstruct_gru


def _prepared(channels=5):
    calendar = pd.bdate_range("2009-07-01", "2011-01-10")
    position = np.arange(len(calendar))
    values = .0001 * (2 + np.sin(position / 7))
    features = np.column_stack((values, np.sin(position), -np.cos(position),
                                .1 * np.sin(position / 3), -.01 * np.cos(position / 5)))[:, :channels]
    inputs = np.lib.stride_tricks.sliding_window_view(features, 3, axis=0)[:-1].transpose(0, 2, 1)
    rows = pd.DataFrame({"return_date": calendar[3:], "realized_bv": values[3:]},
                        index=calendar[2:-1].rename("forecast_date"))
    return ForecastData(rows, inputs.copy()), calendar


def test_raw_bv_scaling_preserves_signed_features_and_ignores_future_values():
    prepared, _ = _prepared()
    training = (prepared.rows.return_date <= "2009-12-31").to_numpy()
    scale = _training_scale(prepared, training)
    assert scale["target_scale"] == pytest.approx(prepared.rows.loc[training, "realized_bv"].mean())
    np.testing.assert_allclose(scale["feature_mean"], prepared.inputs[training].mean(axis=(0, 1)))
    assert (prepared.inputs[:, :, 1:] < 0).any()
    altered = ForecastData(prepared.rows.copy(), prepared.inputs.copy())
    altered.inputs[~training] = 100
    altered.rows.loc[~training, "realized_bv"] = 100
    assert _training_scale(altered, training) == scale


@pytest.mark.parametrize("channels, name", [(1, "gru_univar"), (5, "gru_multivar")])
def test_tiny_cuda_raw_bv_refits_use_past_labels_and_save_independent_layers(tmp_path, channels, name):
    try:
        torch, _ = _require_cuda()
    except RuntimeError as error:
        if "no compatible GPU" in str(error) or "prediction extra" in str(error):
            pytest.skip(str(error))
        raise
    prepared, calendar = _prepared(channels)
    prepared.rows.loc[prepared.rows.index[-1], "realized_bv"] = np.nan
    options = dict(model_name=name, profiles=[(2, 3, 256, .1, .0001)], seeds=(11,),
                   max_epochs=1, patience=1, folds=2, progress=lambda message: None)
    forecasts, tuning, metadata = reconstruct_gru(prepared, calendar, tmp_path / name, **options)
    changed = ForecastData(prepared.rows.copy(), prepared.inputs.copy())
    later = changed.rows.index >= "2010-06-01"
    changed.inputs[later] *= 5
    changed.rows.loc[later, "realized_bv"] *= 5
    altered, changed_tuning, _ = reconstruct_gru(changed, calendar, tmp_path / f"changed_{name}", **options)
    earlier = forecasts.index < "2010-06-01"
    np.testing.assert_array_equal(forecasts.loc[earlier, f"{name}_bv"], altered.loc[earlier, f"{name}_bv"])
    pd.testing.assert_frame_equal(tuning, changed_tuning)
    assert forecasts.loc["2009-12-31", f"{name}_refit_asof"] == pd.Timestamp("2009-12-31")
    assert np.isfinite(forecasts.iloc[-1][f"{name}_bv"])
    assert metadata["device"] == "cuda:0" and metadata["refit_epochs"] == 1
    assert metadata["backend"] == ("ROCm-required PyTorch" if torch.version.hip else "CUDA-required PyTorch")
    assert metadata["hip_version"] == torch.version.hip
    assert metadata["input_channels"] == channels
    assert len(metadata["checkpoints"]) == 2
    for fold in metadata["folds"]:
        assert fold["train_last_target"] <= fold["validation_first_issue"]
    for relative_path in metadata["checkpoints"]:
        saved = torch.load(tmp_path / name / relative_path, map_location="cuda:0", weights_only=True)
        assert saved["model_name"] == name and saved["input_channels"] == channels
        assert saved["hidden_1"] == 2 and saved["hidden_2"] == 3
        assert saved["state_dict"]["first.weight_ih_l0"].shape == (6, channels)
        assert saved["state_dict"]["second.weight_ih_l0"].shape == (9, 2)
        assert saved["train_last_target"] <= saved["refit_asof"]
        expected_backend = "native ROCm/HIP; MIOpen disabled" if torch.version.hip else "native CUDA; cuDNN disabled"
        assert saved["rnn_backend"] == metadata["rnn_backend"] == expected_backend
        assert saved["hip_version"] == metadata["hip_version"]
        assert all(tensor.device == torch.device("cuda:0") for tensor in saved["state_dict"].values())
    with pytest.raises(ValueError, match="never replaced"):
        reconstruct_gru(prepared, calendar, tmp_path / name, **options)


def test_network_pipeline_requires_cuda_without_creating_checkpoints(tmp_path, monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("financial_economics.gru_forecasting.importlib.import_module", lambda name: fake_torch)
    prepared, calendar = _prepared()
    with pytest.raises(RuntimeError, match=r"CUDA:0.*CPU fallback is disabled"):
        reconstruct_gru(prepared, calendar, tmp_path / "missing_cuda", model_name="gru_multivar", folds=2)
    assert not (tmp_path / "missing_cuda").exists()
