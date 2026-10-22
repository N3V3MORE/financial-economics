import numpy as np
import pandas as pd
import pytest

from financial_economics.forecasting import ForecastData
from financial_economics.paper_networks import _require_cuda
from financial_economics.paper_networks_log import _log_inputs, _log_scale, reconstruct_log_gru


def _prepared(channels=5):
    calendar = pd.bdate_range("2009-07-01", "2011-01-10")
    position = np.arange(len(calendar))
    # Very calm values near zero are exactly where raw linear outputs went negative.
    values = 1e-6 * (1.05 + np.sin(position / 7))
    features = np.column_stack((values, np.sin(position), -np.cos(position),
                                .1 * np.sin(position / 3), -.01 * np.cos(position / 5)))[:, :channels]
    inputs = np.lib.stride_tricks.sliding_window_view(features, 3, axis=0)[:-1].transpose(0, 2, 1)
    rows = pd.DataFrame({"return_date": calendar[3:], "realized_bv": values[3:]},
                        index=calendar[2:-1].rename("forecast_date"))
    return ForecastData(rows, inputs.copy()), calendar


def test_log_scaling_uses_training_labels_only_and_keeps_signed_features():
    prepared, _ = _prepared()
    training = (prepared.rows.return_date <= "2009-12-31").to_numpy()
    scale = _log_scale(prepared, training)
    logs = np.log(prepared.rows.loc[training, "realized_bv"])
    assert scale["target_mean"] == pytest.approx(logs.mean())
    assert scale["target_std"] == pytest.approx(logs.std(ddof=0))
    transformed = _log_inputs(prepared.inputs)
    np.testing.assert_allclose(transformed[..., 0], np.log(prepared.inputs[..., 0]))
    np.testing.assert_array_equal(transformed[..., 1:], prepared.inputs[..., 1:])
    altered = ForecastData(prepared.rows.copy(), prepared.inputs.copy())
    altered.inputs[~training] = 100
    altered.rows.loc[~training, "realized_bv"] = 100
    assert _log_scale(altered, training) == scale


@pytest.mark.parametrize("channels, name", [(1, "gru_univar_log"), (5, "gru_multivar_log")])
def test_tiny_cuda_log_refits_are_positive_and_use_past_labels(tmp_path, channels, name):
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
    forecasts, tuning, metadata = reconstruct_log_gru(prepared, calendar, tmp_path / name, **options)
    issued = forecasts[f"{name}_bv"].dropna()
    assert len(issued) and (issued > 0).all() and np.isfinite(issued).all()
    changed = ForecastData(prepared.rows.copy(), prepared.inputs.copy())
    later = changed.rows.index >= "2010-06-01"
    changed.inputs[later] *= 5
    changed.rows.loc[later, "realized_bv"] *= 5
    altered, changed_tuning, _ = reconstruct_log_gru(changed, calendar, tmp_path / f"changed_{name}", **options)
    earlier = forecasts.index < "2010-06-01"
    np.testing.assert_array_equal(forecasts.loc[earlier, f"{name}_bv"], altered.loc[earlier, f"{name}_bv"])
    pd.testing.assert_frame_equal(tuning, changed_tuning)
    assert metadata["floor_events"] == 0 and metadata["input_channels"] == channels
    assert metadata["backend"] == ("ROCm-required PyTorch" if torch.version.hip else "CUDA-required PyTorch")
    assert metadata["hip_version"] == torch.version.hip
    for fold in metadata["folds"]:
        assert fold["train_last_target"] <= fold["validation_first_issue"]
    for relative_path in metadata["checkpoints"]:
        saved = torch.load(tmp_path / name / relative_path, map_location="cuda:0", weights_only=True)
        assert saved["smearing"] > 0 and saved["train_last_target"] <= saved["refit_asof"]
        expected_backend = "native ROCm/HIP; MIOpen disabled" if torch.version.hip else "native CUDA; cuDNN disabled"
        assert saved["rnn_backend"] == metadata["rnn_backend"] == expected_backend
        assert saved["hip_version"] == metadata["hip_version"]
    with pytest.raises(ValueError, match="never replaced"):
        reconstruct_log_gru(prepared, calendar, tmp_path / name, **options)
