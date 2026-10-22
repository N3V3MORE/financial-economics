from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from financial_economics.forecasting import prepare_forecasts
from financial_economics.gru_forecasting import (
    _annual_splits, _gpu_metadata, _native_rnn, _require_cuda, _training_scale, gru_forecasts,
)
from financial_economics.prediction_config import PredictionSettings


def test_annual_refits_group_issue_dates_and_only_use_available_targets():
    dates = pd.bdate_range("2014-11-01", "2018-01-10")
    dates = dates.difference(pd.to_datetime(["2017-01-02", "2018-01-01"]))
    prepared = prepare_forecasts(pd.Series(.01, index=dates), dates)
    splits = _annual_splits(prepared.rows, dates, "2017-01-01")
    assert splits[0][1] == pd.Timestamp("2016-12-30")
    assert splits[1][1] == pd.Timestamp("2017-12-29")
    for _, asof, training, issued in splits:
        assert prepared.rows.loc[training, "return_date"].max() <= asof
        assert prepared.rows.index[issued][0] == asof
    december = prepared.rows.index.get_loc("2017-12-29")
    assert not splits[0][3][december]
    assert splits[1][3][december]
    unavailable = prepared.rows.index.get_loc("2016-12-29")
    assert not splits[0][2][unavailable]


def test_training_scaling_cannot_use_future_returns():
    dates = pd.bdate_range("2014-11-01", "2017-02-28")
    returns = pd.Series(.01 * np.sin(np.arange(len(dates))), index=dates)
    original = prepare_forecasts(returns, dates)
    training = (original.rows.return_date <= "2014-12-31").to_numpy()
    changed = returns.copy()
    changed.loc["2015-01-01":] = -.25
    altered = prepare_forecasts(changed, dates)
    assert _training_scale(original, training) == _training_scale(altered, training)


def test_missing_cuda_fails_clearly_without_cpu_fallback(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr("financial_economics.gru_forecasting.importlib.import_module",
                        lambda name: fake_torch)
    with pytest.raises(RuntimeError, match=r"CUDA:0.*CPU fallback is disabled") as error:
        _require_cuda()
    assert "NVIDIA CUDA or AMD ROCm" in str(error.value)


def test_missing_pytorch_guidance_covers_cuda_and_rocm(monkeypatch):
    def missing(name):
        raise ImportError("Synthetic missing PyTorch")

    monkeypatch.setattr("financial_economics.gru_forecasting.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match=r"prediction extra.*NVIDIA CUDA.*AMD ROCm"):
        _require_cuda()


@pytest.mark.parametrize("native_rnn", [False, True])
@pytest.mark.parametrize("runtime, cuda, hip, gpu, rnn_backend", [
    ("CUDA", "13.0", None, "NVIDIA synthetic GPU", "native CUDA; cuDNN disabled"),
    ("ROCm", None, "7.14.1", "AMD synthetic GPU", "native ROCm/HIP; MIOpen disabled"),
])
def test_gpu_metadata_records_the_runtime_behind_the_cuda_api(runtime, cuda, hip, gpu, rnn_backend,
                                                            native_rnn):
    requested_devices = []

    def device_name(index):
        requested_devices.append(index)
        return gpu

    torch = SimpleNamespace(__version__="2.11.0", version=SimpleNamespace(cuda=cuda, hip=hip),
                            cuda=SimpleNamespace(get_device_name=device_name))
    metadata = _gpu_metadata(torch, native_rnn=native_rnn)
    assert metadata["backend"] == f"{runtime}-required PyTorch"
    assert metadata["device"] == "cuda:0" and requested_devices == [0]
    assert metadata["gpu"] == gpu and metadata["torch_version"] == "2.11.0"
    assert metadata["cuda_version"] == cuda and metadata["hip_version"] == hip
    if native_rnn:
        assert metadata["rnn_backend"] == rnn_backend
    else:
        assert "rnn_backend" not in metadata


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_native_rnn_context_only_toggles_enabled_and_always_restores_it(enabled, raises):
    class Flags:
        def __init__(self):
            for name, value in {"enabled": enabled, "deterministic": True, "benchmark": False,
                                "benchmark_limit": 17, "allow_tf32": False, "writes": []}.items():
                object.__setattr__(self, name, value)

        def __setattr__(self, name, value):
            self.writes.append((name, value))
            object.__setattr__(self, name, value)

    flags = Flags()
    torch = SimpleNamespace(backends=SimpleNamespace(cudnn=flags))
    outcome = pytest.raises(RuntimeError, match="synthetic forward failure") if raises else nullcontext()
    with outcome:
        with _native_rnn(torch):
            assert flags.enabled is False
            assert flags.deterministic is True and flags.benchmark is False
            assert flags.benchmark_limit == 17 and flags.allow_tf32 is False
            if raises:
                raise RuntimeError("synthetic forward failure")
    assert flags.enabled is enabled
    assert flags.writes == [("enabled", False), ("enabled", enabled)]


def test_tiny_cuda_pipeline_saves_checkpoints_and_preserves_earlier_forecasts(tmp_path):
    try:
        torch, _ = _require_cuda()
    except RuntimeError as error:
        if "no compatible GPU" in str(error) or "prediction extra" in str(error):
            pytest.skip(str(error))
        raise
    dates = pd.bdate_range("2014-11-01", "2017-02-28")
    returns = pd.Series(.001 + .01 * np.sin(np.arange(len(dates)) / 3), index=dates)
    settings = replace(PredictionSettings(), lookback=3, hidden_sizes=(2,),
                       learning_rates=(.001,), seeds=(11,), max_epochs=2, patience=2)
    prepared = prepare_forecasts(returns, dates, lookback=3)
    forecasts, tuning, metadata = gru_forecasts(prepared, settings, dates,
                                               tmp_path / "original", progress=lambda message: None)
    altered = returns.copy()
    altered.loc["2017-02-01":] = -.2
    changed, changed_tuning, changed_metadata = gru_forecasts(
        prepare_forecasts(altered, dates, lookback=3), settings, dates,
        tmp_path / "altered", progress=lambda message: None)
    earlier = (forecasts.index >= settings.validation_start) & (forecasts.index < "2017-02-01")
    np.testing.assert_array_equal(forecasts.loc[earlier, "gru_variance"],
                                  changed.loc[earlier, "gru_variance"])
    pd.testing.assert_frame_equal(tuning, changed_tuning)
    assert metadata["refit_epochs"] == changed_metadata["refit_epochs"]
    assert metadata["device"] == "cuda:0"
    assert metadata["backend"] == ("ROCm-required PyTorch" if torch.version.hip else "CUDA-required PyTorch")
    assert metadata["hip_version"] == torch.version.hip
    if torch.version.hip:
        assert metadata["rnn_backend"] == "native ROCm/HIP; MIOpen disabled"
    assert forecasts.loc[forecasts.index < settings.validation_start, "gru_variance"].isna().all()
    assert (forecasts.loc[forecasts.index >= settings.validation_start, "gru_variance"] > 0).all()
    assert forecasts.loc["2016-12-30", "gru_refit_asof"] == pd.Timestamp("2016-12-30")
    for relative_path in metadata["checkpoints"]:
        saved = torch.load(tmp_path / "original" / relative_path, map_location="cuda:0", weights_only=True)
        assert all(value.device == torch.device("cuda:0") for value in saved["state_dict"].values())
        assert saved["train_last_target"] <= saved["refit_asof"]
        assert saved["epochs"] in (1, 2)
        assert saved["target_scale"] > 0
        assert saved["backend"] == metadata["backend"] and saved["hip_version"] == metadata["hip_version"]
        if torch.version.hip:
            assert saved["rnn_backend"] == metadata["rnn_backend"]
    with pytest.raises(ValueError, match="new GRU checkpoint directory"):
        gru_forecasts(prepared, settings, dates, tmp_path / "original")
