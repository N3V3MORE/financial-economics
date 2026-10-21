"""Saved forecasts retain provenance across a reader-only compatibility fix."""

import pandas as pd
import pytest

from financial_economics import paper_reconstruction as reconstruction


def snapshot(directory, index_name=None):
    identity = {"source_manifest": "source", "oxford_archive": "bv", "config": "config"}
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    frame = pd.DataFrame({"return_date": pd.to_datetime(["2010-01-05", "2010-01-06"]),
                          "arfima_bv": [.0001, .0002], "arfima_refit_asof": dates},
                         index=dates.rename(index_name))
    frame.to_csv(directory / "forecasts.csv")
    code = reconstruction._code_hashes("classical")
    code["paper_reconstruction.py"] = "historical-execution-wrapper"
    reconstruction._write_metadata(directory, {"inputs": identity, "code": code})
    return frame, identity


@pytest.mark.parametrize("index_name", [None, "forecast_date"])
def test_named_and_legacy_unnamed_index_caches_preserve_values(tmp_path, index_name):
    expected, identity = snapshot(tmp_path, index_name)
    actual, metadata = reconstruction._read_stage(tmp_path, identity, "classical")
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
    assert actual.index.name == "forecast_date"
    assert metadata["cache_reader"]["original_wrapper_sha256"] == "historical-execution-wrapper"
    assert metadata["cache_reader"]["current_wrapper_sha256"] != "historical-execution-wrapper"


def test_changed_inputs_or_forecasting_code_refuse_cache_reuse(tmp_path, monkeypatch):
    _, identity = snapshot(tmp_path)
    with pytest.raises(ValueError, match="inputs or model code changed"):
        reconstruction._read_stage(tmp_path, identity | {"config": "changed"}, "classical")
    code = reconstruction._code_hashes("classical") | {"paper_models.py": "changed"}
    monkeypatch.setattr(reconstruction, "_code_hashes", lambda stage: code)
    with pytest.raises(ValueError, match="inputs or model code changed"):
        reconstruction._read_stage(tmp_path, identity, "classical")


def test_modified_saved_forecast_is_detected(tmp_path):
    _, identity = snapshot(tmp_path)
    with (tmp_path / "forecasts.csv").open("a") as destination:
        destination.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        reconstruction._read_stage(tmp_path, identity, "classical")
