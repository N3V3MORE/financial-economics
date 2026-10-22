"""Saved forecasts retain provenance across a reader-only compatibility fix."""

import pandas as pd
import pytest

from financial_economics import paper_reconstruction as reconstruction


def snapshot(directory, index_name=None, *, stage="classical", legacy_metadata=None):
    """legacy_metadata picks which reviewed original digest (0 = oldest, -1 = newest) to record."""
    identity = {"source_manifest": "source", "oxford_archive": "bv", "config": "config"}
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    frame = pd.DataFrame({"return_date": pd.to_datetime(["2010-01-05", "2010-01-06"]),
                          "arfima_bv": [.0001, .0002], "arfima_refit_asof": dates},
                         index=dates.rename(index_name))
    frame.to_csv(directory / "forecasts.csv")
    code = reconstruction._code_hashes(stage)
    if legacy_metadata is not None:
        for name, (originals, current) in reconstruction.METADATA_ONLY_CODE_TRANSITIONS.items():
            if name in code:
                assert code[name] == current, f"Review the compatibility pair after changing {name}"
                code[name] = originals[legacy_metadata]
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


@pytest.mark.parametrize("original", [0, -1])
@pytest.mark.parametrize("stage", ["gru_univar", "gru_multivar_log"])
def test_metadata_only_digest_transitions_preserve_archived_forecasts(tmp_path, stage, original):
    expected, identity = snapshot(tmp_path, stage=stage, legacy_metadata=original)
    original_run = (tmp_path / "run.json").read_bytes()
    actual, metadata = reconstruction._read_stage(tmp_path, identity, stage)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
    assert (tmp_path / "run.json").read_bytes() == original_run
    transitions = metadata["cache_reader"]["metadata_only_code_compatibility"]
    assert set(transitions) == set(reconstruction._code_hashes(stage)).intersection(
        reconstruction.METADATA_ONLY_CODE_TRANSITIONS)
    for name, transition in transitions.items():
        originals, current = reconstruction.METADATA_ONLY_CODE_TRANSITIONS[name]
        assert (transition["original_sha256"], transition["current_sha256"]) == (originals[original], current)
        assert "model math unchanged" in transition["reason"]


def test_metadata_only_transitions_never_accept_the_current_digest_as_an_original():
    for originals, current in reconstruction.METADATA_ONLY_CODE_TRANSITIONS.values():
        assert originals and current not in originals and len(set(originals)) == len(originals)


@pytest.mark.parametrize("stage", ["gru_univar", "gru_multivar_log"])
def test_metadata_compatibility_still_rejects_arbitrary_code_or_file_changes(tmp_path, monkeypatch, stage):
    _, identity = snapshot(tmp_path, stage=stage, legacy_metadata=-1)
    code = reconstruction._code_hashes(stage)
    with monkeypatch.context() as changed:
        changed.setattr(reconstruction, "_code_hashes", lambda _: code | {"paper_networks.py": "unreviewed"})
        with pytest.raises(ValueError, match="inputs or model code changed"):
            reconstruction._read_stage(tmp_path, identity, stage)
    with (tmp_path / "forecasts.csv").open("a") as destination:
        destination.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        reconstruction._read_stage(tmp_path, identity, stage)
