"""Saved forecast stages must share verified reconstruction inputs."""

from hashlib import sha256
import json

import pandas as pd
import pytest

from financial_economics.studies.realistic import _load_forecasts


INPUTS = {name: sha256(name.encode()).hexdigest()
          for name in ("source_manifest", "oxford_archive", "config")}
STAGES = {"evaluation": "arfima", "gru_univar_log": "gru_univar_log",
          "gru_multivar_log": "gru_multivar_log"}


def _stage(root, stage, *, inputs=None, dates=None):
    directory = root / stage
    directory.mkdir()
    model = STAGES[stage]
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"] if dates is None else dates)
    frame = pd.DataFrame({f"{model}_bv": [.0001, .0002], f"{model}_refit_asof": dates},
                         index=dates.rename("forecast_date"))
    frame.to_csv(directory / "forecasts.csv")
    metadata = {"inputs": INPUTS.copy() if inputs is None else inputs,
                "files": {"forecasts.csv": sha256((directory / "forecasts.csv").read_bytes()).hexdigest()}}
    (directory / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return frame, metadata


def test_matching_stages_keep_values_and_validated_input_provenance(tmp_path):
    expected = []
    for stage in STAGES:
        frame, _ = _stage(tmp_path, stage)
        expected.append(frame)
    forecasts, provenance = _load_forecasts(tmp_path, list(STAGES.values()))
    pd.testing.assert_frame_equal(forecasts, pd.concat(expected, axis=1))
    assert set(provenance) == set(STAGES)
    for stage, saved in provenance.items():
        assert saved["inputs"] == INPUTS
        assert saved["run_sha256"] == sha256((tmp_path / stage / "run.json").read_bytes()).hexdigest()
        assert saved["forecasts_sha256"] == sha256((tmp_path / stage / "forecasts.csv").read_bytes()).hexdigest()


@pytest.mark.parametrize("stage", ["gru_univar_log", "gru_multivar_log"])
@pytest.mark.parametrize("field", list(INPUTS))
def test_different_inputs_fail_despite_matching_dates_and_valid_file_hashes(tmp_path, stage, field):
    _stage(tmp_path, "evaluation")
    changed = INPUTS | {field: sha256(b"another reconstruction run").hexdigest()}
    _stage(tmp_path, stage, inputs=changed)
    with pytest.raises(ValueError, match=rf"different input identities \({field}\)"):
        _load_forecasts(tmp_path, ["arfima", STAGES[stage]])


def test_matching_digest_case_is_normalized_in_provenance(tmp_path):
    _stage(tmp_path, "evaluation")
    _stage(tmp_path, "gru_univar_log", inputs={name: value.upper() for name, value in INPUTS.items()})
    _, provenance = _load_forecasts(tmp_path, ["arfima", "gru_univar_log"])
    assert provenance["gru_univar_log"]["inputs"] == INPUTS


@pytest.mark.parametrize("missing", ["inputs", *INPUTS])
def test_missing_input_identity_is_rejected(tmp_path, missing):
    _, metadata = _stage(tmp_path, "evaluation")
    if missing == "inputs":
        del metadata["inputs"]
    else:
        del metadata["inputs"][missing]
    (tmp_path / "evaluation" / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation forecasts require valid SHA-256 input identities"):
        _load_forecasts(tmp_path, ["arfima"])


@pytest.mark.parametrize("invalid", [None, [], "not an identity"])
def test_invalid_input_identity_container_is_rejected(tmp_path, invalid):
    _, metadata = _stage(tmp_path, "evaluation")
    metadata["inputs"] = invalid
    (tmp_path / "evaluation" / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="require valid SHA-256 input identities"):
        _load_forecasts(tmp_path, ["arfima"])


@pytest.mark.parametrize("invalid", [None, 123, True, "", "a" * 63, "a" * 65, "z" * 64])
def test_invalid_identity_digest_is_rejected(tmp_path, invalid):
    _stage(tmp_path, "evaluation", inputs=INPUTS | {"source_manifest": invalid})
    with pytest.raises(ValueError, match="require valid SHA-256 input identities"):
        _load_forecasts(tmp_path, ["arfima"])


def test_modified_forecast_bytes_still_fail_hash_verification(tmp_path):
    _stage(tmp_path, "evaluation")
    with (tmp_path / "evaluation" / "forecasts.csv").open("a", encoding="utf-8") as destination:
        destination.write("\n")
    with pytest.raises(ValueError, match="do not match their provenance hash"):
        _load_forecasts(tmp_path, ["arfima"])


def test_different_issue_dates_still_fail_with_valid_hashes_and_matching_inputs(tmp_path):
    _stage(tmp_path, "evaluation")
    _stage(tmp_path, "gru_univar_log", dates=["2010-01-05", "2010-01-06"])
    with pytest.raises(ValueError, match="must cover identical issue dates"):
        _load_forecasts(tmp_path, ["arfima", "gru_univar_log"])
