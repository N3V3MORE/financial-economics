"""The migration checker must catch incomplete inventories and damaged artifacts."""

import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


script = Path(__file__).resolve().parents[1] / "scripts" / "verify_restructure.py"
spec = importlib.util.spec_from_file_location("restructure_verifier", script)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture(tmp_path):
    baseline, current = tmp_path / "baseline", tmp_path / "current"
    names = {"configs/study.toml": b"original config\n",
             "outputs/completed/performance.csv": b"metric,value\nreturn,0.1\n",
             "scripts/make_findings_figures.py": b"# existing user script\n"}
    for root in (baseline, current):
        for name, data in names.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    manifest = {"files": {name: verifier.fingerprint(baseline / name) for name in names}}
    return baseline, current, manifest


def test_incomplete_backup_inventory_cannot_pass(tmp_path):
    baseline, current, manifest = fixture(tmp_path)
    del manifest["files"]["configs/study.toml"]
    result = verifier.protected_check(baseline, current, manifest)
    assert result["status"] == "failed"
    assert any(row["error"] == "absent_from_baseline_manifest" for row in result["errors"])


@pytest.mark.parametrize("damage", ["change", "delete"])
def test_changed_or_missing_completed_results_fail(tmp_path, damage):
    baseline, current, manifest = fixture(tmp_path)
    path = current / "outputs/completed/performance.csv"
    if damage == "change":
        path.write_bytes(b"different result")
    else:
        path.unlink()
    assert verifier.protected_check(baseline, current, manifest)["status"] == "failed"


def test_new_results_are_explicitly_listed(tmp_path):
    baseline, current, manifest = fixture(tmp_path)
    (current / "outputs/completed/additional.csv").write_bytes(b"new result")
    result = verifier.protected_check(baseline, current, manifest)
    assert result["status"] == "passed"
    assert result["new_files"]["current"] == ["outputs/completed/additional.csv"]


def test_live_checkpoint_move_requires_identical_bytes(tmp_path):
    baseline, current, manifest = fixture(tmp_path)
    name = "outputs/run/.revised-abc/stage/checkpoints/2010/seed_11.pt"
    before = baseline / name
    before.parent.mkdir(parents=True)
    before.write_bytes(b"original checkpoint")
    manifest["files"][name] = verifier.fingerprint(before)
    completed = "outputs/run/gru_univar_log/checkpoints/2010/seed_11.pt"
    after = current / completed
    after.parent.mkdir(parents=True)
    after.write_bytes(before.read_bytes())
    transition = {name: {"kind": "completed_stage", "completed_path": completed}}
    result = verifier.protected_check(baseline, current, manifest, transition)
    assert result["status"] == "passed"
    assert result["verified_live_updates"][0]["completed_path"] == completed
    after.write_bytes(b"changed checkpoint")
    assert verifier.protected_check(baseline, current, manifest, transition)["status"] == "failed"


def test_live_log_may_append_but_cannot_rewrite_original_content(tmp_path):
    baseline, current, manifest = fixture(tmp_path)
    name = "outputs/trial-logs/training.log"
    for root in (baseline, current):
        path = root / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"original log\n")
    manifest["files"][name] = verifier.fingerprint(baseline / name)
    transition = {name: {"kind": "append_only_log"}}
    (current / name).write_bytes(b"original log\nnew line\n")
    assert verifier.protected_check(baseline, current, manifest, transition)["status"] == "passed"
    (current / name).write_bytes(b"rewritten log\nnew line\n")
    assert verifier.protected_check(baseline, current, manifest, transition)["status"] == "failed"


def test_table_hash_includes_dates_even_when_table_has_no_columns():
    first = pd.DataFrame(index=pd.date_range("2020-01-01", periods=3))
    second = pd.DataFrame(index=pd.date_range("2020-01-02", periods=3))
    assert verifier.table_hash(first, np, pd) != verifier.table_hash(second, np, pd)


def test_table_hash_detects_sub_rounding_value_changes():
    first = pd.DataFrame({"return": [0.123456789012345]})
    second = first.copy()
    second.iloc[0, 0] = np.nextafter(first.iloc[0, 0], np.inf)
    assert verifier.table_hash(first, np, pd) != verifier.table_hash(second, np, pd)
