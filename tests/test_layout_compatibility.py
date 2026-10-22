"""Relocation preserves old imports and rejects unreviewed cache identities."""

from importlib import import_module, reload
import pickle
import runpy
import subprocess
import sys
from pathlib import Path
import warnings

import pytest

from financial_economics import _layout
from financial_economics.provenance import package_code_digests


@pytest.mark.parametrize("original,relative", sorted(_layout._manifest()["modules"].items()))
def test_old_import_is_the_same_module_as_the_relocated_implementation(original, relative):
    legacy = import_module("financial_economics." + original.removesuffix(".py"))
    canonical = import_module("financial_economics." + relative.removesuffix(".py").replace("/", "."))
    assert legacy is canonical


def test_monkeypatches_through_legacy_names_reach_real_function_globals(monkeypatch):
    from financial_economics import paper_realistic
    from financial_economics.studies import realistic

    marker = object()
    monkeypatch.setattr(paper_realistic, "dynamic_costed_path", marker)
    assert realistic.realistic_portfolios.__globals__["dynamic_costed_path"] is marker


def test_provenance_hashes_nested_implementations_and_the_relocation_manifest():
    code = package_code_digests()
    assert "portfolio/execution.py" in code
    assert "models/gpu.py" in code
    assert "studies/historical.py" in code
    assert "_layout.py" in code
    assert "_relocation.json" in code
    assert code["portfolio/execution.py"] != code["costs.py"]


@pytest.fixture
def reviewed_package(monkeypatch):
    """Simulate the reviewed after-state so these checks survive later package edits."""
    manifest = _layout._manifest()
    monkeypatch.setattr(_layout, "_python_digests", lambda: dict(manifest["after"]))
    return manifest["before"], _layout.current_code_digests()


def test_only_the_exact_reviewed_baseline_is_compatible(reviewed_package):
    before, current = reviewed_package
    assert _layout.code_matches(before, current)
    assert _layout.code_matches(current, current)
    assert not _layout.code_matches(before | {"costs.py": "changed"}, current)
    assert not _layout.code_matches(before, current | {"portfolio/execution.py": "changed"})


def test_editing_any_python_dependency_invalidates_the_whole_package_exception(monkeypatch, reviewed_package):
    before, _ = reviewed_package
    after = _layout._python_digests()
    monkeypatch.setattr(_layout, "_python_digests", lambda: after | {"portfolio/execution.py": "changed"})
    assert not _layout.code_matches(before, _layout.current_code_digests())


def test_single_module_relocation_ignores_edits_elsewhere(monkeypatch, reviewed_package):
    before, _ = reviewed_package
    after = _layout._manifest()["after"]
    monkeypatch.setattr(_layout, "_python_digests", lambda: after | {"portfolio/execution.py": "changed"})
    assert _layout.relocated_digest_matches(
        "paper_sources.py", before["paper_sources.py"], after["datasets/sources/paper.py"])
    assert not _layout.relocated_digest_matches(
        "paper_sources.py", before["paper_sources.py"], "changed")
    assert not _layout.relocated_digest_matches(
        "paper_sources.py", before["costs.py"], after["datasets/sources/paper.py"])
    assert not _layout.relocated_digest_matches("new_module.py", None, after["datasets/sources/paper.py"])


def test_retired_gpu_math_is_not_approved_by_a_module_move():
    current = package_code_digests()["models/gpu.py"]
    assert not _layout.relocated_digest_matches(
        "gru_forecasting.py", "e55887f5c820fc4467f45c9762f193110b3193617de188933e1ca59d36fffd41", current)


@pytest.mark.parametrize("name", ["../costs.py", "portfolio/execution.py", "not-a-module"])
def test_source_resolver_refuses_non_module_names(name):
    with pytest.raises(ValueError, match="module filename"):
        _layout.module_source(name)


def test_legacy_runpy_still_works_after_import():
    import_module("financial_economics.config")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        namespace = runpy.run_module("financial_economics.config", run_name="__main__")
    assert "Study" in namespace
    assert "load_study" in namespace


def test_original_class_pickle_namespace_is_preserved():
    from financial_economics.config import Study

    value = Study("sample", "2020-01-01", "2020-12-31", "SPY", {"SPY": "Stocks"})
    encoded = pickle.dumps(value)
    assert Study.__module__ == "financial_economics.config"
    assert b"financial_economics.config" in encoded
    assert pickle.loads(encoded) == value


def test_reload_preserves_original_pickle_and_runpy_behavior():
    module = reload(import_module("financial_economics.config"))
    assert module.Study.__module__ == "financial_economics.config"
    assert module.load_study.__module__ == "financial_economics.config"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        namespace = runpy.run_module("financial_economics.config", run_name="__main__")
    assert "load_study" in namespace


@pytest.mark.parametrize("run_name,alter_sys", [("__main__", False), (None, False), ("custom", True)])
def test_fresh_runpy_preserves_namespace_and_callers_main(run_name, alter_sys):
    source = Path(__file__).resolve().parents[1] / "src"
    code = """
import pickle, runpy, sys
sys.path.insert(0, sys.argv[1])
def caller_function():
    return 1
original_main = sys.modules['__main__']
pickle.dumps(caller_function)
run_name = None if sys.argv[2] == 'None' else sys.argv[2]
namespace = runpy.run_module('financial_economics.config', run_name=run_name,
                             alter_sys=sys.argv[3] == 'True')
assert 'Study' in namespace and 'load_study' in namespace
assert sys.modules['__main__'] is original_main
pickle.dumps(caller_function)
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(source), str(run_name), str(alter_sys)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
