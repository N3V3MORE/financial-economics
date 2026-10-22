"""Relocation preserves old imports and rejects unreviewed cache identities."""

from importlib import import_module
import pickle
import runpy
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


def test_only_the_exact_reviewed_baseline_is_compatible():
    before = _layout._manifest()["before"]
    current = package_code_digests()
    assert _layout.code_matches(before, current)
    assert _layout.code_matches(current, current)
    assert not _layout.code_matches(before | {"costs.py": "changed"}, current)
    assert not _layout.code_matches(before, current | {"portfolio/execution.py": "changed"})


def test_editing_any_python_dependency_invalidates_the_relocation_exception(monkeypatch):
    before = _layout._manifest()["before"]
    current = package_code_digests()
    actual = _layout._python_digests()
    monkeypatch.setattr(_layout, "_python_digests", lambda: actual | {"portfolio/execution.py": "changed"})
    assert not _layout.code_matches(before, current)
    assert not _layout.relocated_digest_matches(
        "paper_sources.py", before["paper_sources.py"], current["datasets/sources/paper.py"])


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
