"""Provenance names resolve to grouped modules and reject unreviewed cache identities."""

from importlib import import_module

import pytest

from financial_economics import _layout, cli
from financial_economics.provenance import package_code_digests


@pytest.mark.parametrize("original,relative", sorted(_layout._manifest()["modules"].items()))
def test_every_provenance_name_resolves_to_an_existing_module(original, relative):
    assert _layout.module_source(original) == (_layout.PACKAGE_ROOT / relative).resolve()
    assert _layout.module_source(original).is_file()


def test_provenance_hashes_nested_implementations_and_the_relocation_manifest():
    code = package_code_digests()
    assert "portfolio/execution.py" in code
    assert "models/gpu.py" in code
    assert "studies/historical.py" in code
    assert "_layout.py" in code
    assert "_relocation.json" in code
    assert "costs.py" not in code


def test_single_module_relocation_is_checked_per_file():
    manifest = _layout._manifest()
    before, after = manifest["before"], manifest["after"]
    assert _layout.relocated_digest_matches(
        "paper_sources.py", before["paper_sources.py"], after["datasets/sources/paper.py"])
    assert not _layout.relocated_digest_matches("paper_sources.py", before["paper_sources.py"], "changed")
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


@pytest.mark.parametrize("command,module", sorted((name, module) for name, (module, _) in cli.MODULE_COMMANDS.items()))
def test_module_commands_pass_their_arguments_through(monkeypatch, command, module):
    received = []
    monkeypatch.setattr(import_module(f"financial_economics.{module}"), "main",
                        lambda argv: received.append(argv) or 7)
    assert cli.main([command, "stage", "--flag", "value"]) == 7
    assert received == [["stage", "--flag", "value"]]


def test_module_commands_are_listed_in_fecon_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    help_text = capsys.readouterr().out
    assert all(name in help_text for name in cli.MODULE_COMMANDS)
