"""Alternative providers preserve series identities, units, and raw provenance."""
from hashlib import sha256
from io import BytesIO
import json

import pandas as pd
import pytest

from financial_economics.paper_sources import _load_macro_snapshot, download_sources, load_sources


def _snapshot(tmp_path, *, symbol="DAAA", units="percent per year", duplicate=False):
    folder = tmp_path / "macro"
    folder.mkdir()
    dates = ["2000-01-03", "2020-12-31"]
    if duplicate:
        dates.append("2020-12-31")
    frame = pd.DataFrame({"gov_yield": [1.] * len(dates), "corp_yield": [2.] * len(dates),
                          "OIL": [30.] * len(dates)}, index=pd.Index(dates, name="Date"))
    frame.to_csv(folder / "observations.csv")
    (folder / "raw.txt").write_text("Retained synthetic provider response.", encoding="utf-8")
    metadata = {"sources": {
        "gov_yield": {"provider": "wrds", "fred_series": "DGS3", "units": "percent per year"},
        "corp_yield": {"provider": "wrds", "fred_series": symbol, "units": units},
        "OIL": {"provider": "eia", "fred_series": "DCOILBRENTEU", "units": "USD per barrel"}},
        "files": {p.name: sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},
        "limitations": ["Synthetic alternative sources for testing."]}
    (folder / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder


@pytest.mark.parametrize("kwargs", [{"symbol": "DBAA"}, {"units": "decimal"}, {"duplicate": True}])
def test_alternative_rejects_changed_series_units_or_duplicate_dates(tmp_path, kwargs):
    with pytest.raises(ValueError, match="identity or units|unique ordered dates"):
        _load_macro_snapshot(_snapshot(tmp_path, **kwargs))


def test_alternative_rejects_modified_data_before_price_download(tmp_path, monkeypatch):
    folder = _snapshot(tmp_path)
    (folder / "observations.csv").write_text("changed", encoding="utf-8")
    monkeypatch.setattr("financial_economics.paper_sources.urlopen",
                        lambda *args, **kwargs: pytest.fail("Prices must not be fetched for invalid macro data"))
    with pytest.raises(ValueError, match="hash mismatch"):
        download_sources(tmp_path / "prices", macro_dir=folder)
    assert not (tmp_path / "prices").exists()


def test_alternative_rejects_absolute_raw_file_paths(tmp_path):
    folder = _snapshot(tmp_path)
    path = folder / "manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["files"][str((folder / "raw.txt").resolve())] = metadata["files"].pop("raw.txt")
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="relative paths"):
        _load_macro_snapshot(folder)


def test_alternative_copies_and_hashes_raw_inputs_without_contacting_fred(tmp_path, monkeypatch):
    macro = _snapshot(tmp_path)
    requested = []
    raw = json.dumps({"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": "UTC"}, "timestamp": [946857600, 1609372800],
        "indicators": {"quote": [{"close": [100., 120.]}]}}]}}).encode()

    def fetch(request, **kwargs):
        requested.append(request.full_url)
        return BytesIO(raw)

    monkeypatch.setattr("financial_economics.paper_sources.urlopen", fetch)
    monkeypatch.setattr("financial_economics.paper_sources.YAHOO", {"equity_price": ("^GSPC", "S&PCOMP(PI)", "test")})
    destination = tmp_path / "prices"
    download_sources(destination, macro_dir=macro)
    frame, metadata = load_sources(destination)
    assert len(requested) == 1 and "finance.yahoo.com" in requested[0]
    assert set(frame) == {"equity_price", "gov_yield", "corp_yield", "OIL"}
    assert metadata["sources"]["corp_yield"]["provider"] == "wrds"
    assert metadata["macro_snapshot_sha256"] == sha256((macro / "manifest.json").read_bytes()).hexdigest()
    assert "macro_snapshot/raw.txt" in metadata["files"]
    assert (destination / "macro_snapshot/raw.txt").read_bytes() == (macro / "raw.txt").read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        download_sources(destination, macro_dir=macro)
    (destination / "macro_snapshot/raw.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sources(destination)
