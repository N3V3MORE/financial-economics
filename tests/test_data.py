"""Synthetic source fixtures test data integrity without live services."""

from dataclasses import replace
from io import BytesIO
import json
from zipfile import ZipFile, ZipInfo

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from financial_economics.settings.study import Study
from financial_economics.datasets import public as data


def french_zip(observations=None):
    observations = observations or [
        "20231229,1.23,0.10,-0.20,0.01",
        "20240102,0.25,0.10,-0.20,0.01",
        "20240103,-0.50,0.10,-0.20,0.02",
        "20240104,0.75,0.10,-0.20,0.02",
        "20240105,0.20,0.10,-0.20,0.02",
    ]
    content = "\n".join([
        "Created using a synthetic source.",
        "The risk-free return has explanatory metadata.",
        "",
        ",Mkt-RF,SMB,HML,RF",
        *observations,
        "",
        "Annual Factors: January-December",
        ",Mkt-RF,SMB,HML,RF",
        "2024,100,200,300,400",
        "Copyright example",
    ])
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(ZipInfo("factors.csv", date_time=(2024, 1, 1, 0, 0, 0)), content)
    return buffer.getvalue()


@pytest.fixture
def study():
    return Study("test-study", "2024-01-02", "2024-01-04", "SPY", {"SPY": "Market", "QQQ": "Growth"})


@pytest.fixture
def prices():
    return pd.DataFrame(
        {"SPY": [100.0, 101.0, 102.0, 100.0], "QQQ": [50.0, 52.0, 51.0, 53.0]},
        index=pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"]),
    ).rename_axis("Date")


def mock_sources(monkeypatch, prices):
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return pd.concat({"Adj Close": prices, "Close": prices * 0.95}, axis=1)

    monkeypatch.setattr(yf, "download", fake_download)
    monkeypatch.setattr(yf, "set_tz_cache_location", lambda location: None)
    monkeypatch.setattr(data, "urlopen", lambda *args, **kwargs: BytesIO(french_zip()))
    return calls


def test_french_daily_block_ignores_metadata_and_footer_and_converts_units():
    factors = data.parse_french_factors(french_zip())
    assert list(factors.columns) == ["Mkt-RF", "RF"]
    assert len(factors) == 5
    assert factors.loc["2023-12-29", "Mkt-RF"] == pytest.approx(0.0123)
    assert factors.loc["2024-01-03", "RF"] == pytest.approx(0.0002)


@pytest.mark.parametrize("rows", [
    ["20240230,1,2,3,4"],
    ["20240102,-99.99,2,3,4"],
    ["20240102,1,2,3,4", "20240102,2,3,4,5"],
    ["20240103,1,2,3,4", "20240102,2,3,4,5"],
])
def test_french_invalid_dates_missing_codes_and_duplicate_dates_are_rejected(rows):
    with pytest.raises(ValueError):
        data.parse_french_factors(french_zip(rows))


def test_snapshot_roundtrip_is_offline_and_preserves_adjusted_prices(monkeypatch, tmp_path, study, prices):
    calls = mock_sources(monkeypatch, prices)
    directory = tmp_path / "snapshot"
    manifest = data.download_snapshot(study, directory)
    _, request = calls[0]
    assert request["auto_adjust"] is False
    assert request["keepna"] is True
    assert request["start"] == "2023-12-26"
    assert request["end"] == "2024-01-05"
    assert manifest["settings"] == study.settings()
    assert manifest["retrieved_at_utc"].endswith("+00:00")
    assert (directory / data.SOURCE_ZIP).read_bytes() == french_zip()

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline loading must not call a network service.")

    monkeypatch.setattr(yf, "download", forbidden)
    monkeypatch.setattr(data, "urlopen", forbidden)
    loaded, factors, loaded_manifest = data.load_snapshot(study, directory)
    pd.testing.assert_frame_equal(loaded, prices)
    assert loaded_manifest == manifest
    assert factors.loc["2024-01-02", "Mkt-RF"] == pytest.approx(0.0025)
    returns, aligned = data.align_sample(loaded.pct_change(fill_method=None).iloc[1:], factors, study)
    assert len(returns) == len(aligned) == 3


def test_snapshot_rejects_settings_changes_and_file_tampering(monkeypatch, tmp_path, study, prices):
    mock_sources(monkeypatch, prices)
    directory = tmp_path / "snapshot"
    data.download_snapshot(study, directory)
    with pytest.raises(ValueError, match="settings"):
        data.load_snapshot(replace(study, benchmark="QQQ"), directory)
    price_path = directory / data.PRICE_FILE
    price_path.write_bytes(price_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        data.load_snapshot(study, directory)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_prices_never_create_a_snapshot(monkeypatch, tmp_path, study, prices, bad_value):
    prices.iloc[1, 0] = bad_value
    mock_sources(monkeypatch, prices)
    directory = tmp_path / "snapshot"
    with pytest.raises(ValueError, match=r"positive|nonfinite"):
        data.download_snapshot(study, directory)
    assert not directory.exists()
    assert not list(tmp_path.glob(".snapshot-*"))


def test_calendar_mismatch_is_an_error_not_a_silent_inner_join(study, prices):
    returns = prices.pct_change(fill_method=None).iloc[1:]
    factors = data.parse_french_factors(french_zip())
    with pytest.raises(ValueError, match=r"missing ETF dates.*2024-01-03"):
        data.align_sample(returns.drop(index="2024-01-03"), factors, study)
    extra = pd.DataFrame([[0.01, 0.01]], columns=study.tickers, index=pd.to_datetime(["2024-01-01"]))
    with pytest.raises(ValueError, match=r"unexpected ETF dates.*2024-01-01"):
        data.align_sample(pd.concat([extra, returns]), factors, replace(study, start="2024-01-01"))


def test_existing_destination_is_preserved_without_any_download(monkeypatch, tmp_path, study):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("original")

    def forbidden(*args, **kwargs):
        raise AssertionError("An existing snapshot must be rejected before downloading.")

    monkeypatch.setattr(yf, "download", forbidden)
    with pytest.raises(FileExistsError):
        data.download_snapshot(study, directory)
    assert marker.read_text() == "original"


def test_staging_failure_leaves_no_partial_snapshot(monkeypatch, tmp_path, study, prices):
    mock_sources(monkeypatch, prices)

    def fail_validation(*args, **kwargs):
        raise ValueError("synthetic final validation failure")

    monkeypatch.setattr(data, "load_snapshot", fail_validation)
    with pytest.raises(ValueError, match="final validation"):
        data.download_snapshot(study, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    assert not list(tmp_path.glob(".snapshot-*"))


def test_offline_loader_validates_schema_even_with_updated_hash(monkeypatch, tmp_path, study, prices):
    mock_sources(monkeypatch, prices)
    directory = tmp_path / "snapshot"
    data.download_snapshot(study, directory)
    path = directory / data.PRICE_FILE
    path.write_text(path.read_text().replace("2024-01-03", "2024-01-02"))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][data.PRICE_FILE]["sha256"] = data._file_info(path)["sha256"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unique, sorted"):
        data.load_snapshot(study, directory)


def test_seed_must_be_the_immediately_preceding_trading_close(monkeypatch, tmp_path, study, prices):
    mock_sources(monkeypatch, prices)
    directory = tmp_path / "snapshot"
    data.download_snapshot(study, directory)

    # An older price is present, but it would turn the first daily return into
    # a multi-session return without changing any dates inside the sample.
    prices.index = pd.to_datetime(["2023-12-28", "2024-01-02", "2024-01-03", "2024-01-04"])
    prices.index.name = "Date"
    path = directory / data.PRICE_FILE
    prices.to_csv(path, date_format="%Y-%m-%d", float_format="%.17g")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][data.PRICE_FILE] = data._file_info(path, prices)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=r"immediately preceding.*2023-12-29"):
        data.load_snapshot(study, directory)
    with pytest.raises(ValueError, match=r"immediately preceding.*2023-12-29"):
        data.download_snapshot(study, tmp_path / "bad-seed-snapshot")
    assert not (tmp_path / "bad-seed-snapshot").exists()
