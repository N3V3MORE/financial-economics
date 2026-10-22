"""Invented CRSP rows verify the optional source without accessing WRDS."""

from io import BytesIO
from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from financial_economics import cli, data, wrds_source as wrds
from financial_economics.config import Study


@pytest.fixture
def sample():
    study = Study("synthetic", "2024-01-02", "2024-01-04", "AAA",
                  {"AAA": "Stock fund", "BBB": "Bond fund"})
    dates = pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"])
    factors = pd.DataFrame({"Mkt-RF": [.01, .02, -.01, .005], "RF": .0001}, index=dates)
    factors.index.name = "Date"
    history = pd.DataFrame([
        [ticker, ticker, identifier, "2020-01-01", "9999-12-31", "2020-01-01", "9999-12-31", "FUND", "ETF"]
        for ticker, identifier in [("AAA", 1), ("BBB", 2)]
    ], columns=wrds.HISTORY_COLUMNS)
    observations = pd.DataFrame([
        [identifier, day, value, "NA"]
        for identifier, values in [(1, [.10, -.05, .02]), (2, [-.02, .03, 0])]
        for day, value in zip(dates[1:], values, strict=True)
    ], columns=wrds.RETURN_COLUMNS)
    return study, factors, history, observations


def test_total_returns_recover_exactly_without_double_counting_distributions(sample):
    *_, observations = sample
    levels, mapping = wrds.build_return_index(*sample)
    assert mapping == {"AAA": 1, "BBB": 2}
    np.testing.assert_allclose(levels.AAA, [100, 110, 104.5, 106.59])
    np.testing.assert_allclose(levels.BBB, [100, 98, 100.94, 100.94])
    expected = observations.pivot(index="date", columns="permno", values="total_return")
    np.testing.assert_allclose(levels.pct_change(fill_method=None).iloc[1:], expected)


@pytest.mark.parametrize("change,code", [
    ("missing_day", "RETURN_COVERAGE"), ("missing_fund", "RETURN_COVERAGE"),
    ("duplicate", "DUPLICATE_RETURN"), ("missing_value", "MISSING_OR_INVALID"),
    ("missing_flag", "MISSING_OR_INVALID"), ("total_loss", "MISSING_OR_INVALID"),
    ("infinite", "MISSING_OR_INVALID"), ("extra_day", "RETURN_COVERAGE"),
])
def test_incomplete_or_flagged_returns_are_never_filled(sample, change, code):
    study, factors, history, observations = sample
    if change == "missing_day":
        observations = observations.drop(index=0)
    elif change == "missing_fund":
        observations = observations.loc[observations.permno.eq(1)]
    elif change == "duplicate":
        observations = pd.concat([observations, observations.iloc[[0]]])
    elif change == "extra_day":
        observations.loc[0, "date"] = pd.Timestamp("2024-01-01")
    else:
        column = "return_flag" if change == "missing_flag" else "total_return"
        observations.loc[0, column] = {
            "missing_value": np.nan, "missing_flag": "MP", "total_loss": -1, "infinite": np.inf,
        }[change]
    with pytest.raises(wrds.WRDSError, match=code):
        wrds.build_return_index(study, factors, history, observations)


@pytest.mark.parametrize("change,code", [
    ("ambiguous", "AMBIGUOUS"), ("overlap", "IDENTITY_COVERAGE"),
    ("missing", "IDENTITY_MISSING"), ("gap", "IDENTITY_COVERAGE"),
    ("same_security", "DUPLICATE_SECURITY"),
])
def test_historical_identifiers_are_not_chosen_arbitrarily(sample, change, code):
    study, factors, history, _ = sample
    if change in ("ambiguous", "overlap"):
        extra = history.iloc[[0]].copy()
        if change == "ambiguous":
            extra["permno"] = 3
        history = pd.concat([history, extra])
    elif change == "missing":
        history = history.iloc[[0]]
    elif change == "gap":
        history.loc[0, "info_end"] = "2024-01-02"
    else:
        history.loc[1, "permno"] = 1
    with pytest.raises(wrds.WRDSError, match=code):
        wrds.resolve_identifiers(study, factors, history)


def fake_sources(monkeypatch, sample):
    study, factors, history, observations = sample
    archive = BytesIO()
    with ZipFile(archive, "w") as zipped:
        rows = ["", ",Mkt-RF,RF"]
        rows += [f"{day:%Y%m%d},{row['Mkt-RF'] * 100},{row['RF'] * 100}"
                 for day, row in factors.iterrows()]
        rows += ["", "Annual Factors"]
        zipped.writestr("factors.csv", "\n".join(rows))
    monkeypatch.setattr(wrds, "urlopen", lambda *a, **kw: BytesIO(archive.getvalue()))
    monkeypatch.setattr(wrds, "version", lambda name: "synthetic")

    class Connection:
        def close(self):
            pass

    monkeypatch.setattr(wrds, "connect_wrds", Connection)
    calls = []

    def query(connection, sql, params):
        calls.append((sql, params))
        if sql == wrds.HISTORY_SQL:
            return history.loc[history.ticker.isin(study.tickers), wrds.IDENTITY_COLUMNS].copy()
        if sql == wrds.HISTORY_BY_ID_SQL:
            return history[wrds.IDENTITY_COLUMNS].copy()
        return observations.copy()

    monkeypatch.setattr(wrds, "_query", query)
    return calls


def test_snapshot_preserves_sources_and_reloads_offline(sample, monkeypatch, tmp_path):
    calls = fake_sources(monkeypatch, sample)
    study, factors, _, _ = sample
    destination = tmp_path / "wrds"
    manifest = wrds.download_wrds_snapshot(study, destination)
    assert calls[0][1]["tickers"] == study.tickers
    assert calls[2][1]["permnos"] == [1, 2]
    assert "%(permnos)s" in calls[2][0] and "ANY" in calls[2][0]
    assert set(manifest["files"]) == wrds.SNAPSHOT_FILES
    assert not (destination / data.PRICE_FILE).exists()
    monkeypatch.setattr(wrds, "connect_wrds", lambda: pytest.fail("Offline loading contacted WRDS"))
    loaded, loaded_factors, loaded_manifest = data.load_snapshot(study, destination)
    expected, _ = wrds.build_return_index(*sample)
    pd.testing.assert_frame_equal(loaded, expected)
    pd.testing.assert_frame_equal(loaded_factors, factors)
    assert loaded_manifest == manifest
    from financial_economics import return_data
    monkeypatch.setattr(return_data, "prices_to_returns", lambda *a: pytest.fail("CRSP must be consumed directly"))
    direct = return_data.load_returns(study, destination)
    expected_raw = sample[3].pivot(index="date", columns="permno", values="total_return")
    np.testing.assert_array_equal(direct.daily, expected_raw)
    assert direct.initial_date == pd.Timestamp("2023-12-29")
    assert "WRDS / CRSP" in data.price_source_description(manifest)
    assert "not traded prices" in data.price_source_description(manifest)

    # Replacing both index data and its fingerprint must still fail reconstruction.
    loaded.iloc[-1, 0] += 1
    loaded.to_csv(destination / wrds.INDEX_FILE)
    manifest["files"][wrds.INDEX_FILE] = data._file_info(destination / wrds.INDEX_FILE, loaded)
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(wrds.WRDSError, match="RECONSTRUCTION"):
        data.load_snapshot(study, destination)


def test_existing_destination_is_rejected_before_network(sample, monkeypatch, tmp_path):
    monkeypatch.setattr(wrds, "urlopen", lambda *a, **kw: pytest.fail("Unexpected download"))
    with pytest.raises(FileExistsError):
        wrds.download_wrds_snapshot(sample[0], tmp_path)


def test_verified_monthly_snapshot_roundtrip(sample, monkeypatch, tmp_path):
    study, _, history, _ = sample
    study = replace(study, start="2024-01-01", end="2024-01-31")
    dates = pd.bdate_range("2024-01-01", "2024-01-31")
    factors = pd.DataFrame({"Mkt-RF": .001, "RF": .0001},
                           index=pd.DatetimeIndex([pd.Timestamp("2023-12-29"), *dates], name="Date"))
    observations = pd.DataFrame([[identifier, day, .001*identifier, "NA"]
                                 for identifier in [1, 2] for day in dates], columns=wrds.RETURN_COLUMNS)
    fake_sources(monkeypatch, (study, factors, history, observations))
    original_query = wrds._query
    monthly = pd.DataFrame([[identifier, dates[-1], (1+.001*identifier)**len(dates)-1, "CR"]
                            for identifier in [1, 2]], columns=wrds.RETURN_COLUMNS)
    monkeypatch.setattr(wrds, "_query", lambda connection, sql, params:
                        monthly.copy() if sql == wrds.MONTHLY_SQL else original_query(connection, sql, params))
    destination = tmp_path / "verified"
    manifest = wrds.download_wrds_snapshot(study, destination, verify_monthly=True)
    assert manifest["coverage"]["fund_months"] == 2
    assert set(manifest["files"]) == wrds.SNAPSHOT_FILES | {wrds.MONTHLY_FILE, wrds.QUERY_FILE}
    from financial_economics.return_data import load_returns
    loaded = load_returns(study, destination, verified_monthly=True)
    assert loaded.daily.index.equals(dates)
    # Even updating the file fingerprint cannot bypass the reconciliation itself.
    monthly.loc[0, "total_return"] += .01
    monthly.to_csv(destination / wrds.MONTHLY_FILE, index=False)
    manifest["files"][wrds.MONTHLY_FILE] = data._file_info(destination / wrds.MONTHLY_FILE)
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(wrds.WRDSError, match="RECONCILIATION"):
        load_returns(study, destination, verified_monthly=True)


def test_former_ticker_is_retained_under_the_same_security(sample, monkeypatch, tmp_path):
    study, factors, history, observations = sample
    former = history.iloc[[0]].copy()
    former["ticker"] = "OLD"
    former["info_end"] = "2023-12-31"
    history.loc[0, "info_start"] = "2024-01-01"
    history = pd.concat([former, history], ignore_index=True)
    fake_sources(monkeypatch, (study, factors, history, observations))
    destination = tmp_path / "wrds"
    wrds.download_wrds_snapshot(study, destination)
    retained = pd.read_csv(destination / wrds.HISTORY_FILE)
    assert set(retained.loc[retained.requested_ticker.eq("AAA"), "ticker"]) == {"OLD", "AAA"}
    levels, _, _ = data.load_snapshot(study, destination)
    assert levels.AAA.iloc[-1] == pytest.approx(106.59)


def test_failed_coverage_leaves_no_snapshot(sample, monkeypatch, tmp_path):
    study, factors, history, observations = sample
    fake_sources(monkeypatch, (study, factors, history, observations.iloc[:-1]))
    with pytest.raises(wrds.WRDSError, match="COVERAGE"):
        wrds.download_wrds_snapshot(study, tmp_path / "wrds")
    assert not (tmp_path / "wrds").exists()
    assert not list(tmp_path.glob(".wrds-*"))


def test_saved_login_is_read_without_exposing_or_rewriting_password(monkeypatch, tmp_path):
    pgpass = tmp_path / "pgpass"
    content = f"{wrds.HOST}:{wrds.PORT}:wrds:synthetic_user:secret\\:with\\\\escapes\n"
    pgpass.write_text(content, encoding="utf-8")
    monkeypatch.setenv("PGPASSFILE", str(pgpass))
    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    assert wrds._saved_username() == "synthetic_user"
    assert pgpass.read_text(encoding="utf-8") == content
    pgpass.write_text(content + f"{wrds.HOST}:{wrds.PORT}:wrds:second_user:another\n")
    with pytest.raises(wrds.WRDSError, match="SAVED_LOGIN_REQUIRED") as error:
        wrds._saved_username()
    assert "secret" not in str(error.value) and "synthetic_user" not in str(error.value)


def test_query_errors_never_expose_provider_message():
    class Connection:
        def cursor(self):
            raise RuntimeError("secret credentials and licensed observations")

    with pytest.raises(wrds.WRDSError, match="QUERY_FAILED") as error:
        wrds._query(Connection(), "SELECT", {})
    assert "secret" not in str(error.value) and error.value.__suppress_context__


def test_cli_selects_wrds_without_yahoo(sample, monkeypatch, tmp_path):
    config = tmp_path / "study.toml"
    config.write_text('[study]\nname="test"\nstart="2024-01-02"\nend="2024-01-04"\n'
                      'benchmark="AAA"\n[assets]\nAAA="Stocks"\n')
    calls = []
    monkeypatch.setattr(wrds, "download_wrds_snapshot", lambda study, path, verify_monthly: calls.append(path))
    monkeypatch.setattr(cli, "download_snapshot", lambda *a: pytest.fail("Unexpected Yahoo fallback"))
    assert cli.main(["--config", str(config), "download", "--source", "wrds",
                     "--data-dir", str(tmp_path / "wrds")]) == 0
    assert calls == [tmp_path / "wrds"]


def test_phase1_records_and_describes_the_actual_provider(sample, tmp_path):
    from financial_economics.report import write_outputs

    study, factors, _, _ = sample
    levels, _ = wrds.build_return_index(*sample)
    returns = levels.pct_change(fill_method=None).iloc[1:]
    manifest = {"files": {}, "sources": {"prices": {
        "provider": "WRDS / CRSP CIZ", "representation": "total_return_index"}}}
    write_outputs(replace(study, hac_lags=0), returns, factors.loc[returns.index],
                  manifest, tmp_path)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert "WRDS / CRSP CIZ" in summary and "not traded prices" in summary
    assert "Yahoo" not in summary
    assert run["sources"] == manifest["sources"]
