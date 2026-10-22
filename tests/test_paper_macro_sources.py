"""Synthetic WRDS/EIA inputs verify daily alignment and immutable provenance."""

from io import BytesIO
import json

import numpy as np
import pandas as pd
import pytest

from financial_economics.datasets.sources import macro as macro
from financial_economics.datasets.sources.paper import load_sources
from financial_economics.datasets.sources.crsp import WRDSError


def eia_html(rows, headers=("Week Of", "Mon", "Tue", "Wed", "Thu", "Fri")):
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    return ("<html><title>Europe Brent Spot Price FOB (Dollars per Barrel)</title>"
            "<table><tr>" + "".join(f"<th>{value}</th>" for value in headers)
            + "</tr>" + body + "</table></html>").encode()


def test_eia_weekdays_and_year_boundary_keep_missing_cells_unfilled():
    raw = eia_html([("2019 Dec-30 to Jan- 3", "60", "", "62", "63", "64"),
                    ("2020 Jan- 6 to Jan-10", "65", "66", "67", "", "69")])
    observed = macro._parse_brent(raw)
    assert observed.index.tolist() == pd.to_datetime([
        "2019-12-30", "2020-01-01", "2020-01-02", "2020-01-03",
        "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-10"]).tolist()
    assert observed.tolist() == [60, 62, 63, 64, 65, 66, 67, 69]
    assert b"<td></td>" in raw


@pytest.mark.parametrize("row, message", [
    (("2020 Jan- 7 to Jan-11", "1", "2", "3", "4", "5"), "Monday-Friday"),
    (("2020 Jan- 6 to Jan- 9", "1", "2", "3", "4", "5"), "Monday-Friday"),
    (("not a dated week", "1", "2", "3", "4", "5"), "unexpected daily history row"),
    (("2020 Jan- 6 to Jan-10", "nan", "2", "3", "4", "5"), "positive finite"),
    (("2020 Jan- 6 to Jan-10", "inf", "2", "3", "4", "5"), "positive finite"),
    (("2020 Jan- 6 to Jan-10", "-1", "2", "3", "4", "5"), "positive finite"),
    (("2020 Jan- 6 to Jan-10", "ND", "2", "3", "4", "5"), "numeric or blank"),
])
def test_eia_rejects_invalid_dates_and_values(row, message):
    with pytest.raises(ValueError, match=message):
        macro._parse_brent(eia_html([row]))


def test_eia_rejects_maintenance_wrong_frequency_duplicates_and_unordered_rows():
    with pytest.raises(ValueError, match="expected daily Brent price identity"):
        macro._parse_brent(b"<html><title>Site maintenance</title></html>")
    with pytest.raises(ValueError, match="Monday-Friday history table"):
        macro._parse_brent(eia_html([], headers=("Date", "Monthly price")))
    row = ("2020 Jan- 6 to Jan-10", "1", "2", "3", "4", "5")
    later = ("2020 Jan-13 to Jan-17", "6", "7", "8", "9", "10")
    for rows in ([row, row], [later, row]):
        with pytest.raises(ValueError, match="ordered, unique"):
            macro._parse_brent(eia_html(rows))


@pytest.fixture
def sources(monkeypatch):
    dates = pd.date_range(macro.START, macro.END)
    rates = pd.DataFrame({"date": dates, "dgs3": 3., "daaa": 5.})
    rates.loc[dates.weekday >= 5, ["dgs3", "daaa"]] = np.nan
    rows = []
    for monday in pd.date_range("2000-01-03", "2020-12-28", freq="W-MON"):
        friday = monday + pd.Timedelta(days=4)
        label = f"{monday:%Y %b}-{monday.day} to {friday:%b}-{friday.day}"
        rows.append((label, "60", "", "62", "63", "64"))
    raw = eia_html(rows)
    calls, connections = [], []

    class Cursor:
        description = tuple((name,) for name in rates)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params):
            calls.append((sql, params))

        def fetchall(self):
            return list(rates.itertuples(index=False, name=None))

    class Connection:
        closed = False

        def cursor(self):
            return Cursor()

        def close(self):
            self.closed = True

    def connect():
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(macro, "connect_wrds", connect)
    monkeypatch.setattr(macro, "urlopen", lambda *args, **kwargs: BytesIO(raw))
    return rates, raw, calls, connections


def test_macro_snapshot_keeps_raw_nulls_hashes_identities_and_one_read_only_query(tmp_path, sources):
    rates, raw, calls, connections = sources
    destination = tmp_path / "macro"
    metadata = macro.download_macro_sources(destination)
    observations, loaded = load_sources(destination)
    assert metadata == loaded
    assert list(observations) == ["gov_yield", "corp_yield", "OIL"]
    assert len(connections) == 1 and connections[0].closed
    assert calls == [(macro.WRDS_QUERY, (macro.START, macro.END))]
    assert set(metadata["files"]) == {"observations.csv", "wrds_rates_daily.csv", "wrds_query.json", "brent_daily.html"}
    retained = pd.read_csv(destination / "wrds_rates_daily.csv", parse_dates=["date"])
    pd.testing.assert_frame_equal(retained, rates)
    assert (destination / "brent_daily.html").read_bytes() == raw
    query = json.loads((destination / "wrds_query.json").read_text())
    assert query["table"] == "frb_all.rates_daily"
    assert query["params"] == [macro.START, macro.END]
    assert query["rows"] == len(rates)
    assert pd.isna(observations.loc["2000-01-04", "OIL"])
    assert observations.loc["2000-01-05", "OIL"] == 62
    for name, fred_series, symbol, original, units in (
            ("gov_yield", "DGS3", "dgs3", "GBUS03Y", "percent per year"),
            ("corp_yield", "DAAA", "daaa", "TRUCBYC", "percent per year"),
            ("OIL", "DCOILBRENTEU", "RBRTE", "CRUDOIL", "USD per barrel")):
        source = metadata["sources"][name]
        assert (source["fred_series"], source["symbol"], source["original"], source["units"]) == (
            fred_series, symbol, original, units)
        assert source["observations"] == observations[name].count()
    assert not list(tmp_path.glob(".paper-macro-*"))
    with (destination / "observations.csv").open("a") as output:
        output.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sources(destination)


def test_existing_snapshot_is_untouched_and_never_contacts_sources(tmp_path, sources):
    _, _, calls, connections = sources
    destination = tmp_path / "macro"
    destination.mkdir()
    sentinel = destination / "existing.txt"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError, match="never overwritten"):
        macro.download_macro_sources(destination)
    assert sentinel.read_text() == "retain"
    assert not connections and not calls


def test_bad_eia_response_removes_staging_without_attempting_wrds(tmp_path, sources, monkeypatch):
    _, _, calls, connections = sources
    monkeypatch.setattr(macro, "urlopen", lambda *args, **kwargs: BytesIO(b"maintenance"))
    destination = tmp_path / "macro"
    with pytest.raises(ValueError, match="expected daily Brent"):
        macro.download_macro_sources(destination)
    assert not destination.exists() and not calls and not connections
    assert not list(tmp_path.glob(".paper-macro-*"))


def test_wrds_query_failure_closes_connection_and_hides_driver_details(tmp_path, monkeypatch):
    class Connection:
        closed = False

        def cursor(self):
            raise RuntimeError("private_connection_detail")

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(macro, "connect_wrds", lambda: connection)
    with pytest.raises(WRDSError, match="WRDS_MACRO_QUERY_FAILED") as error:
        macro._read_wrds_rates()
    assert "private_connection_detail" not in str(error.value)
    assert connection.closed


@pytest.mark.parametrize("defect", ["truncated", "missing_month", "negative", "infinite", "duplicate", "unsorted"])
def test_incomplete_or_invalid_wrds_macro_observations_never_commit(tmp_path, sources, defect):
    rates, _, _, connections = sources
    if defect == "truncated":
        rates.loc[rates.date > "2016-10-07", "daaa"] = np.nan
    elif defect == "missing_month":
        rates.loc[rates.date.between("2005-05-01", "2005-05-31"), "daaa"] = np.nan
    elif defect == "negative":
        rates.loc[4, "dgs3"] = -1
    elif defect == "infinite":
        rates.loc[4, "dgs3"] = np.inf
    elif defect == "duplicate":
        rates.loc[4, "date"] = rates.loc[3, "date"]
    else:
        rates.loc[[3, 4], "date"] = rates.loc[[4, 3], "date"].to_numpy()
    destination = tmp_path / "macro"
    with pytest.raises(ValueError):
        macro.download_macro_sources(destination)
    assert not destination.exists()
    assert len(connections) == 1 and connections[0].closed
    assert not list(tmp_path.glob(".paper-macro-*"))


def test_cli_requires_output_and_reports_only_snapshot_summary(tmp_path, sources, capsys):
    with pytest.raises(SystemExit):
        macro.main([])
    destination = tmp_path / "macro"
    assert macro.main(["--output-dir", str(destination)]) == 0
    output = capsys.readouterr().out
    assert "Saved immutable macro snapshot:" in output
    assert "Macro source gov_yield:" in output
