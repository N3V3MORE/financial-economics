"""Retain WRDS/FRED yields and official EIA Brent when direct FRED is unavailable.

Run with ``fecon paper-macro-sources --output-dir PATH``.
Downloads make one read-only WRDS query, retain raw evidence, and never fill
missing observations or overwrite an existing snapshot.
"""

import argparse
from datetime import datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from financial_economics.provenance import file_digest, utc_now
from financial_economics.datasets.sources.crsp import WRDSError, connect_wrds


START, END = "2000-01-01", "2020-12-31"
EIA_URL = "https://www.eia.gov/dnav/pet/hist/rbrteD.htm"
WRDS_TABLE = "frb_all.rates_daily"
WRDS_QUERY = ("SELECT date,dgs3,daaa FROM frb_all.rates_daily "
              "WHERE date BETWEEN %s AND %s ORDER BY date")


class _Tables(HTMLParser):
    """Read simple table cells without an optional HTML parser dependency."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.tables = [], []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.stack.append({"rows": [], "row": None, "cell": None})
        elif self.stack:
            table = self.stack[-1]
            if tag == "tr":
                table["row"] = []
            elif tag in ("td", "th") and table["row"] is not None:
                table["cell"] = []

    def handle_data(self, data):
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"].append(data)

    def handle_endtag(self, tag):
        if not self.stack:
            return
        table = self.stack[-1]
        if tag in ("td", "th") and table["cell"] is not None:
            table["row"].append(" ".join("".join(table["cell"]).split()))
            table["cell"] = None
        elif tag == "tr" and table["row"] is not None:
            table["rows"].append(table["row"])
            table["row"], table["cell"] = None, None
        elif tag == "table":
            self.tables.append(self.stack.pop()["rows"])


def _check_observations(series, name):
    dates = series.index
    if (series.empty or not isinstance(dates, pd.DatetimeIndex)
            or dates.hasnans or dates.tz is not None
            or not dates.equals(dates.normalize())
            or not dates.is_unique or not dates.is_monotonic_increasing
            or not np.isfinite(series.to_numpy(dtype=float)).all()
            or (series <= 0).any()):
        raise ValueError(f"{name} requires ordered, unique, positive finite daily observations.")


def _parse_brent(raw):
    """Map each daily value to its weekday; blank source cells stay unobserved."""
    html = raw.decode("utf-8-sig")
    if "Europe Brent Spot Price FOB" not in html or "Dollars per Barrel" not in html:
        raise ValueError("EIA response lacks the expected daily Brent price identity.")
    parser = _Tables()
    parser.feed(html)
    headers = ["Week Of", "Mon", "Tue", "Wed", "Thu", "Fri"]
    tables = [rows for rows in parser.tables if rows and rows[0] == headers]
    if len(tables) != 1:
        raise ValueError("EIA response must contain exactly one daily Monday-Friday history table.")
    observations = []
    pattern = (r"(\d{4})\s+([A-Za-z]{3})-\s*(\d{1,2})\s+to\s+"
               r"([A-Za-z]{3})-\s*(\d{1,2})")
    for row in tables[0][1:]:
        if not any(row):
            continue
        if len(row) != 6 or (match := re.fullmatch(pattern, row[0])) is None:
            raise ValueError("EIA response has an unexpected daily history row.")
        year, month, day, last_month, last_day = match.groups()
        monday = pd.Timestamp(datetime.strptime(f"{year}-{month}-{day}", "%Y-%b-%d"))
        friday = monday + pd.Timedelta(days=4)
        if (monday.weekday() != 0 or friday.month != datetime.strptime(last_month, "%b").month
                or friday.day != int(last_day)):
            raise ValueError("EIA history rows must label the correct Monday-Friday dates.")
        for offset, text in enumerate(row[1:]):
            if text:
                try:
                    value = float(text)
                except ValueError:
                    raise ValueError("EIA daily Brent values must be numeric or blank.") from None
                observations.append((monday + pd.Timedelta(days=offset), value))
    series = pd.Series([value for _, value in observations],
                       index=pd.DatetimeIndex([day for day, _ in observations], name="Date"),
                       name="OIL", dtype=float)
    _check_observations(series, "EIA Brent")
    return series


def _read_wrds_rates():
    connection = connect_wrds()
    try:
        try:
            with connection.cursor() as cursor:
                cursor.execute(WRDS_QUERY, (START, END))
                frame = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
        except Exception:
            raise WRDSError("WRDS_MACRO_QUERY_FAILED: check access to frb_all.rates_daily.") from None
    finally:
        connection.close()
    if list(frame) != ["date", "dgs3", "daaa"] or frame.empty:
        raise ValueError("WRDS macro response must contain date, dgs3 and daaa observations.")
    frame["date"] = pd.to_datetime(frame.date)
    dates = pd.DatetimeIndex(frame.date)
    if (dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize())
            or not dates.is_unique or not dates.is_monotonic_increasing
            or not ((dates >= START) & (dates <= END)).all()):
        raise ValueError("WRDS macro dates must be unique, ordered daily dates within 2000-2020.")
    frame[["dgs3", "daaa"]] = frame[["dgs3", "daaa"]].astype(float)
    return frame


def _covered(series, name):
    result = series.loc[START:END].dropna()
    _check_observations(result, name)
    months = result.index.to_period("M").unique()
    expected = pd.period_range(START, END, freq="M")
    if not months.equals(expected):
        raise ValueError(f"{name} must cover every month from January 2000 through December 2020.")
    return result


def download_macro_sources(destination):
    """Write one immutable macro snapshot compatible with paper_sources.load_sources."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Use a new macro-source directory; snapshots are never overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-macro-", dir=destination.parent) as temporary:
        output = Path(temporary) / "snapshot"
        output.mkdir()
        with urlopen(Request(EIA_URL, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
            raw = response.read()
        oil = _covered(_parse_brent(raw), "EIA Brent")
        rates = _read_wrds_rates()
        indexed = rates.set_index("date")
        gov = _covered(indexed.dgs3.rename("gov_yield"), "WRDS DGS3")
        corp = _covered(indexed.daaa.rename("corp_yield"), "WRDS DAAA")
        observations = pd.concat([gov, corp, oil], axis=1).sort_index().rename_axis("Date")
        observations.to_csv(output / "observations.csv", float_format="%.17g")
        rates.to_csv(output / "wrds_rates_daily.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g")
        query = {"table": WRDS_TABLE, "sql": WRDS_QUERY, "params": [START, END],
                 "columns": list(rates), "rows": len(rates), "connection": "read-only; one query; no retries"}
        (output / "wrds_query.json").write_text(json.dumps(query, indent=2) + "\n", encoding="utf-8")
        (output / "brent_daily.html").write_bytes(raw)
        sources = {}
        for name, series, symbol, original, difference in (
                ("gov_yield", gov, "dgs3", "GBUS03Y", "3-year U.S. constant-maturity government yield, percent"),
                ("corp_yield", corp, "daaa", "TRUCBYC", "seasoned AAA corporate yield; differs from AA 3-year yield")):
            sources[name] = {"provider": "WRDS / FRB FRED mirror", "table": WRDS_TABLE,
                             "symbol": symbol, "fred_series": symbol.upper(), "original": original,
                             "difference": difference, "units": "percent per year",
                             "observations": len(series), "first": str(series.index[0].date()),
                             "last": str(series.index[-1].date())}
        sources["OIL"] = {"provider": "U.S. Energy Information Administration", "symbol": "RBRTE",
                          "fred_series": "DCOILBRENTEU", "original": "CRUDOIL", "url": EIA_URL,
                          "difference": "positive Brent spot price instead of WTI; avoids negative-price logarithms",
                          "units": "USD per barrel", "observations": len(oil),
                          "first": str(oil.index[0].date()), "last": str(oil.index[-1].date())}
        metadata = {"created_at_utc": utc_now(), "sources": sources,
                    "limitations": ["Contemporary historical snapshots are not point-in-time vintages",
                                    "WRDS inputs are licensed/private; retain raw observations locally",
                                    "WRDS FRB/FRED mirror has its own update vintage; current FRED byte equivalence is unverified",
                                    "DAAA has different credit quality and maturity from TRUCBYC",
                                    "EIA Brent replaces the direct FRED transport for DCOILBRENTEU"],
                    "files": {path.name: file_digest(path) for path in output.iterdir()}}
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError("Macro-source destination appeared during download; refusing overwrite.")
        output.rename(destination)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = download_macro_sources(args.output_dir)
    for name, source in metadata["sources"].items():
        print(f"Macro source {name}: {source['observations']} observations, {source['first']} to {source['last']}")
    print(f"Saved immutable macro snapshot: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
