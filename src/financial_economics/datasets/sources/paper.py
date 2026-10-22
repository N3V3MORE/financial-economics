"""Labelled price substitutes and optional verified macro inputs for the paper.

Downloads and alignment are CPU-native. Forecast training and defensive-asset
evaluation remain separate so short ETF histories cannot truncate training.
"""

from datetime import datetime, timezone
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from financial_economics.studies.paper import load_bipower, prepare_paper_forecasts
from financial_economics.models.variance import ForecastData


# Raw Close deliberately excludes reinvested bond-ETF distributions.
YAHOO = {
    "equity_price": ("^GSPC", "S&PCOMP(PI)", "S&P 500 price index; alternate provider"),
    "equity_total": ("^SP500TR", "S&PCOMP(RI)", "S&P 500 total-return index; alternate provider"),
    "GOLD": ("GC=F", "GOLDBLN", "gold futures price instead of London spot"),
    "SILVER": ("SI=F", "GSSITOT", "silver futures price instead of commodity total return"),
    "PLATINUM": ("PL=F", "PLATFRE", "platinum futures price instead of spot"),
    "COPPER": ("HG=F", "LCPCASH", "copper futures price instead of spot"),
    "COCOA": ("CC=F", "COCINUS", "cocoa futures price instead of cash price"),
    "WHEAT": ("ZW=F", "WHEATSF", "wheat futures price instead of cash price"),
    "SOYBEAN": ("ZS=F", "SOYBEAN", "soybean futures price instead of spot"),
    "CORN": ("ZC=F", "COTSCIL", "corn futures price instead of spot"),
    "US10": ("IEF", "BMUS10Y", "7-10 year Treasury ETF price instead of 10-year clean-price index"),
    "US30": ("TLT", "BMUS30Y", "20+ year Treasury ETF price instead of 30-year clean-price index"),
    "SWISS10": ("CSBGC0.SW", "BMSW10Y", "7-15 year Swiss government ETF price; converted CHF to USD"),
    "BUND10": ("EXHD.DE", "BMBD10Y", "5.5-10.5 year German government ETF price; converted EUR to USD"),
    "usd_chf": ("CHF=X", "not specified", "CHF per USD; currency conversion assumption"),
    "eur_usd": ("EURUSD=X", "not specified", "USD per EUR; currency conversion assumption"),
}
FRED = {
    "gov_yield": ("DGS3", "GBUS03Y", "3-year U.S. constant-maturity government yield, percent"),
    "corp_yield": ("DAAA", "TRUCBYC", "seasoned AAA corporate yield; differs from AA 3-year yield"),
    "OIL": ("DCOILBRENTEU", "CRUDOIL", "positive Brent spot price instead of WTI; avoids negative-price logarithms"),
}
ASSETS = ("GOLD", "SILVER", "PLATINUM", "COPPER", "OIL", "COCOA", "WHEAT",
          "SOYBEAN", "CORN", "US10", "US30", "SWISS10", "BUND10")


def _yahoo_close(raw):
    chart = json.loads(raw)["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError("Yahoo returned an error instead of historical prices.")
    result = chart["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(
        result["meta"]["exchangeTimezoneName"]).normalize().tz_localize(None)
    series = pd.Series(result["indicators"]["quote"][0]["close"], index=dates, dtype=float)
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Yahoo dates must be unique and ordered in exchange-local time.")
    series = series.loc["2000":"2020"].dropna()
    if series.empty or not np.isfinite(series).all() or (series <= 0).any():
        raise ValueError("Public reconstruction requires positive observed price levels.")
    return series, result["meta"]


def _load_macro_snapshot(directory):
    """Verify the three unchanged economic series before fetching any prices."""
    directory = Path(directory).resolve()
    manifest_raw = (directory / "manifest.json").read_bytes()
    metadata = json.loads(manifest_raw)
    retained = {"manifest.json": manifest_raw}
    for name, digest in metadata["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or name == "manifest.json":
            raise ValueError("Macro snapshot files must use relative paths within their source directory.")
        source = (directory / name).resolve()
        if not source.is_relative_to(directory) or source == directory:
            raise ValueError("Macro snapshot files must stay within their source directory.")
        raw = source.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"Macro source hash mismatch: {name}.")
        retained[name] = raw
    if "observations.csv" not in metadata["files"]:
        raise ValueError("Macro snapshot must hash observations.csv.")
    frame = pd.read_csv(StringIO(retained["observations.csv"].decode("utf-8")),
                        index_col="Date", parse_dates=True, float_precision="round_trip")
    if (set(frame.columns) != set(FRED) or not isinstance(frame.index, pd.DatetimeIndex)
            or frame.index.hasnans or frame.index.tz is not None
            or not frame.index.equals(frame.index.normalize())
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
        raise ValueError("Macro snapshot needs three exact series and unique ordered dates.")
    for name, (symbol, _, _) in FRED.items():
        details = metadata["sources"][name]
        units = "USD per barrel" if name == "OIL" else "percent per year"
        if details.get("fred_series") != symbol or details.get("units") != units:
            raise ValueError(f"Macro series identity or units do not match {symbol}.")
        observed = frame[name].dropna()
        if (observed.empty or not np.isfinite(observed).all() or (observed <= 0).any()
                or observed.index[0] > pd.Timestamp("2000-01-10")
                or observed.index[-1] < pd.Timestamp("2020-12-24")):
            raise ValueError(f"Macro series {symbol} lacks valid 2000-2020 coverage.")
    return frame, metadata, retained


def download_sources(destination, *, macro_dir=None):
    """Save immutable raw responses, series identities, coverage and hashes."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new public-source directory; snapshots are never overwritten.")
    macro = None if macro_dir is None else _load_macro_snapshot(macro_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-sources-", dir=destination.parent) as temporary:
        output = Path(temporary) / "snapshot"
        output.mkdir()
        columns, descriptions = {}, {}
        providers = (("yahoo", YAHOO), ("fred", FRED)) if macro is None else (("yahoo", YAHOO),)
        for provider, definitions in providers:
            for name, (symbol, original, difference) in definitions.items():
                if provider == "yahoo":
                    query = urlencode({"period1": 946684800, "period2": 1609545600, "interval": "1d"})
                    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}?{query}"
                    suffix = "json"
                else:
                    # The graph CSV endpoint supports historical bounds without an API key.
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{urlencode({'id': symbol, 'cosd': '2000-01-01', 'coed': '2020-12-31'})}"
                    suffix = "csv"
                request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=60) as response:
                    raw = response.read()
                (output / f"{name}.{suffix}").write_bytes(raw)
                details = {}
                if provider == "yahoo":
                    series, details = _yahoo_close(raw)
                else:
                    frame = pd.read_csv(StringIO(raw.decode("utf-8")), na_values=".")
                    if len(frame.columns) != 2 or frame.columns[-1] != symbol:
                        raise ValueError(f"Unexpected FRED response for {symbol}.")
                    series = pd.Series(pd.to_numeric(frame[symbol], errors="raise").to_numpy(),
                                       index=pd.to_datetime(frame.iloc[:, 0])).loc["2000":"2020"].dropna()
                    if series.empty or series.index.has_duplicates or not np.isfinite(series).all():
                        raise ValueError(f"Missing or invalid historical FRED data for {symbol}.")
                columns[name] = series
                descriptions[name] = {"provider": provider, "symbol": symbol, "original": original,
                                      "difference": difference, "url": url, "observations": len(series),
                                      "first": str(series.index[0].date()), "last": str(series.index[-1].date()),
                                      "currency": details.get("currency"), "name": details.get("longName", details.get("shortName"))}
                print(f"Public source {name}: {len(series)} observations", flush=True)
        if macro is not None:
            frame, macro_metadata, retained = macro
            for name in FRED:
                columns[name] = frame[name].dropna()
                descriptions[name] = dict(macro_metadata["sources"][name])
            for name, raw in retained.items():
                copy = output / "macro_snapshot" / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(raw)
        pd.DataFrame(columns).rename_axis("Date").to_csv(output / "observations.csv")
        metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": descriptions,
                    "limitations": ["Reconstruction using substitute sources, not the original paper dataset",
                                    "Futures price changes are not investable futures total returns",
                                    "Bond ETF prices are not constant-maturity clean-price indices",
                                    "DAAA has different credit quality and maturity from TRUCBYC",
                                    "Contemporary historical data are not point-in-time vintages",
                                    "CHF/EUR bond prices converted to USD; original FX treatment unspecified"],
                    "files": {path.relative_to(output).as_posix(): sha256(path.read_bytes()).hexdigest()
                              for path in output.rglob("*") if path.is_file()}}
        if macro is not None:
            metadata["macro_snapshot_sha256"] = sha256(retained["manifest.json"]).hexdigest()
            metadata["limitations"] += macro_metadata.get("limitations", [])
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        output.rename(destination)


def load_sources(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in metadata["files"].items():
        if sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Public source hash mismatch: {name}.")
    frame = pd.read_csv(directory / "observations.csv", index_col="Date", parse_dates=True,
                        float_precision="round_trip")
    return frame, metadata


def _past_values(series, dates, name, max_age=10):
    """Carry the most recent observed level; never backfill or carry stale data."""
    observed = series.dropna()
    positions = observed.index.get_indexer(dates, method="pad")
    if (positions < 0).any():
        raise ValueError(f"{name} lacks a past observation at the requested start.")
    age = (dates - observed.index[positions]).days
    if (age > max_age).any():
        raise ValueError(f"{name} has a gap longer than {max_age} calendar days.")
    values = pd.Series(observed.to_numpy()[positions], index=dates, name=name)
    return values, int((age > 0).sum())


def prepare_sources(observations, archive, lookback=22):
    """Prepare one-step U.S. models and all 13 defensive price-return proxies."""
    calendar = observations.equity_price.dropna().index
    bv = load_bipower(archive, "2000-01-01", "2020-12-31")
    if len(bv.index.difference(calendar)):
        raise ValueError("Oxford SPX observations fall outside the public equity-price calendar.")
    univar = prepare_paper_forecasts(bv, calendar, lookback)
    feature_dates = calendar[calendar >= "2000-09-01"]
    levels, feature_carries = {}, {}
    for name in ("GOLD", "OIL", "gov_yield", "corp_yield"):
        levels[name], feature_carries[name] = _past_values(observations[name], feature_dates, name)
    features = pd.DataFrame({"bv": bv.reindex(feature_dates),
                             "gold_log": np.log(levels["GOLD"]).diff(),
                             "oil_log": np.log(levels["OIL"]).diff(),
                             "gov_change": levels["gov_yield"].diff(),
                             "corp_change": levels["corp_yield"].diff()}).loc["2002-04-11":].dropna()
    issues = features.index[lookback - 1:]
    positions = calendar.get_indexer(issues)
    usable = positions + 1 < len(calendar)
    targets = calendar[positions[usable] + 1]
    windows = np.lib.stride_tricks.sliding_window_view(features.to_numpy(), lookback, axis=0).transpose(0, 2, 1)
    multi = ForecastData(pd.DataFrame({"return_date": targets, "realized_bv": bv.reindex(targets).to_numpy()},
                                     index=issues[usable].rename("forecast_date")), windows[usable].copy())
    evaluation_dates = calendar[(calendar >= "2009-12-31") & (calendar <= "2020-12-31")]
    prices, portfolio_carries = {}, {}
    for name in ("equity_total", *ASSETS, "usd_chf", "eur_usd"):
        prices[name], portfolio_carries[name] = _past_values(observations[name], evaluation_dates, name)
    prices = pd.DataFrame(prices)
    prices["SWISS10"] /= prices.usd_chf
    prices["BUND10"] *= prices.eur_usd
    returns = prices[["equity_total", *ASSETS]].pct_change(fill_method=None).iloc[1:]
    returns = returns.rename(columns={"equity_total": "EQUITY"})
    if not np.isfinite(returns.to_numpy()).all() or (returns <= -1).any().any():
        raise ValueError("All public defensive returns must be finite and above -100%.")
    price_returns = np.log(observations.equity_price.dropna()).diff().dropna()
    return univar, multi, bv, price_returns, returns, calendar, {
        "feature_alignment_carries_since_2000_09_01": feature_carries,
        "portfolio_alignment_carries_including_seed": portfolio_carries,
        "maximum_carry_calendar_days": 10,
        "portfolio_dates": len(returns), "multivar_features": list(features),
        "calendar": "S&P 500 observed trading dates; no filled BV targets",
        "input_assumptions": "raw BV; gold/Brent log changes; percent-yield differences; train-only scaling",
        "garch_input": "S&P 500 price-index log differences; portfolio returns remain arithmetic",
        "currency": "USD; Swiss/German proxy prices converted using contemporaneous observed FX",
    }
