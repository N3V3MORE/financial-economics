"""Official dated FX for foreign bond valuation, never a forecasting feature.

H.10 observations are noon New York quotes published in a weekly release.
Their date identifies the valuation, not same-day public data availability.
"""

from datetime import datetime
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .paper_sources import _past_values
from .provenance import file_digest, utc_now


FX_SOURCES = {"usd_chf": ("sz", "Swiss francs per U.S. dollar"),
              "eur_usd": ("eu", "U.S. dollars per EU euro")}


def _parse_rates(raw):
    pairs = re.findall(r'<tr>\s*<th[^>]*>\s*(\d{1,2}-[A-Z]{3}-\d{2})\s*</th>\s*'
                       r'<td[^>]*>\s*(ND|\d+(?:\.\d+)?)\s*</td>\s*</tr>', raw.decode(), re.I)
    observed = [(datetime.strptime(date, "%d-%b-%y"), float(value))
                for date, value in pairs if value.upper() != "ND"]
    series = pd.Series(dict(observed), dtype=float)
    if (not observed or len(series) != len(observed) or not series.index.is_monotonic_increasing
            or not np.isfinite(series).all() or (series <= 0).any()):
        raise ValueError("Federal Reserve FX needs ordered, unique, positive dated observations.")
    return series.loc["2009-12-31":"2020-12-31"]


def download_fx(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Use a new FX snapshot; existing inputs are never overwritten.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-fx-", dir=destination.parent) as temporary:
        output = Path(temporary) / "snapshot"
        output.mkdir()
        columns, sources = {}, {}
        for name, (code, units) in FX_SOURCES.items():
            url = f"https://www.federalreserve.gov/releases/h10/Hist/dat00_{code}.htm"
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as response:
                raw = response.read()
            (output / f"{name}.html").write_bytes(raw)
            columns[name] = _parse_rates(raw)
            if columns[name].empty:
                raise ValueError("Federal Reserve response lacks the historical test period.")
            sources[name] = {"provider": "Federal Reserve H.10", "url": url, "units": units,
                             "original": "original currency treatment unspecified",
                             "difference": "noon New York FX used for USD valuation; replaces date-misaligned Yahoo FX",
                             "observations": len(columns[name]), "first": str(columns[name].index[0].date()),
                             "last": str(columns[name].index[-1].date())}
        pd.DataFrame(columns).rename_axis("Date").to_csv(output / "observations.csv")
        metadata = {"created_at_utc": utc_now(), "sources": sources,
                    "limitations": ["Noon FX rather than close FX; original currency convention unspecified",
                                    "Weekly publication: used only for historical valuation, never model features"],
                    "files": {path.name: file_digest(path) for path in output.iterdir()}}
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        output.rename(destination)


def _valuation_dates(observations, returns):
    calendar = observations.equity_price.dropna().index
    first, last = calendar.get_indexer([returns.index[0], returns.index[-1]])
    if first < 1 or last < first:
        raise ValueError("FX valuation needs an observed equity seed close and evaluation dates.")
    dates = calendar[first - 1:last + 1]
    if not dates[1:].equals(returns.index):
        raise ValueError("FX valuation must retain the complete equity return calendar.")
    return dates


def currency_returns(observations, returns, fx_rates):
    """Replace only the two foreign bond returns using the dated noon quotes."""
    dates = _valuation_dates(observations, returns)
    result, carries = returns.copy(), {}
    for asset, name in (("SWISS10", "usd_chf"), ("BUND10", "eur_usd")):
        observed = fx_rates[name].dropna()
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError("FX quotes must be finite and positive.")
        price, _ = _past_values(observations[asset], dates, asset)
        rate, carries[name] = _past_values(observed, dates, name)
        dollar_price = price / rate if asset == "SWISS10" else price * rate
        result[asset] = dollar_price.pct_change(fill_method=None).iloc[1:]
    return result, {"currency": "USD using official dated Federal Reserve noon NY FX; valuation only",
                    "fx_carries_including_seed": carries,
                    "reason": "Yahoo FX moved known Swiss currency shocks to the following date; primary quotes restore the correct valuation dates"}


def local_currency_returns(observations, returns):
    """Post hoc: value the two foreign bond proxies in CHF and EUR, with no FX.

    The paper's printed volatilities and correlations for these bonds match
    local-currency returns, not USD-converted ones.
    """
    dates = _valuation_dates(observations, returns)
    result, carries = returns.copy(), {}
    for asset in ("SWISS10", "BUND10"):
        price, carries[asset] = _past_values(observations[asset], dates, asset)
        result[asset] = price.pct_change(fill_method=None).iloc[1:]
    return result, {"currency": "SWISS10 in CHF and BUND10 in EUR; all other assets in USD",
                    "local_price_carries_including_seed": carries,
                    "reason": "post hoc: the paper's Figure 6 correlations and Table A.2 volatilities match local-currency, not USD, bond returns"}
