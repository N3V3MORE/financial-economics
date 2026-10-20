"""Provider-independent, complete daily and monthly return inputs."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .analysis import _check_returns, prices_to_returns
from .data import align_sample, load_snapshot


@dataclass(frozen=True)
class ReturnData:
    daily: pd.DataFrame
    factors: pd.DataFrame
    initial_date: pd.Timestamp
    provenance: dict


def compound_months(daily: pd.DataFrame, calendar: pd.DatetimeIndex,
                    start: str, end: str) -> pd.DataFrame:
    """Require full calendar months and every expected trading day before compounding."""
    _check_returns(daily)
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if first.day != 1 or last != last + pd.offsets.MonthEnd(0):
        raise ValueError("Monthly analysis requires complete calendar months.")
    expected = calendar[(calendar >= first) & (calendar <= last)]
    if not daily.index.equals(expected):
        raise ValueError("Monthly returns require every expected trading date, with no extras.")
    periods = pd.period_range(first, last, freq="M")
    if not daily.index.to_period("M").unique().equals(periods):
        raise ValueError("A requested month is missing.")
    monthly = (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
    monthly.index = monthly.index.to_timestamp("M")
    monthly.index.name = "Date"
    return monthly


def load_returns(study, directory: Path, *, verified_monthly: bool = False) -> ReturnData:
    levels, factors, manifest = load_snapshot(study, directory)
    if manifest["schema_version"] == 2:
        from .wrds_source import RETURN_FILE, validate_monthly_extract, MONTHLY_FILE
        raw = pd.read_csv(directory / RETURN_FILE, keep_default_na=False, float_precision="round_trip")
        raw["date"] = pd.to_datetime(raw.date)
        mapping = manifest["sources"]["prices"]["identifiers"]
        # Consume CRSP returns directly; the index is only a legacy compatibility view.
        daily = raw.pivot(index="date", columns="permno", values="total_return")
        daily = daily.rename(columns={v: k for k, v in mapping.items()})[study.tickers].astype(float)
        daily.index.name, daily.columns.name = "Date", None
        if verified_monthly:
            if MONTHLY_FILE not in manifest["files"]:
                raise ValueError("Protection requires a WRDS snapshot with a monthly cross-check.")
            monthly = pd.read_csv(directory / MONTHLY_FILE, keep_default_na=False)
            validate_monthly_extract(daily, monthly, mapping, factors.index, study)
    else:
        daily = prices_to_returns(levels, study.start, study.end)
    daily, aligned = align_sample(daily, factors, study)
    initial = levels.index[levels.index < daily.index[0]][-1]
    return ReturnData(daily, aligned, initial, manifest)
