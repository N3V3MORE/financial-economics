"""Download once, retain the source evidence, and analyse the same data offline."""

import csv
from datetime import date, datetime, timedelta
from importlib.metadata import version
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from financial_economics.settings.study import Study
from financial_economics.provenance import file_digest, utc_now


FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
SOURCE_ZIP = "french_factors_daily.zip"
PRICE_FILE = "adjusted_close.csv"
FACTOR_FILE = "factors.csv"
FACTOR_COLUMNS = ["Mkt-RF", "RF"]


def _validate_frame(
    frame: pd.DataFrame, columns: list[str], label: str, *, positive: bool = False
) -> None:
    if frame.empty or list(frame.columns) != columns:
        raise ValueError(f"{label} must contain nonempty columns {columns} in that order.")
    dates = frame.index
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError(f"{label} requires calendar dates as its index.")
    if (
        dates.hasnans
        or dates.tz is not None
        or not dates.equals(dates.normalize())
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(f"{label} dates must be unique, sorted, and timezone-free daily dates.")
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric values.") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains missing or nonfinite values; no filling is allowed.")
    if positive and (values <= 0).any():
        raise ValueError(f"{label} prices must be strictly positive.")


def parse_french_factors(archive: bytes) -> pd.DataFrame:
    """Read only the daily table; French returns are percentages, converted once."""
    try:
        with ZipFile(BytesIO(archive)) as source:
            members = [name for name in source.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError("French archive must contain exactly one CSV file.")
            content = source.read(members[0]).decode("utf-8-sig")
    except (BadZipFile, UnicodeDecodeError) as error:
        raise ValueError("French source is not a readable CSV ZIP archive.") from error

    rows = iter(csv.reader(StringIO(content)))
    for row in rows:
        header = [field.strip() for field in row]
        if header and header[0] == "" and all(name in header for name in FACTOR_COLUMNS):
            break
    else:
        raise ValueError("French daily factor header was not found.")

    observations = []
    for row in rows:
        stamp = row[0].strip() if row else ""
        if not stamp or not stamp.isdigit() or len(stamp) == 4:
            break  # The daily table ends before notes or annual summaries.
        if len(stamp) != 8 or len(row) != len(header):
            raise ValueError(f"Malformed French daily observation: {stamp!r}.")
        try:
            day = datetime.strptime(stamp, "%Y%m%d")
            values = [float(row[header.index(name)]) for name in FACTOR_COLUMNS]
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid French observation dated {stamp!r}.") from error
        if any(value in (-99.99, -999) for value in values):
            raise ValueError(f"French source reports a missing value on {stamp}.")
        observations.append((day, *values))

    if not observations:
        raise ValueError("French source contains no daily observations.")
    factors = pd.DataFrame(observations, columns=["Date", *FACTOR_COLUMNS]).set_index("Date")
    factors = factors / 100.0
    _validate_frame(factors, FACTOR_COLUMNS, "Factors")
    return factors


def align_sample(
    returns: pd.DataFrame, factors: pd.DataFrame, study: Study
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require exact trading-date agreement instead of silently dropping dates."""
    _validate_frame(returns, study.tickers, "Returns")
    _validate_frame(factors, FACTOR_COLUMNS, "Factors")
    if (returns <= -1).any().any():
        raise ValueError("Simple ETF returns must be greater than -100%.")
    if factors.index[-1] < pd.Timestamp(study.end):
        raise ValueError("Study end exceeds the published French daily factor history.")
    sample = returns.loc[study.start : study.end]
    matched = factors.loc[study.start : study.end]
    if matched.empty:
        raise ValueError("There are no French trading dates in the requested sample.")
    missing = matched.index.difference(sample.index)
    extra = sample.index.difference(matched.index)
    if len(missing) or len(extra):
        missing_dates = ", ".join(missing[:5].strftime("%Y-%m-%d")) or "none"
        unexpected_dates = ", ".join(extra[:5].strftime("%Y-%m-%d")) or "none"
        raise ValueError(
            "ETF and French trading dates differ: "
            f"missing ETF dates ({len(missing)}): {missing_dates}; "
            f"unexpected ETF dates ({len(extra)}): {unexpected_dates}. No dates were dropped."
        )
    return sample.copy(), matched.copy()


def _file_info(path: Path, frame: pd.DataFrame | None = None) -> dict:
    info = {"sha256": file_digest(path)}
    if frame is not None:
        info.update(
            rows=len(frame),
            start=frame.index[0].strftime("%Y-%m-%d"),
            end=frame.index[-1].strftime("%Y-%m-%d"),
        )
    return info


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, index_col="Date", float_precision="round_trip")
        stamps = frame.index.astype(str)
        if not stamps.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError("Dates must use YYYY-MM-DD.")
        frame.index = pd.to_datetime(stamps, format="%Y-%m-%d", errors="raise")
        frame.index.name = "Date"
        return frame.astype(float)
    except (ValueError, KeyError, pd.errors.ParserError) as error:
        raise ValueError(f"Invalid snapshot table {path.name}: {error}") from error


def load_snapshot(
    study: Study, directory: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read retained files only, checking configuration, hashes, and daily values."""
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read snapshot manifest in {directory}.") from error
    schema = manifest.get("schema_version")
    if schema not in (1, 2) or manifest.get("settings") != study.settings():
        raise ValueError("Snapshot settings do not match this study; use the original settings or fetch a new snapshot.")
    if schema == 2:
        from financial_economics.datasets.sources.crsp import INDEX_FILE, SNAPSHOT_FILES, MONTHLY_FILE, QUERY_FILE, validate_snapshot_returns

        if (manifest.get("sources", {}).get("prices", {}).get("provider") != "WRDS / CRSP CIZ"
                or manifest["sources"]["prices"].get("representation") != "total_return_index"):
            raise ValueError("Invalid WRDS snapshot source definition.")
        price_file, required_files = INDEX_FILE, SNAPSHOT_FILES
        if "coverage" in manifest:
            required_files = SNAPSHOT_FILES | {MONTHLY_FILE, QUERY_FILE}
    else:
        price_file, required_files = PRICE_FILE, {PRICE_FILE, FACTOR_FILE, SOURCE_ZIP}
    files = manifest.get("files", {})
    if set(files) != required_files:
        raise ValueError("Snapshot manifest does not list the required data files.")
    for name, info in files.items():
        path = directory / name
        if not path.is_file() or _file_info(path)["sha256"] != info.get("sha256"):
            raise ValueError(f"Snapshot hash verification failed for {name}.")

    prices = _read_frame(directory / price_file)
    factors = _read_frame(directory / FACTOR_FILE)
    _validate_frame(prices, study.tickers, "Adjusted close", positive=True)
    _validate_frame(factors, FACTOR_COLUMNS, "Factors")
    if prices.index[0] >= pd.Timestamp(study.start):
        raise ValueError("Adjusted close history needs a pre-start price to calculate the first return.")
    sample_days = factors.loc[study.start : study.end].index
    if len(sample_days):
        previous_days = factors.index[factors.index < sample_days[0]]
        seed_days = prices.index[prices.index < sample_days[0]]
        if not len(previous_days):
            raise ValueError("French history must include the trading day before the sample.")
        if not len(seed_days) or seed_days[-1] != previous_days[-1]:
            expected = previous_days[-1].strftime("%Y-%m-%d")
            raise ValueError(f"The seed must be the immediately preceding French trading close: {expected}.")
    for name, frame in ((price_file, prices), (FACTOR_FILE, factors)):
        if _file_info(directory / name, frame) != files[name]:
            raise ValueError(f"Snapshot row count or date bounds disagree for {name}.")
    align_sample(prices.pct_change(fill_method=None).iloc[1:], factors, study)
    if schema == 2:
        validate_snapshot_returns(study, factors, directory, prices, manifest)
    return prices, factors, manifest


def price_source_description(manifest: dict) -> str:
    """Describe the consumed provider instead of hard-coding a Yahoo label."""
    source = manifest.get("sources", {}).get("prices", {})
    provider = source.get("provider", "Unspecified saved provider")
    if source.get("representation") == "total_return_index":
        return f"{provider}; total-return index reconstructed from daily total returns, not traded prices"
    return f"{provider}; adjusted closes"


def download_snapshot(study: Study, directory: Path) -> dict:
    """Save a new validated snapshot atomically; never replace an existing one."""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError(f"Snapshot destination already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    seed_start = (date.fromisoformat(study.start) - timedelta(days=7)).isoformat()
    exclusive_end = (date.fromisoformat(study.end) + timedelta(days=1)).isoformat()

    import yfinance as yf

    yf.set_tz_cache_location(str(directory.parent / ".yfinance-cache"))
    try:
        raw = yf.download(
            study.tickers,
            start=seed_start,
            end=exclusive_end,
            auto_adjust=False,
            keepna=True,
            ignore_tz=True,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise RuntimeError("Yahoo Finance download failed; no snapshot was saved.") from error
    if isinstance(raw.columns, pd.MultiIndex) and "Adj Close" in raw.columns.get_level_values(0):
        prices = raw["Adj Close"].copy()
    elif len(study.tickers) == 1 and "Adj Close" in raw.columns:
        prices = raw[["Adj Close"]].rename(columns={"Adj Close": study.tickers[0]})
    else:
        raise ValueError("Yahoo Finance did not supply the required adjusted-close series.")
    if set(prices.columns) != set(study.tickers):
        raise ValueError("Yahoo Finance did not return every requested ticker.")
    prices = prices[study.tickers].sort_index()
    prices.index.name = "Date"
    prices.columns.name = None
    _validate_frame(prices, study.tickers, "Adjusted close", positive=True)

    try:
        request = Request(FRENCH_URL, headers={"User-Agent": "financial-economics/0.1"})
        with urlopen(request, timeout=60) as response:
            archive = response.read()
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError("French factor download failed; no snapshot was saved.") from error
    factors = parse_french_factors(archive)
    align_sample(prices.pct_change(fill_method=None).iloc[1:], factors, study)

    # TemporaryDirectory owns this new sibling only; failure removes no user files.
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        prices.to_csv(staging / PRICE_FILE, date_format="%Y-%m-%d", float_format="%.17g")
        factors.to_csv(staging / FACTOR_FILE, date_format="%Y-%m-%d", float_format="%.17g")
        (staging / SOURCE_ZIP).write_bytes(archive)
        manifest = {
            "schema_version": 1,
            "retrieved_at_utc": utc_now(),
            "settings": study.settings(),
            "sources": {
                "prices": {
                    "provider": "Yahoo Finance via yfinance",
                    "url": "https://finance.yahoo.com/",
                    "field": "Adj Close",
                    "auto_adjust": False,
                    "start": seed_start,
                    "end_exclusive": exclusive_end,
                },
                "factors": {"url": FRENCH_URL, "units": "decimal returns"},
            },
            "versions": {name: version(name) for name in ("yfinance", "pandas", "numpy")},
            "files": {
                PRICE_FILE: _file_info(staging / PRICE_FILE, prices),
                FACTOR_FILE: _file_info(staging / FACTOR_FILE, factors),
                SOURCE_ZIP: _file_info(staging / SOURCE_ZIP),
            },
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        load_snapshot(study, staging)
        if directory.exists():
            raise FileExistsError(f"Snapshot destination already exists: {directory}")
        staging.rename(directory)
    return manifest
