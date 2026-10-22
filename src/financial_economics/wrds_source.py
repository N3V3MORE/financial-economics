"""Bounded CRSP extracts; licensed observations stay in local snapshot files."""

from contextlib import closing
from datetime import date, datetime, UTC
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import (FACTOR_FILE, FRENCH_URL, SOURCE_ZIP, _file_info,
                   _validate_frame, load_snapshot, parse_french_factors)


HOST, PORT = "wrds-pgdata.wharton.upenn.edu", 9737
INDEX_FILE = "total_return_index.csv"
RETURN_FILE = "crsp_daily_returns.csv"
HISTORY_FILE = "crsp_identifier_history.csv"
SNAPSHOT_FILES = {INDEX_FILE, RETURN_FILE, HISTORY_FILE, FACTOR_FILE, SOURCE_ZIP}
MONTHLY_FILE = "crsp_monthly_returns.csv"
QUERY_FILE = "queries.json"
IDENTITY_COLUMNS = ["ticker", "permno", "info_start", "info_end", "security_start",
                    "security_end", "security_type", "security_subtype"]
HISTORY_COLUMNS = ["requested_ticker", *IDENTITY_COLUMNS]
RETURN_COLUMNS = ["permno", "date", "total_return", "return_flag"]

HISTORY_SQL = """
SELECT ticker, permno,
       to_char(secinfostartdt, 'YYYY-MM-DD') AS info_start,
       to_char(secinfoenddt, 'YYYY-MM-DD') AS info_end,
       to_char(securitybegdt, 'YYYY-MM-DD') AS security_start,
       to_char(securityenddt, 'YYYY-MM-DD') AS security_end,
       securitytype AS security_type, securitysubtype AS security_subtype
FROM crsp.stksecurityinfohist
WHERE ticker = ANY(%(tickers)s)
  AND secinfostartdt <= %(end)s AND secinfoenddt >= %(start)s
ORDER BY ticker, permno, secinfostartdt
"""
HISTORY_BY_ID_SQL = HISTORY_SQL.replace("ticker = ANY(%(tickers)s)", "permno = ANY(%(permnos)s)")
RETURN_SQL = """
SELECT permno, dlycaldt AS date, dlyret AS total_return,
       dlyretmissflg AS return_flag
FROM crsp.dsf_v2
WHERE permno = ANY(%(permnos)s) AND dlycaldt BETWEEN %(start)s AND %(end)s
ORDER BY permno, dlycaldt
"""


class WRDSError(RuntimeError):
    """Fixed error codes prevent provider values or credentials reaching logs."""


MONTHLY_SQL = """
SELECT permno, mthcaldt AS date, mthret AS total_return, mthretflg AS return_flag
FROM crsp.msf_v2
WHERE permno = ANY(%(permnos)s) AND mthcaldt BETWEEN %(start)s AND %(end)s
ORDER BY permno, mthcaldt
"""


def validate_monthly_extract(daily, monthly, mapping, calendar, study):
    """Reject incomplete or inconsistent monthly CRSP data; never repair it silently."""
    from .return_data import compound_months
    try:
        expected = compound_months(daily, calendar, study.start, study.end)
        if list(monthly.columns) != RETURN_COLUMNS:
            raise ValueError()
        rows = monthly.copy()
        # crsp.metaflaginfo: AR / CR = full previous-close-to-current-close
        # compounded daily return with no missing returns (verified 2026-10-20).
        if not rows.return_flag.eq("CR").all():
            raise WRDSError("WRDS_MONTHLY_RETURN_FLAG_INVALID")
        rows["date"] = pd.to_datetime(rows.date).dt.to_period("M").dt.to_timestamp("M")
        if rows.duplicated(["permno", "date"]).any():
            raise ValueError()
        actual = rows.pivot(index="date", columns="permno", values="total_return")
        actual = actual.rename(columns={v: k for k, v in mapping.items()})
        if set(actual.columns) != set(expected.columns) or not actual.index.equals(expected.index):
            raise ValueError()
        difference = np.abs(actual[expected.columns].to_numpy(dtype=float) - expected.to_numpy())
        if not np.isfinite(difference).all() or (difference > .0001).any():
            raise WRDSError("WRDS_MONTHLY_RECONCILIATION_FAILED: investigate differences above one basis point.")
        return {"status": "passed", "fund_months": int(difference.size),
                "tolerance": .0001, "maximum_absolute_difference": float(difference.max()),
                "monthly_return_flags": sorted(rows.return_flag.astype(str).unique().tolist())}
    except WRDSError:
        raise
    except (ValueError, TypeError, KeyError):
        raise WRDSError("WRDS_MONTHLY_COVERAGE_FAILED") from None


def _saved_username():
    if os.environ.get("WRDS_USERNAME"):
        return os.environ["WRDS_USERNAME"]
    default = (Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
               / "postgresql" / "pgpass.conf") if os.name == "nt" else Path.home() / ".pgpass"
    path = Path(os.environ.get("PGPASSFILE", default))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        raise WRDSError("WRDS_SAVED_LOGIN_REQUIRED: on Windows run python -m "
                        "financial_economics.wrds_credentials; otherwise configure pgpass and WRDS_USERNAME.") from None
    users = set()
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields, current, escaped = [], [], False
        for char in line:
            if escaped:
                current.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == ":" and len(fields) < 4:
                fields.append("".join(current))
                current = []
            else:
                current.append(char)
        fields.append("".join(current))
        if (len(fields) == 5 and fields[0] in (HOST, "*")
                and fields[1] in (str(PORT), "*") and fields[2] in ("wrds", "*")
                and fields[3] not in ("", "*")):
            users.add(fields[3])
    if len(users) != 1:
        raise WRDSError("WRDS_SAVED_LOGIN_REQUIRED: set WRDS_USERNAME for one saved login.")
    return users.pop()


def connect_wrds():
    """One read-only connection using Windows Credential Manager or libpq's password file."""
    try:
        import psycopg2
    except ImportError:
        raise WRDSError("WRDS_DRIVER_REQUIRED: run uv sync --extra wrds --frozen.") from None
    from .wrds_credentials import CredentialError, get_saved_login

    try:
        login = get_saved_login()
    except CredentialError:
        raise WRDSError("WRDS_SAVED_LOGIN_FAILED: Windows Credential Manager could not read the WRDS login.") from None
    if login is None:
        credentials = {"user": _saved_username()}
    else:
        username, password = login
        if os.environ.get("WRDS_USERNAME", username) != username:
            raise WRDSError("WRDS_SAVED_LOGIN_MISMATCH: WRDS_USERNAME differs from the saved Windows login.")
        credentials = {"user": username, "password": password}
    connection = None
    try:
        connection = psycopg2.connect(
            host=HOST, port=PORT, dbname="wrds", sslmode="require",
            # Leave time for the WRDS Duo Mobile push before authentication times out.
            connect_timeout=120, application_name="financial-economics",
            options="-c statement_timeout=60000", **credentials,
        )
        connection.set_session(readonly=True, isolation_level="REPEATABLE READ")
        return connection
    except Exception:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        raise WRDSError("WRDS_CONNECTION_FAILED: check saved login, network, and Duo status.") from None


def _query(connection, sql, params):
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
    except Exception:
        raise WRDSError("WRDS_QUERY_FAILED: check CRSP access and required table columns.") from None


def _calendar(study, factors):
    expected = factors.loc[study.start:study.end].index
    if expected.empty or factors.index[-1] < pd.Timestamp(study.end):
        raise WRDSError("WRDS_FACTOR_COVERAGE_INCOMPLETE")
    earlier = factors.index[factors.index < expected[0]]
    if earlier.empty:
        raise WRDSError("WRDS_SEED_DATE_MISSING")
    return expected, earlier[-1]


def resolve_identifiers(study, factors, history):
    """Require one distinct security per fund, with unambiguous dated history."""
    expected, seed = _calendar(study, factors)
    if list(history.columns) != HISTORY_COLUMNS or history.empty:
        raise WRDSError("WRDS_IDENTITY_MISSING")
    if not history.requested_ticker.isin(study.tickers).all():
        raise WRDSError("WRDS_UNEXPECTED_IDENTITY")
    try:
        for column in ("info_start", "info_end", "security_start", "security_end"):
            if not history[column].map(lambda value: date.fromisoformat(value).isoformat() == value).all():
                raise ValueError()
        identifiers = pd.to_numeric(history.permno, errors="raise")
        if (identifiers.isna().any() or not np.isfinite(identifiers).all()
                or (identifiers <= 0).any() or (identifiers % 1 != 0).any()):
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise WRDSError("WRDS_INVALID_IDENTITY") from None
    mapping = {}
    days = np.array([str(seed.date()), *expected.strftime("%Y-%m-%d")])
    for ticker in study.tickers:
        rows = history.loc[history.requested_ticker.eq(ticker)]
        if rows.empty:
            raise WRDSError("WRDS_IDENTITY_MISSING")
        if rows.permno.nunique() != 1:
            raise WRDSError("WRDS_IDENTITY_AMBIGUOUS")
        if not (rows.security_type.eq("FUND") & rows.security_subtype.eq("ETF")).all():
            raise WRDSError("WRDS_IDENTITY_NOT_ETF")
        # Do not apply the other project's common-stock filter: ETFs are funds.
        coverage = np.zeros(len(days), dtype=int)
        for row in rows.itertuples():
            coverage += ((days >= row.info_start) & (days <= row.info_end)
                         & (days >= row.security_start) & (days <= row.security_end))
        if (coverage != 1).any():
            raise WRDSError("WRDS_IDENTITY_COVERAGE_INCOMPLETE")
        mapping[ticker] = int(rows.permno.iloc[0])
    if len(set(mapping.values())) != len(mapping):
        raise WRDSError("WRDS_DUPLICATE_SECURITY")
    return mapping


def build_return_index(study, factors, history, observations):
    """Normalize CRSP total returns to 100; these levels are not traded prices."""
    mapping = resolve_identifiers(study, factors, history)
    expected, seed = _calendar(study, factors)
    if list(observations.columns) != RETURN_COLUMNS or observations.empty:
        raise WRDSError("WRDS_RETURN_COVERAGE_INCOMPLETE")
    frame = observations.copy()
    try:
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        frame["total_return"] = pd.to_numeric(frame.total_return, errors="raise").astype(float)
        if (frame.date.isna().any() or frame.date.dt.tz is not None
                or not frame.date.equals(frame.date.dt.normalize())
                or not frame.permno.isin(mapping.values()).all()):
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise WRDSError("WRDS_INVALID_RETURN_SCHEMA") from None
    if frame.duplicated(["permno", "date"]).any():
        raise WRDSError("WRDS_DUPLICATE_RETURN")
    if (not frame.return_flag.eq("NA").all()
            or not np.isfinite(frame.total_return).all() or frame.total_return.le(-1).any()):
        raise WRDSError("WRDS_MISSING_OR_INVALID_RETURN")
    returns = frame.pivot(index="date", columns="permno", values="total_return")
    if (set(returns.columns) != set(mapping.values()) or not returns.index.equals(expected)
            or returns.isna().any().any()):
        raise WRDSError("WRDS_RETURN_COVERAGE_INCOMPLETE")
    returns = returns.rename(columns={value: key for key, value in mapping.items()})[study.tickers]
    levels = pd.concat([pd.DataFrame(100.0, index=pd.DatetimeIndex([seed]), columns=study.tickers),
                        100 * (1 + returns).cumprod()])
    levels.index.name, levels.columns.name = "Date", None
    _validate_frame(levels, study.tickers, "Total-return index", positive=True)
    return levels, mapping


def validate_snapshot_returns(study, factors, directory, levels, manifest):
    """Reconstruct saved index levels from the retained, hash-checked CRSP rows."""
    try:
        history = pd.read_csv(directory / HISTORY_FILE, keep_default_na=False)
        observations = pd.read_csv(directory / RETURN_FILE, keep_default_na=False)
        reconstructed, mapping = build_return_index(study, factors, history, observations)
        if (manifest["sources"]["prices"]["identifiers"] != mapping
                or not reconstructed.index.equals(levels.index)
                or not np.allclose(reconstructed, levels, rtol=1e-13, atol=1e-13)):
            raise WRDSError("WRDS_INDEX_RECONSTRUCTION_FAILED")
    except WRDSError:
        raise
    except (ValueError, TypeError, KeyError, OSError):
        raise WRDSError("WRDS_SNAPSHOT_VALIDATION_FAILED") from None


def download_wrds_snapshot(study, directory, *, verify_monthly=False):
    """Save a separate immutable WRDS snapshot, never falling back to Yahoo."""
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("WRDS snapshot destination already exists.")
    if not 1 <= len(study.tickers) <= 100:
        raise WRDSError("WRDS_INVALID_TICKER_COUNT")
    try:
        with urlopen(Request(FRENCH_URL, headers={"User-Agent": "financial-economics/0.1"}),
                     timeout=60) as response:
            archive = response.read()
        factors = parse_french_factors(archive)
    except Exception:
        raise WRDSError("WRDS_FRENCH_DOWNLOAD_FAILED") from None
    _, seed = _calendar(study, factors)
    with closing(connect_wrds()) as connection:
        candidates = _query(connection, HISTORY_SQL,
                            {"tickers": study.tickers, "start": str(seed.date()), "end": study.end})
        if list(candidates.columns) != IDENTITY_COLUMNS:
            raise WRDSError("WRDS_INVALID_IDENTITY")
        mapping = {}
        for ticker in study.tickers:
            rows = candidates.loc[candidates.ticker.eq(ticker)]
            if rows.empty:
                raise WRDSError("WRDS_IDENTITY_MISSING")
            if rows.permno.nunique() != 1:
                raise WRDSError("WRDS_IDENTITY_AMBIGUOUS")
            try:
                identifier = int(rows.permno.iloc[0])
                if identifier <= 0 or identifier != rows.permno.iloc[0]:
                    raise ValueError()
            except (ValueError, TypeError, OverflowError):
                raise WRDSError("WRDS_INVALID_IDENTITY") from None
            mapping[ticker] = identifier
        if len(set(mapping.values())) != len(mapping):
            raise WRDSError("WRDS_DUPLICATE_SECURITY")
        # Fetch the same securities' full dated histories, including former tickers.
        history = _query(connection, HISTORY_BY_ID_SQL,
                         {"permnos": list(mapping.values()), "start": str(seed.date()), "end": study.end})
        if list(history.columns) != IDENTITY_COLUMNS:
            raise WRDSError("WRDS_INVALID_IDENTITY")
        labels = {identifier: ticker for ticker, identifier in mapping.items()}
        history.insert(0, "requested_ticker", history.permno.map(labels))
        mapping = resolve_identifiers(study, factors, history)
        observations = _query(connection, RETURN_SQL,
                              {"permnos": list(mapping.values()), "start": study.start, "end": study.end})
        if verify_monthly:
            monthly = _query(connection, MONTHLY_SQL,
                             {"permnos": list(mapping.values()), "start": study.start, "end": study.end})
    levels, mapping = build_return_index(study, factors, history, observations)
    if verify_monthly:
        daily = observations.copy()
        daily["date"] = pd.to_datetime(daily.date)
        daily = daily.pivot(index="date", columns="permno", values="total_return")
        daily = daily.rename(columns={v: k for k, v in mapping.items()})[study.tickers].astype(float)
        coverage = validate_monthly_extract(daily, monthly, mapping, factors.index, study)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{directory.name}-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        levels.to_csv(staging / INDEX_FILE, date_format="%Y-%m-%d", float_format="%.17g")
        observations.to_csv(staging / RETURN_FILE, index=False, date_format="%Y-%m-%d", float_format="%.17g")
        history.to_csv(staging / HISTORY_FILE, index=False)
        factors.to_csv(staging / FACTOR_FILE, date_format="%Y-%m-%d", float_format="%.17g")
        (staging / SOURCE_ZIP).write_bytes(archive)
        if verify_monthly:
            monthly.to_csv(staging / MONTHLY_FILE, index=False, date_format="%Y-%m-%d", float_format="%.17g")
            queries = {"history": HISTORY_SQL, "security_history": HISTORY_BY_ID_SQL,
                       "daily": RETURN_SQL, "monthly": MONTHLY_SQL,
                       "parameters": {"tickers": study.tickers, "permnos": list(mapping.values()),
                                      "seed": str(seed.date()), "start": study.start, "end": study.end}}
            (staging / QUERY_FILE).write_text(json.dumps(queries, indent=2), encoding="utf-8")
        manifest = {
            "schema_version": 2, "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "settings": study.settings(),
            "sources": {
                "prices": {"provider": "WRDS / CRSP CIZ", "table": "crsp.dsf_v2",
                           "history_table": "crsp.stksecurityinfohist", "field": "dlyret",
                           "representation": "total_return_index", "base_level": 100,
                           "identifiers": mapping,
                           "definition": "Decimal daily total returns, including distributions; "
                                         "available-return flag NA. Index levels are not traded prices."},
                "factors": {"url": FRENCH_URL, "units": "decimal returns"},
            },
            "versions": {name: version(name) for name in ("psycopg2-binary", "pandas", "numpy")},
            "files": {name: _file_info(staging / name, frame) for name, frame in
                      [(INDEX_FILE, levels), (RETURN_FILE, None), (HISTORY_FILE, None),
                       (FACTOR_FILE, factors), (SOURCE_ZIP, None)]},
        }
        if verify_monthly:
            manifest["coverage"] = coverage
            manifest["files"].update({name: _file_info(staging / name) for name in [MONTHLY_FILE, QUERY_FILE]})
            manifest["sources"]["prices"]["monthly_table"] = "crsp.msf_v2"
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        load_snapshot(study, staging)
        if directory.exists():
            raise FileExistsError("WRDS snapshot destination already exists.")
        staging.rename(directory)
    return manifest
