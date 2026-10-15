"""Read and cross-check the local outputs used by the Manim film."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from financial_economics.analysis import _check_returns, drawdowns


@dataclass
class PresentationData:
    dates: pd.DatetimeIndex
    wealth: pd.DataFrame
    performance: pd.DataFrame
    capm: pd.DataFrame
    metadata: dict
    benchmark: str


def load_presentation(folder: str | Path) -> PresentationData:
    """Cross-check return statistics; read CAPM estimates from the saved table."""
    folder = Path(folder)
    required = ["daily_returns.csv", "performance.csv", "capm.csv", "run.json"]
    if any(not (folder / name).is_file() for name in required):
        raise FileNotFoundError("Run 'uv run fecon analyze' before rendering the film.")
    metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    returns = pd.read_csv(folder / "daily_returns.csv", index_col=0, parse_dates=True)
    _check_returns(returns, minimum=2)
    performance = pd.read_csv(folder / "performance.csv", index_col=0)
    capm = pd.read_csv(folder / "capm.csv", index_col=0)
    benchmark = metadata["study"]["benchmark"]
    expected = [*metadata["study"]["assets"], "Equal weight"]
    if (list(returns.columns) != expected or list(performance.index) != expected
            or list(capm.index) != expected or benchmark not in returns
            or metadata["observations"] != len(returns)
            or metadata["actual_start"] != returns.index[0].strftime("%Y-%m-%d")
            or metadata["actual_end"] != returns.index[-1].strftime("%Y-%m-%d")
            or not (capm["n_obs"] == len(returns)).all()
            or not (performance["n_obs"] == len(returns)).all()):
        raise ValueError("Saved presentation outputs are inconsistent; rerun analyze.")
    years = len(returns) / metadata["study"]["trading_days"]
    cumulative = (1 + returns).prod() - 1
    checks = {
        "cumulative_return": cumulative,
        "cagr": (1 + cumulative) ** (1 / years) - 1,
        "annualized_volatility": returns.std(ddof=1)
        * np.sqrt(metadata["study"]["trading_days"]),
        "max_drawdown": drawdowns(returns).min(),
    }
    if any(not np.allclose(performance[name], values, rtol=1e-8, atol=1e-10)
           for name, values in checks.items()):
        raise ValueError("Saved presentation statistics are inconsistent; rerun analyze.")
    if (not np.isfinite(capm[["beta", "alpha_pvalue"]]).all().all()
            or not capm["alpha_pvalue"].between(0, 1).all()):
        raise ValueError("Saved CAPM values are inconsistent; rerun analyze.")
    wealth = pd.concat([
        pd.DataFrame(1000.0, index=[0], columns=returns.columns),
        1000 * (1 + returns).cumprod(),
    ], ignore_index=True)
    return PresentationData(returns.index, wealth, performance, capm, metadata, benchmark)
