"""Read the small, explicit study definition."""

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class Study:
    name: str
    start: str
    end: str
    benchmark: str
    assets: dict[str, str]
    trading_days: int = 252
    hac_lags: int = 5

    @property
    def tickers(self) -> list[str]:
        return list(self.assets)

    def settings(self) -> dict:
        return asdict(self)


def load_study(path: Path) -> Study:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    study = Study(**raw["study"], assets=raw["assets"])
    if date.fromisoformat(study.start) >= date.fromisoformat(study.end):
        raise ValueError("Study start must be before its end.")
    if not study.assets or study.benchmark not in study.assets:
        raise ValueError("Choose assets and include the benchmark among them.")
    if any(not ticker or not label for ticker, label in study.assets.items()):
        raise ValueError("Every asset needs a ticker and description.")
    if study.trading_days <= 0 or study.hac_lags < 0:
        raise ValueError("Trading days must be positive and HAC lags nonnegative.")
    return study
