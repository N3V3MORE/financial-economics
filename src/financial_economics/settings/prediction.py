"""Frozen settings for the daily volatility-forecast experiment."""

from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


MAIN_COST_BPS = 10  # the primary tests and episode comparisons use this fee scenario


@dataclass(frozen=True)
class PredictionSettings:
    training_end: str = "2014-12-31"
    validation_start: str = "2015-01-01"
    validation_end: str = "2016-12-31"
    evaluation_start: str = "2017-01-01"
    lookback: int = 22
    ewma_decays: tuple[float, ...] = (.90, .94, .97)
    hidden_sizes: tuple[int, ...] = (16, 32)
    learning_rates: tuple[float, ...] = (.001, .0003)
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10
    volatility_target: float = .10
    defensive_cap: float = .20
    stress_quantile: float = .90
    costs_bps: tuple[float, ...] = (0, 10, 25)
    resamples: int = 10000
    block_lengths: tuple[int, ...] = (6, 3, 12)
    seed: int = 20260101
    forecast_hac_lags: int = 20
    monthly_hac_lags: int = 6
    alpha: float = .05
    windows: tuple[str, ...] = ("2017:2019", "2020:2021", "2022:2025")

    def settings(self) -> dict:
        return asdict(self)

    def validate(self, study) -> None:
        dates = [pd.Timestamp(value) for value in
                 (study.start, self.training_end, self.validation_start,
                  self.validation_end, self.evaluation_start, study.end)]
        if (any(pd.isna(value) or value.tz is not None or value != value.normalize()
                for value in dates)
                or not dates[0] < dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]):
            raise ValueError("Training, validation, and evaluation must be ordered inside the study.")
        if study.benchmark != "SPY" or not {"SPY", "TLT"} <= set(study.tickers):
            raise ValueError("Prediction requires SPY as benchmark and a TLT return series.")
        for name in ("lookback", "batch_size", "max_epochs", "patience", "resamples",
                     "forecast_hac_lags", "monthly_hac_lags", "seed"):
            value = getattr(self, name)
            minimum = 0 if name in ("forecast_hac_lags", "monthly_hac_lags", "seed") else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"Invalid {name}.")
        if self.patience > self.max_epochs:
            raise ValueError("Early-stopping patience must not exceed maximum epochs.")
        for name, lower, upper in (("ewma_decays", 0, 1), ("learning_rates", 0, 1),
                                   ("costs_bps", -1, 10000)):
            values = getattr(self, name)
            if (not values or len(set(values)) != len(values)
                    or any(not isinstance(value, Real) or isinstance(value, bool)
                           or not np.isfinite(value) or not lower < value < upper
                           for value in values)):
                raise ValueError(f"Invalid {name}.")
        if any(value < 0 for value in self.costs_bps) or 0 not in self.costs_bps:
            raise ValueError("Costs must be nonnegative and include zero costs.")
        if MAIN_COST_BPS not in self.costs_bps:
            raise ValueError(f"Costs must include the main {MAIN_COST_BPS:g} bps scenario used by the primary tests.")
        for name, minimum in (("hidden_sizes", 1), ("seeds", 0), ("block_lengths", 1)):
            values = getattr(self, name)
            if (not values or len(set(values)) != len(values)
                    or any(not isinstance(value, int) or isinstance(value, bool)
                           or value < minimum for value in values)):
                raise ValueError(f"Invalid {name}.")
        for name in ("volatility_target", "stress_quantile", "alpha"):
            value = getattr(self, name)
            if (not isinstance(value, Real) or isinstance(value, bool)
                    or not np.isfinite(value) or not 0 < value < 1):
                raise ValueError(f"Invalid {name}.")
        if (not isinstance(self.defensive_cap, Real) or isinstance(self.defensive_cap, bool)
                or not np.isfinite(self.defensive_cap) or not 0 <= self.defensive_cap <= 1):
            raise ValueError("Invalid defensive_cap.")
        for window in self.windows:
            first, last = window.split(":")
            if not self.evaluation_start[:4] <= first <= last <= study.end[:4]:
                raise ValueError("Descriptive windows must lie inside evaluation years.")


def load_prediction(path: Path, study) -> PredictionSettings:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    if "prediction" not in raw:
        raise ValueError("Choose a configuration with frozen prediction settings.")
    values = raw["prediction"]
    for name in ("ewma_decays", "hidden_sizes", "learning_rates", "seeds", "costs_bps",
                 "block_lengths", "windows"):
        if name in values:
            values[name] = tuple(values[name])
    settings = PredictionSettings(**values)
    settings.validate(study)
    return settings
