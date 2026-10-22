"""Explicit frozen settings for the ETF-versus-cash experiment."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtectionSettings:
    allocations: list[float] = field(default_factory=lambda: [.1, .2, .4])
    costs_bps: list[float] = field(default_factory=lambda: [0, 10, 25])
    resamples: int = 10000
    block_lengths: list[int] = field(default_factory=lambda: [6, 3, 12])
    seed: int = 20260101
    hac_lags: int = 6
    alpha: float = .05
    primary_inference: bool = True
    severe_threshold: float = -.05
    training_start: str = "2015-01-01"
    training_end: str = "2019-12-31"
    evaluation_start: str = "2020-01-01"
    windows: list[str] = field(default_factory=lambda: ["2015:2019", "2020:2021", "2022:2025"])
    fixed_portfolios: dict[str, list[str]] = field(default_factory=dict)

    def settings(self):
        return asdict(self)

    def validate(self, study):
        for name, values, lower, upper in [("allocations", self.allocations, 0, 1),
                                           ("costs", self.costs_bps, 0, 10000)]:
            if (not values or len(set(values)) != len(values)
                    or any(not np.isfinite(x) or not lower <= x < upper for x in values)):
                raise ValueError(f"Invalid {name}.")
        if 0 in self.allocations or 0 not in self.costs_bps:
            raise ValueError("Use positive candidate allocations and include zero costs.")
        if (not isinstance(self.resamples, int) or self.resamples < 100
                or not isinstance(self.seed, int) or self.seed < 0
                or not isinstance(self.hac_lags, int) or self.hac_lags < 0
                or not 0 < self.alpha < 1 or not -1 < self.severe_threshold < 0
                or not self.block_lengths or len(set(self.block_lengths)) != len(self.block_lengths)
                or any(not np.isfinite(b) or b < 1 for b in self.block_lengths)):
            raise ValueError("Invalid inference settings.")
        if not (pd.Timestamp(study.start) <= pd.Timestamp(self.training_start)
                <= pd.Timestamp(self.training_end) < pd.Timestamp(self.evaluation_start)
                <= pd.Timestamp(study.end)):
            raise ValueError("Training must precede evaluation inside the study.")
        for window in self.windows:
            start, end = window.split(":")
            if not study.start[:4] <= start <= end <= study.end[:4]:
                raise ValueError("Descriptive windows must be inside the study.")
        for name, funds in self.fixed_portfolios.items():
            if not name or not funds or len(set(funds)) != len(funds) or not set(funds) <= set(study.tickers):
                raise ValueError("Fixed portfolios must contain distinct study funds.")


def load_protection(path: Path, study) -> ProtectionSettings:
    with path.open("rb") as source:
        raw = tomllib.load(source)
    if "protection" not in raw:
        raise ValueError("Choose a configuration with frozen protection settings.")
    settings = ProtectionSettings(**raw["protection"])
    settings.validate(study)
    return settings
