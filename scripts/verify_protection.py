"""Independently check saved protection evidence; print pass/fail, never source rows."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

from financial_economics.config import load_study
from financial_economics.return_data import load_returns


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10, equal_nan=True)


def verify(config, source, output):
    study = load_study(config)
    data = load_returns(study, source, verified_monthly=True)
    run = json.loads((output / "run.json").read_text())
    assert run["source_manifest"] == data.provenance
    assert run["config_sha256"] == sha256(config.read_bytes()).hexdigest()
    for name, fingerprint in run["files"].items():
        assert sha256((output / name).read_bytes()).hexdigest() == fingerprint, name

    def read(name):
        return pd.read_csv(output / f"{name}.csv", float_precision="round_trip")

    daily = data.daily.assign(RF=data.factors.RF)
    # Recompute each month from the source independently of the pipeline's groupby helper.
    months = pd.period_range(study.start, study.end, freq="M")
    monthly = pd.DataFrame([np.prod(1+daily.loc[str(month)].to_numpy(), axis=0)-1 for month in months],
                           index=months.to_timestamp("M"), columns=daily.columns)
    saved_monthly = read("fund_monthly_returns").set_index("Date")
    close(saved_monthly.to_numpy(), monthly.to_numpy())
    for row in read("fund_conditions").itertuples():
        start, end = row.window.split(":")
        subset = monthly.loc[start:end]
        mask = {"all": np.ones(len(subset), dtype=bool), "negative_spy": subset.SPY.lt(0),
                "severe_spy": subset.SPY.le(run["protection"]["severe_threshold"])}[row.condition]
        values = (subset[row.fund]-subset.RF).loc[mask]
        assert row.months == len(values)
        close(row.mean_minus_cash, values.mean())
        close(row.beat_cash_frequency, (values > 0).mean() if len(values) else np.nan)

    paths = read("daily_returns").set_index("Date")
    paths.index = pd.to_datetime(paths.index)
    portfolio_months = pd.DataFrame([np.prod(1+paths.loc[str(month)].to_numpy(), axis=0)-1 for month in months],
                                    index=monthly.index, columns=paths.columns)
    for row in read("portfolio_conditions").itertuples():
        start, end = row.window.split(":")
        mask = monthly.SPY.loc[start:end] < 0 if row.condition == "negative_spy" else monthly.SPY.loc[start:end] <= run["protection"]["severe_threshold"]
        suffix = f" | {row.cost_bps:g}bps"
        gap = (portfolio_months[row.portfolio+suffix]-portfolio_months[row.control+suffix]).loc[start:end].loc[mask]
        assert len(gap) == row.months
        close(row.mean_advantage, gap.mean())
        close(row.beat_control_frequency, (gap > 0).mean() if len(gap) else np.nan)
    for row in read("performance").itertuples():
        start, end = row.window.split(":")
        series = paths.loc[start:end, f"{row.portfolio} | {row.cost_bps:g}bps"]
        values = series.to_numpy()
        excess = values-daily.RF.loc[series.index].to_numpy()
        wealth = np.r_[1., np.cumprod(1+values)]
        close(row.cagr, wealth[-1] ** (study.trading_days/len(values))-1)
        close(row.annualized_volatility, np.std(values, ddof=1)*np.sqrt(study.trading_days))
        close(row.sharpe, excess.mean()/excess.std(ddof=1)*np.sqrt(study.trading_days))
        close(row.max_drawdown, np.min(wealth/np.maximum.accumulate(wealth)-1))
    weights = read("target_weights").set_index("portfolio")
    for name, weight in weights.iterrows():
        actual = [np.prod(1+paths.loc[str(month), f"{name} | 0bps"].to_numpy())-1 for month in months]
        close(actual, monthly[weight.index].to_numpy() @ weight.to_numpy())

    if run["protection"]["primary_inference"]:
        tests = read("primary_tests")
        down = monthly.SPY.lt(0).to_numpy()
        design = np.column_stack([down, ~down]).astype(float)
        inverse = np.linalg.inv(design.T @ design)
        pvalues = []
        n, k = design.shape
        lags = run["protection"]["hac_lags"]
        for row in tests.itertuples():
            y = (monthly[row.fund]-monthly.RF).to_numpy()
            coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
            score = design * (y - design @ coefficient)[:, None]
            meat = score.T @ score
            for lag in range(1, lags+1):
                cross = score[lag:].T @ score[:-lag]
                meat += (1-lag/(lags+1)) * (cross+cross.T)
            covariance = inverse @ meat @ inverse * n/(n-k)
            se = np.sqrt(covariance[0, 0])
            probability = 2*t.sf(abs(coefficient[0]/se), n-k)
            close([row.mean_minus_cash, row.hac_se, row.p_value], [coefficient[0], se, probability])
            pvalues.append(probability)
        order = np.argsort(pvalues)
        corrected = np.minimum(1, np.maximum.accumulate(np.array(pvalues)[order]*(len(order)-np.arange(len(order)))))
        close(tests.holm_p_value.to_numpy()[order], corrected)
        intervals = read("bootstrap_intervals")
        assert set(intervals.mean_block_months) == {3, 6, 12}
        assert intervals.resamples.eq(10000).all() and intervals.seed.eq(20260101).all()
        assert intervals.valid_draws.add(intervals.unavailable_draws).eq(10000).all()

    # Check plotted numbers against independently checked numerical tables.
    conditions = read("fund_conditions")
    full = f"{study.start[:4]}:{study.end[:4]}"
    falling = conditions.loc[conditions.window.eq(full) & conditions.condition.eq("negative_spy")]
    for row in read("figure_protection").itertuples():
        reference = falling.loc[falling.fund.eq(row.fund)].iloc[0]
        close(row.mean_minus_cash, reference.mean_minus_cash)
    perf = read("performance")
    for row in read("figure_tradeoffs").itertuples():
        prefix = f"{1-row.candidate_allocation:.0%} SPY + {row.candidate_allocation:.0%} "
        sub = perf.query("window == @full and cost_bps == @row.cost_bps").set_index("portfolio")[["cagr", "annualized_volatility", "max_drawdown"]]
        close([row.cagr_difference, row.volatility_difference, row.drawdown_difference],
              (sub.loc[prefix+row.fund]-sub.loc[prefix+"cash"]).to_numpy(dtype=float))
    for row in read("figure_declines").itertuples():
        dates = daily.index[(daily.index > row.peak_date) & (daily.index <= row.trough_date)]
        close(row.return_minus_cash, np.prod(1+daily.loc[dates, row.fund])-np.prod(1+daily.RF.loc[dates]))
    print(f"PASS: {study.name}; source/monthly coverage, {len(weights)} zero-cost identities, "
          "all performance/condition rows, HAC/Holm where applicable, output hashes, and figure values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.config, args.data_dir, args.output_dir)
