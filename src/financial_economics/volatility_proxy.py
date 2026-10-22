"""CPU-native Oxford-like volatility estimation from public intraday measures.

This estimates the measurement for an already observed day, not future returns.
Calibration, method selection and the final accuracy test use disjoint dates.
No neural training, GPU backend or filled volatility observations are used here.
"""

import argparse
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .paper import load_bipower
from .provenance import file_digest, utc_now


FIELDS = ("permno", "date", "trade_qmle", "trade_order", "trade_se", "trade_rv5", "trade_rv15",
          "quote_qmle", "quote_order", "quote_se", "quote_rv5", "quote_rv15")
CHANNELS = ("trade_rv5", "quote_rv5", "trade_rv15", "quote_rv15", "trade_qmle", "quote_qmle")
CANDIDATES = (*((method, (channel,)) for method in ("scale", "log_linear") for channel in CHANNELS),
    ("log_linear", ("trade_rv5", "trade_rv15", "trade_qmle")),
    ("log_linear", ("quote_rv5", "quote_rv15", "quote_qmle")),
    ("log_linear", CHANNELS))
PERIODS = {"calibration": ("2000-01-01", "2013-12-31"),
           "selection": ("2014-01-01", "2016-12-31"),
           "final_test": ("2017-01-01", "2022-02-25"),
           "2021_to_archive_end": ("2021-01-01", "2022-02-25"),
           "covid": ("2020-02-19", "2020-03-23"),
           "february_2018_crash_week": ("2018-02-05", "2018-02-09")}
DAILY_CANDIDATES = (("log_linear", ("daily_gk",)),
                    ("log_linear", ("daily_range",)),
                    ("log_linear", ("daily_gk", "daily_gk_lag1", "daily_gk_past22")),
                    ("log_linear", ("daily_gk", "daily_rs_ratio", "daily_range")),
                    ("log_linear", ("daily_gk", "daily_rs_ratio", "daily_gk_lag1", "daily_gk_past22")),
                    ("log_ridge", ("daily_gk", "daily_rs_ratio", "daily_range", "daily_gk_lag1", "daily_gk_past22")),
                    ("log_quadratic", ("daily_gk", "daily_rs_ratio", "daily_range", "daily_gk_lag1", "daily_gk_past22")))


def load_risklab(directory, symbol="SPY", permno=84398):
    """Validate the instrument, raw byte hash, row count and dated 12-field schema."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = directory / f"risklab-{symbol.lower()}.txt"
    if manifest["files"].get(source.name) != file_digest(source):
        raise ValueError("Public volatility source hash does not match its manifest.")
    lines = source.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 7 or lines[:2] != [symbol, str(permno)] or int(lines[3]) != len(lines) - 6:
        raise ValueError(f"Risk Lab source must contain the complete {symbol}/{permno} response.")
    frame = pd.read_csv(StringIO("\n".join(lines[6:])), sep=r"\s+", names=FIELDS)
    dates = pd.DatetimeIndex(pd.to_datetime(frame.pop("date").astype(str), format="%Y%m%d"), name="Date")
    frame.index = dates
    if (not dates.is_unique or not dates.is_monotonic_increasing or dates.hasnans
            or not frame.permno.eq(permno).all() or not np.isfinite(frame.to_numpy()).all()
            or (frame[list(CHANNELS)] < 0).any().any()
            or str(dates[0].date()).replace("-", "") != lines[4]
            or str(dates[-1].date()).replace("-", "") != lines[5]):
        raise ValueError("Invalid public volatility identities, values or dates.")
    # Zero is an unavailable estimate, not an observation of zero market risk.
    return frame[list(CHANNELS)].replace(0., np.nan), manifest


def weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def reliable_measures(frame):
    """Flag gross trade/quote disagreement without deleting a market date."""
    ratio = frame.trade_rv5 / frame.quote_rv5
    unreliable = (ratio > 1.5) | (ratio < 1 / 1.5)
    return frame.mask(unreliable, axis=0), int(unreliable.sum())


def load_daily_measures(directory):
    """Estimate intraday variation from verified SPX daily price ranges; no overnight term."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    source = directory / "equity-prices.json"
    if manifest["files"].get(source.name) != file_digest(source):
        raise ValueError("Daily price source hash does not match its manifest.")
    result = json.loads(source.read_text(encoding="utf-8"))["chart"]["result"][0]
    meta = result["meta"]
    if meta["symbol"] != "^GSPC" or meta["currency"] != "USD":
        raise ValueError("Daily range approximation requires the USD S&P 500 price index.")
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(
        meta["exchangeTimezoneName"]).normalize().tz_localize(None)
    prices = pd.DataFrame(result["indicators"]["quote"][0], index=dates)[["open", "high", "low", "close"]].dropna()
    if (not prices.index.is_unique or not prices.index.is_monotonic_increasing
            or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()
            or (prices.high < prices[["open", "close"]].max(axis=1)).any()
            or (prices.low > prices[["open", "close"]].min(axis=1)).any()):
        raise ValueError("Daily price ranges must be positive, ordered and internally consistent.")
    square_range = np.log(prices.high / prices.low) ** 2
    gk = .5 * square_range - (2 * np.log(2) - 1) * np.log(prices.close / prices.open) ** 2
    gk = gk.where(gk > 0)
    rs = (np.log(prices.high / prices.open) * np.log(prices.high / prices.close)
          + np.log(prices.low / prices.open) * np.log(prices.low / prices.close))
    measures = pd.DataFrame({"daily_gk": np.sqrt(252 * gk),
                             # Zero RS is valid for a monotonic day. This positive
                             # ratio retains that information without inventing variance.
                             "daily_rs_ratio": 1 + rs.clip(lower=0) / gk,
                             "daily_range": np.sqrt(252 * square_range / (4 * np.log(2))),
                             "daily_gk_lag1": np.sqrt(252 * gk.shift(1)),
                             "daily_gk_past22": np.sqrt(252 * gk.shift(1).rolling(22, min_periods=10).mean())})
    return measures, manifest


def _design(values, fitted):
    logged = np.log(values)
    if fitted["method"] in ("log_ridge", "log_quadratic"):
        logged = np.clip((logged - fitted["input_mean"]) / fitted["input_scale"], -6., 6.)
    columns = [np.ones(len(values)), *logged.T]
    if fitted["method"] == "log_quadratic":
        columns.extend(logged[:, i] * logged[:, j] for i in range(logged.shape[1])
                       for j in range(i, logged.shape[1]))
    return np.column_stack(columns)


def fit_proxy(measures, target, method, channels):
    """Fit only supplied calibration rows; minimize percentage error in the scale."""
    joined = measures[list(channels)].join(target.rename("target"), how="inner").dropna()
    if len(joined) < 60 or (joined <= 0).any().any():
        raise ValueError("Proxy calibration needs at least 60 positive complete observations.")
    x, y = joined[list(channels)].to_numpy(), joined.target.to_numpy()
    fitted = {"method": method, "channels": list(channels)}
    if method == "scale" and len(channels) == 1:
        coefficients = []
        unscaled = x[:, 0]
    elif method == "median":
        coefficients = []
        unscaled = np.median(x, axis=1)
    elif method == "log_convex":
        # Positive weights summing to one prevent offsetting coefficients from
        # magnifying a single bad intraday input on a quiet or shortened day.
        logged, outcome = np.log(x), np.log(y)
        centered, residual = logged - logged.mean(axis=0), outcome - outcome.mean()
        initial = np.full(len(channels), 1 / len(channels))

        def loss(weights):
            error = centered @ weights - residual
            return float(error @ error / len(y) + .001 * np.square(weights - initial).sum())

        def gradient(weights):
            return 2 * centered.T @ (centered @ weights - residual) / len(y) + .002 * (weights - initial)

        optimization = minimize(loss, initial, jac=gradient, method="SLSQP",
                                bounds=[(0., 1.)] * len(channels),
                                constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                             "jac": lambda w: np.ones(len(w))},
                                options={"ftol": 1e-12, "maxiter": 1000})
        if not optimization.success:
            raise RuntimeError(f"Convex measurement calibration failed: {optimization.message}")
        weights = optimization.x
        coefficients = [float(outcome.mean() - logged.mean(axis=0) @ weights), *weights.tolist()]
        fitted["ridge_penalty"] = .001
        unscaled = np.exp(_design(x, fitted) @ coefficients)
    elif method in ("log_linear", "log_ridge", "log_quadratic"):
        if method != "log_linear":
            logged = np.log(x)
            fitted.update(input_mean=logged.mean(axis=0).tolist(),
                          input_scale=np.maximum(logged.std(axis=0), 1e-8).tolist(),
                          ridge_penalty=1. if method == "log_ridge" else 10.)
        design = _design(x, fitted)
        if method == "log_linear":
            coefficients = np.linalg.lstsq(design, np.log(y), rcond=None)[0].tolist()
        else:
            penalty = fitted["ridge_penalty"] * np.eye(design.shape[1])
            penalty[0, 0] = 0.
            coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ np.log(y)).tolist()
        unscaled = np.exp(design @ coefficients)
    else:
        raise ValueError("Unknown proxy method or invalid scale channels.")
    ratio = y / unscaled
    scale = weighted_median(ratio, 1 / ratio)
    return {**fitted, "coefficients": coefficients,
            "scale": scale, "calibration_observations": len(joined),
            "calibration_first": str(joined.index[0].date()),
            "calibration_last": str(joined.index[-1].date())}


def estimate_proxy(measures, fitted):
    frame = measures[fitted["channels"]].dropna()
    if (frame <= 0).any().any():
        raise ValueError("Proxy inputs must be positive observed volatility estimates.")
    values = frame.to_numpy()
    if fitted["method"] == "scale":
        predicted = values[:, 0]
    elif fitted["method"] == "median":
        predicted = np.median(values, axis=1)
    else:
        predicted = np.exp(_design(values, fitted) @ fitted["coefficients"])
    predicted *= fitted["scale"]
    if not np.isfinite(predicted).all() or (predicted <= 0).any():
        raise ValueError("Proxy estimates must be positive and finite.")
    return pd.Series(predicted, index=frame.index, name="proxy_annual_volatility")


def available_estimates(estimates, ranking):
    """Use the selection-ranked best available estimator without filling absent dates."""
    empty = pd.DatetimeIndex([], name="Date")
    result = pd.Series(index=empty, dtype=float, name="proxy_annual_volatility")
    chosen = pd.Series(index=empty, dtype="int64", name="measurement_candidate")
    for candidate in ranking:
        extra = estimates[candidate].loc[~estimates[candidate].index.isin(result.index)]
        result = pd.concat([result, extra])
        chosen = pd.concat([chosen, pd.Series(candidate, index=extra.index, dtype="int64")])
    result.index.name = chosen.index.name = "Date"
    return result.sort_index(), chosen.sort_index()


def accuracy(target, predicted):
    """Score volatility levels, with missing expected dates counted in coverage."""
    paired = target.rename("actual").to_frame().join(predicted.rename("predicted")).dropna()
    count = len(paired)
    result = {"expected_days": len(target), "scored_days": count,
              "missing_days": len(target) - count,
              "coverage": count / len(target) if len(target) else None}
    if not count:
        return {**result, "accuracy_score": None, "mean_percentage_error": None,
                "median_percentage_error": None, "p95_percentage_error": None,
                "maximum_percentage_error": None, "minimum_daily_accuracy": None,
                "days_below_85_percent_accuracy": len(target), "share_within_15_percent": 0. if len(target) else None,
                "share_within_10_percent": 0. if len(target) else None,
                "bv_mean_percentage_error": None, "correlation": None}
    error = (paired.predicted / paired.actual - 1).abs()
    return {**result, "accuracy_score": float(1 - error.mean()),
            "mean_percentage_error": float(error.mean()), "median_percentage_error": float(error.median()),
            "p95_percentage_error": float(error.quantile(.95)),
            "maximum_percentage_error": float(error.max()), "minimum_daily_accuracy": float(1 - error.max()),
            "days_below_85_percent_accuracy": int((error > .15 + 1e-12).sum() + len(target) - count),
            "share_within_15_percent": float((error <= .15 + 1e-12).sum() / len(target)),
            "share_within_10_percent": float((error <= .1 + 1e-12).sum() / len(target)),
            "bv_mean_percentage_error": float(((paired.predicted / paired.actual) ** 2 - 1).abs().mean()),
            "correlation": float(paired.actual.corr(paired.predicted)) if (
                count > 1 and paired.actual.nunique() > 1 and paired.predicted.nunique() > 1) else None}


def passes_requested_gate(score):
    """A strong average cannot hide a bad individual day or an omitted date."""
    return bool(score["accuracy_score"] is not None and score["accuracy_score"] >= .90 - 1e-12
                and score["coverage"] == 1. and score["minimum_daily_accuracy"] >= .85 - 1e-12
                and score["days_below_85_percent_accuracy"] == 0)


def passes_historical_gate(scores):
    """The individual-day floor also applies to history offered for training."""
    required = {"calibration", "selection", "final_test"}
    checked = scores.loc[scores.scope.eq("all") & scores.period.isin(required)]
    return (len(checked) == len(required) and set(checked.period) == required
            and all(passes_requested_gate(row) for _, row in checked.iterrows()))


def candidate_specs(auxiliary, *, with_daily=True):
    candidates = list(CANDIDATES)
    daily = ("daily_gk", "daily_rs_ratio", "daily_range", "daily_gk_lag1", "daily_gk_past22")
    groups = [CHANNELS]
    for symbol in auxiliary:
        channels = tuple(f"{symbol}_{channel}" for channel in CHANNELS)
        candidates.extend((method, (channel,)) for method in ("scale", "log_linear") for channel in channels)
        groups.append(channels)
    if auxiliary:
        groups.append(tuple(channel for group in groups for channel in group))
    for channels in groups:
        for augmented in ((channels, (*channels, *daily)) if with_daily else (channels,)):
            candidates.extend((method, augmented) for method in ("log_linear", "log_ridge", "log_quadratic"))
        for augmented in ((channels, (*channels, "daily_gk", "daily_range")) if with_daily else (channels,)):
            candidates.extend((method, augmented) for method in ("log_convex", "median"))
    return list(dict.fromkeys(candidates))


def _ranked_selection(frame):
    eligible = frame.loc[(frame.coverage >= .99) & frame.accuracy_score.ge(.90)]
    if "calibration_accuracy_score" in eligible:
        eligible = eligible.loc[eligible.calibration_accuracy_score.ge(.90)]
    if eligible.empty:
        raise ValueError("No intraday proxy candidate reaches 90% selection accuracy with adequate coverage.")
    ranked = eligible.copy()
    ranked["worst_seen_error"] = ranked.maximum_percentage_error
    if "calibration_maximum_percentage_error" in ranked:
        ranked["worst_seen_error"] = ranked[["maximum_percentage_error", "calibration_maximum_percentage_error"]].max(axis=1)
    return ranked.sort_values(["worst_seen_error", "days_below_85_percent_accuracy", "mean_percentage_error"])


def protocol(source_dir, archive, prices_dir=None, auxiliary=None):
    auxiliary = auxiliary or {}
    return {"created_at_utc": utc_now(), "source_manifest_sha256": file_digest(source_dir / "manifest.json"),
            "oxford_archive_sha256": file_digest(archive), "code_sha256": file_digest(Path(__file__)),
            "periods": PERIODS, "candidates": candidate_specs(auxiliary, with_daily=prices_dir is not None),
            "auxiliary_manifests": {symbol: file_digest(path / "manifest.json") for symbol, path in auxiliary.items()},
            "daily_fallback_candidates": DAILY_CANDIDATES if prices_dir is not None else [],
            "daily_prices_manifest_sha256": file_digest(prices_dir / "manifest.json") if prices_dir else None,
            "selection": "require at least 90% mean accuracy on both calibration and selection dates and 99% selection input coverage; minimize worst daily percentage error across both periods, then selection failing-day count, then selection mean error; calibration fixed before selection",
            "daily_fallback_selection": "minimize worst daily percentage error, then failing-day count, then mean error, using selection dates only",
            "input_quality_rule": "an ETF's inputs are unavailable when its 5-minute trade/quote volatility ratio exceeds 1.5 or falls below 1/1.5; market dates remain in the test with another ETF or a labelled daily-range fallback",
            "missing_estimator_rule": "use the highest selection-ranked candidate with observed inputs; never fill absent source dates",
            "accuracy_definition": "1 minus mean absolute percentage error of annualized sqrt(BV), not correlation or a classification hit rate",
            "accuracy_threshold": .90, "individual_accuracy_min": .85, "coverage_min": 1.,
            "gate_periods": ["calibration", "selection", "final_test"],
            "stress": "Oxford volatility above the calibration-period 90th percentile; missing dates remain missing",
            "units": "Risk Lab annual volatility in decimal units; Oxford annual volatility=sqrt(252*BV)",
            "backend": "CPU-native NumPy least squares/ridge, SciPy convex calibration and weighted-median calibration",
            "limitations": ["Estimates a same-day measurement, not future risk or investment performance",
                            "Historical data have already been inspected; proxy is exploratory",
                            "SPY ETF and SPX index differ; RV/QMLE and bipower variation differ",
                            "Exact historical publication times and post-2022 accuracy are unverified"]}


def run(args):
    root = args.output_dir.resolve()
    source_dir, archive = args.data_dir.resolve(), args.oxford_archive.resolve()
    prices_dir = getattr(args, "prices_dir", None)
    auxiliary = {symbol: path.resolve() for symbol in ("ivv", "voo")
                 if (path := getattr(args, f"{symbol}_dir", None)) is not None}
    protected = [source_dir, archive, *auxiliary.values()]
    if prices_dir is not None:
        prices_dir = prices_dir.resolve()
        protected.append(prices_dir)
    if root.exists() or any(root.is_relative_to(path) or path.is_relative_to(root) for path in protected):
        raise ValueError("Use a new output directory outside the source snapshot.")
    measures, _ = load_risklab(source_dir)
    measures, bad = reliable_measures(measures)
    quality_counts = {"SPY": bad}
    published_first, published_last = measures.index[0], measures.index[-1]
    for symbol, path in auxiliary.items():
        extra, _ = load_risklab(path, symbol.upper(), {"ivv": 88215, "voo": 12305}[symbol])
        extra, bad = reliable_measures(extra)
        quality_counts[symbol.upper()] = bad
        measures = measures.join(extra.add_prefix(f"{symbol}_"), how="outer")
    daily = None
    if prices_dir is not None:
        daily, _ = load_daily_measures(prices_dir)
        measures = measures.join(daily, how="outer")
    frozen = protocol(source_dir, archive, prices_dir, auxiliary)
    root.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    actual_bv = load_bipower(archive, PERIODS["calibration"][0], PERIODS["final_test"][1])
    target = np.sqrt(252 * actual_bv)
    training = target.loc[slice(*PERIODS["calibration"])]
    candidates, fitted_models, estimates = [], [], []
    for number, (method, channels) in enumerate(frozen["candidates"]):
        fitted = fit_proxy(measures, training, method, channels)
        predicted = estimate_proxy(measures, fitted)
        score = accuracy(target.loc[slice(*PERIODS["selection"])], predicted)
        calibration_score = accuracy(training, predicted)
        candidates.append({"candidate": number, "method": method, "channels": ",".join(channels), **score,
                           "calibration_accuracy_score": calibration_score["accuracy_score"],
                           "calibration_maximum_percentage_error": calibration_score["maximum_percentage_error"]})
        fitted_models.append(fitted)
        estimates.append(predicted)
    selection = pd.DataFrame(candidates)
    eligible = _ranked_selection(selection)
    chosen = int(eligible.candidate.iloc[0])
    selection["selected"] = selection.candidate.eq(chosen)
    ranking = eligible.candidate.astype(int).to_list()
    predicted, measurement_candidate = available_estimates(estimates, ranking)
    fitted = fitted_models[chosen]
    daily_fallback = None
    if prices_dir is not None:
        daily_choices, daily_estimates = [], []
        for method, channels in DAILY_CANDIDATES:
            model = fit_proxy(daily, training, method, channels)
            estimate = estimate_proxy(daily, model)
            score = accuracy(target.loc[slice(*PERIODS["selection"])], estimate)
            daily_choices.append({"model": model, "selection_mean_percentage_error": score["mean_percentage_error"],
                                  "selection_maximum_percentage_error": score["maximum_percentage_error"],
                                  "selection_days_below_85_percent_accuracy": score["days_below_85_percent_accuracy"]})
            daily_estimates.append(estimate)
        best_daily = min(range(len(daily_choices)), key=lambda i: (
            daily_choices[i]["selection_maximum_percentage_error"],
            daily_choices[i]["selection_days_below_85_percent_accuracy"],
            daily_choices[i]["selection_mean_percentage_error"]))
        daily_fallback = dict(daily_choices[best_daily])
        daily_fallback["candidates"] = daily_choices.copy()
        extra = daily_estimates[best_daily].loc[lambda x: (~x.index.isin(predicted.index))
                                                  & (x.index >= published_first) & (x.index <= published_last)]
        predicted = pd.concat([predicted, extra]).sort_index()
        measurement_candidate = pd.concat([measurement_candidate, pd.Series(-1, index=extra.index)]).sort_index()
    stress_level = float(training.quantile(.9))
    records = []
    for name, (first, last) in PERIODS.items():
        observed = target.loc[first:last]
        records.append({"period": name, "scope": "all", **accuracy(observed, predicted)})
        records.append({"period": name, "scope": "stress", **accuracy(observed.loc[observed > stress_level], predicted)})
    scores = pd.DataFrame(records)
    final = scores.loc[scores.period.eq("final_test") & scores.scope.eq("all")].iloc[0]
    passed = passes_historical_gate(scores)
    final_target = target.loc[slice(*PERIODS["final_test"])]
    missing = final_target.index.difference(predicted.index).strftime("%Y-%m-%d").tolist()
    comparison = target.rename("oxford_annual_volatility").to_frame().join(predicted)
    comparison.index.name = "Date"
    comparison["percentage_error"] = (comparison.proxy_annual_volatility / comparison.oxford_annual_volatility - 1).abs()
    comparison["daily_accuracy"] = 1 - comparison.percentage_error
    comparison["measurement_candidate"] = measurement_candidate
    comparison["below_85_percent_accuracy"] = comparison.percentage_error.isna() | comparison.percentage_error.gt(.15 + 1e-12)
    proxy = (predicted ** 2 / 252).rename("proxy_bv").to_frame()
    proxy["estimated_measurement"] = True
    proxy["measurement_candidate"] = measurement_candidate
    proxy["calibration_last_label"] = fitted["calibration_last"]
    # A later-year export remains gated; the full diagnostic series is always clearly labelled estimated.
    selection.to_csv(root / "selection.csv", index=False)
    scores.to_csv(root / "accuracy.csv", index=False)
    comparison.to_csv(root / "comparison.csv")
    comparison.loc[comparison.below_85_percent_accuracy].to_csv(root / "failures.csv")
    proxy.to_csv(root / "estimated_bv.csv")
    for year in sorted(set(final_target.index.year)):
        records.append({"period": str(year), "scope": "all", **accuracy(target.loc[str(year)], predicted)})
    pd.DataFrame(records).to_csv(root / "accuracy_by_period.csv", index=False)
    metadata = {"created_at_utc": utc_now(), "protocol_sha256": file_digest(root / "protocol.json"),
                "selected_candidate": chosen, "fitted_model": fitted, "stress_threshold": stress_level,
                "flagged_trade_quote_disagreements": quality_counts,
                "ranked_candidates": [{"candidate": candidate, **fitted_models[candidate]} for candidate in ranking],
                "fallback_observations": int(measurement_candidate.ne(chosen).sum()),
                "daily_price_fallback": daily_fallback,
                "daily_price_fallback_observations": int(measurement_candidate.eq(-1).sum()),
                "calibration_last_label": str(training.index[-1].date()),
                "passes_90_percent_volatility_gate": bool(final.accuracy_score >= .90
                    and final.median_percentage_error <= .10 and final.coverage >= .99),
                "passes_requested_accuracy_gate": passed,
                "gate_is_not_90_percent_days_within_10_percent": True,
                "final_test_missing_dates": missing,
                "later_years_available": {"first": str(proxy.loc["2022-02-26":].index[0].date()),
                                         "last": str(proxy.index[-1].date()),
                                         "observations": len(proxy.loc["2022-02-26":])},
                "files": {path.name: file_digest(path) for path in root.iterdir() if path.is_file()}}
    (root / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(scores.loc[scores.scope.eq("all")].to_string(index=False))
    print(f"Selected: {fitted['method']} / {','.join(fitted['channels'])}; 90% average / 85% every-day gate passed: {passed}")
    print(f"Saved labelled diagnostic estimates: {root}")
    return passed


def load_accepted_proxy(directory):
    """Load only a hash-verified proxy that passed its stated accuracy/coverage gate."""
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    for name, digest in metadata["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_digest(path) != digest:
            raise ValueError("Proxy output does not match its recorded byte hashes.")
    if not metadata.get("passes_90_percent_volatility_gate"):
        raise ValueError("Proxy did not pass the 90% volatility accuracy and coverage gate.")
    if not metadata.get("passes_requested_accuracy_gate"):
        raise ValueError("Proxy has not passed the required 90% average / 85% every-day accuracy gate; diagnostic use only.")
    frozen = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(directory / "accuracy.csv")
    final = scores.loc[scores.period.eq("final_test") & scores.scope.eq("all")]
    if (len(final) != 1 or frozen.get("individual_accuracy_min") != .85
            or frozen.get("accuracy_threshold") != .90 or frozen.get("coverage_min") != 1.
            or not passes_historical_gate(scores)):
        raise ValueError("Saved scores do not satisfy the 90% average / 85% every-day accuracy gate.")
    frame = pd.read_csv(directory / "estimated_bv.csv", index_col=0, parse_dates=True,
                        float_precision="round_trip")
    values = frame.proxy_bv
    if (not values.index.is_unique or not values.index.is_monotonic_increasing
            or not np.isfinite(values).all() or (values <= 0).any()
            or not frame.estimated_measurement.eq(True).all()):
        raise ValueError("Proxy must contain positive observed estimates on unique ordered dates.")
    return values.rename("bv"), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--oxford-archive", type=Path, required=True)
    parser.add_argument("--prices-dir", type=Path, help="Verified daily SPX ranges for explicitly labelled missing-day estimates")
    parser.add_argument("--ivv-dir", type=Path, help="Verified Risk Lab IVV/88215 snapshot")
    parser.add_argument("--voo-dir", type=Path, help="Verified Risk Lab VOO/12305 snapshot")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        passed = run(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(1, f"Volatility proxy: {error}\n")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
