"""Run isolated, reusable stages of the public-data U.S. reconstruction.

Example: python -m financial_economics.paper_reconstruction classical --help
Classical fits are CPU-native; GRU fits require cuda:0 via CUDA or ROCm, without CPU fallback.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import pandas as pd

from .paper_sources import load_sources, prepare_sources
from .provenance import file_digest, git_branch, git_head, utc_now


STAGES = ("classical", "gru_univar", "gru_multivar")
# Post hoc, labelled log-BV fixes; the raw-BV stages above stay frozen.
LOG_STAGES = ("gru_univar_log", "gru_multivar_log")

# Exact reviewed transitions add dispatch provenance only; model math is unchanged.
# Any later edit requires a fresh review and cannot pass these fixed digests.
# Each file maps its reviewed original digests to the one current digest.
# gru_forecasting.py has no entry: turning off NVIDIA TF32 changed cuDNN training math,
# so GRU stages cached by any earlier version (e55887f5..., 5be8eadf..., b2212cfc...)
# need a fresh run.
METADATA_ONLY_CODE_TRANSITIONS = {
    "paper_gru.py": (("565e94d2bbd433118dc41fee67b849cf5504c3489f5f5a75dd528ce37bba2564",),
                     "b0dee9a9a30a773ddd79405fe22124ca56156022729d516c509894da90f951da"),
    "paper_networks.py": (("0018d3d7645ac02e9031a2a4bc87c812bdb48d81ed3cfa3b76ecac7b7d24184e",),
                          "80ca9710081bed0aaee1d2dfc07c5546e98b3949d265798f7cbe16d12c073dd1"),
    "paper_networks_log.py": (("6e132375daa7c3e2cf774a2924071ab97ff43abf126112c66cf6b86c1f36f5c6",),
                              "185a93030c3a780fe0d3ad16ab48578a126206b9e75d13cf52372a614d7c2865"),
}


def _code_hashes(stage):
    names = ["paper_sources.py", "paper.py", "forecasting.py", "paper_reconstruction.py"]
    if stage == "classical":
        names += ["paper_models.py"]
    elif stage in LOG_STAGES:
        names += ["paper_networks.py", "paper_networks_log.py", "paper_gru.py", "gru_forecasting.py"]
    else:
        names += ["paper_networks.py", "paper_gru.py", "gru_forecasting.py"]
    return {name: file_digest(Path(__file__).with_name(name)) for name in names}


def _input_identity(args):
    return {"source_manifest": file_digest(args.data_dir / "manifest.json"),
            "oxford_archive": file_digest(args.oxford_archive), "config": file_digest(args.config)}


def _write_metadata(output, metadata):
    metadata["files"] = {str(path.relative_to(output)): file_digest(path)
                         for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _read_stage(directory, identity, stage):
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    current_code = _code_hashes(stage)
    # Reading old CSV headers can change without changing any fitted model.
    dependencies = {name: digest for name, digest in current_code.items()
                    if name != "paper_reconstruction.py"}
    compatible_code = {
        name: {"original_sha256": metadata["code"].get(name), "current_sha256": digest,
               "reason": "reviewed metadata-only addition: cuDNN/MIOpen library flags/version and "
                         "first training-forward dispatch evidence; model math unchanged"}
        for name, digest in dependencies.items()
        if name in METADATA_ONLY_CODE_TRANSITIONS
        and metadata["code"].get(name) in METADATA_ONLY_CODE_TRANSITIONS[name][0]
        and digest == METADATA_ONLY_CODE_TRANSITIONS[name][1]
    }
    if (metadata["inputs"] != identity
            or any(metadata["code"].get(name) != digest and name not in compatible_code
                   for name, digest in dependencies.items())):
        raise ValueError(f"Saved {stage} settings, inputs or model code changed; use a fresh run.")
    for name, digest in metadata["files"].items():
        if file_digest(directory / name) != digest:
            raise ValueError(f"Saved {stage} hash mismatch: {name}.")
    forecasts = pd.read_csv(directory / "forecasts.csv", index_col=0,
                            parse_dates=True, float_precision="round_trip")
    if not isinstance(forecasts.index, pd.DatetimeIndex) or not forecasts.index.is_unique:
        raise ValueError("Saved forecasts require unique dated issue rows.")
    forecasts.index.name = "forecast_date"
    for name in forecasts:
        if name == "return_date" or name.endswith("_refit_asof"):
            forecasts[name] = pd.to_datetime(forecasts[name])
    metadata["cache_reader"] = {"original_wrapper_sha256": metadata["code"]["paper_reconstruction.py"],
                                "current_wrapper_sha256": current_code["paper_reconstruction.py"],
                                "metadata_only_code_compatibility": compatible_code,
                                "rule": "input/settings and every saved file must match; forecasting "
                                        "dependencies match exactly or use a reviewed exact metadata-only "
                                        "digest transition; wrapper-only reader compatibility accepted"}
    return forecasts, metadata


def run(args):
    with args.config.open("rb") as source:
        settings = tomllib.load(source)["reconstruction"]
    destination = args.output_dir.resolve()
    if (destination.is_relative_to(args.data_dir.resolve())
            or args.oxford_archive.resolve().is_relative_to(destination)
            or (args.fx_dir is not None and destination.is_relative_to(args.fx_dir.resolve()))):
        raise ValueError("Keep new outputs outside the immutable source inputs.")
    if args.stage in ("evaluate", "evaluate_fixed") and args.fx_dir is None:
        raise ValueError("Evaluation requires --fx-dir with the official dated H.10 snapshot.")
    observations, source_metadata = load_sources(args.data_dir)
    univar, multi, bv, price_returns, returns, calendar, quality = prepare_sources(
        observations, args.oxford_archive, settings["lookback"])
    if pd.Timestamp(settings["evaluation_end"]) > calendar[-1]:
        raise ValueError("Requested evaluation end is not covered by the public snapshot.")
    identity = _input_identity(args)
    # Use exactly the same available issues for all five models; labels may be missing.
    common = univar.rows.index.intersection(multi.rows.index).intersection(price_returns.index)
    eligible = univar.rows.loc[common]
    eligible = eligible.loc[eligible.return_date.between(settings["evaluation_start"],
                                                        settings["evaluation_end"])]
    stage = args.stage
    target = destination / {"evaluate": "evaluation", "evaluate_fixed": "evaluation_fixed",
                            "evaluate_local": "evaluation_local"}.get(stage, stage)
    if target.exists():
        raise ValueError(f"Existing results are never overwritten: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{stage}-", dir=destination) as temporary:
        output = Path(temporary) / "run"
        output.mkdir()
        metadata = {"created_at_utc": utc_now(),
                    "study": "Public-data U.S. reconstruction; not an exact numerical replication",
                    "inputs": identity, "settings": settings, "source_quality": quality,
                    "calculation_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")},
                    "branch": git_branch(),
                    "head": git_head()}
        if stage == "classical":
            from .paper_models import classical_forecasts

            forecasts, fits, model = classical_forecasts(eligible, bv, price_returns,
                window=settings["window"], window_start=settings["window_start"], window_end=settings["window_end"])
            fits.to_csv(output / "daily_fits.csv", index=False)
            metadata["model"] = model
        elif stage in STAGES:
            from .paper_networks import reconstruct_gru

            forecasts, tuning, model = reconstruct_gru(
                univar if stage == "gru_univar" else multi, calendar, output / "checkpoints", model_name=stage,
                **{name: settings[name] for name in ("profiles", "seeds", "training_end", "evaluation_start",
                                                    "max_epochs", "patience", "learning_rate", "folds")})
            forecasts = forecasts.reindex(eligible.index)
            tuning.to_csv(output / "tuning.csv", index=False)
            metadata["model"] = model
        elif stage in LOG_STAGES:
            from .paper_networks_log import reconstruct_log_gru

            forecasts, tuning, model = reconstruct_log_gru(
                univar if stage == "gru_univar_log" else multi, calendar, output / "checkpoints",
                model_name=stage, **{name: settings[name] for name in (
                    "profiles", "seeds", "training_end", "evaluation_start",
                    "max_epochs", "patience", "learning_rate", "folds")})
            forecasts = forecasts.reindex(eligible.index)
            tuning.to_csv(output / "tuning.csv", index=False)
            metadata["model"] = model
        else:
            from .paper_evaluation import FIXED_MODELS, MODELS, evaluate_paper
            from .paper_fx import FX_SOURCES, currency_returns, local_currency_returns

            if stage == "evaluate_local":
                returns, fx_quality = local_currency_returns(observations, returns)
                fx_metadata = {"sources": {}, "limitations": [
                    "Post hoc: Swiss and German bond proxies earn unhedged local-currency returns, as the paper's"
                    " statistics imply; a USD investor would also bear currency risk"]}
                metadata["post_hoc"] = ("Swiss/German bonds valued in CHF/EUR after the paper-figure check;"
                                        " the USD evaluation is retained unchanged beside it")
            else:
                fx_rates, fx_metadata = load_sources(args.fx_dir)
                returns, fx_quality = currency_returns(observations, returns, fx_rates)
                metadata["evaluation_fx"] = {"manifest_sha256": file_digest(args.fx_dir / "manifest.json"),
                                              "sources": fx_metadata["sources"],
                                              "unused_original_snapshot_series": list(fx_metadata["sources"])}
            quality.update(fx_quality)
            quality["portfolio_alignment_carries_including_seed"] = {
                name: count for name, count in quality["portfolio_alignment_carries_including_seed"].items()
                if name not in FX_SOURCES}
            forecasts = eligible.copy()
            metadata["model_runs"] = {}
            saved_stages = (*STAGES, *LOG_STAGES) if stage == "evaluate_fixed" else STAGES
            for saved_stage in saved_stages:
                saved, model_metadata = _read_stage(destination / saved_stage, identity, saved_stage)
                if not saved.index.equals(eligible.index) or not saved.return_date.equals(eligible.return_date):
                    raise ValueError("All model caches must use the same forecast and return dates.")
                for name in saved:
                    if name not in forecasts:
                        forecasts[name] = saved[name]
                metadata["model_runs"][saved_stage] = {
                    "manifest_sha256": file_digest(destination / saved_stage / "run.json"),
                    "model": model_metadata["model"], "cache_reader": model_metadata["cache_reader"]}
            tables, assumptions = evaluate_paper(returns, forecasts, settings,
                                                 FIXED_MODELS if stage == "evaluate_fixed" else MODELS)
            for name, frame in tables.items():
                frame.to_csv(output / f"{name}.csv", index=True)
            metadata.update({"assumptions": assumptions,
                             "source_substitutions": source_metadata["sources"] | fx_metadata["sources"],
                             "limitations": source_metadata["limitations"] + fx_metadata["limitations"]})
        forecasts.to_csv(output / "forecasts.csv", index_label="forecast_date")
        metadata["code"] = (_code_hashes(stage) if stage in (*STAGES, *LOG_STAGES) else
                            {name: file_digest(Path(__file__).with_name(name))
                             for name in ("paper_evaluation.py", "paper_reconstruction.py", "paper_fx.py",
                                          "paper_sources.py", "analysis.py", "costs.py", "prediction_inference.py")})
        _write_metadata(output, metadata)
        output.rename(target)
    print(f"Saved {stage}: {target}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, *LOG_STAGES, "evaluate", "evaluate_fixed", "evaluate_local"))
    parser.add_argument("--config", type=Path, default=Path("configs/paper-reconstruction.toml"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--fx-dir", type=Path, help="Official H.10 snapshot; required for evaluation only")
    parser.add_argument("--oxford-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
