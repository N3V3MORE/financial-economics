"""CUDA-only log-BV GRUs: a labelled fix for negative raw-BV forecasts.

The frozen raw-BV runs in paper_networks.py let a linear output fall below zero
in very calm markets. These GRUs learn standardized log BV, so every forecast
is positive. Profiles, seeds, folds and epochs rules stay identical; selection
uses only pre-2010 labels. The fix was added after the raw results were seen.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .gru_forecasting import _annual_splits, _gpu_inputs, _require_cuda
from .paper_gru import _cv_folds
from .paper_networks import DEFAULT_PROFILES, _model, _profiles


LOG_FLOOR = 1e-12


def _log_inputs(inputs):
    """Replace the BV channel by its logarithm; signed explanatory changes stay as they are."""
    values = np.array(inputs, dtype=float, copy=True)
    values[..., 0] = np.log(np.maximum(values[..., 0], LOG_FLOOR))
    return values


def _log_scale(prepared, training):
    inputs = _log_inputs(prepared.inputs[training])
    targets = prepared.rows.realized_bv.to_numpy(dtype=float)[training]
    if (not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(targets).all()
            or (targets <= 0).any()):
        raise ValueError("Log-BV scaling requires finite training features and positive labels.")
    deviation = inputs.std(axis=(0, 1))
    logs = np.log(targets)
    return {"feature_mean": inputs.mean(axis=(0, 1)).tolist(),
            "feature_std": np.where(deviation > 0, deviation, 1.).tolist(),
            "target_mean": float(logs.mean()), "target_std": max(float(logs.std()), 1e-12)}


def _encoded_targets(torch, prepared, rows, scaling, device):
    logs = np.log(prepared.rows.realized_bv.to_numpy(dtype=float)[rows])
    return torch.as_tensor((logs - scaling["target_mean"]) / scaling["target_std"],
                           dtype=torch.float32, device=device)


def _log_predictions(torch, model, inputs, scaling):
    if inputs.device != torch.device("cuda:0"):
        raise RuntimeError("Log-BV GRU prediction inputs must reside on CUDA:0.")
    model.eval()
    with torch.no_grad():
        encoded = model(inputs).cpu().numpy().astype(float)
    values = scaling["target_mean"] + scaling["target_std"] * encoded
    if not np.isfinite(values).all():
        raise RuntimeError("Log-BV GRU produced nonfinite forecasts.")
    return values


def _fit(torch, prepared, training, validation, scaling, profile, seed, epochs, patience, learning_rate):
    """Return the model, its best epoch and a training-only smearing factor."""
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = _gpu_inputs(torch, _log_inputs(prepared.inputs[training]), scaling, device)
    targets = _encoded_targets(torch, prepared, training, scaling, device)
    if validation is not None:
        validation_inputs = _gpu_inputs(torch, _log_inputs(prepared.inputs[validation]), scaling, device)
        validation_targets = _encoded_targets(torch, prepared, validation, scaling, device)
    model = _model(torch, prepared.inputs.shape[2], profile, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_epoch, best_state, stale = np.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(inputs), device=device)
        for offset in range(0, len(order), profile["batch_size"]):
            selected = order[offset:offset + profile["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = (model(inputs[selected]) - targets[selected]).square().mean()
            if profile["l2"]:
                loss = loss + profile["l2"] * sum(parameter.square().sum() for parameter in model.parameters())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Log-BV GRU training produced nonfinite loss on CUDA:0.")
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                score = float((model(validation_inputs) - validation_targets).square().mean().item())
            if not np.isfinite(score):
                raise RuntimeError("Log-BV GRU validation produced nonfinite MSE on CUDA:0.")
            if score < best_loss:
                best_loss, best_epoch, stale = score, epoch, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    # Duan smearing restores the mean BV level lost by exponentiating a log forecast.
    fitted = _log_predictions(torch, model, inputs, scaling)
    actual = np.log(prepared.rows.realized_bv.to_numpy(dtype=float)[training])
    smearing = float(np.mean(np.exp(actual - fitted)))
    return model, best_epoch, smearing


def reconstruct_log_gru(prepared, calendar, checkpoint_dir, *, model_name, profiles=None,
                        seeds=(11, 22, 33), training_end="2009-12-31", evaluation_start="2010-01-01",
                        max_epochs=60, patience=8, learning_rate=.001, folds=5, progress=print):
    """Select by seed-mean log-BV MSE, then freeze settings for annual refits."""
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise ValueError("Use a new GRU checkpoint directory; existing files are never replaced.")
    rows, inputs = prepared.rows, prepared.inputs
    if (not isinstance(model_name, str) or not model_name.isidentifier()
            or any(type(value) is not int or value < 1 for value in (max_epochs, patience, folds))
            or not np.isfinite(learning_rate) or learning_rate <= 0
            or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("GRU requires a model identifier and valid epochs, folds, seeds and learning rate.")
    if (inputs.ndim != 3 or inputs.shape[0] != len(rows) or inputs.shape[1] < 1
            or inputs.shape[2] not in (1, 5) or not np.isfinite(inputs).all()
            or (inputs[:, :, 0] < 0).any() or not isinstance(rows.index, pd.DatetimeIndex)
            or not rows.index.is_unique or not rows.index.is_monotonic_increasing
            or rows.return_date.isna().any() or not (rows.return_date > rows.index).all()):
        raise ValueError("GRU requires ordered one- or five-channel histories and future return dates.")
    targets = rows.realized_bv.to_numpy(dtype=float)
    available = np.isfinite(targets)
    if np.isinf(targets).any() or (targets[available] <= 0).any():
        raise ValueError("Log-BV targets must be positive finite observations or missing.")
    candidates = _profiles(DEFAULT_PROFILES if profiles is None else profiles)
    splits = _annual_splits(rows, calendar, evaluation_start)
    if pd.Timestamp(training_end) > splits[0][1]:
        raise ValueError("GRU tuning cannot use targets after the first evaluation model's refit date.")
    initial_training = (rows.return_date <= pd.Timestamp(training_end)).to_numpy() & available
    validation_folds = _cv_folds(rows, initial_training, folds)
    torch, device = _require_cuda()
    records, fold_metadata = [], []
    for number, (training, validation) in enumerate(validation_folds, 1):
        fold_metadata.append({"fold": number, "training_observations": int(training.sum()),
                              "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                              "validation_first_issue": str(rows.index[validation][0].date()),
                              "validation_last_target": str(rows.loc[validation, "return_date"].max().date()),
                              "validation_observations": int(validation.sum())})
    for candidate, profile in enumerate(candidates):
        squared_error, observations, best_epochs, fold_mse = 0., 0, [], []
        for number, (training, validation) in enumerate(validation_folds, 1):
            scaling = _log_scale(prepared, training)
            validation_inputs = _gpu_inputs(torch, _log_inputs(inputs[validation]), scaling, device)
            predictions = []
            for seed in seeds:
                progress(f"{model_name} validation: profile={candidate}, fold={number}, seed={seed}")
                model, epoch, smearing = _fit(torch, prepared, training, validation, scaling, profile,
                                              seed, max_epochs, patience, learning_rate)
                predictions.append(np.exp(_log_predictions(torch, model, validation_inputs, scaling)) * smearing)
                best_epochs.append(epoch)
            # Score on the log scale so the 2008 crisis fold cannot decide the selection alone.
            errors = np.log(np.mean(predictions, axis=0)) - np.log(targets[validation])
            squared_error += float(errors @ errors)
            observations += len(errors)
            fold_mse.append(float(np.mean(errors ** 2)))
        records.append({"profile": candidate, **profile, "log_mse": squared_error / observations,
                        "validation_observations": observations, "best_epochs": best_epochs,
                        "fold_log_mse": fold_mse})
    tuning = pd.DataFrame(records)
    selected = int(tuning.log_mse.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    profile = candidates[selected]
    epochs = max(1, int(np.median(tuning.iloc[selected].best_epochs)))
    result = rows.copy()
    result[f"{model_name}_bv"], result[f"{model_name}_refit_asof"] = np.nan, pd.NaT
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoints, annual_metadata = [], []
    for year, asof, training, issued in splits:
        training = training & available
        scaling = _log_scale(prepared, training)
        issued_inputs = _gpu_inputs(torch, _log_inputs(inputs[issued]), scaling, device)
        predictions, paths, smearings = [], [], []
        for seed in seeds:
            progress(f"{model_name} annual refit: year={year}, seed={seed}, epochs={epochs}")
            model, _, smearing = _fit(torch, prepared, training, None, scaling, profile,
                                      seed, epochs, patience, learning_rate)
            predictions.append(np.exp(_log_predictions(torch, model, issued_inputs, scaling)) * smearing)
            smearings.append(smearing)
            path = checkpoint_dir / str(year) / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                       **scaling, **profile, "model_name": model_name, "num_layers": 2,
                       "input_channels": inputs.shape[2], "lookback": inputs.shape[1],
                       "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
                       "smearing": smearing, "refit_asof": str(asof.date()),
                       "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                       "training_observations": int(training.sum()), "device": str(device),
                       "input_units": "log BV and signed explanatory changes; train-only channel standardization",
                       "target_units": "standardized log BV; exp(forecast) times training smearing factor",
                       "torch_version": str(torch.__version__), "rnn_backend": "native CUDA; cuDNN disabled"}
            with path.open("xb") as destination:
                torch.save(payload, destination)
            paths.append(str(path.relative_to(checkpoint_dir)))
        mean_prediction = np.mean(predictions, axis=0)
        if not np.isfinite(mean_prediction).all() or (mean_prediction <= 0).any():
            raise RuntimeError("Log-BV GRU forecasts must be finite and positive.")
        result.loc[issued, f"{model_name}_bv"] = mean_prediction
        result.loc[issued, f"{model_name}_refit_asof"] = asof
        checkpoints.extend(paths)
        annual_metadata.append({"evaluation_year": int(year), "refit_asof": str(asof.date()),
                                "train_last_target": payload["train_last_target"],
                                "training_observations": int(training.sum()),
                                "forecast_observations": int(issued.sum()),
                                "smearing": smearings, "scaling": scaling, "checkpoints": paths})
    metadata = {"model_name": model_name, "backend": "CUDA-required PyTorch",
                "device": str(device), "gpu": torch.cuda.get_device_name(0),
                "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
                "rnn_backend": "native CUDA; cuDNN disabled", "num_layers": 2,
                "input_channels": inputs.shape[2], "lookback": inputs.shape[1],
                "profiles": candidates, "selected_profile": dict(profile), "seeds": list(seeds),
                "learning_rate": learning_rate, "max_epochs": max_epochs, "patience": patience,
                "refit_epochs": epochs, "validation_log_mse": float(tuning.iloc[selected].log_mse),
                "selected_best_epochs": tuning.iloc[selected].best_epochs,
                "selection_rule": "pooled log-BV MSE of seed-mean forecasts across expanding purged folds",
                "epoch_rule": "median selected-profile fold/seed best epoch, rounded down",
                "training_loss": "MSE of standardized log BV, plus L2 on all weights and biases",
                "dropout_rule": "dropout on the sequence between the two single-layer GRUs",
                "forecast_units": "mean over seeds of exp(log forecast) times training smearing; always positive",
                "floor_events": 0, "folds": fold_metadata,
                "annual_refits": annual_metadata, "checkpoints": checkpoints,
                "deterministic_algorithms": True,
                "fix": "added after raw-BV results showed negative forecasts; labelled post hoc, raw runs retained",
                "reconstruction_choices": "same predeclared profiles, seeds, folds and optimizer settings as raw-BV runs"}
    return result, tuning, metadata
