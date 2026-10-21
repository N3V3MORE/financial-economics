"""CUDA-only raw-BV GRUs for the public-data paper reconstruction.

The paper publishes tuning categories, not its grid or seeds. These small,
predeclared profiles are reconstruction choices, frozen before evaluation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .gru_forecasting import _annual_splits, _gpu_inputs, _require_cuda
from .paper_gru import _cv_folds


PROFILE_FIELDS = ("hidden_1", "hidden_2", "batch_size", "dropout", "l2")
DEFAULT_PROFILES = ((16, 16, 64, 0., 0.), (16, 32, 32, .1, .0001),
                    (32, 16, 64, .1, 0.), (32, 32, 32, 0., .0001))


def _profiles(values):
    result = []
    for value in values:
        profile = dict(value) if isinstance(value, dict) else dict(zip(PROFILE_FIELDS, value, strict=True))
        if (set(profile) != set(PROFILE_FIELDS)
                or any(type(profile[name]) is not int or profile[name] < 1
                       for name in PROFILE_FIELDS[:3])
                or not np.isfinite([profile["dropout"], profile["l2"]]).all()
                or not 0 <= profile["dropout"] < 1 or profile["l2"] < 0):
            raise ValueError("GRU profiles need positive widths/batch size, dropout in [0,1), and nonnegative L2.")
        result.append(profile)
    if not result:
        raise ValueError("At least one predeclared GRU profile is required.")
    return result


def _training_scale(prepared, training):
    inputs = prepared.inputs[training]
    targets = prepared.rows.realized_bv.to_numpy(dtype=float)[training]
    if not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Raw-BV scaling requires finite training features and labels.")
    deviation = inputs.std(axis=(0, 1))
    return {"feature_mean": inputs.mean(axis=(0, 1)).tolist(),
            "feature_std": np.where(deviation > 0, deviation, 1.).tolist(),
            "target_scale": max(float(targets.mean()), 1e-12)}


def _model(torch, channels, profile, device):
    class RawBVGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.GRU(channels, profile["hidden_1"], batch_first=True)
            self.second = torch.nn.GRU(profile["hidden_1"], profile["hidden_2"], batch_first=True)
            self.dropout = torch.nn.Dropout(profile["dropout"])
            self.output = torch.nn.Linear(profile["hidden_2"], 1)

        def forward(self, inputs):
            # Native CUDA retains GPU execution and avoids the known Windows cuDNN crash.
            with torch.backends.cudnn.flags(enabled=False):
                sequence, _ = self.first(inputs)
                _, hidden = self.second(self.dropout(sequence))
            return self.output(hidden[-1]).squeeze(-1)

    model = RawBVGRU().to(device)
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("Every reconstructed GRU parameter must reside on CUDA:0.")
    return model


def _raw_predictions(torch, model, inputs, scaling):
    if inputs.device != torch.device("cuda:0"):
        raise RuntimeError("Reconstructed GRU prediction inputs must reside on CUDA:0.")
    model.eval()
    with torch.no_grad():
        values = model(inputs).cpu().numpy().astype(float) * scaling["target_scale"]
    if not np.isfinite(values).all():
        raise RuntimeError("Reconstructed GRU produced nonfinite raw-BV forecasts.")
    return values


def _fit(torch, prepared, training, validation, scaling, profile, seed, epochs,
         patience, learning_rate):
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = _gpu_inputs(torch, prepared.inputs[training], scaling, device)
    targets = torch.as_tensor(prepared.rows.realized_bv.to_numpy(dtype=float)[training]
                              / scaling["target_scale"], dtype=torch.float32, device=device)
    if validation is not None:
        validation_inputs = _gpu_inputs(torch, prepared.inputs[validation], scaling, device)
        validation_targets = torch.as_tensor(prepared.rows.realized_bv.to_numpy(dtype=float)[validation]
                                              / scaling["target_scale"], dtype=torch.float32, device=device)
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
            # Apply the stated penalty to biases as well as both GRUs' weights.
            if profile["l2"]:
                loss = loss + profile["l2"] * sum(parameter.square().sum() for parameter in model.parameters())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Raw-BV GRU training produced nonfinite loss on CUDA:0.")
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                score = float((model(validation_inputs) - validation_targets).square().mean().item())
            if not np.isfinite(score):
                raise RuntimeError("Raw-BV GRU validation produced nonfinite MSE on CUDA:0.")
            if score < best_loss:
                best_loss, best_epoch, stale = score, epoch, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def reconstruct_gru(prepared, calendar, checkpoint_dir, *, model_name, profiles=None,
                    seeds=(11, 22, 33), training_end="2009-12-31", evaluation_start="2010-01-01",
                    max_epochs=60, patience=8, learning_rate=.001, folds=5, progress=print):
    """Select by seed-mean raw-BV MSE, then freeze settings for annual refits."""
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
    if np.isinf(targets).any() or (targets[available] < 0).any():
        raise ValueError("Raw-BV targets must be nonnegative finite observations or missing.")
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
            scaling = _training_scale(prepared, training)
            validation_inputs = _gpu_inputs(torch, inputs[validation], scaling, device)
            predictions = []
            for seed in seeds:
                progress(f"{model_name} validation: profile={candidate}, fold={number}, seed={seed}")
                model, epoch = _fit(torch, prepared, training, validation, scaling, profile,
                                    seed, max_epochs, patience, learning_rate)
                predictions.append(_raw_predictions(torch, model, validation_inputs, scaling))
                best_epochs.append(epoch)
            errors = np.mean(predictions, axis=0) - targets[validation]
            squared_error += float(errors @ errors)
            observations += len(errors)
            fold_mse.append(float(np.mean(errors ** 2)))
        records.append({"profile": candidate, **profile, "mse": squared_error / observations,
                        "validation_observations": observations, "best_epochs": best_epochs,
                        "fold_mse": fold_mse})
    tuning = pd.DataFrame(records)
    selected = int(tuning.mse.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    profile = candidates[selected]
    epochs = max(1, int(np.median(tuning.iloc[selected].best_epochs)))
    result = rows.copy()
    result[f"{model_name}_bv"], result[f"{model_name}_refit_asof"] = np.nan, pd.NaT
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoints, annual_metadata, floor_events = [], [], 0
    for year, asof, training, issued in splits:
        training = training & available
        scaling = _training_scale(prepared, training)
        issued_inputs = _gpu_inputs(torch, inputs[issued], scaling, device)
        predictions, paths = [], []
        for seed in seeds:
            progress(f"{model_name} annual refit: year={year}, seed={seed}, epochs={epochs}")
            model, _ = _fit(torch, prepared, training, None, scaling, profile,
                            seed, epochs, patience, learning_rate)
            predictions.append(_raw_predictions(torch, model, issued_inputs, scaling))
            path = checkpoint_dir / str(year) / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                       **scaling, **profile, "model_name": model_name, "num_layers": 2,
                       "input_channels": inputs.shape[2], "lookback": inputs.shape[1],
                       "seed": seed, "epochs": epochs, "learning_rate": learning_rate,
                       "refit_asof": str(asof.date()),
                       "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                       "training_observations": int(training.sum()), "device": str(device),
                       "input_units": "raw BV and signed explanatory changes; train-only channel standardization",
                       "target_units": "raw BV divided by training-mean BV; linear output",
                       "torch_version": str(torch.__version__), "rnn_backend": "native CUDA; cuDNN disabled"}
            with path.open("xb") as destination:
                torch.save(payload, destination)
            paths.append(str(path.relative_to(checkpoint_dir)))
        mean_prediction = np.mean(predictions, axis=0)
        floored = int((mean_prediction < 1e-12).sum())
        floor_events += floored
        result.loc[issued, f"{model_name}_bv"] = np.maximum(mean_prediction, 1e-12)
        result.loc[issued, f"{model_name}_refit_asof"] = asof
        checkpoints.extend(paths)
        annual_metadata.append({"evaluation_year": int(year), "refit_asof": str(asof.date()),
                                "train_last_target": payload["train_last_target"],
                                "training_observations": int(training.sum()),
                                "forecast_observations": int(issued.sum()),
                                "floor_events": floored, "scaling": scaling, "checkpoints": paths})
    metadata = {"model_name": model_name, "backend": "CUDA-required PyTorch",
                "device": str(device), "gpu": torch.cuda.get_device_name(0),
                "torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda),
                "rnn_backend": "native CUDA; cuDNN disabled", "num_layers": 2,
                "input_channels": inputs.shape[2], "lookback": inputs.shape[1],
                "profiles": candidates, "selected_profile": dict(profile), "seeds": list(seeds),
                "learning_rate": learning_rate, "max_epochs": max_epochs, "patience": patience,
                "refit_epochs": epochs, "validation_mse": float(tuning.iloc[selected].mse),
                "selected_best_epochs": tuning.iloc[selected].best_epochs,
                "selection_rule": "pooled raw-BV MSE of seed-mean forecasts across expanding purged folds",
                "epoch_rule": "median selected-profile fold/seed best epoch, rounded down",
                "training_loss": "MSE of raw BV divided by training-mean BV, plus L2 on all weights and biases",
                "dropout_rule": "dropout on the sequence between the two single-layer GRUs",
                "forecast_units": "mean raw-BV seed predictions, then floor at 1e-12",
                "floor_events": floor_events, "folds": fold_metadata,
                "annual_refits": annual_metadata, "checkpoints": checkpoints,
                "deterministic_algorithms": True,
                "reconstruction_choices": "predeclared profiles, seeds, scaling, folds, linear output and optimizer settings; original values unavailable"}
    return result, tuning, metadata
