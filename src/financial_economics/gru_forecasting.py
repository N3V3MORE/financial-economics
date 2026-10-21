"""CUDA-only GRU variance forecasts, frozen validation settings, and annual refits."""

import importlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .forecasting import ForecastData, qlike

# CUDA deterministic matrix operations require this before importing PyTorch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _require_cuda():
    try:
        torch = importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("GRU requires the prediction extra: uv sync --extra prediction.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("GRU requires CUDA:0; no CUDA GPU is available. CPU fallback is disabled.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if torch.empty(1, device=device).device != device:
        raise RuntimeError("GRU could not allocate tensors on CUDA:0.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch, device


def _annual_splits(rows, calendar, evaluation_start):
    """Assign each issue date to the model fitted at its preceding annual close."""
    if (not isinstance(calendar, pd.DatetimeIndex) or not calendar.is_unique
            or not calendar.is_monotonic_increasing):
        raise ValueError("Annual refits require a sorted, unique trading calendar.")
    years = sorted(set(calendar[calendar >= pd.Timestamp(evaluation_start)].year))
    asofs = []
    for year in years:
        previous = calendar[calendar < pd.Timestamp(year=year, month=1, day=1)]
        if not len(previous):
            raise ValueError("Annual refits need a trading close before evaluation starts.")
        asofs.append(previous[-1])
    splits = []
    for position, (year, asof) in enumerate(zip(years, asofs)):
        issued = rows.index >= asof
        if position + 1 < len(asofs):
            issued &= rows.index < asofs[position + 1]
        training = (rows.return_date <= asof).to_numpy()
        if issued.any():
            if not training.any():
                raise ValueError("Annual refits need target-available training observations.")
            splits.append((year, asof, training, issued))
    if not splits:
        raise ValueError("No evaluation forecast dates are available for annual refits.")
    return splits


def _training_scale(prepared, training):
    inputs = prepared.inputs[training]
    targets = prepared.rows.squared_return.to_numpy(dtype=float)[training]
    if not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Training requires finite, target-available inputs and squared returns.")
    deviation = inputs.std(axis=(0, 1))
    return {"feature_mean": inputs.mean(axis=(0, 1)).tolist(),
            "feature_std": np.where(deviation > 0, deviation, 1.).tolist(),
            "target_scale": max(float(targets.mean()), 1e-12)}


def _gpu_inputs(torch, inputs, scaling, device):
    standardized = (inputs - scaling["feature_mean"]) / scaling["feature_std"]
    return torch.as_tensor(standardized, dtype=torch.float32, device=device)


def _make_model(torch, hidden_size, target_scale, device):
    class VarianceGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(2, hidden_size, num_layers=1, batch_first=True)
            self.output = torch.nn.Linear(hidden_size, 1)
            self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))

        def forward(self, inputs):
            _, hidden = self.gru(inputs)
            return torch.nn.functional.softplus(self.output(hidden[-1]).squeeze(-1)) * self.target_scale + 1e-12

    model = VarianceGRU().to(device)
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("Every GRU parameter must reside on CUDA:0.")
    return model


def _fit(torch, train_inputs, train_targets, validation, scaling,
         hidden_size, learning_rate, seed, epochs, batch_size, patience):
    device = torch.device("cuda:0")
    tensors = [train_inputs, train_targets, *(validation or ())]
    if any(tensor.device != device for tensor in tensors):
        raise RuntimeError("GRU training and validation tensors must reside on CUDA:0.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _make_model(torch, hidden_size, scaling["target_scale"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_epoch, best_state, stale = np.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_inputs), device=device)
        epoch_loss = torch.zeros((), device=device)
        for offset in range(0, len(order), batch_size):
            indices = order[offset:offset + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_inputs[indices])
            loss = (prediction.log() + train_targets[indices] / prediction).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(indices)
        if not bool(torch.isfinite(epoch_loss).item()):
            raise RuntimeError("GRU training produced a nonfinite loss on CUDA:0.")
        if validation is not None:
            model.eval()
            with torch.no_grad():
                predicted = model(validation[0])
                score = float((predicted.log() + validation[1] / predicted).mean().item())
            if not np.isfinite(score):
                raise RuntimeError("GRU validation produced a nonfinite loss on CUDA:0.")
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


def _predict(torch, model, inputs):
    if inputs.device != torch.device("cuda:0"):
        raise RuntimeError("GRU prediction tensors must reside on CUDA:0.")
    with torch.no_grad():
        prediction = model(inputs).cpu().numpy().astype(float)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise RuntimeError("GRU variance forecasts must be finite and strictly positive.")
    return prediction


def _save_checkpoint(torch, model, path, scaling, prepared, training,
                     asof, seed, hidden_size, learning_rate, epochs, batch_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
               **scaling, "hidden_size": hidden_size, "learning_rate": learning_rate,
               "lookback": prepared.inputs.shape[1], "seed": seed, "epochs": epochs,
               "batch_size": batch_size, "refit_asof": str(asof.date()),
               "train_last_target": str(prepared.rows.loc[training, "return_date"].max().date()),
               "training_observations": int(np.sum(training)), "device": "cuda:0",
               "torch_version": str(torch.__version__)}
    with path.open("xb") as destination:
        torch.save(payload, destination)


def gru_forecasts(prepared: ForecastData, settings, calendar: pd.DatetimeIndex,
                  checkpoint_dir: Path, progress=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune on validation only, then refit annually for frozen median-best epochs.

    The selected initial model supplies the retrospective validation threshold
    history. Annual models replace it from their as-of issue close onward.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise ValueError("Use a new GRU checkpoint directory; existing files are never replaced.")
    rows, inputs = prepared.rows, prepared.inputs
    if inputs.shape != (len(rows), settings.lookback, 2) or not np.isfinite(inputs).all():
        raise ValueError("GRU input dimensions and finite features must match forecast rows.")
    splits = _annual_splits(rows, calendar, settings.evaluation_start)
    initial_training = (rows.return_date <= pd.Timestamp(settings.training_end)).to_numpy()
    validation = ((rows.index >= pd.Timestamp(settings.validation_start))
                  & (rows.return_date <= pd.Timestamp(settings.validation_end))).to_numpy()
    history = ((rows.index >= pd.Timestamp(settings.validation_start))
               & (rows.index < splits[0][1]))
    if not initial_training.any() or not validation.any() or not history.any():
        raise ValueError("GRU needs initial training and a validation forecast history.")
    initial_asof = calendar[calendar <= pd.Timestamp(settings.training_end)][-1]
    torch, device = _require_cuda()
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    scaling = _training_scale(prepared, initial_training)
    train_inputs = _gpu_inputs(torch, inputs[initial_training], scaling, device)
    train_targets = torch.as_tensor(rows.squared_return.to_numpy()[initial_training], dtype=torch.float32, device=device)
    validation_inputs = _gpu_inputs(torch, inputs[validation], scaling, device)
    validation_targets = torch.as_tensor(rows.squared_return.to_numpy()[validation], dtype=torch.float32, device=device)
    history_inputs = _gpu_inputs(torch, inputs[history], scaling, device)
    tuning_rows, candidates, checkpoints = [], [], []
    for hidden_size in settings.hidden_sizes:
        for learning_rate in settings.learning_rates:
            predictions, best_epochs, paths = [], [], []
            candidate = len(tuning_rows)
            for seed in settings.seeds:
                progress(f"GRU validation: hidden={hidden_size}, rate={learning_rate:g}, seed={seed}")
                model, best_epoch = _fit(torch, train_inputs, train_targets,
                                         (validation_inputs, validation_targets), scaling,
                                         hidden_size, learning_rate, seed, settings.max_epochs,
                                         settings.batch_size, settings.patience)
                predictions.append(_predict(torch, model, history_inputs))
                best_epochs.append(best_epoch)
                path = checkpoint_dir / "initial" / f"grid_{candidate}" / f"seed_{seed}.pt"
                _save_checkpoint(torch, model, path, scaling, prepared, initial_training,
                                 initial_asof, seed, hidden_size, learning_rate, best_epoch, settings.batch_size)
                paths.append(str(path.relative_to(checkpoint_dir)))
            mean_prediction = np.mean(predictions, axis=0)
            scored = mean_prediction[validation[history]]
            observed = rows.squared_return.to_numpy()[validation]
            tuning_rows.append({"hidden_size": hidden_size, "learning_rate": learning_rate,
                                "qlike": float(qlike(observed, scored).mean()),
                                "mse": float(np.mean((observed - scored) ** 2)),
                                "validation_observations": int(validation.sum()),
                                "best_epochs": best_epochs})
            candidates.append((mean_prediction, best_epochs, paths))
            checkpoints.extend(paths)
    tuning = pd.DataFrame(tuning_rows)
    selected = int(tuning.qlike.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    selected_row = tuning.iloc[selected]
    selected_history, selected_epochs, selected_paths = candidates[selected]
    refit_epochs = int(np.median(selected_epochs))
    result = rows.copy()
    result["gru_variance"] = np.nan
    result["gru_refit_asof"] = pd.NaT
    result.loc[history, "gru_variance"] = selected_history
    result.loc[history, "gru_refit_asof"] = initial_asof
    annual_metadata = []
    for year, asof, training, issued in splits:
        progress(f"GRU annual refit: year={year}, asof={asof.date()}, epochs={refit_epochs}")
        scaling = _training_scale(prepared, training)
        train_inputs = _gpu_inputs(torch, inputs[training], scaling, device)
        train_targets = torch.as_tensor(rows.squared_return.to_numpy()[training], dtype=torch.float32, device=device)
        issued_inputs = _gpu_inputs(torch, inputs[issued], scaling, device)
        predictions, paths = [], []
        for seed in settings.seeds:
            progress(f"GRU annual seed: year={year}, seed={seed}")
            model, _ = _fit(torch, train_inputs, train_targets, None, scaling,
                            int(selected_row.hidden_size), float(selected_row.learning_rate), seed,
                            refit_epochs, settings.batch_size, settings.patience)
            predictions.append(_predict(torch, model, issued_inputs))
            path = checkpoint_dir / "annual" / str(year) / f"seed_{seed}.pt"
            _save_checkpoint(torch, model, path, scaling, prepared, training, asof, seed,
                             int(selected_row.hidden_size), float(selected_row.learning_rate),
                             refit_epochs, settings.batch_size)
            paths.append(str(path.relative_to(checkpoint_dir)))
        result.loc[issued, "gru_variance"] = np.mean(predictions, axis=0)
        result.loc[issued, "gru_refit_asof"] = asof
        checkpoints.extend(paths)
        annual_metadata.append({"evaluation_year": year, "refit_asof": str(asof.date()),
                                "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                                "training_observations": int(training.sum()), "checkpoints": paths})
    metadata = {"backend": "CUDA-required PyTorch", "device": str(device),
                "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda), "seeds": list(settings.seeds),
                "selected_hidden_size": int(selected_row.hidden_size),
                "selected_learning_rate": float(selected_row.learning_rate),
                "validation_qlike": float(selected_row.qlike), "refit_epochs": refit_epochs,
                "selected_best_epochs": selected_epochs,
                "epoch_rule": "median of selected-grid seeds' 1-based best validation epochs",
                "selection_rule": "QLIKE of the mean seed variance forecast",
                "validation_history": "selected initial model; retrospective threshold calibration",
                "initial_refit_asof": str(initial_asof.date()),
                "selected_initial_checkpoints": selected_paths, "annual_refits": annual_metadata,
                "checkpoints": checkpoints, "deterministic_algorithms": True,
                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}
    return result, tuning, metadata
