"""CUDA-only two-layer GRU for a closer, univariate paper replication."""

from pathlib import Path

import numpy as np
import pandas as pd

from .gru_forecasting import _annual_splits, _gpu_inputs, _predict, _require_cuda


def _cv_folds(rows, initial_training, folds):
    """Expanding folds, purged so training labels exist at validation issue time."""
    positions = np.flatnonzero(initial_training)
    if folds < 2 or len(positions) < folds + 1:
        raise ValueError("Chronological validation needs at least two nonempty folds.")
    chunks = np.array_split(positions, folds + 1)
    result = []
    for number in range(1, folds + 1):
        validation = np.zeros(len(rows), dtype=bool)
        validation[chunks[number]] = True
        training = np.zeros(len(rows), dtype=bool)
        training[np.concatenate(chunks[:number])] = True
        training &= (rows.return_date <= rows.index[chunks[number][0]]).to_numpy()
        if not training.any():
            raise ValueError("Purged chronological validation has no available training labels.")
        result.append((training, validation))
    return result


def _training_scale(prepared, training):
    inputs = np.sqrt(prepared.inputs[training])
    targets = np.sqrt(prepared.rows.realized_bv.to_numpy(dtype=float)[training])
    if not len(inputs) or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Scaling requires finite training features and targets.")
    deviation = inputs.std(axis=(0, 1))
    return {"feature_mean": inputs.mean(axis=(0, 1)).tolist(),
            "feature_std": np.where(deviation > 0, deviation, 1.).tolist(),
            "target_scale": max(float(targets.mean()), 1e-12)}


def _model(torch, hidden_size, target_scale, dropout, device):
    class VolatilityGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(1, hidden_size, num_layers=2,
                                    batch_first=True, dropout=dropout)
            self.output = torch.nn.Linear(hidden_size, 1)
            self.register_buffer("target_scale", torch.tensor(target_scale, dtype=torch.float32))

        def forward(self, inputs):
            # Native CUDA avoids a reproducible Windows cuDNN shutdown crash.
            # Parameters, activations, and gradients still reside on CUDA:0.
            with torch.backends.cudnn.flags(enabled=False):
                _, hidden = self.gru(inputs)
            return (torch.nn.functional.softplus(self.output(hidden[-1]).squeeze(-1))
                    * self.target_scale + 1e-12)

    model = VolatilityGRU().to(device)
    if any(parameter.device != device for parameter in model.parameters()):
        raise RuntimeError("Every paper GRU parameter must reside on CUDA:0.")
    return model


def _fit(torch, prepared, training, validation, scaling, hidden_size, seed, epochs, settings):
    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = _gpu_inputs(torch, np.sqrt(prepared.inputs[training]), scaling, device)
    targets = torch.as_tensor(np.sqrt(prepared.rows.realized_bv.to_numpy()[training]),
                              dtype=torch.float32, device=device)
    validation_inputs = (None if validation is None else
                         _gpu_inputs(torch, np.sqrt(prepared.inputs[validation]), scaling, device))
    validation_targets = (None if validation is None else
                          torch.as_tensor(np.sqrt(prepared.rows.realized_bv.to_numpy()[validation]),
                                          dtype=torch.float32, device=device))
    model = _model(torch, hidden_size, scaling["target_scale"], settings.dropout, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate,
                                 weight_decay=settings.weight_decay)
    best_loss, best_epoch, best_state, stale = np.inf, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(inputs), device=device)
        for offset in range(0, len(order), settings.batch_size):
            selected = order[offset:offset + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs[selected]) - targets[selected]) / scaling["target_scale"]).square().mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Paper GRU training produced nonfinite MSE on CUDA:0.")
            loss.backward()
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                score = float(((model(validation_inputs) - validation_targets)
                               / scaling["target_scale"]).square().mean().item())
            if not np.isfinite(score):
                raise RuntimeError("Paper GRU validation produced nonfinite MSE on CUDA:0.")
            if score < best_loss:
                best_loss, best_epoch, stale = score, epoch, 0
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= settings.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def paper_gru_forecasts(prepared, settings, calendar, checkpoint_dir, progress=print):
    """Predict sqrt(BV), select by its MSE, and freeze annual-fit epochs.

    This interpretation matches the paper's Table 1 scaling, 22 * sqrt(BV).
    Seed forecasts are averaged in volatility space before squaring for storage.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise ValueError("Use a new paper GRU checkpoint directory; existing files are never replaced.")
    rows, inputs = prepared.rows, prepared.inputs
    if (inputs.shape != (len(rows), settings.lookback, 1) or not np.isfinite(inputs).all()
            or (inputs < 0).any() or not rows.index.is_unique or not rows.index.is_monotonic_increasing):
        raise ValueError("Paper GRU requires ordered dates and finite nonnegative univariate BV histories.")
    targets = rows.realized_bv.to_numpy(dtype=float)
    available = np.isfinite(targets)
    if np.isinf(targets).any() or (targets[available] < 0).any():
        raise ValueError("BV labels must be nonnegative finite values or missing.")
    initial_training = ((rows.return_date <= pd.Timestamp(settings.training_end)).to_numpy()
                        & available)
    folds = _cv_folds(rows, initial_training, settings.folds)
    splits = _annual_splits(rows, calendar, settings.evaluation_start)
    if not settings.hidden_sizes or not settings.seeds:
        raise ValueError("Paper GRU needs at least one hidden size and one seed.")
    torch, device = _require_cuda()
    fold_metadata = [{"fold": number,
                      "training_observations": int(training.sum()),
                      "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                      "validation_first_issue": str(rows.index[validation][0].date()),
                      "validation_last_target": str(rows.loc[validation, "return_date"].max().date()),
                      "validation_observations": int(validation.sum())}
                     for number, (training, validation) in enumerate(folds, 1)]
    records = []
    for hidden_size in settings.hidden_sizes:
        squared_error, observations, best_epochs = 0., 0, []
        for number, (training, validation) in enumerate(folds, 1):
            scaling = _training_scale(prepared, training)
            validation_inputs = _gpu_inputs(torch, np.sqrt(inputs[validation]), scaling, device)
            predictions = []
            for seed in settings.seeds:
                progress(f"Paper GRU validation: hidden={hidden_size}, fold={number}, seed={seed}")
                model, epoch = _fit(torch, prepared, training, validation, scaling,
                                    hidden_size, seed, settings.max_epochs, settings)
                predictions.append(_predict(torch, model, validation_inputs))
                best_epochs.append(epoch)
            errors = np.mean(predictions, axis=0) - np.sqrt(targets[validation])
            squared_error += float(errors @ errors)
            observations += len(errors)
        records.append({"hidden_size": hidden_size, "mse": squared_error / observations,
                        "validation_observations": observations, "best_epochs": best_epochs})
    tuning = pd.DataFrame(records)
    selected = int(tuning.mse.argmin())
    tuning["selected"] = np.arange(len(tuning)) == selected
    hidden_size = int(tuning.iloc[selected].hidden_size)
    epochs = max(1, int(np.median(tuning.iloc[selected].best_epochs)))
    result = rows.copy()
    result["gru_bv"], result["gru_refit_asof"] = np.nan, pd.NaT
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoints, annual_metadata = [], []
    for year, asof, training, issued in splits:
        training &= available
        if not training.any():
            raise ValueError("Annual paper GRU refits need available BV labels.")
        scaling = _training_scale(prepared, training)
        issued_inputs = _gpu_inputs(torch, np.sqrt(inputs[issued]), scaling, device)
        predictions, paths = [], []
        for seed in settings.seeds:
            progress(f"Paper GRU annual refit: year={year}, seed={seed}, epochs={epochs}")
            model, _ = _fit(torch, prepared, training, None, scaling,
                            hidden_size, seed, epochs, settings)
            predictions.append(_predict(torch, model, issued_inputs))
            path = checkpoint_dir / str(year) / f"seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                       **scaling, "hidden_size": hidden_size, "num_layers": 2,
                       "lookback": settings.lookback, "seed": seed, "epochs": epochs,
                       "dropout": settings.dropout, "weight_decay": settings.weight_decay,
                       "learning_rate": settings.learning_rate, "batch_size": settings.batch_size,
                       "refit_asof": str(asof.date()),
                       "train_last_target": str(rows.loc[training, "return_date"].max().date()),
                       "training_observations": int(training.sum()), "device": str(device),
                       "input_units": "sqrt(raw daily BV)", "target_units": "sqrt(raw daily BV)",
                       "torch_version": str(torch.__version__), "rnn_backend": "native CUDA; cuDNN disabled"}
            with path.open("xb") as destination:
                torch.save(payload, destination)
            paths.append(str(path.relative_to(checkpoint_dir)))
        result.loc[issued, "gru_bv"] = np.mean(predictions, axis=0) ** 2
        result.loc[issued, "gru_refit_asof"] = asof
        checkpoints.extend(paths)
        annual_metadata.append({"evaluation_year": year, "refit_asof": str(asof.date()),
                                "train_last_target": payload["train_last_target"],
                                "training_observations": int(training.sum()), "checkpoints": paths})
    metadata = {"backend": "CUDA-required PyTorch", "device": str(device),
                "gpu": torch.cuda.get_device_name(0), "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda), "num_layers": 2,
                "rnn_backend": "native CUDA; cuDNN disabled to avoid Windows shutdown failure",
                "selected_hidden_size": hidden_size, "learning_rate": settings.learning_rate,
                "dropout": settings.dropout, "weight_decay": settings.weight_decay,
                "seeds": list(settings.seeds), "validation_mse": float(tuning.iloc[selected].mse),
                "selected_best_epochs": tuning.iloc[selected].best_epochs, "refit_epochs": epochs,
                "selection_rule": "pooled daily-volatility MSE of seed-mean forecasts across expanding folds",
                "epoch_rule": "median selected-grid fold/seed best epoch, rounded down",
                "training_loss": "sqrt(BV) MSE divided by squared training-mean sqrt(BV)",
                "forecast_units": "seed-mean sqrt(BV) squared for gru_bv storage",
                "folds": fold_metadata, "annual_refits": annual_metadata, "checkpoints": checkpoints,
                "deterministic_algorithms": True}
    return result, tuning, metadata
