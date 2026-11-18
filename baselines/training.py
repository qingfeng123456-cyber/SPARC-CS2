"""Reproducible full-prefix baseline fitting, checkpointing and prediction.

Validation selects checkpoints by series/round macro joint NLL. Passing
val_dataset=None performs a fixed-epoch final fit and never reads test labels.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .models import (EVENT_CLASSES, INPUT_DIM, MARK_COUNT, OUTPUT_DIM, TAIL_CLASS, TIME_BINS,
                         make_model, parameter_count)
except ImportError:
    from models import (EVENT_CLASSES, INPUT_DIM, MARK_COUNT, OUTPUT_DIM, TAIL_CLASS, TIME_BINS,
                        make_model, parameter_count)


DEFAULTS = dict(
    max_epochs=60, min_epochs=10, patience=10, min_delta=0., batch_size=32,
    eval_batch_size=64, learning_rate=.001, min_lr=.00015, weight_decay=.015,
    scheduler_epochs=60, clip_grad_norm=2., amp=True, deterministic=True,
    progress_every_seconds=30., initialize_class_prior=True,
    prior_smoothing=1. / 512., allow_cpu_full=False,
)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def _rng_state() -> dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False, default=_json_value), encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(state), temporary)
    temporary.replace(path)


def _record_arrays(dataset: Any) -> Mapping[str, Any] | None:
    records = getattr(dataset, "records", None)
    return records if isinstance(records, Mapping) else None


def _dataset_signature(dataset: Any) -> str | None:
    if dataset is None:
        return None
    digest = hashlib.sha256(str(len(dataset)).encode())
    manifest = getattr(dataset, "manifest", None)
    if isinstance(manifest, Mapping):
        cache_identity = {k: manifest[k] for k in
                          ("schema", "dimensions", "n_classes", "files", "index_sha256")
                          if k in manifest}
        digest.update(json.dumps(cache_identity, sort_keys=True, separators=(",", ":"),
                                 default=_json_value).encode())
    records = _record_arrays(dataset)
    if records is not None:
        for key in ("series_id", "round_id", "bins", "marks", "cutoff_index", "global_index"):
            if key in records:
                digest.update(key.encode())
                a = np.asarray(records[key])
                digest.update(json.dumps(a.tolist(), ensure_ascii=False, separators=(",", ":")).encode())
    else:
        digest.update(type(dataset).__qualname__.encode())
    return digest.hexdigest()


def _training_prior(dataset: Any, smoothing: float) -> np.ndarray:
    if smoothing <= 0:
        raise ValueError("prior_smoothing must be positive for full output support")
    records = _record_arrays(dataset)
    if records is not None and all(k in records for k in ("bins", "marks", "series_id", "round_id")):
        bins, marks = np.asarray(records["bins"]), np.asarray(records["marks"])
        targets = np.where(bins == TIME_BINS, TAIL_CLASS, bins * MARK_COUNT + marks - 1)
        series, rounds = np.asarray(records["series_id"]).astype(str), np.asarray(records["round_id"]).astype(str)
        pairs = np.char.add(np.char.add(series, "/"), rounds)
        _, round_inverse, round_counts = np.unique(pairs, return_inverse=True, return_counts=True)
        first = np.unique(pairs, return_index=True)[1]
        round_series = series[first]
        series_values, series_rounds = np.unique(round_series, return_counts=True)
        counts_by_series = dict(zip(series_values, series_rounds))
        per_round = np.asarray([counts_by_series[s] for s in round_series], dtype=float)
        weights = len(dataset) / (len(series_values) * per_round[round_inverse] * round_counts[round_inverse])
    else:
        rows = [dataset[i] for i in range(len(dataset))]
        targets = np.asarray([int(row["y"]) for row in rows])
        weights = np.asarray([float(row["weight"]) for row in rows])
    if hasattr(dataset, "weights"):
        weights = np.asarray(dataset.weights, dtype=float)
    if weights.shape != (len(dataset),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid fitted-cohort weights for class-prior initialization")
    if np.any((targets < 0) | (targets >= OUTPUT_DIM)):
        raise ValueError("Training targets lie outside the shared joint event support")
    counts = np.bincount(targets.astype(np.int64), weights=weights, minlength=OUTPUT_DIM) + smoothing
    return counts / counts.sum()


def _make_loader(dataset: Any, batch_size: int, shuffle: bool, seed: int):
    try:
        from .data import make_loader
    except ImportError:
        from data import make_loader
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle, seed=seed)


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    out = {k: torch.as_tensor(batch[k]).to(device, non_blocking=True)
           for k in ("x", "lengths", "y", "weight", "index")}
    out["x"] = out["x"].float()
    out["lengths"], out["y"], out["index"] = (out[k].long() for k in ("lengths", "y", "index"))
    out["weight"] = out["weight"].float()
    if not torch.isfinite(out["x"]).all() or not torch.isfinite(out["weight"]).all():
        raise FloatingPointError("Nonfinite baseline input or training weight")
    if torch.any(out["weight"] <= 0):
        raise ValueError("Training weights must be positive")
    return out


def predict(model: nn.Module, loader: Any, device: str | torch.device,
            *, amp: bool = True) -> dict[str, np.ndarray]:
    """Return compact per-cutoff evidence, sorted by the dataset-local index.

    Full 10221-class probabilities remain reproducible from the checkpoint; they
    are deliberately not accumulated in RAM or serialized for every cutoff.
    """
    device = torch.device(device)
    model.to(device)
    was_training = model.training
    model.eval()
    collected: dict[str, list[np.ndarray]] = {k: [] for k in (
        "index", "y", "nll", "time_nll", "probability_a_b_10s", "time_mass", "tail")}
    marks = torch.arange(1, MARK_COUNT + 1, device=device)
    ab = torch.stack(((marks & 1) != 0, (marks & 2) != 0), dim=-1).float()
    started = last_progress = time.perf_counter()
    seen = 0
    try:
        with torch.inference_mode():
            for raw in loader:
                batch = _device_batch(raw, device)
                try:
                    with _autocast(device, amp):
                        logits = model(batch["x"], batch["lengths"])
                except torch.cuda.OutOfMemoryError as error:
                    raise RuntimeError(
                        f"CUDA OOM in prediction, batch={len(batch['y'])}, "
                        f"prefix={batch['x'].shape[1]}; reduce eval_batch_size. "
                        "No examples/history were truncated."
                    ) from error
                if logits.shape != (len(batch["y"]), OUTPUT_DIM) or not torch.isfinite(logits).all():
                    raise FloatingPointError("Invalid/nonfinite direct joint logits")
                logp = logits.float().log_softmax(-1)
                probability = logp.exp()
                if not torch.allclose(probability.sum(1), torch.ones(len(logp), device=device),
                                      atol=3e-5, rtol=3e-5):
                    raise FloatingPointError("Predicted joint probability does not sum to one")
                event_logp = logp[:, :EVENT_CLASSES].reshape(-1, TIME_BINS, MARK_COUNT)
                time_logp = event_logp.logsumexp(2)
                all_time_logp = torch.cat((time_logp, logp[:, -1:]), dim=1)
                row = torch.arange(len(logp), device=device)
                true_time = torch.where(batch["y"] == TAIL_CLASS, TIME_BINS,
                                        torch.div(batch["y"], MARK_COUNT, rounding_mode="floor"))
                output = dict(index=batch["index"], y=batch["y"],
                              nll=-logp[row, batch["y"]],
                              time_nll=-all_time_logp[row, true_time],
                              probability_a_b_10s=event_logp[:, :10].exp().sum(1) @ ab,
                              time_mass=time_logp.exp(), tail=probability[:, -1])
                for key, value in output.items():
                    collected[key].append(value.detach().cpu().numpy())
                seen += len(batch["y"])
                now = time.perf_counter()
                if now - last_progress >= 30.:
                    print(f"  PREDICT samples={seen} elapsed_s={now-started:.1f}", flush=True)
                    last_progress = now
    finally:
        model.train(was_training)
    if not collected["index"]:
        raise ValueError("Cannot predict an empty dataset")
    result = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(result["index"])) != len(result["index"]):
        raise ValueError("Prediction loader produced duplicate cutoff indices")
    order = np.argsort(result["index"], kind="stable")
    result = {key: value[order] for key, value in result.items()}
    true_bin = np.where(result["y"] == TAIL_CLASS, TIME_BINS, result["y"] // MARK_COUNT)
    true_mark = np.where(result["y"] == TAIL_CLASS, 0, result["y"] % MARK_COUNT + 1)
    binary = (true_bin[:, None] < 10) & ((true_mark[:, None] & np.asarray([1, 2])) != 0)
    result["brier_a_b_10s"] = np.square(result["probability_a_b_10s"] - binary)
    result["brier_ab10"] = result["brier_a_b_10s"].mean(1)
    return result


def _aggregate(dataset: Any, indices: np.ndarray, values: np.ndarray) -> Mapping[str, Any]:
    if hasattr(dataset, "aggregate"):
        return dataset.aggregate(indices, values)
    try:
        from .data import aggregate
    except ImportError:
        from data import aggregate
    return aggregate(dataset, indices, values)


def evaluate(model: nn.Module, dataset: Any, device: str | torch.device,
             *, batch_size: int = 64, seed: int = 0, amp: bool = True,
             loader: Any = None, return_predictions: bool = False) -> dict[str, Any]:
    if loader is None:
        loader = _make_loader(dataset, batch_size, False, seed)
    prediction = predict(model, loader, device, amp=amp)
    if len(prediction["index"]) != len(dataset):
        raise ValueError("Evaluation loader did not cover every cutoff exactly once")
    metrics: dict[str, Any] = {}
    for key in ("nll", "time_nll", "brier_ab10"):
        aggregate = dict(_aggregate(dataset, prediction["index"], prediction[key]))
        value = aggregate.get("macro_nll", aggregate.get("mean"))
        if value is None or not np.isfinite(float(value)):
            raise FloatingPointError(f"Nonfinite/missing macro score for {key}")
        metrics["macro_" + key] = float(value)
        metrics[key + "_aggregation"] = aggregate
    if return_predictions:
        metrics["predictions"] = prediction
    return metrics


def load_model(checkpoint_path: str | Path, device: str | torch.device = "cpu",
               *, return_state: bool = False):
    """Load a trusted checkpoint generated by this study and rebuild its model."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(state["model_name"], state["model_config"])
    model.load_state_dict(state["model"])
    model = model.to(device)
    return (model, state) if return_state else model


def train_model(train_dataset: Any, val_dataset: Any, model_name: str,
                model_config: Mapping[str, Any], run_dir: str | Path, seed: int,
                config: Mapping[str, Any] | None, device: str | torch.device,
                resume: bool = True) -> dict[str, Any]:
    """Fit one declared candidate; write best.pt/last.pt and visible epoch logs.

    With validation, all completed epochs may be selected; min_epochs constrains
    stopping only. Final refits (validation=None) use config['epochs'] exactly.
    An OOM fails clearly; this function never truncates samples or prefixes.
    """
    options = {**DEFAULTS, **dict(config or {})}
    device = torch.device(device)
    if len(train_dataset) == 0 or (val_dataset is not None and len(val_dataset) == 0):
        raise ValueError("Empty training or validation cohort")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; the full study is not run on CPU")
    if device.type == "cpu" and len(train_dataset) > 512 and not options["allow_cpu_full"]:
        raise RuntimeError("CPU is reserved for small checks; choose CUDA for the full study")
    requested_epochs = options.get("epochs")
    epochs = int(options["max_epochs"] if requested_epochs is None else requested_epochs)
    if epochs < 1 or int(options["batch_size"]) < 1:
        raise ValueError("epochs and batch_size must be positive")
    if val_dataset is not None and epochs < int(options["min_epochs"]):
        raise ValueError("Validation fitting requires max_epochs >= min_epochs")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    signature = dict(train=_dataset_signature(train_dataset), val=_dataset_signature(val_dataset))
    seed_everything(int(seed), bool(options["deterministic"]))
    model = make_model(model_name, model_config).to(device)
    if options["initialize_class_prior"]:
        prior = _training_prior(train_dataset, float(options["prior_smoothing"]))
        model.initialize_class_prior(torch.as_tensor(prior))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(options["learning_rate"]),
                                  weight_decay=float(options["weight_decay"]))
    period = int(options["scheduler_epochs"])
    if period < 1 or float(options["learning_rate"]) <= 0:
        raise ValueError("Invalid learning-rate schedule")
    floor = float(options["min_lr"]) / float(options["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: floor + (1. - floor) * (1. + math.cos(math.pi * min(step, period) / period)) / 2.)
    fp16 = device.type == "cuda" and bool(options["amp"]) and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
    history: list[dict[str, Any]] = []
    start_epoch, best_epoch, bad_epochs, total_updates = 1, 0, 0, 0
    best_metric: float | None = None
    if resume and last_path.exists():
        previous = torch.load(last_path, map_location="cpu", weights_only=False)
        if (previous["model_name"] != model_name or previous["model_config"] != dict(model_config)
                or previous["seed"] != seed or previous["dataset_signature"] != signature):
            raise ValueError("Resume checkpoint does not match model, seed, or data cohort")
        fixed_keys = ("batch_size", "learning_rate", "min_lr", "weight_decay", "scheduler_epochs",
                      "amp", "clip_grad_norm", "initialize_class_prior", "prior_smoothing", "deterministic")
        if any(previous["config"].get(k) != options[k] for k in fixed_keys):
            raise ValueError("Resume attempted with changed optimization configuration")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        _restore_rng(previous["rng"])
        start_epoch = previous["epoch"] + 1
        best_epoch, best_metric, bad_epochs = (previous[k] for k in ("best_epoch", "best_metric", "bad_epochs"))
        total_updates, history = previous["total_updates"], previous["history"]
        print(f"RESUME {model_name} seed={seed} completed={start_epoch-1} best={best_epoch}", flush=True)
    elif not resume and (last_path.exists() or best_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing run without resume: {run_dir}")
    metadata = dict(model_name=model_name, model_config=dict(model_config), config=options,
                    seed=seed, parameters=parameter_count(model), dataset_signature=signature,
                    train_cutoffs=len(train_dataset), validation_cutoffs=0 if val_dataset is None else len(val_dataset),
                    selection="fixed_final_epoch" if val_dataset is None else "series_round_macro_joint_nll",
                    output_dim=OUTPUT_DIM,
                    prefix_policy=("last_valid_matched_summary" if model.__class__.__name__ == "MLPJoint"
                                   else "complete_observed_prefix"),
                    execution_backend=dict(gru_cudnn=getattr(model, "use_cudnn", None)),
                    torch_version=torch.__version__, device=str(device))
    _write_json(run_dir / "run_config.json", metadata)
    validation_loader = None if val_dataset is None else _make_loader(
        val_dataset, int(options["eval_batch_size"]), False, int(seed))
    print(f"TRAIN {model_name} seed={seed} params={metadata['parameters']:,} "
          f"cutoffs={len(train_dataset):,} epochs<= {epochs} device={device}", flush=True)
    stop_reason = "fixed_epochs" if val_dataset is None else "max_epochs"
    if (val_dataset is not None and start_epoch > int(options["min_epochs"])
            and bad_epochs >= int(options["patience"])):
        epochs = start_epoch - 1
        stop_reason = "early_stopping"
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        loader = _make_loader(train_dataset, int(options["batch_size"]), True, int(seed) + epoch)
        if hasattr(loader, "batch_sampler") and hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)
        seen, epoch_updates, loss_sum = 0, 0, 0.
        visited = np.zeros(len(train_dataset), dtype=bool)
        last_progress = started
        lr = float(optimizer.param_groups[0]["lr"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for raw in loader:
            indices = np.asarray(raw["index"], dtype=np.int64)
            if (np.any(indices < 0) or np.any(indices >= len(visited))
                    or len(np.unique(indices)) != len(indices) or visited[indices].any()):
                raise ValueError("Training loader repeated or invalid cutoff indices")
            visited[indices] = True
            batch = _device_batch(raw, device)
            try:
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, bool(options["amp"])):
                    logits = model(batch["x"], batch["lengths"])
                    loss = (F.cross_entropy(logits.float(), batch["y"], reduction="none") * batch["weight"]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, update {epoch_updates}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(options["clip_grad_norm"]), error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            except torch.cuda.OutOfMemoryError as error:
                raise RuntimeError(
                    f"CUDA OOM in {model_name}, batch={len(batch['y'])}, prefix={batch['x'].shape[1]}. "
                    "Reduce batch_size and start a fresh run; no examples/history were truncated."
                ) from error
            count = len(batch["y"])
            seen += count
            epoch_updates += 1
            total_updates += 1
            loss_sum += float(loss.detach().item()) * count
            now = time.perf_counter()
            if now - last_progress >= float(options["progress_every_seconds"]):
                memory = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.
                print(f"  {model_name} epoch={epoch}/{epochs} samples={seen}/{len(train_dataset)} "
                      f"updates={epoch_updates} lr={lr:.6g} peak_MB={memory:.0f} "
                      f"elapsed_s={now-started:.1f}", flush=True)
                last_progress = now
        if seen != len(train_dataset):
            raise ValueError(f"Training loader covered {seen}/{len(train_dataset)} samples")
        validation = None if val_dataset is None else evaluate(
            model, val_dataset, device, batch_size=int(options["eval_batch_size"]), seed=int(seed),
            amp=bool(options["amp"]), loader=validation_loader)
        metric = None if validation is None else float(validation["macro_nll"])
        improved = (val_dataset is None or best_metric is None
                    or metric < best_metric - float(options["min_delta"]))
        if improved:
            best_metric, best_epoch, bad_epochs = metric, epoch, 0
        else:
            bad_epochs += 1
        scheduler.step()
        elapsed = time.perf_counter() - started
        memory = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.
        row = dict(epoch=epoch, samples=seen, updates=epoch_updates, total_updates=total_updates,
                   training_weighted_nll=loss_sum / seen, validation_macro_nll=metric,
                   learning_rate=lr, peak_cuda_mb=memory, seconds=elapsed,
                   best_epoch=best_epoch, improved=improved)
        history.append(row)
        state = dict(**metadata, epoch=epoch, best_epoch=best_epoch, best_metric=best_metric,
                     bad_epochs=bad_epochs, total_updates=total_updates, history=history,
                     model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     scaler=scaler.state_dict(), rng=_rng_state())
        _save_checkpoint(last_path, state)
        if improved:
            _save_checkpoint(best_path, state)
        _write_json(run_dir / "history.json", history)
        if validation is not None:
            _write_json(run_dir / "last_validation.json", validation)
        with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"EPOCH {model_name} seed={seed} {epoch}/{epochs} samples={seen} updates={epoch_updates} "
              f"train={loss_sum/seen:.6f} val_macro={metric if metric is not None else 'none'} "
              f"best_epoch={best_epoch} lr={lr:.6g} peak_MB={memory:.0f} seconds={elapsed:.1f}", flush=True)
        if val_dataset is not None and epoch >= int(options["min_epochs"]) and bad_epochs >= int(options["patience"]):
            stop_reason = "early_stopping"
            break
    if not best_path.exists():
        raise RuntimeError("Training produced no selected checkpoint")
    result = dict(checkpoint_path=str(best_path), selected_checkpoint=str(best_path),
                  last_checkpoint=str(last_path), selected_epoch=int(best_epoch),
                  best_validation_macro_nll=best_metric, completed_epochs=len(history),
                  total_updates=total_updates, stop_reason=stop_reason, metadata=metadata)
    _write_json(run_dir / "selection.json", result)
    return result


def numerical_checks(device: str = "cpu") -> dict[str, Any]:
    """Small implementation checks; never a baseline performance experiment."""
    torch.set_num_threads(min(2, torch.get_num_threads()))
    seed_everything(2718)
    target = torch.device(device)
    x = torch.randn(3, 9, INPUT_DIM, device=target)
    lengths = torch.tensor([9, 4, 7], device=target)
    polluted = F.pad(x, (0, 0, 0, 4), value=19.)
    results: dict[str, Any] = {}
    for name in ("mlp", "gru", "tcn", "transformer"):
        options = dict(width=32, dropout=0.)
        if name == "tcn":
            options["num_layers"] = 4
        elif name in ("gru", "transformer"):
            options["num_layers"] = 1
        model = make_model(name, options).to(target)
        model.eval()
        with torch.no_grad():
            output = model(x, lengths)
            padded = model(polluted, lengths)
            torch.testing.assert_close(output, padded, atol=3e-5, rtol=3e-5)
            p = output.softmax(-1)
            torch.testing.assert_close(p.sum(1), torch.ones(3, device=target), atol=2e-6, rtol=2e-6)
        model.train()
        loss = F.cross_entropy(model(x, lengths), torch.tensor([0, 100, TAIL_CLASS], device=target))
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not grads or not all(torch.isfinite(gradient).all() for gradient in grads):
            raise AssertionError("Missing/nonfinite baseline gradients")
        batch = dict(x=x.cpu(), lengths=lengths.cpu(), y=torch.tensor([0, 100, TAIL_CLASS]),
                     weight=torch.ones(3), index=torch.tensor([2, 0, 1]))
        evidence = predict(model, [batch], target, amp=False)
        np.testing.assert_array_equal(evidence["index"], [0, 1, 2])
        np.testing.assert_array_equal(evidence["y"], [100, TAIL_CLASS, 0])
        np.testing.assert_allclose(evidence["time_mass"].sum(1) + evidence["tail"], 1., atol=2e-6)
        results[name] = dict(parameters=parameter_count(model), loss=float(loss.detach().cpu()),
                             pad_invariance=True, normalization=True, finite_gradients=True,
                             sorted_predictions=True)
    return results


if __name__ == "__main__":
    print(json.dumps(numerical_checks(), indent=2))
