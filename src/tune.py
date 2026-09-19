"""Hyperparameter search, done in stages so each choice builds on the last."""

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from config import CellType
from dataset import build_dataset, unscale
from train import Config, TrainedModel, pick_device, predict, train_model

import torch

# Tried one group at a time. The winner of each stage is carried into the next, so
# this is far cheaper than trying every combination and it shows the reasoning.
STAGES: dict[str, list[dict[str, Any]]] = {
    "sequence_length": [{"sequence_length": n} for n in (12, 36, 72, 144)],
    "scaling": [{"scaling": s} for s in ("Standard", "MinMax", "LogStandard")],
    "capacity": [
        {"hidden_size": 16},
        {"hidden_size": 32},
        {"hidden_size": 64},
        {"hidden_size": 64, "num_layers": 2, "dropout": 0.2},
    ],
    "optimisation": [
        {"learning_rate": 3e-3},
        {"learning_rate": 1e-3},
        {"learning_rate": 3e-4},
        {"learning_rate": 1e-3, "batch_size": 128},
    ],
}


def run_once(
    series: pd.Series, config: Config, device: torch.device
) -> tuple[dict[str, Any], TrainedModel]:
    """Train one configuration and score it on the validation week."""
    dataset = build_dataset(series, config.sequence_length, config.scaling)
    trained = train_model(dataset, config, device=device, verbose=False)

    scaled, _ = predict(trained.model, dataset["X_val"], device)
    predicted = unscale(scaled, dataset["mean"], dataset["std"], dataset["scaling"])
    scores = evaluate.score(dataset["val_actual"], predicted)

    row = {
        "cell": config.cell,
        "sequence_length": config.sequence_length,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "scaling": config.scaling,
        "val_MAE": scores["mean_absolute_error"],
        "val_RMSE": scores["root_mean_squared_error"],
        "parameters": trained.parameters,
        "epochs_run": trained.epochs_run,
        "train_seconds": round(trained.train_seconds, 1),
    }
    return row, trained


def search(
    series: pd.Series,
    cell: CellType,
    base: Config | None = None,
    stages: dict[str, list[dict[str, Any]]] | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[Config, pd.DataFrame]:
    """Walk the stages and keep whatever setting gave the lowest validation error."""
    device = device or pick_device()
    stages = stages or STAGES
    best_config = replace(base or Config(), cell=cell)
    best_mae = float("inf")
    rows: list[dict[str, Any]] = []

    for stage_name, options in stages.items():
        if verbose:
            print(f"\n[{cell}] tuning {stage_name}")

        for option in options:
            candidate = replace(best_config, **option)
            row, _ = run_once(series, candidate, device)
            better = row["val_MAE"] < best_mae
            rows.append({**row, "stage": stage_name, "tried": str(option), "kept": better})

            if verbose:
                mark = "  <- best so far" if better else ""
                print(f"  {str(option):<48} val MAE {row['val_MAE']:9.2f}{mark}")

            if better:
                best_mae, best_config = row["val_MAE"], candidate

    return best_config, pd.DataFrame(rows)


def final_fit(
    series: pd.Series, config: Config, device: torch.device | None = None, verbose: bool = False
) -> dict[str, Any]:
    """Train the chosen setup and score it on the December 16 to 22 test week."""
    device = device or pick_device()
    dataset = build_dataset(series, config.sequence_length, config.scaling)
    trained = train_model(dataset, config, device=device, verbose=verbose)

    scaled, seconds = predict(trained.model, dataset["X_test"], device)
    predicted = unscale(scaled, dataset["mean"], dataset["std"], dataset["scaling"])

    return {
        "trained": trained,
        "index": dataset["test_index"],
        "actual": dataset["test_actual"],
        "predicted": predicted,
        "scores": evaluate.score(dataset["test_actual"], predicted),
        "predict_seconds": seconds,
    }
