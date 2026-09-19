"""Runs the whole comparison and saves the tables and figures for the report."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import argparse
from dataclasses import asdict
from typing import Any

import matplotlib
import pandas as pd

import matplotlib.pyplot as plt

import evaluate
import tune
from config import FIGURE_DIRECTORY, RESULTS_DIRECTORY, CellType
import dataset, models, train

CELLS: list[CellType] = ["RNN", "LSTM", "GRU"]

QUICK_STAGES = {
    "capacity": [{"hidden_size": 16}, {"hidden_size": 32}],
    "sequence_length": [{"sequence_length": 12}, {"sequence_length": 36}],
}


def tune_every_model(
    series: pd.Series, base: train.Config, stages: dict, device
) -> dict[str, train.Config]:
    chosen: dict[str, train.Config] = {}

    for cell in CELLS:
        best, table = tune.search(series, cell, base=base, stages=stages, device=device)

        table.to_csv(RESULTS_DIRECTORY / f"tuning_{cell}.csv", index=False)

        chosen[cell] = best

        print(f"\n[{cell}] chosen - {best.label()}")

    return chosen


def evaluate_area(
    square_id: int, chosen: dict[str, train.Config], device
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """Train each model on one area and score them all on the test week."""

    series = dataset.load_series(square_id)

    scores: dict[str, Any] = {}

    timings: list[dict[str, Any]] = []

    predictions: list[pd.DataFrame] = []

    for cell in CELLS:
        print(f"\n[Square {square_id}] Training {cell}")

        outcome = tune.final_fit(series, chosen[cell], device=device)

        trained = outcome["trained"]

        scores[cell] = outcome["scores"]

        timings.append(
            {
                "model": cell,
                "square_id": square_id,
                "epochs_run": trained.epochs_run,
                "parameters": trained.parameters,
                "train_seconds": round(trained.train_seconds, 2),
                "sequence_length": trained.config.sequence_length,
                "seconds_per_epoch": round(
                    trained.train_seconds / trained.epochs_run, 3
                ),
                "predict_seconds_full_week": round(outcome["predict_seconds"], 4),
            }
        )

        predictions.append(
            pd.DataFrame(
                {
                    "timestamp": outcome["index"],
                    "square_id": square_id,
                    "model": cell,
                    "actual": outcome["actual"],
                    "predicted": outcome["predicted"],
                }
            )
        )

        print(
            f"\nMean Absolute Error - {outcome['scores']['mean_absolute_error']:.2f},  Root Mean Squared Error - {outcome['scores']['root_mean_squared_error']:.2f}"
        )

    # Baseline, for context rather than as a fourth model.
    first = predictions[0]

    naive = models.seasonal_naive(series, pd.DatetimeIndex(first["timestamp"]))

    scores["SeasonalNaive"] = evaluate.score(first["actual"].to_numpy(), naive)

    predictions.append(
        pd.DataFrame(
            {
                "predicted": naive,
                "square_id": square_id,
                "model": "SeasonalNaive",
                "actual": first["actual"],
                "timestamp": first["timestamp"],
            }
        )
    )

    table = evaluate.metrics_table(scores)

    table.to_csv(RESULTS_DIRECTORY / f"metrics_{square_id}.csv")

    return table, timings, pd.concat(predictions, ignore_index=True)


def save_figures(predictions: pd.DataFrame) -> int:
    """One actual-vs-predicted plot per model per area, nine in total."""

    count = 0

    for square_id in sorted(predictions["square_id"].unique()):
        for cell in CELLS:
            rows = predictions[
                (predictions["square_id"] == square_id)
                & (predictions["model"] == cell)
            ].sort_values("timestamp")

            figure, ax = plt.subplots(figsize=(13, 3.6))

            evaluate.plot_forecast(
                pd.DatetimeIndex(rows["timestamp"]),
                rows["actual"].to_numpy(),
                rows["predicted"].to_numpy(),
                f"Square {square_id}, {cell}, 16 to 22 December",
                ax,
            )

            figure.tight_layout()

            figure.savefig(
                FIGURE_DIRECTORY / f"forecast_{square_id}_{cell}.png", dpi=150
            )

            plt.close(figure)

            count += 1

    return count


def main() -> None:
    matplotlib.use("Agg")  # no screen when run as a script

    # Print whole tables. Without max_columns pandas hides columns behind an ellipsis.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--quick", action="store_true", help="short run to check the pipeline"
    )

    parser.add_argument("--areas", type=int, default=3)

    args = parser.parse_args()

    device = train.pick_device()

    train.warm_up(device)

    hardware = train.describe_hardware(device)

    print("hardware:", hardware)

    squares = dataset.top_squares(args.areas)

    print(f"busiest areas: {squares}")

    base = (
        train.Config(max_epochs=3, patience=2, batch_size=256)
        if args.quick
        else train.Config()
    )

    stages = QUICK_STAGES if args.quick else tune.STAGES

    chosen = tune_every_model(dataset.load_series(squares[0]), base, stages, device)

    all_timings: list[dict[str, Any]] = []

    all_predictions: list[pd.DataFrame] = []

    for square_id in squares:
        table, timings, predictions = evaluate_area(square_id, chosen, device)

        all_timings.extend(timings)

        all_predictions.append(predictions)

        print(f"\n{'==' * 10} Square {square_id} {'==' * 10}\n{table}")

    predictions = pd.concat(all_predictions, ignore_index=True)

    predictions.to_parquet(RESULTS_DIRECTORY / "predictions.parquet", index=False)

    pd.DataFrame(all_timings).to_csv(RESULTS_DIRECTORY / "timings.csv", index=False)

    summary = RESULTS_DIRECTORY / "summary.json"

    summary.write_text(
        json.dumps(
            {
                "areas": squares,
                "hardware": hardware,
                "tuned_on": squares[0],
                "chosen": {cell: asdict(config) for cell, config in chosen.items()},
            },
            indent=2,
        )
    )

    print(f"\nSaved {save_figures(predictions)} figures to {FIGURE_DIRECTORY}")

    print("\n" + pd.DataFrame(all_timings).to_string(index=False))


if __name__ == "__main__":
    main()
