"""A quick side experiment: would a convolutional model do better here?

The main study compares three recurrent cells. This script asks a different question,
whether a dilated causal convolution, which reads the whole window at once instead of
one step at a time, does any better on the same task.

It is deliberately small. Nothing in the main pipeline is changed. The convolution is
written to present the same interface as an RNN layer, so it drops straight into the
existing Forecaster, training loop, scaling and metrics. That means the comparison is
fair for the same reason the recurrent comparison is fair: only the cell differs.

Usage:
    python src/cnn_test.py --data-dir data/processed/colab
    python src/cnn_test.py --data-dir /content/data --quick
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset
import evaluate
import models
import tune
from train import Config, describe_hardware, pick_device, warm_up

# Five seed means from the main notebook run, for comparison.
RECURRENT_RESULTS = pd.DataFrame({
    "Persistence": {5161: 92.80, 5059: 81.52, 5259: 75.97},
    "RNN": {5161: 86.08, 5059: 73.96, 5259: 66.67},
    "LSTM": {5161: 84.58, 5059: 67.57, 5259: 66.00},
    "GRU": {5161: 83.70, 5059: 69.12, 5259: 67.33},
})


class ConvCell(nn.Module):
    """Stacked dilated causal convolutions, shaped like an RNN layer.

    A recurrent cell walks the window one step at a time. This reads the whole window
    at once through convolutions whose gaps double each layer, so four layers with a
    kernel of three can see 31 steps back. Causal means each position only ever sees
    the past, which is enforced by padding on the left and trimming the right.

    Returns (output, None) with output shaped (batch, steps, hidden) so that the
    existing Forecaster can take the final step exactly as it does for an RNN.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 4,
        batch_first: bool = True,
        dropout: float = 0.0,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.dilations = [2**layer for layer in range(num_layers)]

        channels = [input_size] + [hidden_size] * num_layers

        self.convolutions = nn.ModuleList(
            nn.Conv1d(channels[i], channels[i + 1], kernel_size, dilation=dilation)
            for i, dilation in enumerate(self.dilations)
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

    def receptive_field(self) -> int:
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Conv1d wants (batch, channels, steps), the loaders give (batch, steps, channels)
        h = x.transpose(1, 2)

        for convolution, dilation in zip(self.convolutions, self.dilations):
            padding = (self.kernel_size - 1) * dilation

            # Pad only on the left so no position can see its own future
            h = F.relu(convolution(F.pad(h, (padding, 0))))

            if self.dropout is not None:
                h = self.dropout(h)

        return h.transpose(1, 2), None


def layers_needed(sequence_length: int, kernel_size: int = 3) -> int:
    """How many dilated layers before the model can see the whole window.

    Each layer doubles its gap, so the reach grows roughly twice as fast per layer.
    Fewer layers than this and the convolution silently ignores the oldest part of
    its input, which would make the comparison against the recurrent models unfair.
    """

    layers, reach = 0, 1

    while reach < sequence_length:
        reach += (kernel_size - 1) * 2**layers
        layers += 1

    return layers


def register_cnn() -> None:
    """Make "CNN" available to build_model without editing the main modules."""

    models.CELLS["CNN"] = ConvCell


def run_seeds(
    square_id: int, config: Config, seeds: list[int], device
) -> tuple[list[float], float]:
    """Train the same setup on several seeds, return the errors and total time."""

    series = dataset.load_series(square_id)
    errors = []

    start = time.perf_counter()
    for seed in seeds:
        outcome = tune.final_fit(series, replace(config, seed=seed), device=device)
        errors.append(outcome["scores"]["mean_absolute_error"])

    return errors, time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/processed/colab",
        help="folder holding internet_matrix.npy and its companions",
    )
    parser.add_argument("--areas", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="two seeds and a short budget, only to check the script runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset.set_data_dir(args.data_dir)
    register_cnn()

    device = pick_device()
    warm_up(device)
    print("hardware:", describe_hardware(device))

    seeds = [0, 1] if args.quick else list(range(args.seeds))

    if args.quick:
        print("\n  QUICK MODE: the budget is too short for the model to converge.")
        print("  The recurrent models needed 19 to 45 epochs. Do not read results from this.\n")

    # Same window as the LSTM chose, with enough depth to actually see all of it.
    sequence_length = 144

    config = Config(
        cell="CNN",
        sequence_length=sequence_length,
        hidden_size=64,
        num_layers=layers_needed(sequence_length),
        scaling="Standard",
        max_epochs=10 if args.quick else 60,
        patience=3 if args.quick else 8,
    )

    field = ConvCell(num_layers=config.num_layers).receptive_field()
    assert field >= config.sequence_length, "the convolution cannot see its whole window"
    parameters = models.build_model("CNN", config.hidden_size, config.num_layers).parameter_count()
    print(f"\nCNN: {config.num_layers} dilated layers, receptive field {field} steps, "
          f"{parameters:,} parameters")
    print(f"seeds: {seeds}\n")

    squares = dataset.top_squares(args.areas)
    rows = []

    for square_id in squares:
        errors, seconds = run_seeds(square_id, config, seeds, device)
        mean = sum(errors) / len(errors)
        spread = pd.Series(errors).std()
        rows.append({
            "square_id": square_id,
            "CNN": round(mean, 2),
            "CNN_std": round(float(spread), 2),
            "seconds_total": round(seconds, 1),
        })
        print(f"square {square_id}: CNN mean absolute error {mean:.2f} (sd {spread:.2f}), "
              f"{seconds:.0f}s for {len(seeds)} seeds")

    results = pd.DataFrame(rows).set_index("square_id")
    comparison = RECURRENT_RESULTS.join(results[["CNN", "CNN_std"]])

    print("\n\nMean absolute error, five seed means for the recurrent models\n")
    print(comparison.to_string())

    best_recurrent = comparison[["RNN", "LSTM", "GRU"]].min(axis=1)
    verdict = pd.DataFrame({
        "best recurrent": best_recurrent.round(2),
        "CNN": comparison["CNN"],
        "CNN better by": (best_recurrent - comparison["CNN"]).round(2),
        "against persistence": (
            100 * (comparison["Persistence"] - comparison["CNN"]) / comparison["Persistence"]
        ).round(1),
    })
    print("\n\nHow the CNN compares\n")
    print(verdict.to_string())
    print("\n  positive 'CNN better by' means the convolution won that area")
    print("  'against persistence' is the percentage of the naive error it removed")


if __name__ == "__main__":
    main()
