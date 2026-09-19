"""The three recurrent models and a simple baseline to compare them against."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import STEPS_PER_DAY, CellType

# All three read a window of past values. They differ only in the cell:
#   RNN   no gates, fewest parameters, forgets the most
#   LSTM  three gates plus a cell state, remembers longest, most parameters
#   GRU   two gates, no cell state, sits between the other two
CELLS: dict[CellType, type[nn.Module]] = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}


class Forecaster(nn.Module):
    """Takes a window of past traffic, predicts the next value."""

    def __init__(
        self,
        cell: CellType = "LSTM",
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.cell = cell

        self.recurrent = CELLS[cell](
            input_size=1,
            batch_first=True,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, steps, hidden)
        output, _ = self.recurrent(x)

        # only the final step matters
        last_step = output[:, -1, :]

        return self.readout(last_step).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    cell: CellType,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    seed: int | None = None,
) -> Forecaster:
    if seed is not None:
        torch.manual_seed(seed)

    return Forecaster(cell, hidden_size, num_layers, dropout)


def persistence(series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """Baseline: guess that the next value equals the one just seen.

    This is the baseline that matters for one step ahead forecasting. Traffic
    correlates 0.987 with itself ten minutes earlier, so simply repeating the last
    reading is already accurate and any model has to beat it to be worth training.
    """

    positions = series.index.get_indexer(test_index)

    return series.to_numpy(dtype=np.float32)[positions - 1]


def seasonal_naive(series: pd.Series, test_index: pd.DatetimeIndex) -> np.ndarray:
    """A naive model that just guesses that today looks exactly like yesterday."""

    positions = series.index.get_indexer(test_index)

    return series.to_numpy(dtype=np.float32)[positions - STEPS_PER_DAY]
