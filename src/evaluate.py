"""Error metrics and the actual-vs-predicted plot."""

from typing import TypedDict

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


class Scores(TypedDict):
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float
    mean_absolute_percentage_error_skipped: int


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mean_absolute_percentage_error(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[float, int]:
    floor = 0.01 * float(np.median(actual))

    usable = actual > floor

    errors = np.abs((actual[usable] - predicted[usable]) / actual[usable])

    return float(100 * errors.mean()), int((~usable).sum())


def score(actual: np.ndarray, predicted: np.ndarray) -> Scores:
    actual = np.asarray(actual, dtype=np.float64)

    predicted = np.asarray(predicted, dtype=np.float64)

    percentage, skipped = mean_absolute_percentage_error(actual, predicted)

    return Scores(
        mean_absolute_percentage_error_skipped=skipped,
        mean_absolute_percentage_error=round(percentage, 3),
        mean_absolute_error=round(mean_absolute_error(actual, predicted), 3),
        root_mean_squared_error=round(root_mean_squared_error(actual, predicted), 3),
    )


def metrics_table(scores_by_model: dict[str, Scores]) -> pd.DataFrame:
    """One row per model, best first, ready to paste into the report."""

    # Fixed column order so the headline errors come first and the table reads the
    # same every run. Without this the dict order decides, and pandas hides the
    # important columns behind an ellipsis when the frame is wide.
    columns = [
        "mean_absolute_error",
        "root_mean_squared_error",
        "mean_absolute_percentage_error",
        "mean_absolute_percentage_error_skipped",
    ]

    table = pd.DataFrame(scores_by_model).T[columns]

    table.index.name = "model"

    return table.sort_values("mean_absolute_error")


def plot_forecast(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str,
    ax: Axes,
) -> Axes:
    ax.plot(index, actual, linewidth=1.3, color="#1f3b73", label="actual")

    ax.plot(index, predicted, linewidth=1.1, color="#d1495b", label="predicted")

    ax.set_title(title)

    ax.set_ylabel("Internet activity")

    ax.legend(loc="upper right", frameon=False)

    ax.grid(alpha=0.25)

    ax.margins(x=0)

    return ax


def error_frame(
    index: pd.DatetimeIndex, actual: np.ndarray, predicted: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {"actual": actual, "predicted": predicted, "error": np.abs(actual - predicted)},
        index=index,
    )


def worst_hours(frame: pd.DataFrame, top: int = 5) -> pd.DataFrame:
    rolling = frame["error"].rolling(6).mean().sort_values(ascending=False)

    return rolling.head(top).to_frame("mean_error_over_hour")
