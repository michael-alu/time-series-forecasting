"""Load a single area's traffic and turn it into training windows."""

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from config import ScalingType

DATA_DIRECTORY: Path = config.COLAB_DIRECTORY

SCALING_OPTIONS: tuple[str, ...] = ("None", "Standard", "MinMax", "LogStandard")


class Dataset(TypedDict):
    """Everything a model needs for one area."""

    X_train: np.ndarray  # (samples, sequence_length, 1)
    y_train: np.ndarray  # (samples,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_index: pd.DatetimeIndex
    test_actual: np.ndarray  # unscaled, for the metrics
    val_actual: np.ndarray
    mean: float
    std: float
    scaling: ScalingType


def set_data_dir(path: str | Path) -> None:
    """Point the loaders somewhere else, used for Google Drive on Colab."""

    global DATA_DIRECTORY

    DATA_DIRECTORY = Path(path)


def load_matrix() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    """The whole dataset as a (time steps, areas) grid."""

    matrix = np.load(DATA_DIRECTORY / "internet_matrix.npy", mmap_mode="r")

    saved = np.load(DATA_DIRECTORY / "matrix_index.npz")

    index = pd.to_datetime(saved["timestamps"], unit="ns", utc=True).tz_convert(
        config.TIMEZONE
    )

    return matrix, index, saved["square_ids"]


def load_square_totals() -> pd.DataFrame:
    """One row per area, already sorted by total traffic."""

    return pd.read_parquet(DATA_DIRECTORY / "square_totals.parquet")


def top_squares(n: int = 3) -> list[int]:
    """The n busiest areas."""

    return load_square_totals()["square_id"].head(n).tolist()


def load_series(square_id: int) -> pd.Series:
    """One area's traffic over the full two months."""
    matrix, index, square_ids = load_matrix()

    column = int(np.where(square_ids == square_id)[0][0])

    return pd.Series(
        np.asarray(matrix[:, column]), index=index, name=f"square_{square_id}"
    )


def check_scaling(scaling: str) -> None:
    """Catch a misspelled name instead of silently falling back to Standard."""

    if scaling not in SCALING_OPTIONS:
        raise ValueError(f"scaling must be one of {SCALING_OPTIONS}, got {scaling!r}")


def scale(
    values: np.ndarray, mean: float, std: float, scaling: ScalingType
) -> np.ndarray:
    """Put values on a small, roughly centred range so training is stable."""
    check_scaling(scaling)

    if scaling == "LogStandard":
        values = np.log1p(values)

    if scaling == "None":
        return values.astype(np.float32)

    return ((values - mean) / std).astype(np.float32)


def unscale(
    values: np.ndarray, mean: float, std: float, scaling: ScalingType
) -> np.ndarray:
    """Undo scale(), so predictions are back in real traffic units."""
    if scaling == "None":
        return values.astype(np.float32)

    values = values * std + mean

    if scaling == "LogStandard":
        values = np.expm1(values)

    return values.astype(np.float32)


def scaling_stats(train: np.ndarray, scaling: ScalingType) -> tuple[float, float]:
    """
    Work out the two scaling numbers from the training data only.

    Using the whole series here would leak information from the weeks we test on.
    """

    check_scaling(scaling)

    if scaling == "LogStandard":
        train = np.log1p(train)

    if scaling == "MinMax":
        return float(train.min()), float(train.max() - train.min()) or 1.0

    return float(train.mean()), float(train.std()) or 1.0


def make_windows(
    values: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cut a series into (past window -> next value) pairs.

    With sequence_length 3, [1,2,3,4,5] becomes X=[[1,2,3],[2,3,4]] and y=[4,5].
    """

    windows = np.lib.stride_tricks.sliding_window_view(values, sequence_length)

    X = windows[:-1, :, None].astype(np.float32)

    y = values[sequence_length:].astype(np.float32)

    return X, y


def windows_for(
    series: pd.Series,
    scaled: pd.Series,
    start: str,
    end: str,
    sequence_length: int = 144,
) -> tuple[np.ndarray, np.ndarray]:
    first = series.index.get_loc(series.loc[start:end].index[0])

    last = series.index.get_loc(series.loc[start:end].index[-1])

    return make_windows(
        scaled.to_numpy()[first - sequence_length : last + 1], sequence_length
    )


def build_dataset(
    series: pd.Series,
    sequence_length: int = 144,
    scaling: ScalingType = "Standard",
) -> Dataset:
    """
    Split by date, scale, and cut into windows.

    Validation and test windows are allowed to start in the days before them. That is
    fine for one step ahead forecasting because those values really have been seen by
    the time we predict.
    """

    train = series.loc[: config.TRAIN_END]

    mean, std = scaling_stats(train.to_numpy(), scaling)

    scaled = pd.Series(scale(series.to_numpy(), mean, std, scaling), index=series.index)

    X_train, y_train = make_windows(
        scaled.loc[: config.TRAIN_END].to_numpy(), sequence_length
    )

    X_val, y_val = windows_for(
        series, scaled, config.VAL_START, config.VAL_END, sequence_length
    )

    X_test, y_test = windows_for(
        series, scaled, config.TEST_START, config.TEST_END, sequence_length
    )

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_index=series.loc[config.TEST_START : config.TEST_END].index,
        test_actual=series.loc[config.TEST_START : config.TEST_END].to_numpy(),
        val_actual=series.loc[config.VAL_START : config.VAL_END].to_numpy(),
        mean=mean,
        std=std,
        scaling=scaling,
    )
