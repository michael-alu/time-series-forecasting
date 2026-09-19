"""Turn the 62 daily parquet files into a few small files for Colab.

Reading 62 files from Google Drive every time a Colab session restarts is slow, so
this does the work once and saves:

    internet_matrix.npy    time steps x areas, the whole Internet series
    matrix_index.npz       which timestamp and area each row and column is
    square_totals.parquet  totals per area, for the exploratory analysis
    focus_squares.parquet  just the five areas we look at closely

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    COLAB_DIRECTORY,
    DAILY_DIRECTORY,
    NUMBER_OF_SQUARES,
    STEPS_PER_DAY,
    TIMEZONE,
)

EXTRA_SQUARES = [4159, 4556]  # the assignment asks for these two by name


# This will stack every day into one giant matrix so that it can be easily accessible
# This ensures we don't need to hold all 62 files at all, great way to compress the data and retain relevance
def build_matrix() -> tuple[np.ndarray, pd.DatetimeIndex]:
    files = sorted(DAILY_DIRECTORY.glob("*.parquet"))

    if not files:
        sys.exit(
            f"no parquet files in {DAILY_DIRECTORY}, run src/download_data.py first"
        )

    blocks: list[np.ndarray] = []

    stamps: list[pd.DatetimeIndex] = []

    for file in files:
        file_day = pd.read_parquet(file, columns=["square_id", "timestamp", "internet"])

        file_day["timestamp"] = file_day["timestamp"].dt.tz_convert(TIMEZONE)

        times = pd.DatetimeIndex(sorted(file_day["timestamp"].unique()))

        block = np.zeros((len(times), NUMBER_OF_SQUARES), dtype=np.float32)

        rows = times.get_indexer(file_day["timestamp"])

        # ids start at 1, columns at 0
        columns = file_day["square_id"].to_numpy() - 1

        block[rows, columns] = file_day["internet"].to_numpy(dtype=np.float32)

        blocks.append(block)

        stamps.append(times)

    index = stamps[0].append(stamps[1:])

    order = np.argsort(index.values)

    return np.concatenate(blocks)[order], index[order]


def summarise(matrix: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "std": matrix.std(axis=0),
                "total": matrix.sum(axis=0),
                "mean": matrix.mean(axis=0),
                "maximum": matrix.max(axis=0),
                "zero_fraction": (matrix == 0).mean(axis=0),
                "square_id": np.arange(1, NUMBER_OF_SQUARES + 1),
            }
        )
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    matrix, index = build_matrix()

    print(
        f"\nMatrix shape - {matrix.shape}, Matrix size - {matrix.nbytes / 1e6:.0f} MB"
    )

    expected = pd.date_range(index[0], index[-1], freq="10min", tz=TIMEZONE)

    print(f"\n{index[0]} to {index[-1]}")
    print(
        f"{len(index)} time steps, expected {len(expected)}, {len(expected.difference(index))} missing"
    )

    totals = summarise(matrix)

    focus = totals["square_id"].head(3).tolist() + EXTRA_SQUARES

    print(f"\nBusiest three areas - {focus[:3]}")

    # Saves the internet traffic matrix into a .npy file
    np.save(COLAB_DIRECTORY / "internet_matrix.npy", matrix)

    # Saves the internet traffic matrix index (for fast lookups) into a .npz file
    np.savez_compressed(
        COLAB_DIRECTORY / "matrix_index.npz",
        timestamps=index.tz_convert("UTC").astype("int64").to_numpy(),
        square_ids=np.arange(1, NUMBER_OF_SQUARES + 1, dtype=np.int16),
    )

    # Saves the totals of the internet traffic for each square into a parquet file
    totals.to_parquet(COLAB_DIRECTORY / "square_totals.parquet", index=False)

    focus_frame = pd.DataFrame(
        matrix[:, [s - 1 for s in focus]], index=index, columns=focus
    )

    focus_frame.rename_axis("timestamp").reset_index().melt(
        id_vars="timestamp",
        var_name="square_id",
        value_name="internet",
        # Saves the focus squares into a parquet file
    ).to_parquet(COLAB_DIRECTORY / "focus_squares.parquet", index=False)

    print(f"\nWrote to {COLAB_DIRECTORY}")

    for file in sorted(COLAB_DIRECTORY.iterdir()):
        print(f"  {file.name:<24} {file.stat().st_size / 1e6:>7.1f} MB")


if __name__ == "__main__":
    main()
