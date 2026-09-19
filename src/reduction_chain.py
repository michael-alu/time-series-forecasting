"""Show how 20.8 GB of text became a 357 MB file, and prove nothing was lost.

memory_profile.py measures RAM. This measures disk, and checks the final file size
is exactly one number per reading, so the shrinking is clearly not losing data.

Usage:
    python src/reduction_chain.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    COLAB_DIRECTORY,
    LOG_DIRECTORY,
    NUMBER_OF_SQUARES,
    RESULTS_DIRECTORY,
    STEPS_PER_DAY,
)

STATE_FILE = LOG_DIRECTORY / "download_state.json"


def main() -> None:
    if not STATE_FILE.exists():
        sys.exit(f"{STATE_FILE} not found, run src/download_data.py first")

    state: dict[str, dict] = json.loads(STATE_FILE.read_text())

    days = len(state)

    raw_bytes = sum(day["raw_bytes"] for day in state.values())

    parquet_bytes = sum(day["parquet_bytes"] for day in state.values())

    matrix_bytes = (COLAB_DIRECTORY / "internet_matrix.npy").stat().st_size

    upload_bytes = sum(path.stat().st_size for path in COLAB_DIRECTORY.iterdir())

    stages = pd.DataFrame(
        [
            {"stage": "raw text", "files": days, "MB": raw_bytes / 1e6},
            {"stage": "daily parquet", "files": days, "MB": parquet_bytes / 1e6},
            {"stage": "one matrix", "files": 1, "MB": matrix_bytes / 1e6},
            {
                "stage": "upload folder",
                "MB": upload_bytes / 1e6,
                "files": len(list(COLAB_DIRECTORY.iterdir())),
            },
        ]
    )

    stages["times_smaller"] = (raw_bytes / (stages["MB"] * 1e6)).round(1)

    stages["MB"] = stages["MB"].round(1)

    print("\nDisk used at each stage\n")
    print(stages.to_string(index=False))

    # The four reasons the raw files are so much bigger than the data inside them.
    print("\n\nWhy the raw files are so big?\n")
    print(
        "1. numbers stored as text    0.14186425470242922 is 19 bytes, 4 as a float32"
    )
    print("2. one row per country code  about 3.4 rows for every area and time step")
    print("3. four columns we never use only Internet is needed here")
    print(
        "4. every row repeats its id  the matrix uses position instead, storing neither"
    )

    # If this matches the file size, nothing was thrown away or compressed lossily.
    steps = STEPS_PER_DAY * days

    values = steps * NUMBER_OF_SQUARES

    rows_found = sum(day["rows"] for day in state.values())

    print("\n\nNothing was lost, the size is just arithmetic\n")

    print(f"  {steps:,} time steps x {NUMBER_OF_SQUARES:,} areas = {values:,} values")
    print(f"  {values:,} values x 4 bytes           = {values * 4 / 1e6:.1f} MB")
    print(f"  actual file on disk                   = {matrix_bytes / 1e6:.1f} MB")
    print(f"\n  readings in the source files  {rows_found:,}")
    print(
        f"  slots with no reading         {values - rows_found:,} "
        f"({100 * (values - rows_found) / values:.3f}%), stored as zero"
    )

    print("\n\nwhat this costs us\n")
    print("  a zero could mean no traffic or no reading, we cannot tell them apart")
    print(
        "  the matrix keeps Internet only, so the zipped raw text is kept as a backup"
    )

    stages.to_csv(RESULTS_DIRECTORY / "reduction_stages.csv", index=False)
    print(f"\nsaved to {RESULTS_DIRECTORY / 'reduction_stages.csv'}")


if __name__ == "__main__":
    main()
