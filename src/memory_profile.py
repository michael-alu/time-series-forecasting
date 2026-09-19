"""Measure how much memory each way of loading the data actually needs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import resource
import subprocess
import time
from typing import Callable

import pandas as pd


from config import (
    ACTIVITY_COLUMNS,
    DAILY_DIRECTORY,
    RAW_COLUMNS,
    RAW_DIRECTORY,
    RESULTS_DIRECTORY,
)

PARQUET_DAY = DAILY_DIRECTORY / "2013-11-01.parquet"

RAW_DAY = RAW_DIRECTORY / "sms-call-internet-mi-2013-11-01.txt"


def peak_memory_mb() -> float:
    """Highest memory this process has used. macOS reports bytes, Linux kilobytes."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def naive_read() -> pd.DataFrame:
    """Just read it. Pandas picks float64 for every number."""
    return pd.read_csv(RAW_DAY, sep="\t", names=RAW_COLUMNS)


def small_dtypes() -> pd.DataFrame:
    """Same rows, but tell pandas the numbers fit in smaller types."""
    dtypes: dict[str, str] = {
        "square_id": "int16",
        "time_ms": "int64",
        "country_code": "int16",
    }
    dtypes.update({column: "float32" for column in ACTIVITY_COLUMNS})
    return pd.read_csv(RAW_DAY, sep="\t", names=RAW_COLUMNS, dtype=dtypes)


def needed_columns() -> pd.DataFrame:
    """Only read the three columns this project uses."""
    dtypes = {"square_id": "int16", "time_ms": "int64", "internet": "float32"}
    return pd.read_csv(
        RAW_DAY,
        sep="\t",
        names=RAW_COLUMNS,
        usecols=["square_id", "time_ms", "internet"],
        dtype=dtypes,
    )


def chunked_reduce() -> pd.DataFrame:
    """What the download script does: read a bit at a time and add it up as we go."""
    from download_data import reduce_day

    out = Path("/tmp/memory_profile_day.parquet")
    reduce_day(RAW_DAY, out)
    return pd.read_parquet(out)


def read_parquet() -> pd.DataFrame:
    """Read the shrunk file instead. This is what every later step does."""
    return pd.read_parquet(PARQUET_DAY, columns=["square_id", "timestamp", "internet"])


def one_area() -> pd.Series:
    """Pull a single area out of the big matrix without loading the rest."""
    from dataset import load_series

    return load_series(5161)


APPROACHES: dict[str, Callable[[], pd.DataFrame | pd.Series]] = {
    "1. read it all, default types": naive_read,
    "2. smaller number types": small_dtypes,
    "3. only the columns we need": needed_columns,
    "4. chunked read and add up": chunked_reduce,
    "5. read the shrunk parquet": read_parquet,
    "6. one area from the matrix": one_area,
}


def run_child(name: str) -> None:
    """Run one approach and report what it cost. Called in a fresh process."""
    before = peak_memory_mb()
    start = time.perf_counter()
    result = APPROACHES[name]()
    print(
        json.dumps(
            {
                "peak_MB": round(peak_memory_mb() - before, 1),
                "seconds": round(time.perf_counter() - start, 2),
                "rows": len(result),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--approach", help="internal, runs one approach in this process"
    )
    args = parser.parse_args()

    if args.approach:
        run_child(args.approach)
        return

    if not RAW_DAY.exists():
        sys.exit(
            f"{RAW_DAY} not found, get one with:\n  python src/download_data.py --limit 1 --keep-raw"
        )

    print(f"raw day      {RAW_DAY.stat().st_size / 1e6:.0f} MB on disk")
    print(f"parquet day  {PARQUET_DAY.stat().st_size / 1e6:.1f} MB on disk\n")

    rows = []
    for name in APPROACHES:
        finished = subprocess.run(
            [sys.executable, __file__, "--approach", name],
            capture_output=True,
            text=True,
        )
        measured = json.loads(finished.stdout)
        rows.append({"approach": name, **measured})
        print(
            f"{name:<32} peak {measured['peak_MB']:>7.1f} MB  "
            f"{measured['seconds']:>5.2f}s  {measured['rows']:>9,} rows"
        )

    table = pd.DataFrame(rows)
    table["times_smaller"] = (table["peak_MB"].iloc[0] / table["peak_MB"]).round(1)
    table.to_csv(RESULTS_DIRECTORY / "memory_profile.csv", index=False)

    worst = table["peak_MB"].iloc[0]
    print(f"\nall 62 days the naive way would peak near {worst * 62 / 1000:.0f} GB")
    print(f"saved to {RESULTS_DIRECTORY / 'memory_profile.csv'}")


if __name__ == "__main__":
    main()
