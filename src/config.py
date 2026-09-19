"""Paths and constants used everywhere else."""

from pathlib import Path
from typing import Literal

ROOT: Path = Path(__file__).resolve().parents[1]

RAW_DIRECTORY: Path = ROOT / "data" / "raw"
DAILY_DIRECTORY: Path = ROOT / "data" / "processed" / "daily"
COLAB_DIRECTORY: Path = ROOT / "data" / "processed" / "colab"
RESULTS_DIRECTORY: Path = ROOT / "results"
FIGURE_DIRECTORY: Path = RESULTS_DIRECTORY / "figures"
LOG_DIRECTORY: Path = ROOT / "logs"

# Columns in the raw text files, in order.
RAW_COLUMNS: list[str] = [
    "square_id",
    "time_ms",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet",
]

ACTIVITY_COLUMNS: list[str] = ["sms_in", "sms_out", "call_in", "call_out", "internet"]

NUMBER_OF_SQUARES: int = 10000

# The two areas the assignment names alongside the three busiest.
EXTRA_SQUARES: list[int] = [4159, 4556]

# one reading every 10 minutes
STEPS_PER_DAY: int = 144

TIMEZONE: str = "Europe/Rome"

# The assignment fixes the test week. Validation is the week before it.
TRAIN_END: str = "2013-12-08 23:50"
VAL_START: str = "2013-12-09 00:00"
VAL_END: str = "2013-12-15 23:50"
TEST_START: str = "2013-12-16 00:00"
TEST_END: str = "2013-12-22 23:50"

CellType = Literal["RNN", "LSTM", "GRU"]

ScalingType = Literal["None", "Standard", "MinMax", "LogStandard"]

folders = (
    RAW_DIRECTORY,
    DAILY_DIRECTORY,
    COLAB_DIRECTORY,
    FIGURE_DIRECTORY,
    LOG_DIRECTORY,
)

for folder in folders:
    folder.mkdir(parents=True, exist_ok=True)
