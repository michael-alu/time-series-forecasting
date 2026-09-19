"""Exploratory analysis: what does this traffic actually look like?"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


import matplotlib
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from config import FIGURE_DIRECTORY, RESULTS_DIRECTORY, STEPS_PER_DAY
import dataset

EXTRA_SQUARES = [4159, 4556]

TWO_WEEKS = ("2013-11-01", "2013-11-14 23:50")

BLUE = "#1f3b73"

RED = "#d1495b"


def tidy(ax: plt.Axes) -> plt.Axes:
    ax.grid(alpha=0.25)

    ax.margins(x=0)

    ax.spines[["top", "right"]].set_visible(False)

    return ax


def save(figure: Figure, name: str) -> None:
    figure.tight_layout()

    figure.savefig(FIGURE_DIRECTORY / name, dpi=150)

    plt.close(figure)


def traffic_spread(totals: pd.DataFrame) -> pd.Series:
    """This functions asks how uneven is traffic across areas"""
    values = totals["total"].to_numpy()

    ranked = np.sort(values)[::-1]

    figure, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].hist(values, bins=80, color=BLUE)

    axes[0].set_yscale("log")

    axes[0].set_xlabel("total Internet activity")

    axes[0].set_ylabel("number of areas (log)")

    axes[1].hist(np.log10(values), bins=80, color=BLUE)

    axes[1].axvline(
        np.log10(np.median(values)), color=RED, linestyle="--", label="median"
    )

    axes[1].set_xlabel("log10 of total Internet activity")

    axes[1].legend(frameon=False)

    for ax in axes:
        tidy(ax)

    save(figure, "eda_traffic_distribution.png")

    return pd.Series(
        {
            "max": values.max(),
            "areas": len(values),
            "mean": values.mean(),
            "median": np.median(values),
            "skew": pd.Series(values).skew(),
            "max_over_median": values.max() / np.median(values),
            "share_held_by_top_100": ranked[:100].sum() / values.sum(),
            "share_held_by_top_1000": ranked[:1000].sum() / values.sum(),
        }
    )


def compare_areas() -> pd.DataFrame:
    """First two weeks for the busiest three areas plus 4159 and 4556."""
    rows = []

    start, end = TWO_WEEKS

    squares = dataset.top_squares(3) + EXTRA_SQUARES

    figure, axes = plt.subplots(len(squares), 1, figsize=(13, 11), sharex=True)

    for ax, square_id in zip(axes, squares):
        series = dataset.load_series(square_id).loc[start:end]

        ax.plot(series.index, series.to_numpy(), linewidth=0.9, color=BLUE)

        ax.set_title(f"Square {square_id}", fontsize=10, loc="left")

        tidy(ax)

        weekday = series[series.index.dayofweek < 5]

        weekend = series[series.index.dayofweek >= 5]

        rows.append(
            {
                "peak": series.max(),
                "mean": series.mean(),
                "square_id": square_id,
                "peak_over_trough": series.max() / series.min(),
                "weekend_vs_weekday": weekend.mean() / weekday.mean(),
                "busiest_hour": series.groupby(series.index.hour).mean().idxmax(),
            }
        )

    save(figure, "eda_focus_series.png")

    return pd.DataFrame(rows).round(2)


def autocorrelation(series: pd.Series) -> pd.Series:
    """This function tells us how much the past tells us about the next value"""

    lags = STEPS_PER_DAY * 3

    values = series.to_numpy(dtype=np.float64)

    pacf_values = pacf(values, nlags=48)

    acf_values = acf(values, nlags=lags, fft=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 3.8))

    axes[0].stem(range(len(acf_values)), acf_values, markerfmt=" ", basefmt=" ")

    for day in range(1, 4):
        axes[0].axvline(day * STEPS_PER_DAY, color=RED, linestyle="--", linewidth=0.9)

    axes[0].set_title("ACF, dashed lines one day apart")

    axes[0].set_xlabel("lag (10 minute steps)")

    axes[1].stem(range(len(pacf_values)), pacf_values, markerfmt=" ", basefmt=" ")

    axes[1].set_title("PACF, first 48 lags")

    axes[1].set_xlabel("lag (10 minute steps)")

    for ax in axes:
        tidy(ax)

    save(figure, "eda_autocorrelation.png")

    return pd.Series(
        {
            "lag_6_one_hour": acf_values[6],
            "lag_1_ten_minutes": acf_values[1],
            "lag_144_one_day": acf_values[STEPS_PER_DAY],
            "lag_288_two_days": acf_values[2 * STEPS_PER_DAY],
        }
    ).round(4)


def decompose(series: pd.Series) -> pd.Series:
    """Split the series into daily pattern, slow trend and leftover noise."""
    from statsmodels.tsa.seasonal import STL

    result = STL(
        series.to_numpy(dtype=np.float64), period=STEPS_PER_DAY, robust=True
    ).fit()
    pieces = {
        "observed": series.to_numpy(),
        "trend": result.trend,
        "daily pattern": result.seasonal,
        "leftover": result.resid,
    }

    figure, axes = plt.subplots(4, 1, figsize=(13, 8), sharex=True)
    for ax, (name, values) in zip(axes, pieces.items()):
        ax.plot(series.index, values, linewidth=0.8, color=BLUE)
        ax.set_ylabel(name, fontsize=9)
        tidy(ax)
    save(figure, "eda_decomposition.png")

    total = np.var(series.to_numpy(dtype=np.float64))
    return pd.Series(
        {
            "share_daily_pattern": np.var(result.seasonal) / total,
            "share_trend": np.var(result.trend) / total,
            "share_leftover": np.var(result.resid) / total,
        }
    ).round(4)


def stationarity(series: pd.Series) -> pd.DataFrame:
    """Is the series stable over time, or does it drift?

    Two tests, because they ask opposite questions. ADF asks "is it drifting", KPSS
    asks "is it stable", so agreeing answers are more convincing than one test alone.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    values = series.to_numpy(dtype=np.float64)
    versions = {
        "raw": values,
        "difference": np.diff(values),
        "day over day difference": values[STEPS_PER_DAY:] - values[:-STEPS_PER_DAY],
    }

    rows = []
    for name, data in versions.items():
        adf_p = adfuller(data)[1]
        with warnings.catch_warnings():  # statsmodels warns when p hits the table edge
            warnings.simplefilter("ignore")
            kpss_p = kpss(data, nlags="auto")[1]
        rows.append(
            {
                "series": name,
                "ADF_p": round(adf_p, 4),
                "ADF_says_stable": adf_p < 0.05,
                # The table only goes to 0.1, so report the edge rather than a fake number.
                "KPSS_p": "> 0.1" if kpss_p >= 0.1 else round(kpss_p, 4),
                "KPSS_says_stable": kpss_p > 0.05,
            }
        )
    return pd.DataFrame(rows)


def daily_shape(series: pd.Series) -> None:
    """Average day, one line per weekday, to show weekends differ."""
    frame = pd.DataFrame({"value": series.to_numpy()}, index=series.index)
    frame["hour"] = frame.index.hour + frame.index.minute / 60
    frame["day"] = frame.index.dayofweek

    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    figure, ax = plt.subplots(figsize=(11, 4))
    for day, colour in zip(range(7), plt.cm.viridis(np.linspace(0, 0.9, 7))):
        profile = frame[frame["day"] == day].groupby("hour")["value"].mean()
        ax.plot(profile.index, profile.to_numpy(), label=names[day], color=colour)
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean Internet activity")
    ax.legend(frameon=False, ncol=7, fontsize=9)
    tidy(ax)
    save(figure, "eda_weekly_profile.png")


def main() -> None:
    matplotlib.use("Agg")  # no screen when run as a script

    pd.set_option("display.width", 160)
    busiest_id = dataset.top_squares(1)[0]
    busiest = dataset.load_series(busiest_id)

    print("traffic spread across areas")
    print(traffic_spread(dataset.load_square_totals()).to_string(), "\n")

    print("first two weeks, five areas")
    comparison = compare_areas()
    comparison.to_csv(RESULTS_DIRECTORY / "eda_focus_summary.csv", index=False)
    print(comparison.to_string(index=False), "\n")

    print(f"autocorrelation, square {busiest_id}")
    print(autocorrelation(busiest).to_string(), "\n")

    print(f"decomposition, square {busiest_id}")
    print(decompose(busiest).to_string(), "\n")

    print(f"stationarity, square {busiest_id}")
    tests = stationarity(busiest)
    tests.to_csv(RESULTS_DIRECTORY / "eda_stationarity.csv", index=False)
    print(tests.to_string(index=False))

    daily_shape(busiest)
    print(f"\nfigures saved to {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
