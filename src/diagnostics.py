"""Where the models go wrong, and whether they learned anything beyond the obvious.

The headline errors say which model is best. These checks say why a model is wrong
and whether it deserves credit for being right:

    1. error through the day, to see if mistakes cluster at the busy hours
    2. error against traffic level, to separate big errors from big values
    3. bias, to see whether a model runs high or low rather than just noisy
    4. the echo test, which asks whether a model is really just repeating the last
       value it saw, the trap in any one step ahead problem on smooth data
"""

import numpy as np
import pandas as pd


def error_by_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Average error for each hour of the day, next to the average traffic."""

    grouped = frame.groupby(frame.index.hour)

    return pd.DataFrame({
        "mean_error": grouped["error"].mean(),
        "mean_actual": grouped["actual"].mean(),
        "error_as_percent": 100 * grouped["error"].mean() / grouped["actual"].mean(),
    }).round(2).rename_axis("hour")


def error_by_traffic_level(frame: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Split the week into quiet through busy and compare errors.

    Absolute errors almost always grow with the size of the value, so the percentage
    column is the one that says whether the busy hours are genuinely harder.
    """

    banded = frame.assign(band=pd.qcut(frame["actual"], bins, labels=False) + 1)
    grouped = banded.groupby("band")

    return pd.DataFrame({
        "traffic_from": grouped["actual"].min().round(0),
        "traffic_to": grouped["actual"].max().round(0),
        "mean_error": grouped["error"].mean().round(2),
        "error_as_percent": (100 * grouped["error"].mean() / grouped["actual"].mean()).round(2),
    })


def bias(frame: pd.DataFrame) -> pd.Series:
    """Does the model sit above or below the truth, and where?

    A model that lags a rising signal will under predict while traffic climbs and
    over predict while it falls, which shows up here as opposite signs.
    """

    signed = frame["predicted"] - frame["actual"]
    rising = frame["actual"].diff() > 0

    return pd.Series({
        "mean_signed_error": signed.mean(),
        "while_traffic_rising": signed[rising].mean(),
        "while_traffic_falling": signed[~rising].mean(),
        "share_over_predicted": (signed > 0).mean(),
    }).round(2)


def echo_test(frame: pd.DataFrame) -> pd.Series:
    """Is the prediction really just the previous observed value?

    On a smooth series a model can score well by copying the last reading. If the
    prediction matches the previous value more closely than it matches the value it
    was meant to predict, the model has learned very little.
    """

    previous = frame["actual"].shift(1)
    usable = previous.notna()

    distance_to_previous = (frame["predicted"][usable] - previous[usable]).abs().mean()
    distance_to_target = (frame["predicted"][usable] - frame["actual"][usable]).abs().mean()

    return pd.Series({
        "matches_previous_value": round(distance_to_previous, 2),
        "matches_the_target": round(distance_to_target, 2),
        "correlation_with_previous": round(frame["predicted"][usable].corr(previous[usable]), 4),
        "correlation_with_target": round(frame["predicted"][usable].corr(frame["actual"][usable]), 4),
        "is_just_an_echo": bool(distance_to_previous < distance_to_target),
    })


def summarise(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Run the echo test and bias check for every model and area at once."""

    rows = []
    for (square_id, model), group in predictions.groupby(["square_id", "model"]):
        if model not in models:
            continue
        group = group.sort_values("timestamp").set_index("timestamp")
        frame = group.assign(error=(group["actual"] - group["predicted"]).abs())
        rows.append({"square_id": square_id, "model": model, **echo_test(frame), **bias(frame)})

    return pd.DataFrame(rows)
