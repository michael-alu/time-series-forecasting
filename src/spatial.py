"""Where the traffic is, not just when.

The 10,000 areas are a 100 x 100 grid laid over Milan, so square id 1 is the top left
cell and ids run along each row. That means the totals can be reshaped into an image
and looked at as a map.

Three questions, each one relevant to the forecasting problem:
    1. is traffic spread out or concentrated in one place
    2. do neighbouring areas behave alike, measured with Moran's I
    3. does that similarity fade with distance, which says whether a model for one
       area could borrow information from its neighbours

Usage:
    python src/spatial.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt

import dataset
from config import FIGURE_DIRECTORY, NUMBER_OF_SQUARES, RESULTS_DIRECTORY

GRID_SIDE = 100
EXTRA_SQUARES = [4159, 4556]


def square_to_cell(square_id: int) -> tuple[int, int]:
    """Square id to its (row, column) on the grid. Ids start at 1, cells at 0."""

    return divmod(square_id - 1, GRID_SIDE)


def totals_grid(totals: pd.DataFrame) -> np.ndarray:
    """Total traffic per area, shaped like the map."""

    values = np.zeros(NUMBER_OF_SQUARES, dtype=np.float64)

    values[totals["square_id"].to_numpy() - 1] = totals["total"].to_numpy()

    return values.reshape(GRID_SIDE, GRID_SIDE)


def morans_i(grid: np.ndarray) -> float:
    """Spatial autocorrelation: do high areas sit next to other high areas?

    Uses the four side by side neighbours. The result runs from about -1 (a
    checkerboard) through 0 (no pattern) to +1 (one smooth blob).
    """

    values = grid - grid.mean()

    # Sum of each cell times its neighbour, counted once per direction.
    pairs = (values[:, :-1] * values[:, 1:]).sum() + (values[:-1, :] * values[1:, :]).sum()

    neighbour_count = 2 * (GRID_SIDE * (GRID_SIDE - 1)) * 2

    return float(grid.size * (2 * pairs) / (neighbour_count * (values**2).sum()))


def draw_map(grid: np.ndarray, focus: list[int]) -> None:
    """Whole city on the left, the busy centre zoomed in on the right."""

    shown = np.log10(grid + 1)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    image = axes[0].imshow(shown, cmap="magma", origin="upper")
    axes[0].set_title("Total Internet activity, log10 scale")
    figure.colorbar(image, ax=axes[0], fraction=0.046)

    for square_id in focus:
        row, col = square_to_cell(square_id)
        axes[0].plot(col, row, "o", markersize=5, markerfacecolor="none",
                     markeredgecolor="#4cc9f0", markeredgewidth=1.4)

    top_row, top_col = square_to_cell(focus[0])
    half = 12
    window = shown[top_row - half:top_row + half + 1, top_col - half:top_col + half + 1]

    zoomed = axes[1].imshow(window, cmap="magma", origin="upper",
                            extent=[top_col - half, top_col + half, top_row + half, top_row - half])
    axes[1].set_title(f"Zoom on the busiest area, square {focus[0]}")
    figure.colorbar(zoomed, ax=axes[1], fraction=0.046)

    for square_id in focus:
        row, col = square_to_cell(square_id)
        if abs(row - top_row) <= half and abs(col - top_col) <= half:
            axes[1].plot(col, row, "o", markersize=7, markerfacecolor="none",
                         markeredgecolor="#4cc9f0", markeredgewidth=1.6)
            axes[1].annotate(str(square_id), (col, row), textcoords="offset points",
                             xytext=(7, 4), color="#4cc9f0", fontsize=9)

    for ax in axes:
        ax.set_xlabel("grid column")
        ax.set_ylabel("grid row")

    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "eda_spatial_map.png", dpi=150)
    plt.close(figure)


def correlation_against_distance(sample_size: int = 250, seed: int = 0) -> pd.DataFrame:
    """Do nearby areas rise and fall together?

    Correlating every pair of 10,000 areas is 50 million comparisons, so a random
    sample of the busier areas is used. Quiet areas are skipped because a mostly
    empty series has an unstable correlation.
    """

    matrix, _, _ = dataset.load_matrix()
    totals = dataset.load_square_totals()

    busy = totals.head(2000)["square_id"].to_numpy()
    chosen = np.random.default_rng(seed).choice(busy, size=sample_size, replace=False)

    series = np.asarray(matrix[:, chosen - 1], dtype=np.float32)
    correlation = np.corrcoef(series, rowvar=False)

    rows, cols = np.triu_indices(sample_size, k=1)
    positions = np.array([square_to_cell(s) for s in chosen])
    distance = np.hypot(*(positions[rows] - positions[cols]).T)

    frame = pd.DataFrame({"distance": distance, "correlation": correlation[rows, cols]})
    bins = [0, 1.5, 3, 6, 12, 25, 50, 200]
    frame["band"] = pd.cut(frame["distance"], bins)

    summary = frame.groupby("band", observed=True)["correlation"].agg(["mean", "count"]).round(3)
    summary.index.name = "distance in cells"
    return summary


def draw_correlation_decay(summary: pd.DataFrame) -> None:
    centres = [interval.mid for interval in summary.index]

    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(centres, summary["mean"], marker="o", color="#1f3b73")
    ax.set_xscale("log")
    ax.set_xlabel("distance between areas, in grid cells")
    ax.set_ylabel("mean correlation of their traffic")
    ax.set_title("Nearby areas behave alike, distant ones less so")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    figure.tight_layout()
    figure.savefig(FIGURE_DIRECTORY / "eda_spatial_correlation.png", dpi=150)
    plt.close(figure)


def main() -> None:
    matplotlib.use("Agg")  # no screen when run as a script
    pd.set_option("display.width", 170)

    totals = dataset.load_square_totals()
    grid = totals_grid(totals)
    focus = dataset.top_squares(3) + EXTRA_SQUARES

    print("where the busiest areas sit\n")
    for square_id in focus:
        row, col = square_to_cell(square_id)
        rank = int(totals.index[totals["square_id"] == square_id][0]) + 1
        print(f"  square {square_id:>5}  row {row:>3}  col {col:>3}   rank {rank:>5} of 10000")

    top_cells = np.array([square_to_cell(s) for s in dataset.top_squares(5)])
    span = top_cells.max(axis=0) - top_cells.min(axis=0)
    print(f"\n  the five busiest areas fit inside a {span[0] + 1} by {span[1] + 1} block")

    print(f"\nspatial autocorrelation (Moran's I): {morans_i(grid):.3f}")
    print("  near 0 would mean traffic is scattered at random")
    print("  near 1 means busy areas sit next to other busy areas")

    share = np.sort(grid.ravel())[::-1]
    print(f"\n  busiest 100 cells of 10000 hold {100 * share[:100].sum() / share.sum():.1f}% of all traffic")

    draw_map(grid, focus)

    print("\ncorrelation between areas, by how far apart they are\n")
    summary = correlation_against_distance()
    print(summary.to_string())
    draw_correlation_decay(summary)

    summary.to_csv(RESULTS_DIRECTORY / "spatial_correlation.csv")
    pd.Series({
        "morans_i": round(morans_i(grid), 4),
        "top_100_share": round(float(share[:100].sum() / share.sum()), 4),
        "busiest_square": int(dataset.top_squares(1)[0]),
    }).to_frame("value").to_csv(RESULTS_DIRECTORY / "spatial_summary.csv")

    print(f"\nfigures saved to {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
