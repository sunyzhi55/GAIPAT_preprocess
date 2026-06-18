from __future__ import annotations

import argparse
import importlib
import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STEP3_FILE_PATTERN = re.compile(
    r"^(?P<subject_id>[^_]+)_(?P<task>[^_]+)_(?P<step_id>-?\d+)_(?P<event>grasp|release)_(?P<block_id>-?\d+)_(?P<label>[01])$"
)

STEP3_X_CANDIDATES = ["timestamp"]
STEP3_Y_CANDIDATES = ["deviation_cm", "deviation_dst_cm", "deviation_srt_cm"]

STEP4_X_CANDIDATES = ["normalized_progress", "timestamp"]
STEP4_Y_CANDIDATES = [
    "arithmetic_mean_deviation_dst_cm",
    "dba_deviation_dst_cm",
    "deviation_cm",
    "deviation_dst_cm",
    "deviation_srt_cm",
]

STD_CANDIDATES = ["std_deviation_dst_cm"]


def load_scipy_interpolator():
    """Load scipy interpolator lazily, returning None if scipy is unavailable."""

    try:
        interpolate_module = importlib.import_module("scipy.interpolate")
    except ImportError:
        return None

    make_interp_spline = getattr(interpolate_module, "make_interp_spline", None)
    if make_interp_spline is not None:
        return make_interp_spline
    return getattr(interpolate_module, "interp1d", None)


def parse_step3_metadata(file_path: Path) -> Optional[dict[str, str]]:
    """Parse the event-aware slice filename if possible."""

    match = STEP3_FILE_PATTERN.match(file_path.stem)
    if match is None:
        return None
    return match.groupdict()


def find_csv_files(input_dir: Path) -> list[Path]:
    """Find all CSV files under a directory."""

    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def coerce_numeric_series(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and drop invalid rows later."""

    return pd.to_numeric(series, errors="coerce")


def pick_first_existing(columns: list[str], available_columns: list[str]) -> Optional[str]:
    """Return the first column that exists in the available set."""

    available = set(available_columns)
    for column in columns:
        if column in available:
            return column
    return None


def detect_plot_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Detect the x, y, and optional std columns for a deviation CSV."""

    columns = list(df.columns)
    x_column = pick_first_existing(STEP4_X_CANDIDATES, columns)
    y_column = pick_first_existing(STEP4_Y_CANDIDATES, columns)
    std_column = pick_first_existing(STD_CANDIDATES, columns)
    return x_column, y_column, std_column


def prepare_plot_frame(df: pd.DataFrame, x_column: str, y_column: str, std_column: Optional[str] = None) -> pd.DataFrame:
    """Keep only numeric plotting columns and sort by x."""

    required_columns = [x_column, y_column]
    if std_column is not None:
        required_columns.append(std_column)

    plot_df = df[required_columns].copy()
    plot_df[x_column] = coerce_numeric_series(plot_df[x_column])
    plot_df[y_column] = coerce_numeric_series(plot_df[y_column])
    if std_column is not None:
        plot_df[std_column] = coerce_numeric_series(plot_df[std_column])

    plot_df = plot_df.dropna(subset=[x_column, y_column])
    if plot_df.empty:
        return plot_df

    plot_df = plot_df.sort_values(x_column, kind="mergesort")
    plot_df = plot_df.groupby(x_column, as_index=False, sort=True).mean(numeric_only=True)
    return plot_df


def build_smooth_curve(x_values: np.ndarray, y_values: np.ndarray, dense_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Create a smooth curve using SciPy if available, otherwise fall back to interpolation."""

    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size < 2 or y_values.size < 2:
        return x_values, y_values

    order = np.argsort(x_values, kind="mergesort")
    x_sorted = x_values[order]
    y_sorted = y_values[order]

    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]
    if unique_x.size < 2:
        return unique_x, unique_y

    smooth_x = np.linspace(float(unique_x.min()), float(unique_x.max()), max(dense_points, unique_x.size * 20))
    interpolator = load_scipy_interpolator()
    if interpolator is None:
        smooth_y = np.interp(smooth_x, unique_x, unique_y)
        return smooth_x, smooth_y

    try:
        if interpolator.__name__ == "make_interp_spline":
            spline_order = min(3, unique_x.size - 1)
            spline = interpolator(unique_x, unique_y, k=spline_order)
            smooth_y = spline(smooth_x)
        else:
            kind = "cubic" if unique_x.size >= 4 else "linear"
            spline = interpolator(unique_x, unique_y, kind=kind, fill_value="extrapolate")
            smooth_y = spline(smooth_x)
    except Exception:
        smooth_y = np.interp(smooth_x, unique_x, unique_y)

    return smooth_x, np.asarray(smooth_y, dtype=np.float64)


def draw_deviation_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    x_column: str,
    y_column: str,
    std_column: Optional[str] = None,
) -> None:
    """Draw one deviation plot with scatter points and a smooth curve."""

    plot_df = prepare_plot_frame(df, x_column, y_column, std_column)
    if plot_df.empty:
        return

    x_values = plot_df[x_column].to_numpy(dtype=np.float64)
    y_values = plot_df[y_column].to_numpy(dtype=np.float64)
    smooth_x, smooth_y = build_smooth_curve(x_values, y_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, y_values, s=12, alpha=0.45, color="#2b6cb0", edgecolors="none", label="points")
    ax.plot(smooth_x, smooth_y, color="#e76f51", linewidth=2.2, label="smooth curve")

    if std_column is not None and std_column in plot_df.columns:
        std_values = plot_df[std_column].to_numpy(dtype=np.float64)
        if std_values.size == y_values.size and np.isfinite(std_values).any():
            ax.fill_between(
                x_values,
                y_values - std_values,
                y_values + std_values,
                color="#f4a261",
                alpha=0.18,
                label="std band",
            )

    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel("deviation")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def visualize_step3(step3_dir: Path, output_dir: Path) -> int:
    """Visualize every step-3 deviation CSV file."""

    if not step3_dir.exists():
        logging.warning("Skip missing step-3 directory: %s", step3_dir)
        return 0

    exported = 0
    for file_path in find_csv_files(step3_dir):
        metadata = parse_step3_metadata(file_path)
        if metadata is None:
            logging.info("Skip unexpected step-3 file name: %s", file_path.name)
            continue

        df = pd.read_csv(file_path)
        x_column = pick_first_existing(STEP4_X_CANDIDATES, list(df.columns)) or pick_first_existing(STEP3_X_CANDIDATES, list(df.columns))
        y_column = pick_first_existing(STEP3_Y_CANDIDATES, list(df.columns))
        if x_column is None or y_column is None:
            logging.info("Skip non-plotable step-3 file: %s", file_path.name)
            continue

        output_path = output_dir / f"{file_path.stem}.png"
        title = f"step3: {file_path.stem}"
        draw_deviation_plot(df, output_path, title, x_column, y_column)
        exported += 1

    return exported


def visualize_step4(step4_dir: Path, output_dir: Path) -> int:
    """Visualize every step-4 CSV file while preserving the method subfolders."""

    if not step4_dir.exists():
        logging.warning("Skip missing step-4 directory: %s", step4_dir)
        return 0

    exported = 0
    for file_path in find_csv_files(step4_dir):
        relative_path = file_path.relative_to(step4_dir)
        df = pd.read_csv(file_path)

        x_column = pick_first_existing(STEP4_X_CANDIDATES, list(df.columns))
        y_column = pick_first_existing(STEP4_Y_CANDIDATES, list(df.columns))
        std_column = pick_first_existing(STD_CANDIDATES, list(df.columns))
        if x_column is None or y_column is None:
            logging.info("Skip non-plotable step-4 file: %s", file_path)
            continue

        output_path = output_dir / relative_path.with_suffix(".png")
        title = f"step4: {relative_path.as_posix()}"
        draw_deviation_plot(df, output_path, title, x_column, y_column, std_column=std_column)
        exported += 1

    return exported


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Visualize all step-3 and step-4 deviation CSV files as scatter plots with smooth curves."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--step3-dir",
        type=Path,
        default=None,
        help="Directory containing step-3 deviation CSV files. Defaults to <repo-root>/block_deviation_sequences.",
    )
    parser.add_argument(
        "--step4-dir",
        type=Path,
        default=None,
        help="Directory containing step-4 deviation CSV files. Defaults to <repo-root>/mean_deviation_sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Base directory for visualizations. Defaults to <repo-root>/deviation_visualizations.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    repo_root = args.repo_root
    step3_dir = args.step3_dir or (repo_root / "block_deviation_sequences")
    step4_dir = args.step4_dir or (repo_root / "mean_deviation_sequences")
    output_base_dir = args.output_dir or (repo_root / "deviation_visualizations")
    step3_output_dir = output_base_dir / "step3"
    step4_output_dir = output_base_dir / "step4"
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_base_dir / "visualization.log"),
            logging.StreamHandler(),
        ],
    )

    step3_count = visualize_step3(step3_dir, step3_output_dir)
    step4_count = visualize_step4(step4_dir, step4_output_dir)

    logging.info("Done. Exported %d step-3 plots and %d step-4 plots.", step3_count, step4_count)


if __name__ == "__main__":
    main()