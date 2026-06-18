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


FILE_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>[^_]+)_(?P<task>[^_]+)_(?P<step_id>-?\d+)_(?P<event>grasp|release)_(?P<block_id>-?\d+)_(?P<label>[01])$"
)

METHOD_TEMPLATE_COLUMNS = {
    "arithmetic_mean": "arithmetic_mean_deviation_dst_cm",
    "dba_mean": "dba_deviation_dst_cm",
}


def require_fastdtw() -> None:
    """Fail fast with a clear message when fastdtw is unavailable."""

    try:
        importlib.import_module("fastdtw")
    except ImportError as exc:
        raise RuntimeError(
            "fastdtw is required for DTW scoring. Install it with: pip install fastdtw"
        ) from exc


def load_fastdtw():
    """Load fastdtw lazily."""

    require_fastdtw()
    return importlib.import_module("fastdtw").fastdtw


def load_pearsonr():
    """Load scipy.stats.pearsonr lazily."""

    try:
        return importlib.import_module("scipy.stats").pearsonr
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Pearson correlation. Install it with: pip install scipy"
        ) from exc


def parse_file_metadata(file_path: Path) -> Optional[dict[str, str]]:
    """Parse metadata encoded in the file name."""

    match = FILE_NAME_PATTERN.match(file_path.stem)
    if match is None:
        return None
    return match.groupdict()


def find_method_dirs(input_base_dir: Path, selected_method: str) -> list[Path]:
    """Return the method folders to analyze."""

    if selected_method == "all":
        return [input_base_dir / method_name for method_name in METHOD_TEMPLATE_COLUMNS]
    return [input_base_dir / selected_method]


def load_template(method_dir: Path) -> tuple[str, np.ndarray]:
    """Load the standard template sequence for one method."""

    method_name = method_dir.name
    template_column = METHOD_TEMPLATE_COLUMNS.get(method_name)
    if template_column is None:
        raise ValueError(f"Unsupported method directory: {method_dir}")

    template_path = method_dir / "mean_deviation_sequence.csv"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template file: {template_path}")

    template_df = pd.read_csv(template_path)
    if template_column not in template_df.columns:
        available_columns = ", ".join(template_df.columns)
        raise ValueError(
            f"Template file {template_path} does not contain {template_column}. Available columns: {available_columns}"
        )

    template_values = pd.to_numeric(template_df[template_column], errors="coerce").dropna().to_numpy(dtype=np.float64)
    if template_values.size == 0:
        raise ValueError(f"Template file {template_path} does not contain valid values in {template_column}")
    return template_column, template_values


def load_sequence_values(file_path: Path) -> np.ndarray:
    """Load one normalized deviation sequence."""

    df = pd.read_csv(file_path)
    if "deviation_dst_cm" not in df.columns:
        raise ValueError(f"Missing deviation_dst_cm column in {file_path}")

    values = pd.to_numeric(df["deviation_dst_cm"], errors="coerce").dropna().to_numpy(dtype=np.float64)
    return values


def compute_dtw_distance(template_values: np.ndarray, sequence_values: np.ndarray) -> float:
    """Compute DTW distance between the template and one sequence."""

    fastdtw = load_fastdtw()
    if template_values.size == 0 or sequence_values.size == 0:
        return float("nan")

    distance, _ = fastdtw(
        template_values.tolist(),
        sequence_values.tolist(),
        dist=lambda a, b: abs(float(a) - float(b)),
    )
    return float(distance)


def build_method_scatter_dataframe(method_dir: Path) -> pd.DataFrame:
    """Build per-sequence DTW metrics for one method."""

    template_column, template_values = load_template(method_dir)
    normalized_dir = method_dir / "normalized_sequences"
    if not normalized_dir.exists():
        raise FileNotFoundError(f"Missing normalized sequence directory: {normalized_dir}")

    records: list[dict[str, object]] = []
    for file_path in sorted(normalized_dir.glob("*.csv")):
        metadata = parse_file_metadata(file_path)
        if metadata is None:
            logging.info("Skip unexpected file name: %s", file_path.name)
            continue

        sequence_values = load_sequence_values(file_path)
        if sequence_values.size == 0:
            logging.info("Skip empty normalized sequence: %s", file_path.name)
            continue

        dtw_distance = compute_dtw_distance(template_values, sequence_values)
        label = int(metadata["label"])
        records.append(
            {
                "subject_id": metadata["subject_id"],
                "task": metadata["task"],
                "step_id": int(metadata["step_id"]),
                "event": metadata["event"],
                "block_id": int(metadata["block_id"]),
                "label": label,
                "is_success": label,
                "sequence_length": int(sequence_values.size),
                "dtw_distance": dtw_distance,
                "template_column": template_column,
                "sequence_file": file_path.name,
            }
        )

    return pd.DataFrame(records)


def compute_pearson_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation between DTW distance and success label."""

    pearsonr = load_pearsonr()
    valid_df = df.loc[df["dtw_distance"].notna() & df["label"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame(
            [
                {
                    "sequence_count": 0,
                    "pearson_r": np.nan,
                    "pearson_p_value": np.nan,
                    "mean_distance_label_0": np.nan,
                    "mean_distance_label_1": np.nan,
                }
            ]
        )

    x_values = valid_df["dtw_distance"].to_numpy(dtype=np.float64)
    y_values = valid_df["label"].to_numpy(dtype=np.float64)

    if np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        pearson_r = np.nan
        pearson_p = np.nan
    else:
        pearson_r, pearson_p = pearsonr(x_values, y_values)

    summary = {
        "sequence_count": int(valid_df.shape[0]),
        "pearson_r": float(pearson_r) if np.isfinite(pearson_r) else np.nan,
        "pearson_p_value": float(pearson_p) if np.isfinite(pearson_p) else np.nan,
        "mean_distance_label_0": float(valid_df.loc[valid_df["label"] == 0, "dtw_distance"].mean()),
        "mean_distance_label_1": float(valid_df.loc[valid_df["label"] == 1, "dtw_distance"].mean()),
    }
    return pd.DataFrame([summary])


def plot_scatter(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path, method_name: str) -> None:
    """Draw a scatter plot for DTW distance versus success label."""

    if df.empty:
        return

    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.05, 0.05, size=len(df))
    y_values = df["label"].to_numpy(dtype=np.float64) + jitter

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = np.where(df["label"].to_numpy(dtype=np.int64) == 1, "#2ca02c", "#d62728")
    ax.scatter(
        df["dtw_distance"],
        y_values,
        c=colors,
        s=42,
        alpha=0.78,
        edgecolors="white",
        linewidth=0.5,
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["failure / 0", "success / 1"])
    ax.set_xlabel("DTW distance to standard template")
    ax.set_ylabel("Success label")
    ax.set_title(f"{method_name} DTW distance vs success label")
    ax.grid(True, linestyle="--", alpha=0.25)

    r_value = summary_df.iloc[0]["pearson_r"]
    p_value = summary_df.iloc[0]["pearson_p_value"]
    sequence_count = int(summary_df.iloc[0]["sequence_count"])
    annotation = f"n={sequence_count}\npearson_r={r_value:.4f}\np={p_value:.4g}"
    ax.text(
        0.98,
        0.02,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#666666"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def process_method(method_dir: Path) -> None:
    """Run the full DTW scoring pipeline for one method folder."""

    if not method_dir.exists():
        logging.warning("Skip missing method directory: %s", method_dir)
        return

    scatter_df = build_method_scatter_dataframe(method_dir)
    if scatter_df.empty:
        logging.warning("No normalized sequences found under %s", method_dir)
        return

    summary_df = compute_pearson_summary(scatter_df)

    scatter_csv_path = method_dir / "dtw_scatter_data.csv"
    summary_csv_path = method_dir / "dtw_pearson_summary.csv"
    scatter_png_path = method_dir / "dtw_distance_scatter.png"

    scatter_df.to_csv(scatter_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    plot_scatter(scatter_df, summary_df, scatter_png_path, method_dir.name)

    r_value = summary_df.iloc[0]["pearson_r"]
    p_value = summary_df.iloc[0]["pearson_p_value"]
    logging.info(
        "[%s] Saved %d DTW scores, Pearson r=%s, p=%s",
        method_dir.name,
        len(scatter_df),
        "nan" if pd.isna(r_value) else f"{float(r_value):.6f}",
        "nan" if pd.isna(p_value) else f"{float(p_value):.6g}",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute DTW-based ADF scores and Pearson correlation scatter plots for GAIPAT mean deviation sequences."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing mean_deviation_sequences. Defaults to <repo-root>/mean_deviation_sequences.",
    )
    parser.add_argument(
        "--method",
        choices=["all", "arithmetic_mean", "dba_mean"],
        default="all",
        help="Which method folder to process.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    input_base_dir = args.input_dir or (args.repo_root / "mean_deviation_sequences")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    method_dirs = find_method_dirs(input_base_dir, args.method)
    for method_dir in method_dirs:
        process_method(method_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()