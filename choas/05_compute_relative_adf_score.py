from __future__ import annotations

import argparse
import importlib
import logging
import re
from collections import defaultdict
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

DISTANCE_COLUMN_CANDIDATES = ["gaze_target_distance", "deviation_cm", "deviation_dst_cm"]


def require_scipy_gamma() -> None:
    """Fail fast with a clear message when scipy is unavailable."""

    try:
        importlib.import_module("scipy.stats")
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for gamma fitting. Install it with: pip install scipy"
        ) from exc


def load_gamma():
    """Load scipy.stats.gamma lazily."""

    require_scipy_gamma()
    return importlib.import_module("scipy.stats").gamma


def parse_file_metadata(file_path: Path) -> Optional[dict[str, str]]:
    """Parse metadata encoded in the file name."""

    match = FILE_NAME_PATTERN.match(file_path.stem)
    if match is None:
        return None
    return match.groupdict()


def find_input_files(input_dir: Path) -> list[Path]:
    """Find all candidate CSV files under the input directory."""

    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def load_input_dataframe(file_path: Path) -> pd.DataFrame:
    """Load one preprocessed deviation CSV file."""

    return pd.read_csv(file_path)


def resolve_distance_column(df: pd.DataFrame) -> Optional[str]:
    """Choose the preferred distance column from an input file."""

    for column in DISTANCE_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def extract_valid_distance_series(df: pd.DataFrame, distance_column: str) -> pd.DataFrame:
    """Return rows with valid timestamps and gaze-target distances only."""

    working_df = df.loc[df[distance_column].notna()].copy()
    if working_df.empty:
        return pd.DataFrame(columns=["frame", "timestamp", "gaze_target_distance"])

    working_df[distance_column] = pd.to_numeric(working_df[distance_column], errors="coerce")
    working_df = working_df.dropna(subset=[distance_column])

    if "timestamp" in working_df.columns:
        working_df["timestamp"] = pd.to_numeric(working_df["timestamp"], errors="coerce")
        working_df = working_df.dropna(subset=["timestamp"])
        working_df = working_df.sort_values("timestamp", kind="mergesort")
        working_df = working_df.drop_duplicates(subset=["timestamp"], keep="first")
        working_df = working_df.reset_index(drop=True)
        timestamp_values = working_df["timestamp"].to_numpy(dtype=np.float64)
    else:
        working_df = working_df.reset_index(drop=True)
        timestamp_values = np.arange(len(working_df), dtype=np.float64)

    working_df = working_df.reset_index(drop=True)
    working_df = working_df.assign(
        frame=np.arange(len(working_df), dtype=np.int64),
        timestamp=timestamp_values,
        gaze_target_distance=working_df[distance_column].to_numpy(dtype=np.float64),
    )
    return working_df[["frame", "timestamp", "gaze_target_distance"]]


def collect_file_infos(input_files: list[Path]) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    """Load all valid files and collect success-only values per subject."""

    file_infos: list[dict[str, object]] = []
    success_values_by_subject: dict[str, list[np.ndarray]] = defaultdict(list)

    for file_path in input_files:
        metadata = parse_file_metadata(file_path)
        if metadata is None:
            logging.info("Skip unexpected file name: %s", file_path.name)
            continue

        df = load_input_dataframe(file_path)
        if df.empty:
            logging.info("Skip empty file: %s", file_path)
            continue

        distance_column = resolve_distance_column(df)
        if distance_column is None:
            logging.warning(
                "Skip %s because none of the supported distance columns were found: %s",
                file_path,
                ", ".join(DISTANCE_COLUMN_CANDIDATES),
            )
            continue

        valid_df = extract_valid_distance_series(df, distance_column)
        if valid_df.empty:
            logging.info("No valid distance rows in %s", file_path)
            continue

        file_infos.append(
            {
                "file_path": file_path,
                "metadata": metadata,
                "distance_column": distance_column,
                "valid_df": valid_df,
            }
        )

        if metadata["label"] == "1":
            success_values_by_subject[metadata["subject_id"]].append(
                valid_df["gaze_target_distance"].to_numpy(dtype=np.float64)
            )

    return file_infos, success_values_by_subject


def fit_subject_gamma(values: np.ndarray) -> dict[str, object]:
    """Fit a Gamma distribution and return its mean baseline."""

    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]

    if finite_values.size == 0:
        return {
            "gamma_shape": np.nan,
            "gamma_loc": np.nan,
            "gamma_scale": np.nan,
            "individual_mean_baseline": np.nan,
            "fit_status": "no_success_samples",
            "fit_sample_count": 0,
        }

    positive_values = finite_values[finite_values > 0]
    if positive_values.size > 0:
        finite_values = positive_values

    if finite_values.size == 1 or np.allclose(finite_values, finite_values[0]):
        mean_value = float(np.mean(finite_values))
        return {
            "gamma_shape": np.nan,
            "gamma_loc": np.nan,
            "gamma_scale": np.nan,
            "individual_mean_baseline": mean_value,
            "fit_status": "degenerate_empirical_mean",
            "fit_sample_count": int(finite_values.size),
        }

    gamma = load_gamma()
    try:
        gamma_shape, gamma_loc, gamma_scale = gamma.fit(finite_values)
        individual_mean = float(gamma_shape * gamma_scale + gamma_loc)
        if not np.isfinite(individual_mean):
            raise ValueError("Non-finite gamma mean")
        return {
            "gamma_shape": float(gamma_shape),
            "gamma_loc": float(gamma_loc),
            "gamma_scale": float(gamma_scale),
            "individual_mean_baseline": individual_mean,
            "fit_status": "gamma_fit_success",
            "fit_sample_count": int(finite_values.size),
        }
    except Exception as exc:
        logging.warning("Gamma fit failed; using empirical mean fallback: %s", exc)
        return {
            "gamma_shape": np.nan,
            "gamma_loc": np.nan,
            "gamma_scale": np.nan,
            "individual_mean_baseline": float(np.mean(finite_values)),
            "fit_status": "gamma_fit_failed_empirical_mean",
            "fit_sample_count": int(finite_values.size),
        }


def build_subject_baseline_table(
    file_infos: list[dict[str, object]],
    success_values_by_subject: dict[str, list[np.ndarray]],
) -> pd.DataFrame:
    """Fit one Gamma baseline per subject."""

    subject_ids = sorted(
        {str(info["metadata"]["subject_id"]) for info in file_infos} | set(success_values_by_subject)
    )

    subject_file_counts: dict[str, int] = defaultdict(int)
    subject_success_file_counts: dict[str, int] = defaultdict(int)
    subject_total_valid_counts: dict[str, int] = defaultdict(int)
    subject_success_valid_counts: dict[str, int] = defaultdict(int)

    for info in file_infos:
        metadata = info["metadata"]
        subject_id = str(metadata["subject_id"])
        valid_df = info["valid_df"]
        subject_file_counts[subject_id] += 1
        subject_total_valid_counts[subject_id] += int(len(valid_df))
        if metadata["label"] == "1":
            subject_success_file_counts[subject_id] += 1
            subject_success_valid_counts[subject_id] += int(len(valid_df))

    records: list[dict[str, object]] = []
    for subject_id in subject_ids:
        success_values = success_values_by_subject.get(subject_id, [])
        if success_values:
            fit_input = np.concatenate(success_values)
        else:
            fit_input = np.asarray([], dtype=np.float64)

        fit_result = fit_subject_gamma(fit_input)
        records.append(
            {
                "subject_id": subject_id,
                "file_count": int(subject_file_counts.get(subject_id, 0)),
                "success_file_count": int(subject_success_file_counts.get(subject_id, 0)),
                "valid_frame_count": int(subject_total_valid_counts.get(subject_id, 0)),
                "success_valid_frame_count": int(subject_success_valid_counts.get(subject_id, 0)),
                **fit_result,
            }
        )

    return pd.DataFrame(records)


def build_relative_distance_dataframe(
    file_infos: list[dict[str, object]],
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast each subject baseline to all of their valid samples."""

    baseline_lookup = baseline_df.set_index("subject_id").to_dict(orient="index")
    records: list[dict[str, object]] = []

    for info in file_infos:
        metadata = info["metadata"]
        file_path = info["file_path"]
        valid_df = info["valid_df"]
        subject_id = str(metadata["subject_id"])
        baseline_info = baseline_lookup.get(subject_id)

        if baseline_info is None:
            logging.warning("Missing baseline for subject %s; relative distance will be NaN.", subject_id)

        baseline_value = (
            float(baseline_info["individual_mean_baseline"])
            if baseline_info is not None and pd.notna(baseline_info["individual_mean_baseline"])
            else np.nan
        )

        gamma_shape = float(baseline_info["gamma_shape"]) if baseline_info is not None and pd.notna(baseline_info["gamma_shape"]) else np.nan
        gamma_loc = float(baseline_info["gamma_loc"]) if baseline_info is not None and pd.notna(baseline_info["gamma_loc"]) else np.nan
        gamma_scale = float(baseline_info["gamma_scale"]) if baseline_info is not None and pd.notna(baseline_info["gamma_scale"]) else np.nan
        fit_status = str(baseline_info["fit_status"]) if baseline_info is not None else "missing_baseline"

        for row in valid_df.itertuples(index=False):
            gaze_target_distance = float(row.gaze_target_distance)
            relative_distance = gaze_target_distance - baseline_value if np.isfinite(baseline_value) else np.nan

            records.append(
                {
                    "subject_id": subject_id,
                    "task": metadata["task"],
                    "step_id": int(metadata["step_id"]),
                    "event": metadata["event"],
                    "block_id": int(metadata["block_id"]),
                    "label": int(metadata["label"]),
                    "is_success": int(metadata["label"]),
                    "frame": int(row.frame),
                    "timestamp": float(row.timestamp),
                    "gaze_target_distance": gaze_target_distance,
                    "individual_mean_baseline": baseline_value,
                    "adf_relative_distance": relative_distance,
                    "gamma_shape": gamma_shape,
                    "gamma_loc": gamma_loc,
                    "gamma_scale": gamma_scale,
                    "fit_status": fit_status,
                    "distance_column": info["distance_column"],
                    "sequence_length": int(len(valid_df)),
                    "sequence_file": file_path.name,
                }
            )

    return pd.DataFrame(records)


def summarize_relative_distances(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row summary statistics for the relative distance table."""

    valid_df = df.loc[df["adf_relative_distance"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame(
            [
                {
                    "subject_count": int(baseline_df.shape[0]),
                    "row_count": 0,
                    "success_row_count": 0,
                    "failure_row_count": 0,
                    "mean_relative_distance_success": np.nan,
                    "mean_relative_distance_failure": np.nan,
                    "median_relative_distance_success": np.nan,
                    "median_relative_distance_failure": np.nan,
                }
            ]
        )

    success_df = valid_df.loc[valid_df["label"] == 1]
    failure_df = valid_df.loc[valid_df["label"] == 0]

    return pd.DataFrame(
        [
            {
                "subject_count": int(baseline_df.shape[0]),
                "row_count": int(valid_df.shape[0]),
                "success_row_count": int(success_df.shape[0]),
                "failure_row_count": int(failure_df.shape[0]),
                "mean_relative_distance_success": float(success_df["adf_relative_distance"].mean()) if not success_df.empty else np.nan,
                "mean_relative_distance_failure": float(failure_df["adf_relative_distance"].mean()) if not failure_df.empty else np.nan,
                "median_relative_distance_success": float(success_df["adf_relative_distance"].median()) if not success_df.empty else np.nan,
                "median_relative_distance_failure": float(failure_df["adf_relative_distance"].median()) if not failure_df.empty else np.nan,
            }
        ]
    )


def plot_relative_distance_distribution(df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    """Draw a distribution plot for the relative ADF distance."""

    plot_df = df.loc[df["adf_relative_distance"].notna()].copy()
    if plot_df.empty:
        return

    success_values = plot_df.loc[plot_df["label"] == 1, "adf_relative_distance"].to_numpy(dtype=np.float64)
    failure_values = plot_df.loc[plot_df["label"] == 0, "adf_relative_distance"].to_numpy(dtype=np.float64)
    all_values = plot_df["adf_relative_distance"].to_numpy(dtype=np.float64)

    lower = float(np.nanpercentile(all_values, 1))
    upper = float(np.nanpercentile(all_values, 99))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(all_values)) if np.isfinite(np.nanmin(all_values)) else -1.0
        upper = float(np.nanmax(all_values)) if np.isfinite(np.nanmax(all_values)) else 1.0
    padding = max((upper - lower) * 0.08, 1e-6)
    lower -= padding
    upper += padding
    bins = np.linspace(lower, upper, 60)

    fig, ax = plt.subplots(figsize=(10, 6))

    if success_values.size > 0:
        ax.hist(
            success_values,
            bins=bins,
            density=True,
            alpha=0.60,
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.4,
            label=f"success / 1 (n={success_values.size})",
        )

    if failure_values.size > 0:
        ax.hist(
            failure_values,
            bins=bins,
            density=True,
            alpha=0.55,
            color="#d62728",
            edgecolor="white",
            linewidth=0.4,
            label=f"failure / 0 (n={failure_values.size})",
        )

    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.2, label="subject baseline")
    ax.set_xlabel("ADF Relative Distance = gaze_target_distance - μ_individual")
    ax.set_ylabel("Density")
    ax.set_title("Relative ADF distance distribution")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True)

    summary_row = summary_df.iloc[0]
    annotation = (
        f"subjects={int(summary_row['subject_count'])}\n"
        f"rows={int(summary_row['row_count'])}\n"
        f"success_mean={summary_row['mean_relative_distance_success']:.4f}\n"
        f"failure_mean={summary_row['mean_relative_distance_failure']:.4f}"
    )
    ax.text(
        0.98,
        0.98,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.88, edgecolor="#666666"),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_subject_relative_distance_distribution(subject_df: pd.DataFrame, output_path: Path, subject_id: str) -> None:
    """Draw a per-subject distribution plot for the relative ADF distance."""

    plot_df = subject_df.loc[subject_df["adf_relative_distance"].notna()].copy()
    if plot_df.empty:
        return

    success_values = plot_df.loc[plot_df["label"] == 1, "adf_relative_distance"].to_numpy(dtype=np.float64)
    failure_values = plot_df.loc[plot_df["label"] == 0, "adf_relative_distance"].to_numpy(dtype=np.float64)
    all_values = plot_df["adf_relative_distance"].to_numpy(dtype=np.float64)

    lower = float(np.nanpercentile(all_values, 1))
    upper = float(np.nanpercentile(all_values, 99))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(all_values)) if np.isfinite(np.nanmin(all_values)) else -1.0
        upper = float(np.nanmax(all_values)) if np.isfinite(np.nanmax(all_values)) else 1.0
    padding = max((upper - lower) * 0.08, 1e-6)
    lower -= padding
    upper += padding
    bins = np.linspace(lower, upper, 50)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if success_values.size > 0:
        ax.hist(
            success_values,
            bins=bins,
            density=True,
            alpha=0.62,
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.4,
            label=f"success / 1 (n={success_values.size})",
        )

    if failure_values.size > 0:
        ax.hist(
            failure_values,
            bins=bins,
            density=True,
            alpha=0.58,
            color="#d62728",
            edgecolor="white",
            linewidth=0.4,
            label=f"failure / 0 (n={failure_values.size})",
        )

    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.2, label="subject baseline")
    ax.set_xlabel("ADF Relative Distance = gaze_target_distance - μ_individual")
    ax.set_ylabel("Density")
    ax.set_title(f"Subject {subject_id} relative ADF distance distribution")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_subject_summary_table(relative_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-subject relative distance statistics."""

    valid_df = relative_df.loc[relative_df["adf_relative_distance"].notna()].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "row_count",
                "success_row_count",
                "failure_row_count",
                "mean_relative_distance",
                "median_relative_distance",
                "mean_relative_distance_success",
                "mean_relative_distance_failure",
            ]
        )

    records: list[dict[str, object]] = []
    for subject_id, subject_df in valid_df.groupby("subject_id", sort=True):
        success_df = subject_df.loc[subject_df["label"] == 1]
        failure_df = subject_df.loc[subject_df["label"] == 0]
        records.append(
            {
                "subject_id": subject_id,
                "row_count": int(subject_df.shape[0]),
                "success_row_count": int(success_df.shape[0]),
                "failure_row_count": int(failure_df.shape[0]),
                "mean_relative_distance": float(subject_df["adf_relative_distance"].mean()),
                "median_relative_distance": float(subject_df["adf_relative_distance"].median()),
                "mean_relative_distance_success": float(success_df["adf_relative_distance"].mean()) if not success_df.empty else np.nan,
                "mean_relative_distance_failure": float(failure_df["adf_relative_distance"].mean()) if not failure_df.empty else np.nan,
            }
        )

    return pd.DataFrame(records)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute subject-wise Gamma baselines and relative ADF distances for GAIPAT deviation sequences."
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
        help="Directory containing step-3 deviation CSV files. Defaults to <repo-root>/block_deviation_sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the relative ADF outputs will be written. Defaults to <repo-root>/relative_adf_sequences.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    repo_root = args.repo_root
    input_dir = args.input_dir or (repo_root / "block_deviation_sequences")
    output_dir = args.output_dir or (repo_root / "relative_adf_sequences")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "relative_adf.log"),
            logging.StreamHandler(),
        ],
    )

    input_files = find_input_files(input_dir)
    if not input_files:
        logging.warning("No deviation CSV files found under %s", input_dir)
        return

    file_infos, success_values_by_subject = collect_file_infos(input_files)
    if not file_infos:
        logging.warning("No valid deviation rows were found under %s", input_dir)
        return

    baseline_df = build_subject_baseline_table(file_infos, success_values_by_subject)
    relative_df = build_relative_distance_dataframe(file_infos, baseline_df)

    summary_df = summarize_relative_distances(relative_df, baseline_df)
    subject_summary_df = build_subject_summary_table(relative_df)

    baseline_path = output_dir / "subject_gamma_baselines.csv"
    relative_csv_path = output_dir / "relative_adf_distance_data.csv"
    summary_csv_path = output_dir / "relative_adf_distance_summary.csv"
    subject_summary_path = output_dir / "subject_relative_adf_summary.csv"
    plot_path = output_dir / "relative_adf_distance_distribution.png"
    subject_plot_dir = output_dir / "subject_relative_adf_plots"

    baseline_df.to_csv(baseline_path, index=False)
    relative_df.to_csv(relative_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    subject_summary_df.to_csv(subject_summary_path, index=False)
    plot_relative_distance_distribution(relative_df, summary_df, plot_path)

    subject_plot_dir.mkdir(parents=True, exist_ok=True)
    for subject_id, subject_df in relative_df.groupby("subject_id", sort=True):
        subject_plot_path = subject_plot_dir / f"relative_adf_distance_distribution_{subject_id}.png"
        plot_subject_relative_distance_distribution(subject_df, subject_plot_path, str(subject_id))

    logging.info("Saved %d subject baselines to %s", len(baseline_df), baseline_path.name)
    logging.info("Saved %d relative ADF rows to %s", len(relative_df), relative_csv_path.name)
    logging.info("Saved %d subject summary rows to %s", len(subject_summary_df), subject_summary_path.name)
    logging.info("Done.")


if __name__ == "__main__":
    main()