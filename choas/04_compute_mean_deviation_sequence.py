from __future__ import annotations

import argparse
import importlib
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


FILE_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>[^_]+)_(?P<task>[^_]+)_(?P<step_id>-?\d+)_(?P<event>grasp|release)_(?P<block_id>-?\d+)_(?P<label>[01])$"
)

DEVIATION_COLUMN_CANDIDATES = ["deviation_cm", "deviation_dst_cm"]

INPUT_COLUMNS = ["timestamp", "deviation_dst_cm"]

OUTPUT_COLUMNS = ["normalized_progress", "deviation_dst_cm"]


def require_dba_backend() -> None:
    """Fail fast with a clear message when tslearn is unavailable."""

    try:
        importlib.import_module("tslearn.barycenters")
    except ImportError as exc:
        raise RuntimeError(
            "tslearn is required for DBA correction. Install it with: pip install tslearn"
        ) from exc


def load_dtw_barycenter_averaging():
    """Load tslearn's DBA implementation lazily."""

    require_dba_backend()
    return importlib.import_module("tslearn.barycenters").dtw_barycenter_averaging


def find_deviation_files(input_dir: Path) -> List[Path]:
    """Find deviation sequence CSV files from step 3."""

    return sorted(path for path in input_dir.rglob("*.csv") if path.is_file())


def parse_file_metadata(file_path: Path) -> Optional[dict[str, str]]:
    """Parse metadata encoded in the file name."""

    match = FILE_NAME_PATTERN.match(file_path.stem)
    if match is None:
        return None
    return match.groupdict()


def load_deviation_dataframe(file_path: Path) -> pd.DataFrame:
    """Load one deviation sequence file."""

    return pd.read_csv(file_path)


def resolve_deviation_column(df: pd.DataFrame) -> Optional[str]:
    """Choose the preferred deviation column from an input file."""

    for column in DEVIATION_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def extract_valid_deviation_series(df: pd.DataFrame, deviation_column: str) -> pd.DataFrame:
    """Return the timestamped deviation rows with valid values only."""

    valid_df = df.loc[df[deviation_column].notna(), ["timestamp", deviation_column]].copy()
    if valid_df.empty:
        return valid_df

    valid_df["timestamp"] = pd.to_numeric(valid_df["timestamp"], errors="coerce")
    valid_df[deviation_column] = pd.to_numeric(valid_df[deviation_column], errors="coerce")
    valid_df = valid_df.dropna(subset=["timestamp", deviation_column])
    valid_df = valid_df.sort_values("timestamp", kind="mergesort")
    valid_df = valid_df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    valid_df = valid_df.rename(columns={deviation_column: "deviation_dst_cm"})
    return valid_df


def normalize_deviation_sequence(valid_df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """Normalize one deviation sequence to the target length using progress-based interpolation."""

    if valid_df.empty or target_length <= 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    timestamps = valid_df["timestamp"].to_numpy(dtype=np.float64)
    values = valid_df["deviation_dst_cm"].to_numpy(dtype=np.float64)

    if len(values) == 1:
        normalized_progress = np.linspace(0.0, 1.0, target_length)
        normalized_values = np.full(target_length, values[0], dtype=np.float64)
    else:
        time_span = timestamps[-1] - timestamps[0]
        if time_span == 0:
            normalized_progress = np.linspace(0.0, 1.0, target_length)
            normalized_values = np.full(target_length, values[-1], dtype=np.float64)
        else:
            progress = (timestamps - timestamps[0]) / time_span
            standard_grid = np.linspace(0.0, 1.0, target_length)
            normalized_progress = standard_grid
            normalized_values = np.interp(standard_grid, progress, values)

    return pd.DataFrame(
        {
            "normalized_progress": normalized_progress,
            "deviation_dst_cm": normalized_values,
        }
    )


def resample_deviation_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    """Resample a raw deviation sequence to a fixed length using linear interpolation."""

    if target_length <= 0:
        return np.asarray([], dtype=np.float64)

    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)

    if values.size == 1:
        return np.full(target_length, values[0], dtype=np.float64)

    source_progress = np.linspace(0.0, 1.0, values.size)
    target_progress = np.linspace(0.0, 1.0, target_length)
    return np.interp(target_progress, source_progress, values)


def summarize_lengths(lengths: List[int], target_length: int, source_label: str) -> pd.DataFrame:
    """Build a one-row summary table for sequence lengths."""

    if not lengths:
        return pd.DataFrame(
            [
                {
                    "source_label": source_label,
                    "sequence_count": 0,
                    "min_length": 0,
                    "q25_length": 0,
                    "median_length": 0,
                    "mean_length": 0.0,
                    "q75_length": 0,
                    "max_length": 0,
                    "target_length_L": target_length,
                }
            ]
        )

    series = pd.Series(lengths, dtype="float64")
    return pd.DataFrame(
        [
            {
                "source_label": source_label,
                "sequence_count": int(series.size),
                "min_length": int(series.min()),
                "q25_length": int(series.quantile(0.25)),
                "median_length": int(series.median()),
                "mean_length": float(series.mean()),
                "q75_length": int(series.quantile(0.75)),
                "max_length": int(series.max()),
                "target_length_L": int(target_length),
            }
        ]
    )


def choose_target_length(lengths: List[int], target_length: Optional[int] = None) -> int:
    """Choose a reasonable fixed length based on observed sequence lengths."""

    if target_length is not None and target_length > 0:
        return int(target_length)

    if not lengths:
        return 100

    median_length = float(np.median(lengths))
    rounded_length = int(round(median_length / 10.0) * 10)
    return max(20, rounded_length)


def resolve_method_target_length(
    lengths: List[int],
    method_target_length: Optional[int],
    shared_target_length: Optional[int],
) -> int:
    """Resolve a method-specific length, falling back to a shared override or data-driven choice."""

    if method_target_length is not None and method_target_length > 0:
        return int(method_target_length)
    if shared_target_length is not None and shared_target_length > 0:
        return int(shared_target_length)
    return choose_target_length(lengths, None)


def process_file(
    file_path: Path,
    output_root: Path,
    target_length: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[dict[str, str]], int]:
    """Normalize one deviation file and return its normalized values if it is a success sample."""

    metadata = parse_file_metadata(file_path)
    if metadata is None:
        logging.info("Skip unexpected file name: %s", file_path.name)
        return None, None, None, 0

    df = load_deviation_dataframe(file_path)
    if df.empty:
        logging.info("Skip empty file: %s", file_path)
        return None, None, metadata, 0

    deviation_column = resolve_deviation_column(df)
    if deviation_column is None or "timestamp" not in df.columns:
        missing_columns = ["timestamp"] if "timestamp" not in df.columns else []
        if deviation_column is None:
            missing_columns.append("deviation_cm/deviation_dst_cm")
        logging.warning("Skip %s because columns are missing: %s", file_path, ", ".join(missing_columns))
        return None, None, metadata, 0

    valid_df = extract_valid_deviation_series(df, deviation_column)
    if valid_df.empty:
        logging.info("No valid deviation rows in %s", file_path)
        return None, None, metadata, 0

    normalized_df = normalize_deviation_sequence(valid_df, target_length)
    if normalized_df.empty:
        return None, None, metadata, 0

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / file_path.name
    normalized_df.to_csv(output_path, index=False)

    normalized_vector = normalized_df["deviation_dst_cm"].to_numpy(dtype=np.float64)
    raw_vector = valid_df["deviation_dst_cm"].to_numpy(dtype=np.float64)
    return normalized_vector, raw_vector, metadata, len(valid_df)


def collect_success_lengths(deviation_files: List[Path]) -> List[int]:
    """Collect valid sequence lengths from successful slices only."""

    success_lengths: List[int] = []
    for file_path in deviation_files:
        metadata = parse_file_metadata(file_path)
        if metadata is None or metadata.get("label") != "1":
            continue

        df = load_deviation_dataframe(file_path)
        deviation_column = resolve_deviation_column(df)
        if deviation_column is None:
            continue

        valid_df = extract_valid_deviation_series(df, deviation_column)
        if valid_df.empty:
            continue

        success_lengths.append(int(len(valid_df)))

    return success_lengths


def compute_dba_barycenter(success_sequences: List[np.ndarray], target_length: int) -> np.ndarray:
    """Compute the DBA barycenter from successful raw sequences."""

    dtw_barycenter_averaging = load_dtw_barycenter_averaging()

    if not success_sequences:
        return np.asarray([], dtype=np.float64)

    raw_sequences = [np.asarray(sequence, dtype=np.float64) for sequence in success_sequences if np.asarray(sequence).size > 0]
    if not raw_sequences:
        return np.asarray([], dtype=np.float64)

    raw_sequences_3d = [sequence[:, np.newaxis] for sequence in raw_sequences]
    barycenter_kwargs = {
        "max_iter": 50,
    }
    try:
        barycenter_kwargs["barycenter_size"] = target_length
        barycenter = dtw_barycenter_averaging(raw_sequences_3d, **barycenter_kwargs)
    except (TypeError, ValueError):
        barycenter_kwargs.pop("barycenter_size", None)
        try:
            barycenter = dtw_barycenter_averaging(raw_sequences_3d, **barycenter_kwargs)
        except Exception:
            fallback_sequences = [
                resample_deviation_sequence(sequence, target_length)[:, np.newaxis]
                for sequence in raw_sequences
            ]
            fallback_sequences = [sequence for sequence in fallback_sequences if sequence.size > 0]
            if not fallback_sequences:
                return np.asarray([], dtype=np.float64)
            stacked = np.asarray(fallback_sequences, dtype=np.float64)
            barycenter = dtw_barycenter_averaging(stacked, **barycenter_kwargs)

    barycenter = np.asarray(barycenter, dtype=np.float64).reshape(-1)
    if barycenter.size != target_length:
        barycenter = resample_deviation_sequence(barycenter, target_length)
    return barycenter


def compute_dba_barycenter_auto(success_sequences: List[np.ndarray]) -> np.ndarray:
    """Compute DBA barycenter with tslearn-chosen output length."""

    dtw_barycenter_averaging = load_dtw_barycenter_averaging()

    raw_sequences = [np.asarray(sequence, dtype=np.float64) for sequence in success_sequences if np.asarray(sequence).size > 0]
    if not raw_sequences:
        return np.asarray([], dtype=np.float64)

    raw_sequences_3d = [sequence[:, np.newaxis] for sequence in raw_sequences]
    barycenter = dtw_barycenter_averaging(raw_sequences_3d, max_iter=50)
    barycenter = np.asarray(barycenter, dtype=np.float64).reshape(-1)
    return barycenter



def write_method_outputs(
    output_dir: Path,
    template_column_name: str,
    template_values: np.ndarray,
    std_values: np.ndarray,
    target_length: int,
    sequence_count: int,
    source_label: str,
    file_lengths: List[int],
    method_name: str,
) -> None:
    """Write one method-specific result folder."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mean_df = pd.DataFrame(
        {
            "normalized_progress": np.linspace(0.0, 1.0, target_length),
            template_column_name: template_values,
            "std_deviation_dst_cm": std_values,
            "sequence_count": sequence_count,
        }
    )
    mean_df.to_csv(output_dir / "mean_deviation_sequence.csv", index=False)

    summarize_lengths(file_lengths, target_length, source_label).to_csv(
        output_dir / "length_statistics.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "method_name": method_name,
                "result_type": template_column_name,
                "sequence_count": sequence_count,
                "target_length_L": target_length,
            }
        ]
    ).to_csv(output_dir / "processing_summary.csv", index=False)


def run_method_pipeline(
    *,
    method_name: str,
    input_dir: Path,
    output_dir: Path,
    deviation_files: List[Path],
    shared_target_length: Optional[int],
    method_target_length: Optional[int],
    use_dba: bool,
) -> None:
    """Run one independent mean-sequence pipeline."""

    normalized_output_dir = output_dir / "normalized_sequences"
    success_lengths = collect_success_lengths(deviation_files)
    target_length = resolve_method_target_length(success_lengths, method_target_length, shared_target_length)
    source_label = "success_only" if success_lengths else "no_success"

    logging.info("[%s] Observed %d successful sequences", method_name, len(success_lengths))
    logging.info("[%s] Chosen fixed length L = %d", method_name, target_length)

    normalized_success_vectors: List[np.ndarray] = []
    raw_success_vectors: List[np.ndarray] = []
    processed_count = 0

    for file_path in deviation_files:
        normalized_vector, raw_vector, metadata, _ = process_file(file_path, normalized_output_dir, target_length)
        if metadata is None:
            continue
        processed_count += 1
        if metadata["label"] == "1" and normalized_vector is not None and raw_vector is not None:
            normalized_success_vectors.append(normalized_vector)
            raw_success_vectors.append(raw_vector)

    if not normalized_success_vectors:
        logging.warning("[%s] No successful sequences available for mean computation.", method_name)
        return

    stacked = np.vstack(normalized_success_vectors)
    arithmetic_mean_values = stacked.mean(axis=0)
    std_values = stacked.std(axis=0, ddof=0)

    if use_dba:
        template_column_name = "dba_deviation_dst_cm"
        try:
            template_values = compute_dba_barycenter(raw_success_vectors, target_length)
        except RuntimeError as exc:
            logging.warning("[%s] DBA computation skipped: %s", method_name, exc)
            return

        if template_values.size == 0:
            logging.warning("[%s] DBA barycenter is empty; no DBA folder written.", method_name)
            return
    else:
        template_column_name = "arithmetic_mean_deviation_dst_cm"
        template_values = arithmetic_mean_values

    write_method_outputs(
        output_dir=output_dir,
        template_column_name=template_column_name,
        template_values=template_values,
        std_values=std_values,
        target_length=target_length,
        sequence_count=stacked.shape[0],
        source_label=source_label,
        file_lengths=success_lengths,
        method_name=method_name,
    )

    logging.info("[%s] Processed %d files; success sequences used for mean: %d.", method_name, processed_count, len(normalized_success_vectors))


def run_arithmetic_mean_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    deviation_files: List[Path],
    arithmetic_target_length: Optional[int],
    shared_target_length: Optional[int],
) -> None:
    """Run the arithmetic mean pipeline independently."""

    normalized_output_dir = output_dir / "normalized_sequences"
    success_lengths = collect_success_lengths(deviation_files)
    target_length = resolve_method_target_length(success_lengths, arithmetic_target_length, shared_target_length)
    source_label = "success_only" if success_lengths else "no_success"

    logging.info("[arithmetic_mean] Observed %d successful sequences", len(success_lengths))
    logging.info("[arithmetic_mean] Chosen fixed length L = %d", target_length)

    normalized_success_vectors: List[np.ndarray] = []
    processed_count = 0

    for file_path in deviation_files:
        normalized_vector, _, metadata, _ = process_file(file_path, normalized_output_dir, target_length)
        if metadata is None:
            continue
        processed_count += 1
        if metadata["label"] == "1" and normalized_vector is not None:
            normalized_success_vectors.append(normalized_vector)

    if not normalized_success_vectors:
        logging.warning("[arithmetic_mean] No successful sequences available for mean computation.")
        return

    stacked = np.vstack(normalized_success_vectors)
    arithmetic_mean_values = stacked.mean(axis=0)

    write_method_outputs(
        output_dir=output_dir,
        template_column_name="arithmetic_mean_deviation_dst_cm",
        template_values=arithmetic_mean_values,
        std_values=stacked.std(axis=0, ddof=0),
        target_length=target_length,
        sequence_count=stacked.shape[0],
        source_label=source_label,
        file_lengths=success_lengths,
        method_name="arithmetic_mean",
    )

    pd.DataFrame(
        [
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "processed_files": processed_count,
                "success_sequences_used_for_mean": len(normalized_success_vectors),
                "target_length_L": target_length,
                "length_mode": "median_or_override",
            }
        ]
    ).to_csv(output_dir / "processing_summary.csv", index=False)


def run_dba_mean_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    deviation_files: List[Path],
    dba_target_length: Optional[int],
) -> None:
    """Run the DBA mean pipeline independently."""

    normalized_output_dir = output_dir / "normalized_sequences"
    success_lengths = collect_success_lengths(deviation_files)
    raw_success_sequences: List[np.ndarray] = []

    for file_path in deviation_files:
        metadata = parse_file_metadata(file_path)
        if metadata is None or metadata.get("label") != "1":
            continue

        df = load_deviation_dataframe(file_path)
        deviation_column = resolve_deviation_column(df)
        if deviation_column is None:
            continue

        valid_df = extract_valid_deviation_series(df, deviation_column)
        if valid_df.empty:
            continue

        raw_success_sequences.append(valid_df["deviation_dst_cm"].to_numpy(dtype=np.float64))

    if not raw_success_sequences:
        logging.warning("[dba_mean] No successful sequences available for DBA computation.")
        return

    if dba_target_length is not None and dba_target_length > 0:
        target_length = int(dba_target_length)
        dba_template = compute_dba_barycenter(raw_success_sequences, target_length)
    else:
        dba_template = compute_dba_barycenter_auto(raw_success_sequences)
        target_length = int(dba_template.size)

    if target_length <= 0 or dba_template.size == 0:
        logging.warning("[dba_mean] DBA barycenter is empty; DBA folder will not be written.")
        return

    logging.info("[dba_mean] Observed %d successful sequences", len(success_lengths))
    logging.info("[dba_mean] Chosen DBA length L = %d", target_length)

    normalized_success_vectors: List[np.ndarray] = []
    processed_count = 0
    for file_path in deviation_files:
        normalized_vector, _, metadata, _ = process_file(file_path, normalized_output_dir, target_length)
        if metadata is None:
            continue
        processed_count += 1
        if metadata["label"] == "1" and normalized_vector is not None:
            normalized_success_vectors.append(normalized_vector)

    stacked = np.vstack(normalized_success_vectors)
    std_values = stacked.std(axis=0, ddof=0)
    write_method_outputs(
        output_dir=output_dir,
        template_column_name="dba_deviation_dst_cm",
        template_values=dba_template,
        std_values=std_values,
        target_length=target_length,
        sequence_count=stacked.shape[0],
        source_label="success_only" if success_lengths else "no_success",
        file_lengths=success_lengths,
        method_name="dba_mean",
    )

    pd.DataFrame(
        [
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "processed_files": processed_count,
                "success_sequences_used_for_mean": len(normalized_success_vectors),
                "target_length_L": target_length,
                "length_mode": "dba_auto" if dba_target_length is None else "override",
            }
        ]
    ).to_csv(output_dir / "processing_summary.csv", index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Compute mean deviation sequences from normalized GAIPAT deviation files."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default='/root/autodl-tmp/shenxy/XDU/Dataset/gaipat-main',
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
        help="Directory where normalized files and mean sequence will be written. Defaults to <repo-root>/mean_deviation_sequences.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Optional override for the arithmetic-mean pipeline. DBA uses its own auto length unless overridden separately.",
    )
    parser.add_argument(
        "--arithmetic-target-length",
        type=int,
        default=None,
        help="Optional fixed length L for the arithmetic-mean pipeline.",
    )
    parser.add_argument(
        "--dba-target-length",
        type=int,
        default=None,
        help="Optional fixed length L for the DBA pipeline.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    repo_root = args.repo_root
    input_dir = args.input_dir or (repo_root / "block_deviation_sequences")
    output_base_dir = args.output_dir or (repo_root / "mean_deviation_sequences")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    arithmetic_output_dir = output_base_dir / "arithmetic_mean"
    dba_output_dir = output_base_dir / "dba_mean"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_base_dir / "mean_deviation.log"),
            logging.StreamHandler(),
        ],
    )

    deviation_files = find_deviation_files(input_dir)
    if not deviation_files:
        logging.warning("No deviation CSV files found under %s", input_dir)
        return

    run_arithmetic_mean_pipeline(
        input_dir=input_dir,
        output_dir=arithmetic_output_dir,
        deviation_files=deviation_files,
        arithmetic_target_length=args.arithmetic_target_length,
        shared_target_length=args.target_length,
    )

    run_dba_mean_pipeline(
        input_dir=input_dir,
        output_dir=dba_output_dir,
        deviation_files=deviation_files,
        dba_target_length=args.dba_target_length,
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()