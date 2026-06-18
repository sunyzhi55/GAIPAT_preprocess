from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FILE_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>[^_]+)_(?P<task>[^_]+)_(?P<step_id>-?\d+)_(?P<event>grasp|release)_(?P<block_id>-?\d+)_(?P<label>[01])$"
)

DEVIATION_COLUMN_CANDIDATES = ["deviation_cm", "deviation_dst_cm"]

ALWAYS_CONSTANT_FIELDS = {"event"}
GEOMETRY_CONSTANT_FIELDS = {
    "target_x_y_table_srt_cm",
    "target_x_y_table_dst_cm",
}
METHOD_NAMES = ("arithmetic_mean", "dba_mean")


@dataclass(frozen=True)
class FieldPlan:
    name: str
    strategy: str
    reason: str
    shape: tuple[int, ...] | None = None
    constant_value: Any = None


def find_jsonl_files(input_dir: Path) -> list[Path]:
    """Find step-03 deviation sequence JSONL files."""

    return sorted(path for path in input_dir.rglob("*.jsonl") if path.is_file())


def parse_file_metadata(file_path: Path) -> dict[str, str] | None:
    """Parse metadata encoded in the sequence file name."""

    match = FILE_NAME_PATTERN.match(file_path.stem)
    if match is None:
        return None
    return match.groupdict()


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


def json_safe_value(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python objects."""

    if isinstance(value, np.ndarray):
        return [json_safe_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def stable_json(value: Any) -> str:
    """Build a stable string representation for equality checks."""

    return json.dumps(json_safe_value(value), sort_keys=True, ensure_ascii=False)


def is_finite_number(value: Any) -> bool:
    """Return True for finite real scalars, excluding booleans."""

    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return False


def value_to_numeric_array(value: Any) -> np.ndarray | None:
    """Convert a numeric scalar/list tree to a float array, or None if not numeric."""

    if is_finite_number(value):
        return np.asarray(value, dtype="float64")

    if not isinstance(value, list):
        return None

    try:
        array = np.asarray(value, dtype="float64")
    except (TypeError, ValueError):
        return None

    if array.dtype == object or not np.all(np.isfinite(array)):
        return None
    return array


def read_jsonl_records(file_path: Path) -> list[dict[str, Any]]:
    """Read one JSON object per line."""

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number} contains invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def write_jsonl_records(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    """Write records as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe_value(record), ensure_ascii=False))
            handle.write("\n")


def extract_progress_axis(records: list[dict[str, Any]]) -> np.ndarray:
    """Build a monotonic 0..1 source progress axis from timestamps or row index."""

    if not records:
        return np.asarray([], dtype="float64")
    if len(records) == 1:
        return np.asarray([0.0], dtype="float64")

    timestamps = [record.get("timestamp") for record in records]
    if all(is_finite_number(value) for value in timestamps):
        timestamp_values = np.asarray(timestamps, dtype="float64")
        span = float(timestamp_values[-1] - timestamp_values[0])
        if span > 0:
            progress = (timestamp_values - timestamp_values[0]) / span
            if np.all(np.diff(progress) >= 0):
                return progress

    return np.linspace(0.0, 1.0, len(records), dtype="float64")


def make_target_grid(target_length: int) -> np.ndarray:
    """Create the fixed 0..1 output progress grid."""

    if target_length <= 0:
        return np.asarray([], dtype="float64")
    if target_length == 1:
        return np.asarray([0.0], dtype="float64")
    return np.linspace(0.0, 1.0, target_length, dtype="float64")


def interpolate_1d(source_progress: np.ndarray, values: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate one numeric vector while skipping missing values."""

    values = values.astype("float64")
    valid_mask = np.isfinite(values) & np.isfinite(source_progress)
    if not valid_mask.any():
        return np.full(len(target_grid), np.nan, dtype="float64")
    if valid_mask.sum() == 1:
        return np.full(len(target_grid), float(values[valid_mask][0]), dtype="float64")
    return np.interp(target_grid, source_progress[valid_mask], values[valid_mask])


def interpolate_numeric_array_field(
    records: list[dict[str, Any]],
    field_name: str,
    field_shape: tuple[int, ...],
    source_progress: np.ndarray,
    target_grid: np.ndarray,
) -> list[Any]:
    """Interpolate a scalar or fixed-shape numeric array field."""

    if field_shape == ():
        values = np.asarray(
            [
                float(record[field_name])
                if field_name in record and is_finite_number(record.get(field_name))
                else np.nan
                for record in records
            ],
            dtype="float64",
        )
        return [json_safe_value(value) for value in interpolate_1d(source_progress, values, target_grid)]

    source_values = np.full((len(records), *field_shape), np.nan, dtype="float64")
    for index, record in enumerate(records):
        if field_name not in record:
            continue
        array = value_to_numeric_array(record.get(field_name))
        if array is not None and array.shape == field_shape:
            source_values[index] = array

    output = np.full((len(target_grid), *field_shape), np.nan, dtype="float64")
    for array_index in np.ndindex(field_shape):
        values = source_values[(slice(None), *array_index)]
        output[(slice(None), *array_index)] = interpolate_1d(source_progress, values, target_grid)
    return [json_safe_value(item) for item in output]


def non_null_values(records: list[dict[str, Any]], field_name: str) -> list[Any]:
    """Collect non-null field values in sequence order."""

    values: list[Any] = []
    for record in records:
        if field_name not in record:
            continue
        value = record[field_name]
        if value is not None:
            values.append(value)
    return values


def is_constant_values(values: list[Any]) -> bool:
    """Return True when all provided non-null values are identical."""

    if not values:
        return False
    first = stable_json(values[0])
    return all(stable_json(value) == first for value in values[1:])


def infer_field_plan(records: list[dict[str, Any]], field_name: str) -> FieldPlan:
    """Infer how a field should be normalized."""

    values = non_null_values(records, field_name)
    if not values:
        return FieldPlan(field_name, "null", "field is missing or null in every row")

    if field_name in ALWAYS_CONSTANT_FIELDS:
        if is_constant_values(values):
            return FieldPlan(field_name, "constant", "categorical field is constant", constant_value=values[0])
        return FieldPlan(field_name, "null", "categorical field has inconsistent values")

    if field_name in GEOMETRY_CONSTANT_FIELDS:
        if is_constant_values(values):
            return FieldPlan(field_name, "constant", "geometry field is constant within the slice", constant_value=values[0])
        return FieldPlan(field_name, "null", "geometry field changed within the slice")

    numeric_arrays = [value_to_numeric_array(value) for value in values]
    if all(array is not None for array in numeric_arrays):
        shapes = [array.shape for array in numeric_arrays if array is not None]
        if len(set(shapes)) == 1:
            return FieldPlan(field_name, "linear", "finite numeric scalar/array field", shape=shapes[0])
        return FieldPlan(field_name, "null", "numeric array shape is inconsistent")

    if is_constant_values(values):
        return FieldPlan(field_name, "constant", "non-numeric field is constant", constant_value=values[0])

    return FieldPlan(field_name, "null", "non-numeric or mixed field cannot be interpolated")


def infer_field_plans(records: list[dict[str, Any]]) -> list[FieldPlan]:
    """Infer normalization plans for all fields in stable first-seen order."""

    field_names: list[str] = []
    seen: set[str] = set()
    for record in records:
        for field_name in record:
            if field_name not in seen:
                seen.add(field_name)
                field_names.append(field_name)
    return [infer_field_plan(records, field_name) for field_name in field_names]


def build_normalized_records(
    records: list[dict[str, Any]],
    target_length: int,
    *,
    metadata: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[FieldPlan]]:
    """Normalize one sequence to fixed length while preserving JSONL fields."""

    if not records or target_length <= 0:
        return [], []

    source_progress = extract_progress_axis(records)
    target_grid = make_target_grid(target_length)
    field_plans = infer_field_plans(records)
    field_values: dict[str, list[Any]] = {}

    for plan in field_plans:
        if plan.strategy == "linear" and plan.shape is not None:
            field_values[plan.name] = interpolate_numeric_array_field(
                records,
                plan.name,
                plan.shape,
                source_progress,
                target_grid,
            )
        elif plan.strategy == "constant":
            field_values[plan.name] = [json_safe_value(plan.constant_value) for _ in target_grid]
        else:
            field_values[plan.name] = [None for _ in target_grid]

    output_records: list[dict[str, Any]] = []
    for index, normalized_progress in enumerate(target_grid):
        output_record: dict[str, Any] = {
            "normalized_progress": float(normalized_progress),
        }
        if metadata is not None:
            output_record["sequence_id"] = metadata.get("sequence_id")
            output_record["subject_id"] = metadata.get("subject_id")
            output_record["task"] = metadata.get("task")
            output_record["step_id"] = int(metadata["step_id"]) if metadata.get("step_id") is not None else None
            output_record["block_id"] = int(metadata["block_id"]) if metadata.get("block_id") is not None else None
            output_record["label"] = int(metadata["label"]) if metadata.get("label") is not None else None

        for plan in field_plans:
            output_record[plan.name] = field_values[plan.name][index]
        output_records.append(output_record)

    return output_records, field_plans


def extract_deviation_values(records: list[dict[str, Any]]) -> np.ndarray:
    """Extract the preferred finite deviation sequence."""

    for column_name in DEVIATION_COLUMN_CANDIDATES:
        values: list[float] = []
        for record in records:
            value = record.get(column_name)
            if is_finite_number(value):
                values.append(float(value))
        if values:
            return np.asarray(values, dtype="float64")
    return np.asarray([], dtype="float64")


def collect_success_lengths(jsonl_files: list[Path]) -> list[int]:
    """Collect valid sequence lengths from successful samples only."""

    lengths: list[int] = []
    for file_path in jsonl_files:
        metadata = parse_file_metadata(file_path)
        if metadata is None or metadata.get("label") != "1":
            continue
        records = read_jsonl_records(file_path)
        deviation_values = extract_deviation_values(records)
        if deviation_values.size:
            lengths.append(int(len(records)))
    return lengths


def choose_target_length(lengths: list[int], target_length: int | None = None) -> int:
    """Choose a fixed length based on successful sequence lengths."""

    if target_length is not None and target_length > 0:
        return int(target_length)
    if not lengths:
        return 100
    median_length = float(np.median(lengths))
    rounded_length = int(round(median_length / 10.0) * 10)
    return max(20, rounded_length)


def resolve_method_target_length(
    lengths: list[int],
    method_target_length: int | None,
    shared_target_length: int | None,
) -> int:
    """Resolve method-specific target length."""

    if method_target_length is not None and method_target_length > 0:
        return int(method_target_length)
    if shared_target_length is not None and shared_target_length > 0:
        return int(shared_target_length)
    return choose_target_length(lengths, None)


def resample_deviation_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    """Resample one raw deviation sequence to fixed length."""

    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if target_length <= 0 or values.size == 0:
        return np.asarray([], dtype="float64")
    if values.size == 1:
        return np.full(target_length, float(values[0]), dtype="float64")

    source_grid = np.linspace(0.0, 1.0, values.size, dtype="float64")
    target_grid = make_target_grid(target_length)
    return np.interp(target_grid, source_grid, values)


def compute_dba_barycenter(success_sequences: list[np.ndarray], target_length: int | None) -> np.ndarray:
    """Compute DBA barycenter, optionally forcing a target length."""

    dtw_barycenter_averaging = load_dtw_barycenter_averaging()
    raw_sequences = [np.asarray(sequence, dtype="float64") for sequence in success_sequences if np.asarray(sequence).size]
    if not raw_sequences:
        return np.asarray([], dtype="float64")

    raw_sequences_3d = [sequence[:, np.newaxis] for sequence in raw_sequences]
    kwargs: dict[str, Any] = {"max_iter": 50}
    if target_length is not None and target_length > 0:
        kwargs["barycenter_size"] = int(target_length)

    try:
        barycenter = dtw_barycenter_averaging(raw_sequences_3d, **kwargs)
    except (TypeError, ValueError):
        if "barycenter_size" in kwargs:
            kwargs.pop("barycenter_size")
            barycenter = dtw_barycenter_averaging(raw_sequences_3d, **kwargs)
        else:
            raise

    barycenter = np.asarray(barycenter, dtype="float64").reshape(-1)
    if target_length is not None and target_length > 0 and barycenter.size != int(target_length):
        barycenter = resample_deviation_sequence(barycenter, int(target_length))
    return barycenter


def collect_success_deviation_sequences(jsonl_files: list[Path]) -> list[np.ndarray]:
    """Collect raw deviation vectors from successful samples only."""

    sequences: list[np.ndarray] = []
    for file_path in jsonl_files:
        metadata = parse_file_metadata(file_path)
        if metadata is None or metadata.get("label") != "1":
            continue
        values = extract_deviation_values(read_jsonl_records(file_path))
        if values.size:
            sequences.append(values)
    return sequences


def summarize_lengths(lengths: list[int], target_length: int, source_label: str) -> pd.DataFrame:
    """Build a one-row length statistics table."""

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


def write_mean_deviation_jsonl(
    output_path: Path,
    column_name: str,
    template_values: np.ndarray,
    std_values: np.ndarray,
    sequence_count: int,
) -> None:
    """Write mean deviation template as JSONL."""

    target_length = int(len(template_values))
    grid = make_target_grid(target_length)
    records = []
    for index in range(target_length):
        records.append(
            {
                "normalized_progress": float(grid[index]),
                column_name: json_safe_value(float(template_values[index])),
                "std_deviation_cm": json_safe_value(float(std_values[index])),
                "sequence_count": int(sequence_count),
            }
        )
    write_jsonl_records(records, output_path)


def write_field_plan_summary(output_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write field strategy audit table."""

    pd.DataFrame(rows).to_csv(output_path, index=False)


def process_jsonl_file(
    file_path: Path,
    output_dir: Path,
    target_length: int,
) -> tuple[np.ndarray | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Normalize one JSONL sequence and write a same-name JSONL output."""

    metadata = parse_file_metadata(file_path)
    if metadata is None:
        logging.info("Skip unexpected file name: %s", file_path.name)
        return None, None, []

    records = read_jsonl_records(file_path)
    if not records:
        logging.info("Skip empty JSONL file: %s", file_path)
        return None, metadata, []

    enriched_metadata = {**metadata, "sequence_id": file_path.stem}
    normalized_records, field_plans = build_normalized_records(
        records,
        target_length,
        metadata=enriched_metadata,
    )
    if not normalized_records:
        return None, metadata, []

    output_path = output_dir / file_path.name
    write_jsonl_records(normalized_records, output_path)

    field_rows = [
        {
            "sequence_id": file_path.stem,
            "field_name": plan.name,
            "strategy": plan.strategy,
            "reason": plan.reason,
            "shape": "" if plan.shape is None else str(plan.shape),
        }
        for plan in field_plans
    ]

    normalized_deviation_values = np.asarray(
        [
            record.get("deviation_cm")
            if is_finite_number(record.get("deviation_cm"))
            else record.get("deviation_dst_cm")
            for record in normalized_records
        ],
        dtype="float64",
    )
    return normalized_deviation_values, metadata, field_rows


def run_arithmetic_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    jsonl_files: list[Path],
    arithmetic_target_length: int | None,
    shared_target_length: int | None,
) -> None:
    """Run progress-based interpolation and arithmetic mean template export."""

    normalized_output_dir = output_dir / "normalized_sequences"
    success_lengths = collect_success_lengths(jsonl_files)
    target_length = resolve_method_target_length(success_lengths, arithmetic_target_length, shared_target_length)

    logging.info("[arithmetic_mean] Observed %d successful sequences", len(success_lengths))
    logging.info("[arithmetic_mean] Chosen fixed length L = %d", target_length)

    normalized_success_vectors: list[np.ndarray] = []
    field_rows: list[dict[str, Any]] = []
    processed_count = 0

    for file_path in jsonl_files:
        normalized_vector, metadata, file_field_rows = process_jsonl_file(file_path, normalized_output_dir, target_length)
        if metadata is None:
            continue
        processed_count += 1
        field_rows.extend(file_field_rows)
        if metadata.get("label") == "1" and normalized_vector is not None and np.isfinite(normalized_vector).any():
            normalized_success_vectors.append(normalized_vector)

    if normalized_success_vectors:
        stacked = np.vstack(normalized_success_vectors)
        write_mean_deviation_jsonl(
            output_dir / "mean_deviation_sequence.jsonl",
            "arithmetic_mean_deviation_cm",
            np.nanmean(stacked, axis=0),
            np.nanstd(stacked, axis=0),
            len(normalized_success_vectors),
        )
    else:
        logging.warning("[arithmetic_mean] No successful normalized sequences available for mean computation")

    summarize_lengths(success_lengths, target_length, "success_only" if success_lengths else "no_success").to_csv(
        output_dir / "length_statistics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "method_name": "arithmetic_mean",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "processed_files": processed_count,
                "success_sequences_used_for_mean": len(normalized_success_vectors),
                "target_length_L": target_length,
                "length_mode": "median_or_override",
            }
        ]
    ).to_csv(output_dir / "processing_summary.csv", index=False)
    write_field_plan_summary(output_dir / "field_interpolation_summary.csv", field_rows)


def run_dba_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    jsonl_files: list[Path],
    dba_target_length: int | None,
) -> None:
    """Run DBA-length normalization and DBA mean template export."""

    normalized_output_dir = output_dir / "normalized_sequences"
    success_lengths = collect_success_lengths(jsonl_files)
    success_sequences = collect_success_deviation_sequences(jsonl_files)
    if not success_sequences:
        logging.warning("[dba_mean] No successful sequences available for DBA computation")
        return

    try:
        dba_template = compute_dba_barycenter(success_sequences, dba_target_length)
    except RuntimeError as exc:
        logging.warning("[dba_mean] DBA computation skipped: %s", exc)
        return

    if dba_template.size == 0:
        logging.warning("[dba_mean] DBA barycenter is empty")
        return

    target_length = int(dba_template.size)
    logging.info("[dba_mean] Observed %d successful sequences", len(success_lengths))
    logging.info("[dba_mean] Chosen DBA length L = %d", target_length)

    normalized_success_vectors: list[np.ndarray] = []
    field_rows: list[dict[str, Any]] = []
    processed_count = 0
    for file_path in jsonl_files:
        normalized_vector, metadata, file_field_rows = process_jsonl_file(file_path, normalized_output_dir, target_length)
        if metadata is None:
            continue
        processed_count += 1
        field_rows.extend(file_field_rows)
        if metadata.get("label") == "1" and normalized_vector is not None and np.isfinite(normalized_vector).any():
            normalized_success_vectors.append(normalized_vector)

    std_values = np.zeros(target_length, dtype="float64")
    if normalized_success_vectors:
        std_values = np.nanstd(np.vstack(normalized_success_vectors), axis=0)

    write_mean_deviation_jsonl(
        output_dir / "mean_deviation_sequence.jsonl",
        "dba_deviation_cm",
        dba_template,
        std_values,
        len(success_sequences),
    )
    summarize_lengths(success_lengths, target_length, "success_only" if success_lengths else "no_success").to_csv(
        output_dir / "length_statistics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "method_name": "dba_mean",
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "processed_files": processed_count,
                "success_sequences_used_for_mean": len(success_sequences),
                "target_length_L": target_length,
                "length_mode": "dba_auto" if dba_target_length is None else "override",
            }
        ]
    ).to_csv(output_dir / "processing_summary.csv", index=False)
    write_field_plan_summary(output_dir / "field_interpolation_summary.csv", field_rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize GAIPAT step-03 JSONL deviation sequences to fixed lengths while preserving JSONL fields."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default="/root/autodl-tmp/shenxy/Data/gaipat",
        help="Repository root directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing step-03 JSONL files. Defaults to <repo-root>/block_deviation_sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where normalized JSONL files will be written. Defaults to <repo-root>/normalized_deviation_jsonl_sequences.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Shared fixed length override used by the arithmetic pipeline.",
    )
    parser.add_argument(
        "--arithmetic-target-length",
        type=int,
        default=None,
        help="Fixed length L for the arithmetic/progress interpolation pipeline.",
    )
    parser.add_argument(
        "--dba-target-length",
        type=int,
        default=None,
        help="Fixed length L for the DBA pipeline. If omitted, tslearn chooses DBA length.",
    )
    parser.add_argument(
        "--method",
        choices=["all", *METHOD_NAMES],
        default="all",
        help="Which normalization method to run.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    repo_root = args.repo_root
    input_dir = args.input_dir or (repo_root / "block_deviation_sequences")
    output_base_dir = args.output_dir or (repo_root / "normalized_deviation_jsonl_sequences")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_base_dir / "normalize_jsonl_sequences.log"),
            logging.StreamHandler(),
        ],
    )

    jsonl_files = find_jsonl_files(input_dir)
    if not jsonl_files:
        logging.warning("No JSONL deviation files found under %s", input_dir)
        return

    if args.method in {"all", "arithmetic_mean"}:
        run_arithmetic_pipeline(
            input_dir=input_dir,
            output_dir=output_base_dir / "arithmetic_mean",
            jsonl_files=jsonl_files,
            arithmetic_target_length=args.arithmetic_target_length,
            shared_target_length=args.target_length,
        )

    if args.method in {"all", "dba_mean"}:
        run_dba_pipeline(
            input_dir=input_dir,
            output_dir=output_base_dir / "dba_mean",
            jsonl_files=jsonl_files,
            dba_target_length=args.dba_target_length,
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
