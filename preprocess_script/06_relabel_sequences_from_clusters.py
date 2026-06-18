from __future__ import annotations

import argparse
import logging
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


FILE_NAME_PATTERN = re.compile(
    r"^(?P<subject_id>[^_]+)_(?P<task>[^_]+)_(?P<step_id>-?\d+)_(?P<event>grasp|release)_(?P<block_id>-?\d+)_(?P<label>[0123])$"
)

RELABEL_RULES = {
    ("release", "Focused"): 1,
    ("release", "Distracted"): 0,
    ("release", "Wandering"): 0,
    ("release", "Searching"): 2,
    ("grasp", "Focused"): 1,
    ("grasp", "Distracted"): 0,
    ("grasp", "Wandering"): 2,
    ("grasp", "Searching"): 3,
}

REQUIRED_ASSIGNMENT_COLUMNS = {"sequence_id", "event", "cluster_name"}


def parse_sequence_name(sequence_id: str) -> dict[str, str] | None:
    """Parse sequence id encoded as [subject]_[task]_[step]_[event]_[block]_[label]."""

    match = FILE_NAME_PATTERN.match(sequence_id)
    if match is None:
        return None
    return match.groupdict()


def build_relabelled_name(sequence_id: str, new_label: int) -> str:
    """Build a same-format output file name with the new label."""

    metadata = parse_sequence_name(sequence_id)
    if metadata is None:
        raise ValueError(f"Unexpected sequence_id format: {sequence_id}")
    return (
        f"{metadata['subject_id']}_{metadata['task']}_{metadata['step_id']}_"
        f"{metadata['event']}_{metadata['block_id']}_{int(new_label)}.jsonl"
    )


def resolve_assignments_path(cluster_results_dir: Path | None, cluster_assignments: Path | None) -> Path:
    """Resolve the cluster assignments CSV path."""

    if cluster_assignments is not None:
        return cluster_assignments
    if cluster_results_dir is None:
        raise ValueError("Either --cluster-results-dir or --cluster-assignments must be provided")
    return cluster_results_dir / "cluster_assignments.csv"


def load_assignments(assignments_path: Path) -> pd.DataFrame:
    """Load and validate cluster assignment rows."""

    if not assignments_path.exists():
        raise FileNotFoundError(f"cluster assignments file does not exist: {assignments_path}")

    assignments = pd.read_csv(assignments_path)
    missing_columns = sorted(REQUIRED_ASSIGNMENT_COLUMNS - set(assignments.columns))
    if missing_columns:
        raise ValueError(
            f"{assignments_path} is missing required columns: {', '.join(missing_columns)}"
        )
    return assignments


def normalize_cluster_name(value: Any) -> str:
    """Normalize a cluster name for rule lookup."""

    return str(value).strip()


def infer_new_label(event: str, cluster_name: str) -> int | None:
    """Return the relabelled class id, or None when no rule exists."""

    return RELABEL_RULES.get((str(event).strip().lower(), normalize_cluster_name(cluster_name)))


def resolve_source_file(row: pd.Series, source_dir: Path | None) -> Path | None:
    """Resolve the JSONL source path for one assignment row."""

    if source_dir is not None:
        candidate = source_dir / f"{row['sequence_id']}.jsonl"
        if candidate.exists():
            return candidate

    source_file_value = row.get("source_file")
    if isinstance(source_file_value, str) and source_file_value.strip():
        candidate = Path(source_file_value)
        if candidate.exists():
            return candidate
        if source_dir is not None:
            fallback = source_dir / candidate.name
            if fallback.exists():
                return fallback

    return None


def ensure_unique_destination(destination_path: Path, duplicate_policy: str) -> Path | None:
    """Handle output file collisions."""

    if not destination_path.exists():
        return destination_path
    if duplicate_policy == "skip":
        return None
    if duplicate_policy == "overwrite":
        return destination_path

    stem = destination_path.stem
    suffix = destination_path.suffix
    parent = destination_path.parent
    for index in range(1, 100000):
        candidate = parent / f"{stem}__dup{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique duplicate name for {destination_path}")


def build_output_subdir(output_dir: Path, event: str, new_label: int, copy_discarded: bool) -> Path | None:
    """Build the event-separated output folder for a relabelled sample."""

    event_dir = output_dir / str(event).strip().lower()
    if int(new_label) in {0, 1}:
        return event_dir
    if copy_discarded:
        return event_dir / "discarded"
    return None


def relabel_from_assignments(
    assignments: pd.DataFrame,
    output_dir: Path,
    source_dir: Path | None = None,
    copy_discarded: bool = False,
    duplicate_policy: str = "error",
) -> pd.DataFrame:
    """Copy source JSONL files into event folders with labels rewritten by cluster rules."""

    audit_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in assignments.iterrows():
        sequence_id = str(row["sequence_id"]).strip()
        event = str(row["event"]).strip().lower()
        cluster_name = normalize_cluster_name(row["cluster_name"])
        new_label = infer_new_label(event, cluster_name)

        audit_row: dict[str, Any] = {
            "sequence_id": sequence_id,
            "event": event,
            "cluster_name": cluster_name,
            "old_label": "",
            "new_label": "" if new_label is None else int(new_label),
            "action": "",
            "source_file": "",
            "output_file": "",
            "reason": "",
        }

        metadata = parse_sequence_name(sequence_id)
        if metadata is None:
            audit_row["action"] = "skipped"
            audit_row["reason"] = "unexpected_sequence_id"
            audit_rows.append(audit_row)
            continue
        audit_row["old_label"] = int(metadata["label"])

        if new_label is None:
            audit_row["action"] = "skipped"
            audit_row["reason"] = "no_relabel_rule"
            audit_rows.append(audit_row)
            continue

        output_subdir = build_output_subdir(output_dir, event, new_label, copy_discarded)
        if output_subdir is None:
            audit_row["action"] = "discarded"
            audit_row["reason"] = "discard_label_not_copied"
            audit_rows.append(audit_row)
            continue

        source_file = resolve_source_file(row, source_dir)
        if source_file is None:
            audit_row["action"] = "skipped"
            audit_row["reason"] = "source_file_not_found"
            audit_rows.append(audit_row)
            continue

        output_subdir.mkdir(parents=True, exist_ok=True)
        output_name = build_relabelled_name(sequence_id, new_label)
        destination_path = ensure_unique_destination(output_subdir / output_name, duplicate_policy)
        if destination_path is None:
            audit_row["action"] = "skipped"
            audit_row["source_file"] = str(source_file)
            audit_row["reason"] = "duplicate_destination"
            audit_rows.append(audit_row)
            continue

        shutil.copy2(source_file, destination_path)
        audit_row["action"] = "copied"
        audit_row["source_file"] = str(source_file)
        audit_row["output_file"] = str(destination_path)
        audit_row["reason"] = "ok"
        audit_rows.append(audit_row)

    return pd.DataFrame(audit_rows)


def write_summary(audit_df: pd.DataFrame, output_dir: Path) -> None:
    """Write compact relabel summary tables."""

    audit_path = output_dir / "relabel_audit.csv"
    audit_df.to_csv(audit_path, index=False)

    if audit_df.empty:
        pd.DataFrame().to_csv(output_dir / "relabel_summary.csv", index=False)
        return

    summary = (
        audit_df.groupby(["event", "cluster_name", "new_label", "action"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["event", "new_label", "cluster_name", "action"], kind="mergesort")
    )
    summary.to_csv(output_dir / "relabel_summary.csv", index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command line arguments."""

    parser = argparse.ArgumentParser(
        description="Relabel and copy normalized GAIPAT JSONL sequences according to cluster_assignments.csv."
    )
    parser.add_argument(
        "--cluster-results-dir",
        type=Path,
        default=None,
        help="Directory containing cluster_assignments.csv.",
    )
    parser.add_argument(
        "--cluster-assignments",
        type=Path,
        default=None,
        help="Direct path to cluster_assignments.csv. Overrides --cluster-results-dir.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional directory containing source JSONL files. If omitted, source_file in cluster_assignments.csv is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="/root/autodl-tmp/shenxy/Data/gaipat/final_relabelled",
        help="Directory where relabelled event-separated JSONL files will be copied.",
    )
    parser.add_argument(
        "--copy-discarded",
        action="store_true",
        help="Also copy discard-label samples into <output>/<event>/discarded. By default label 2/3 rows are only audited.",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=["error", "skip", "overwrite", "suffix"],
        default="error",
        help="How to handle destination name collisions.",
    )
    return parser


def main() -> None:
    """Entry point."""

    args = build_argument_parser().parse_args()
    assignments_path = resolve_assignments_path(args.cluster_results_dir, args.cluster_assignments)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "relabel_sequences.log"),
            logging.StreamHandler(),
        ],
    )

    assignments = load_assignments(assignments_path)
    audit_df = relabel_from_assignments(
        assignments,
        output_dir=args.output_dir,
        source_dir=args.source_dir,
        copy_discarded=args.copy_discarded,
        duplicate_policy=args.duplicate_policy,
    )
    write_summary(audit_df, args.output_dir)

    copied_count = int(audit_df["action"].eq("copied").sum()) if not audit_df.empty else 0
    discarded_count = int(audit_df["action"].eq("discarded").sum()) if not audit_df.empty else 0
    skipped_count = int(audit_df["action"].eq("skipped").sum()) if not audit_df.empty else 0
    logging.info(
        "Done. copied=%d discarded=%d skipped=%d output=%s",
        copied_count,
        discarded_count,
        skipped_count,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
