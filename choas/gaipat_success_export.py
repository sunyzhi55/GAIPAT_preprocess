from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TASK_NAMES = ["car", "tb", "house", "sc", "tc", "tsb"]
POSITION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BlockState:
    corners: Tuple[float, float, float, float, float, float, float, float]
    level: int
    holding: int


@dataclass(frozen=True)
class StepInstruction:
    step_id: int
    block_id: int
    origin_corners: Tuple[float, float, float, float, float, float, float, float]
    origin_level: int
    destination_corners: Tuple[float, float, float, float, float, float, float, float]
    destination_level: int


@dataclass(frozen=True)
class TaskResult:
    subject_id: str
    setup: str
    position: str
    pupil: str
    task: str
    recorded: bool
    success: bool
    steps_expected: int
    steps_checked: int
    matched_steps: int
    failure_reason: str


def read_csv_rows(csv_path: Path) -> List[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: str) -> int:
    if value is None:
        return 0
    text = value.strip()
    if not text or text.lower() == "nan":
        return 0
    return int(float(text))


def parse_float(value: str) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text or text.lower() == "nan":
        return float("nan")
    return float(text)


def is_close(a: float, b: float, tolerance: float = POSITION_TOLERANCE) -> bool:
    return abs(a - b) <= tolerance


def load_participant_manifest(setup_dir: Path) -> Dict[str, dict[str, str]]:
    manifest_path = setup_dir / "participants.csv"
    rows = read_csv_rows(manifest_path)
    manifest: Dict[str, dict[str, str]] = {}
    for row in rows:
        manifest[row["id"]] = row
    return manifest


def load_task_instructions(setup_dir: Path, task_name: str) -> List[StepInstruction]:
    instruction_path = setup_dir / f"instructions_{task_name}.csv"
    rows = read_csv_rows(instruction_path)
    steps: List[StepInstruction] = []

    for row in rows:
        step_id = parse_int(row["id"])

        origin_corners = (
            parse_float(row["origin_x0"]),
            parse_float(row["origin_y0"]),
            parse_float(row["origin_x1"]),
            parse_float(row["origin_y1"]),
            parse_float(row["origin_x2"]),
            parse_float(row["origin_y2"]),
            parse_float(row["origin_x3"]),
            parse_float(row["origin_y3"]),
        )

        destination_corners = (
            parse_float(row["destin_x0"]),
            parse_float(row["destin_y0"]),
            parse_float(row["destin_x1"]),
            parse_float(row["destin_y1"]),
            parse_float(row["destin_x2"]),
            parse_float(row["destin_y2"]),
            parse_float(row["destin_x3"]),
            parse_float(row["destin_y3"]),
        )

        steps.append(
            StepInstruction(
                step_id=step_id,
                block_id=parse_int(row["block"]),
                origin_corners=origin_corners,
                origin_level=parse_int(row["origin_level"]),
                destination_corners=destination_corners,
                destination_level=parse_int(row["destin_level"]),
            )
        )

    steps.sort(key=lambda item: item.step_id)
    return steps


def load_table_states(participant_dir: Path, task_name: str) -> List[tuple[int, Dict[int, BlockState]]]:
    states_path = participant_dir / task_name / "table" / "states.csv"
    if not states_path.exists():
        return []

    rows = read_csv_rows(states_path)
    if not rows:
        return []

    header = rows[0].keys()
    block_ids = sorted({int(column.split("_")[0]) for column in header if column.endswith("_x0")})
    snapshots: List[tuple[int, Dict[int, BlockState]]] = []

    for row in rows:
        timestamp = parse_int(row["timestamp"])
        snapshot: Dict[int, BlockState] = {}
        for block_id in block_ids:
            x0 = parse_float(row[f"{block_id}_x0"])
            y0 = parse_float(row[f"{block_id}_y0"])
            x1 = parse_float(row[f"{block_id}_x1"])
            y1 = parse_float(row[f"{block_id}_y1"])
            x2 = parse_float(row[f"{block_id}_x2"])
            y2 = parse_float(row[f"{block_id}_y2"])
            x3 = parse_float(row[f"{block_id}_x3"])
            y3 = parse_float(row[f"{block_id}_y3"])
            level = parse_int(row[f"{block_id}_level"])
            holding = parse_int(row[f"{block_id}_holding"])
            snapshot[block_id] = BlockState(
                corners=(x0, y0, x1, y1, x2, y2, x3, y3),
                level=level,
                holding=holding,
            )
        snapshots.append((timestamp, snapshot))

    return snapshots


def get_initial_state(snapshots: List[tuple[int, Dict[int, BlockState]]]) -> Optional[Dict[int, BlockState]]:
    if not snapshots:
        return None
    return snapshots[0][1]


def get_final_state(snapshots: List[tuple[int, Dict[int, BlockState]]]) -> Optional[Dict[int, BlockState]]:
    if not snapshots:
        return None
    return snapshots[-1][1]


def block_matches_corners_and_level(
    state: BlockState,
    target_corners: Tuple[float, float, float, float, float, float, float, float],
    target_level: int,
    target_holding: int = 0,
) -> bool:
    for observed, expected in zip(state.corners, target_corners):
        if not is_close(observed, expected):
            return False
    return state.level == target_level and state.holding == target_holding


def simulate_expected_final_state(
    initial_state: Dict[int, BlockState],
    steps: List[StepInstruction],
) -> Dict[int, BlockState]:
    expected_state = dict(initial_state)

    for step in steps:
        if step.block_id not in expected_state:
            continue
        expected_state[step.block_id] = BlockState(
            corners=step.destination_corners,
            level=step.destination_level,
            holding=0,
        )

    return expected_state


def compare_final_state(
    observed_state: Dict[int, BlockState],
    expected_state: Dict[int, BlockState],
    moved_block_ids: set[int],
) -> tuple[bool, str, int]:
    matched_blocks = 0

    for block_id in sorted(expected_state.keys()):
        expected_block = expected_state[block_id]
        observed_block = observed_state.get(block_id)

        if observed_block is None:
            if block_id in moved_block_ids:
                return False, f"moved_block_missing_state(block={block_id})", matched_blocks
            return False, f"unused_block_missing_state(block={block_id})", matched_blocks

        if not block_matches_corners_and_level(
            observed_block,
            expected_block.corners,
            expected_block.level,
            expected_block.holding,
        ):
            if block_id in moved_block_ids:
                return False, f"moved_block_wrong_position(block={block_id})", matched_blocks
            return False, f"unused_block_wrong_position(block={block_id})", matched_blocks

        matched_blocks += 1

    return True, "", matched_blocks


def evaluate_task(participant_dir: Path, subject_row: dict[str, str], task_name: str) -> TaskResult:
    steps = load_task_instructions(participant_dir.parent.parent / "setup", task_name)
    recorded_flag = parse_int(subject_row.get(task_name, "0")) == 1

    if not recorded_flag:
        return TaskResult(
            subject_id=subject_row["id"],
            setup=subject_row.get("setup", ""),
            position=subject_row.get("position", ""),
            pupil=subject_row.get("pupil", ""),
            task=task_name,
            recorded=False,
            success=False,
            steps_expected=len(steps),
            steps_checked=0,
            matched_steps=0,
            failure_reason="not_recorded",
        )

    snapshots = load_table_states(participant_dir, task_name)
    if not snapshots:
        return TaskResult(
            subject_id=subject_row["id"],
            setup=subject_row.get("setup", ""),
            position=subject_row.get("position", ""),
            pupil=subject_row.get("pupil", ""),
            task=task_name,
            recorded=True,
            success=False,
            steps_expected=len(steps),
            steps_checked=0,
            matched_steps=0,
            failure_reason="missing_states",
        )

    if len(snapshots) < 1:
        return TaskResult(
            subject_id=subject_row["id"],
            setup=subject_row.get("setup", ""),
            position=subject_row.get("position", ""),
            pupil=subject_row.get("pupil", ""),
            task=task_name,
            recorded=True,
            success=False,
            steps_expected=len(steps),
            steps_checked=0,
            matched_steps=0,
            failure_reason="missing_states",
        )

    initial_state = get_initial_state(snapshots)
    final_state = get_final_state(snapshots)

    if initial_state is None or final_state is None:
        return TaskResult(
            subject_id=subject_row["id"],
            setup=subject_row.get("setup", ""),
            position=subject_row.get("position", ""),
            pupil=subject_row.get("pupil", ""),
            task=task_name,
            recorded=True,
            success=False,
            steps_expected=len(steps),
            steps_checked=0,
            matched_steps=0,
            failure_reason="missing_initial_or_final_state",
        )

    expected_final_state = simulate_expected_final_state(initial_state, steps)
    moved_block_ids = {step.block_id for step in steps}
    success, failure_reason, matched_blocks = compare_final_state(final_state, expected_final_state, moved_block_ids)

    if not success and not failure_reason:
        failure_reason = "unknown_failure"

    return TaskResult(
        subject_id=subject_row["id"],
        setup=subject_row.get("setup", ""),
        position=subject_row.get("position", ""),
        pupil=subject_row.get("pupil", ""),
        task=task_name,
        recorded=True,
        success=success,
        steps_expected=len(steps),
        steps_checked=len(steps),
        matched_steps=matched_blocks,
        failure_reason=failure_reason,
    )


def build_outputs(repo_root: Path) -> tuple[List[TaskResult], List[dict[str, str]]]:
    setup_dir = repo_root / "setup"
    participants_dir = repo_root / "participants"
    manifest = load_participant_manifest(setup_dir)

    task_results: List[TaskResult] = []
    participant_summaries: List[dict[str, str]] = []

    for subject_id, subject_row in manifest.items():
        participant_dir = participants_dir / subject_id
        per_task_results: List[TaskResult] = []

        for task_name in TASK_NAMES:
            task_result = evaluate_task(participant_dir, subject_row, task_name)
            task_results.append(task_result)
            per_task_results.append(task_result)

        recorded_tasks = sum(1 for item in per_task_results if item.recorded)
        successful_tasks = sum(1 for item in per_task_results if item.success)
        failed_tasks = sum(1 for item in per_task_results if item.recorded and not item.success)
        not_recorded_tasks = sum(1 for item in per_task_results if not item.recorded)
        overall_success = recorded_tasks == len(TASK_NAMES) and successful_tasks == len(TASK_NAMES)

        participant_summaries.append(
            {
                "subject_id": subject_id,
                "setup": subject_row.get("setup", ""),
                "position": subject_row.get("position", ""),
                "pupil": subject_row.get("pupil", ""),
                "car": _task_status(per_task_results, "car"),
                "tb": _task_status(per_task_results, "tb"),
                "house": _task_status(per_task_results, "house"),
                "sc": _task_status(per_task_results, "sc"),
                "tc": _task_status(per_task_results, "tc"),
                "tsb": _task_status(per_task_results, "tsb"),
                "recorded_tasks": str(recorded_tasks),
                "successful_tasks": str(successful_tasks),
                "failed_tasks": str(failed_tasks),
                "not_recorded_tasks": str(not_recorded_tasks),
                "overall_success": "1" if overall_success else "0",
            }
        )

    return task_results, participant_summaries


def _task_status(results: Iterable[TaskResult], task_name: str) -> str:
    for result in results:
        if result.task == task_name:
            if not result.recorded:
                return "not_recorded"
            return "success" if result.success else "failure"
    return "not_recorded"


def write_csv(csv_path: Path, rows: List[dict[str, str]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export GAIPAT participant success summaries from setup/instructions and participant table logs."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory (default: script location).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "processed",
        help="Directory where CSV files will be written.",
    )
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Write only participants with overall_success = 1 to the participant summary CSV.",
    )
    args = parser.parse_args()

    task_results, participant_summaries = build_outputs(args.repo_root)

    summary_path = args.output_dir / "participant_success_summary.csv"
    filtered_path = args.output_dir / "participant_success_summary_only.csv"
    task_path = args.output_dir / "participant_task_success.csv"

    participant_rows = participant_summaries
    if args.only_success:
        participant_rows = [row for row in participant_rows if row["overall_success"] == "1"]

    write_csv(
        summary_path,
        participant_rows,
        [
            "subject_id",
            "setup",
            "position",
            "pupil",
            "car",
            "tb",
            "house",
            "sc",
            "tc",
            "tsb",
            "recorded_tasks",
            "successful_tasks",
            "failed_tasks",
            "not_recorded_tasks",
            "overall_success",
        ],
    )

    successful_rows = [row for row in participant_summaries if row["overall_success"] == "1"]
    write_csv(
        filtered_path,
        successful_rows,
        [
            "subject_id",
            "setup",
            "position",
            "pupil",
            "car",
            "tb",
            "house",
            "sc",
            "tc",
            "tsb",
            "recorded_tasks",
            "successful_tasks",
            "failed_tasks",
            "not_recorded_tasks",
            "overall_success",
        ],
    )

    write_csv(
        task_path,
        [
            {
                "subject_id": result.subject_id,
                "setup": result.setup,
                "position": result.position,
                "pupil": result.pupil,
                "task": result.task,
                "recorded": "1" if result.recorded else "0",
                "success": "1" if result.success else "0",
                "steps_expected": str(result.steps_expected),
                "steps_checked": str(result.steps_checked),
                "matched_steps": str(result.matched_steps),
                "failure_reason": result.failure_reason,
            }
            for result in task_results
        ],
        [
            "subject_id",
            "setup",
            "position",
            "pupil",
            "task",
            "recorded",
            "success",
            "steps_expected",
            "steps_checked",
            "matched_steps",
            "failure_reason",
        ],
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {filtered_path}")
    print(f"Wrote {task_path}")


if __name__ == "__main__":
    main()