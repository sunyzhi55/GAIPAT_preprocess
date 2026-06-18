"""
GAIPAT Master DataFrame Construction Pipeline (Practical Implementation)
Implements multimodal synchronization following spec, adapted to actual data format.

This version uses actual GAIPAT data structure:
- participants/{id}/{task}/table/gazepoints.csv + events.csv + states.csv
- participants/{id}/{task}/screen/gazepoints.csv
- setup/instructions_{task}.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


TASK_NAMES = ["car", "tb", "house", "sc", "tc", "tsb"]
POSITION_TOLERANCE = 1e-6
SCREEN_SWITCH_WINDOW_MS = 2000

TABLE_PUPIL_COLUMNS = [
    "table_pupil_timestamp",
    "table_confidence_right",
    "table_confidence_left",
    "table_diameter_right",
    "table_diameter_left",
]

SCREEN_PUPIL_COLUMNS = [
    "screen_pupil_timestamp",
    "screen_confidence_right",
    "screen_confidence_left",
    "screen_diameter_right",
    "screen_diameter_left",
]


def build_screen_to_table_affine_matrix() -> np.ndarray:
    """Solve the affine transform from screen centimeters to table centimeters."""

    source_points = np.float32(
        [
            [0.0, 0.0],
            [EnvironmentConstants.DISPLAY_WIDTH_CM, 0.0],
            [0.0, EnvironmentConstants.DISPLAY_HEIGHT_CM],
        ]
    )
    target_points = np.float32(
        [
            [0.0, 0.0],
            [EnvironmentConstants.LEGO_WIDTH_CM, 0.0],
            [0.0, EnvironmentConstants.LEGO_HEIGHT_CM],
        ]
    )
    return cv2.getAffineTransform(source_points, target_points)


def extract_failure_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Return contiguous rows where is_success == 0, annotated with segment metadata."""

    if df.empty or "is_success" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    failure_mask = df["is_success"].fillna(0).astype(int).eq(0)
    if not failure_mask.any():
        return df.iloc[0:0].copy()

    segment_ids = failure_mask.ne(failure_mask.shift(fill_value=False)).cumsum()
    failure_df = df.loc[failure_mask].copy()
    failure_df["failure_segment_id"] = segment_ids.loc[failure_mask].astype("int32")
    failure_df["failure_segment_row_id"] = failure_df.groupby("failure_segment_id").cumcount().astype("int32")
    failure_df["failure_segment_start_timestamp"] = failure_df.groupby("failure_segment_id")["timestamp"].transform("min")
    failure_df["failure_segment_end_timestamp"] = failure_df.groupby("failure_segment_id")["timestamp"].transform("max")
    failure_df["failure_segment_length"] = failure_df.groupby("failure_segment_id")["timestamp"].transform("size").astype("int32")
    return failure_df


def build_failure_export_summary(
    subject_id: str,
    task: str,
    master_rows: int,
    failure_segments: pd.DataFrame,
    failure_output_file: Optional[Path],
) -> dict[str, str]:
    """Build one summary row for the failure export of a participant/task."""

    failure_rows = int(len(failure_segments))
    segment_count = int(failure_segments["failure_segment_id"].nunique()) if failure_rows else 0
    return {
        "subject_id": subject_id,
        "task": task,
        "master_rows": str(master_rows),
        "failure_rows": str(failure_rows),
        "failure_segments": str(segment_count),
        "failure_exported": "1" if failure_rows > 0 else "0",
        "failure_file": str(failure_output_file) if failure_output_file is not None else "",
    }


# ============================================================================
# ENVIRONMENT CONSTANTS
# ============================================================================

class EnvironmentConstants:
    """Physical constants per spec."""
    
    # Display: Dell P2416D (physical dimensions)
    DISPLAY_WIDTH_CM = 52.7
    DISPLAY_HEIGHT_CM = 29.6
    DISPLAY_WIDTH_PX = 2560
    DISPLAY_HEIGHT_PX = 1440
    
    # LEGO assembly surface
    LEGO_WIDTH_CM = 76.0
    LEGO_HEIGHT_CM = 38.0
    
    # Static reference block (experimenter-placed)
    STATIC_BLOCK_ID = 0


SCREEN_TO_TABLE_AFFINE_MATRIX = build_screen_to_table_affine_matrix()


# ============================================================================
# HELPERS
# ============================================================================

def read_csv_rows(csv_path: Path) -> List[dict]:
    """Read CSV file safely."""
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        logging.warning(f"Error reading {csv_path}: {e}")
        return []


def safe_float(value) -> float:
    """Safely parse float, return NaN on failure."""
    if value is None or value == "":
        return np.nan
    try:
        f = float(value)
        return f
    except (ValueError, TypeError):
        return np.nan


def safe_int(value) -> int:
    """Safely parse int, return 0 on failure."""
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def extract_corners(row: dict, prefix: str) -> Tuple[float, float, float, float, float, float, float, float]:
    """Extract x/y corner coordinates from a CSV row."""
    corners: List[float] = []
    for corner in range(4):
        corners.append(safe_float(row.get(f"{prefix}_x{corner}")))
        corners.append(safe_float(row.get(f"{prefix}_y{corner}")))
    return tuple(corners)


def relative_corners_to_physical(
    corners: Tuple[float, float, float, float, float, float, float, float],
    width_cm: float,
    height_cm: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Convert relative corner coordinates to physical centimeters."""
    converted: List[float] = []
    for corner in range(4):
        converted.append(corners[corner * 2] * width_cm)
        converted.append(corners[corner * 2 + 1] * height_cm)
    return tuple(converted)


def state_row_to_block_state(row: pd.Series, block_id: int) -> Tuple[Tuple[float, float, float, float, float, float, float, float], int, int]:
    """Read one block state snapshot from a table/states row."""
    prefix = f"{block_id}_"
    corners = []
    for corner in range(4):
        corners.append(safe_float(row.get(f"{prefix}x{corner}")))
        corners.append(safe_float(row.get(f"{prefix}y{corner}")))
    level = safe_int(row.get(f"{prefix}level"))
    holding = safe_int(row.get(f"{prefix}holding"))
    return tuple(corners), level, holding


def corners_match(observed: Tuple[float, float, float, float, float, float, float, float], expected: Tuple[float, float, float, float, float, float, float, float]) -> bool:
    """Compare two corner tuples with tolerance."""
    for observed_value, expected_value in zip(observed, expected):
        if np.isnan(observed_value) or np.isnan(expected_value):
            return False
        if not np.isclose(observed_value, expected_value, atol=POSITION_TOLERANCE, rtol=0.0):
            return False
    return True


# ============================================================================
# DATA LOADERS
# ============================================================================

def load_gazepoints_table(participant_dir: Path, task_name: str) -> pd.DataFrame:
    """Load table gazepoints."""
    path = participant_dir / task_name / "table" / "gazepoints.csv"
    rows = read_csv_rows(path)
    
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gaze_x_table_rel", "gaze_y_table_rel"])
    
    data = []
    for row in rows:
        ts = safe_int(row.get("timestamp"))
        x = safe_float(row.get("x"))
        y = safe_float(row.get("y"))
        data.append({"timestamp": ts, "gaze_x_table_rel": x, "gaze_y_table_rel": y})
    
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def load_gazepoints_screen(participant_dir: Path, task_name: str) -> pd.DataFrame:
    """Load screen gazepoints."""
    path = participant_dir / task_name / "screen" / "gazepoints.csv"
    rows = read_csv_rows(path)
    
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gaze_x_screen_rel", "gaze_y_screen_rel"])
    
    data = []
    for row in rows:
        ts = safe_int(row.get("timestamp"))
        x = safe_float(row.get("x"))
        y = safe_float(row.get("y"))
        data.append({"timestamp": ts, "gaze_x_screen_rel": x, "gaze_y_screen_rel": y})
    
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df


def load_pupil_info(participant_dir: Path, task_name: str, source_name: str) -> pd.DataFrame:
    """Load pupil_info.csv from one source directory and prefix its fields."""

    path = participant_dir / task_name / source_name / "pupil_info.csv"
    rows = read_csv_rows(path)
    pupil_columns = TABLE_PUPIL_COLUMNS if source_name == "table" else SCREEN_PUPIL_COLUMNS

    if not rows:
        return pd.DataFrame(columns=pupil_columns)

    prefix = f"{source_name}_"
    data = []
    for row in rows:
        data.append(
            {
                f"{prefix}pupil_timestamp": safe_float(row.get("timestamp")),
                f"{prefix}confidence_right": safe_float(row.get("confidence_right")),
                f"{prefix}confidence_left": safe_float(row.get("confidence_left")),
                f"{prefix}diameter_right": safe_float(row.get("diameter_right")),
                f"{prefix}diameter_left": safe_float(row.get("diameter_left")),
            }
        )

    df = pd.DataFrame(data)
    if not df.empty:
        timestamp_column = f"{prefix}pupil_timestamp"
        df = df.sort_values(timestamp_column).drop_duplicates(subset=[timestamp_column], keep="first").reset_index(drop=True)
        df[timestamp_column] = pd.to_numeric(df[timestamp_column], errors="coerce").astype("float64")
        for column in [
            f"{prefix}confidence_right",
            f"{prefix}confidence_left",
            f"{prefix}diameter_right",
            f"{prefix}diameter_left",
        ]:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float32")
    return df


def load_events_table(participant_dir: Path, task_name: str) -> pd.DataFrame:
    """Load table events."""
    path = participant_dir / task_name / "table" / "events.csv"
    rows = read_csv_rows(path)
    
    if not rows:
        return pd.DataFrame(columns=["timestamp", "event", "block_id"])
    
    data = []
    for row in rows:
        ts = safe_int(row.get("timestamp"))
        event = str(row.get("event", "")).strip()
        block_id = safe_int(row.get("block_id"))
        data.append({"timestamp": ts, "event": event, "block_id": block_id})
    
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].astype("int64")
    df["block_id"] = df["block_id"].astype("int16")
    return df


def load_table_states(participant_dir: Path, task_name: str) -> pd.DataFrame:
    """Load table states for final-state comparison."""
    path = participant_dir / task_name / "table" / "states.csv"
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logging.warning(f"Error reading {path}: {exc}")
        return pd.DataFrame()

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].map(safe_int).astype("int64")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def load_screen_states(participant_dir: Path, task_name: str) -> pd.DataFrame:
    """Load screen states so slide_id can be forward-filled."""
    path = participant_dir / task_name / "screen" / "states.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "slide"])

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        logging.warning(f"Error reading {path}: {exc}")
        return pd.DataFrame(columns=["timestamp", "slide"])

    if df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "slide"])

    df = df[[column for column in df.columns if column in {"timestamp", "slide"}]].copy()
    df["timestamp"] = df["timestamp"].map(safe_int).astype("int64")
    if "slide" not in df.columns:
        df["slide"] = 0
    df["slide"] = df["slide"].map(safe_int).astype("int16")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return df


def load_slides(setup_dir: Path, task_name: str) -> Dict[int, Dict[str, Tuple[float, float, float, float, float, float, float, float]]]:
    """Load slide AOI definitions."""
    path = setup_dir / f"slides_{task_name}.csv"
    rows = read_csv_rows(path)
    slides: Dict[int, Dict[str, Tuple[float, float, float, float, float, float, float, float]]] = {}

    for row in rows:
        slide_id = safe_int(row.get("id"))
        slides[slide_id] = {
            "title_corners": extract_corners(row, "title"),
            "released_corners": extract_corners(row, "released"),
            "grasp_corners": extract_corners(row, "grasp"),
            "release_corners": extract_corners(row, "release"),
        }

    return slides


def load_instructions(setup_dir: Path, task_name: str) -> Dict[int, Dict]:
    """
    Load instructions and map step_id -> block metadata.
    Returns: {step_id: {"block_id": int, "origin_corners": tuple, "dest_corners": tuple, ...}}
    """
    path = setup_dir / f"instructions_{task_name}.csv"
    rows = read_csv_rows(path)
    
    instructions = {}
    
    for row in rows:
        step_id = safe_int(row.get("id"))
        block_id = safe_int(row.get("block"))
        
        instructions[step_id] = {
            "block_id": block_id,
            "origin_corners": extract_corners(row, "origin"),
            "origin_level": safe_int(row.get("origin_level")),
            "dest_corners": extract_corners(row, "destin"),
            "dest_level": safe_int(row.get("destin_level")),
        }
    
    return dict(sorted(instructions.items(), key=lambda item: item[0]))


# ============================================================================
# MASTER DATAFRAME BUILDER
# ============================================================================

class MasterDataframeBuilder:
    """Build master dataframe for single participant/task."""
    
    def __init__(self, subject_id: str, task_name: str, repo_root: Path):
        self.subject_id = subject_id
        self.task_name = task_name
        self.repo_root = repo_root
        self.setup_dir = repo_root / "setup"
        self.participant_dir = repo_root / "participants" / subject_id
        
        # Load data
        self.table_gaze = load_gazepoints_table(self.participant_dir, task_name)
        self.screen_gaze = load_gazepoints_screen(self.participant_dir, task_name)
        self.table_pupil_info = load_pupil_info(self.participant_dir, task_name, "table")
        self.screen_pupil_info = load_pupil_info(self.participant_dir, task_name, "screen")
        self.table_events = load_events_table(self.participant_dir, task_name)
        self.table_states = load_table_states(self.participant_dir, task_name)
        self.screen_states = load_screen_states(self.participant_dir, task_name)
        self.instructions = load_instructions(self.setup_dir, task_name)
        self.slides = load_slides(self.setup_dir, task_name)
        
        self.df = None
        self.slide_start_timestamp_by_slide: Dict[int, int] = {}
        self.grasp_timestamp_by_slide: Dict[int, float] = {}
        self.success_by_slide: Dict[int, int] = {}
    
    def build(self) -> pd.DataFrame:
        """Build master dataframe."""
        
        logging.info(f"[{self.subject_id}] {self.task_name}: Building master dataframe...")
        
        # Logic 1: Timestamp union
        self._build_timestamp_union()
        
        if len(self.df) == 0:
            logging.warning(f"[{self.subject_id}] {self.task_name}: No data")
            return self.df
        
        # Add metadata
        self.df["subject_id"] = self.subject_id
        self.df["task"] = self.task_name
        
        # Build slide/step context before feature construction
        self._build_step_context()

        # Convert coordinates to physical units
        self._convert_coordinates()
        
        # Logic 2: Cognitive focus FSM
        self._compute_active_target_type()

        # Logic 3: Step-level success label
        self._compute_is_success()

        self._cast_dtypes()
        
        logging.info(f"[{self.subject_id}] {self.task_name}: Generated {len(self.df)} rows")
        
        return self.df
    
    def _build_timestamp_union(self):
        """Logic 1: Build timestamp union without interpolation."""
        
        # Collect all unique timestamps
        timestamps = set()
        timestamps.update(self.table_gaze["timestamp"].values)
        timestamps.update(self.screen_gaze["timestamp"].values)
        if not self.table_pupil_info.empty:
            timestamps.update(self.table_pupil_info["table_pupil_timestamp"].dropna().values)
        if not self.screen_pupil_info.empty:
            timestamps.update(self.screen_pupil_info["screen_pupil_timestamp"].dropna().values)
        timestamps.update(self.table_events["timestamp"].values)
        if not self.table_states.empty:
            timestamps.update(self.table_states["timestamp"].values)
        if not self.screen_states.empty:
            timestamps.update(self.screen_states["timestamp"].values)
        
        if not timestamps:
            self.df = pd.DataFrame()
            return
        
        timestamps = sorted(timestamps)
        self.df = pd.DataFrame({"timestamp": timestamps})
        self.df["timestamp"] = self.df["timestamp"].astype("int64")
        
        # Merge gazepoints (no interpolation - preserve NaN)
        table_gaze = self.table_gaze.copy()
        self.df = self.df.merge(table_gaze, on="timestamp", how="left")
        
        screen_gaze = self.screen_gaze.copy()
        self.df = self.df.merge(screen_gaze, on="timestamp", how="left")

        table_pupil_info = self.table_pupil_info.copy()
        self.df = self.df.merge(table_pupil_info, left_on="timestamp", right_on="table_pupil_timestamp", how="left")
        if "table_pupil_timestamp" in self.df.columns:
            self.df["table_pupil_timestamp"] = self.df["table_pupil_timestamp"].astype("float64")

        screen_pupil_info = self.screen_pupil_info.copy()
        self.df = self.df.merge(screen_pupil_info, left_on="timestamp", right_on="screen_pupil_timestamp", how="left")
        if "screen_pupil_timestamp" in self.df.columns:
            self.df["screen_pupil_timestamp"] = self.df["screen_pupil_timestamp"].astype("float64")
        
        # Merge and forward-fill events
        events = self.table_events.copy()
        self.df = self.df.merge(events, on="timestamp", how="left")
        self.df["event"] = self.df["event"].ffill().fillna("none")

        # Merge and forward-fill slide id from screen states
        slide_states = self.screen_states[["timestamp", "slide"]].copy() if not self.screen_states.empty else pd.DataFrame(columns=["timestamp", "slide"])
        self.df = self.df.merge(slide_states, on="timestamp", how="left")
        self.df["slide_id"] = self.df["slide"].ffill().fillna(0).astype("int16")
        self.df.drop(columns=["slide"], inplace=True)
        self.df["step_id"] = self.df["slide_id"].astype("int16")

        # Map current block from instruction metadata
        self.df["block_id"] = self.df["slide_id"].map(
            lambda slide_id: self.instructions.get(int(slide_id), {}).get("block_id", -1)
        ).astype("int16")

        # Step start timestamps are used by the active target state machine
        self.slide_start_timestamp_by_slide = (
            self.df.groupby("slide_id", dropna=False)["timestamp"].min().astype("int64").to_dict()
        )

    def _build_step_context(self):
        """Pre-compute slide boundary and success metadata."""

        if not self.instructions:
            self.slide_start_timestamp_by_slide = {}
            self.grasp_timestamp_by_slide = {}
            self.success_by_slide = {}
            return

        slide_state_df = self.screen_states.sort_values("timestamp").drop_duplicates("timestamp", keep="last") if not self.screen_states.empty else pd.DataFrame(columns=["timestamp", "slide"])
        if not slide_state_df.empty:
            self.slide_start_timestamp_by_slide = slide_state_df.groupby("slide")["timestamp"].min().astype("int64").to_dict()

        grasp_events = self.table_events[self.table_events["event"].isin(["grasp", "pick_up", "grasp_success", "assemble_start"])].copy()
        if not grasp_events.empty:
            grasp_events = grasp_events.sort_values("timestamp")
            grasp_by_block = grasp_events.groupby("block_id")["timestamp"].min().to_dict()
        else:
            grasp_by_block = {}

        for slide_id, instruction in self.instructions.items():
            block_id = instruction.get("block_id", -1)
            self.grasp_timestamp_by_slide[slide_id] = float(grasp_by_block.get(block_id, np.nan))

        self.success_by_slide = self._compute_is_success_by_slide(slide_state_df)

    def _compute_is_success_by_slide(self, slide_state_df: pd.DataFrame) -> Dict[int, int]:
        """Compare the post-step table state to the destination instruction for each slide."""

        if self.table_states.empty or not self.instructions:
            return {slide_id: 0 for slide_id in self.instructions}

        state_df = self.table_states.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        if state_df.empty:
            return {slide_id: 0 for slide_id in self.instructions}

        state_timestamps = state_df["timestamp"].to_numpy()
        timeline_max = int(max(self.df["timestamp"].max(), state_timestamps[-1])) if len(self.df) else int(state_timestamps[-1])

        success_by_slide: Dict[int, int] = {}
        ordered_slides = list(slide_state_df.sort_values("timestamp")["slide"].drop_duplicates().astype(int)) if not slide_state_df.empty else sorted(self.instructions.keys())

        for index, slide_id in enumerate(ordered_slides):
            instruction = self.instructions.get(slide_id)
            if instruction is None:
                continue

            next_slide_ts = timeline_max
            if not slide_state_df.empty:
                slide_rows = slide_state_df.reset_index(drop=True)
                matching = slide_rows.index[slide_rows["slide"].astype(int) == slide_id].tolist()
                if matching:
                    current_idx = matching[0]
                    if current_idx + 1 < len(slide_rows):
                        next_slide_ts = int(slide_rows.loc[current_idx + 1, "timestamp"])

            state_idx = state_timestamps.searchsorted(next_slide_ts, side="left")
            if state_idx >= len(state_df):
                state_idx = len(state_df) - 1

            observed_row = state_df.iloc[state_idx]
            block_id = instruction["block_id"]
            if block_id < 0:
                success_by_slide[slide_id] = 0
                continue

            observed_corners, observed_level, observed_holding = state_row_to_block_state(observed_row, block_id)
            observed_corners_cm = relative_corners_to_physical(
                observed_corners,
                EnvironmentConstants.LEGO_WIDTH_CM,
                EnvironmentConstants.LEGO_HEIGHT_CM,
            )
            expected_corners = relative_corners_to_physical(
                instruction["dest_corners"],
                EnvironmentConstants.LEGO_WIDTH_CM,
                EnvironmentConstants.LEGO_HEIGHT_CM,
            )
            expected_level = instruction["dest_level"]

            if corners_match(observed_corners_cm, expected_corners) and observed_level == expected_level and observed_holding == 0:
                success_by_slide[slide_id] = 1
            else:
                success_by_slide[slide_id] = 0

        return success_by_slide

    def _step_end_timestamp(self, slide_id: int) -> float:
        """Return the timestamp at which a slide segment ends."""

        if self.screen_states.empty:
            if len(self.df):
                return float(self.df["timestamp"].max())
            return float("nan")

        slide_rows = self.screen_states.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        matching = slide_rows.index[slide_rows["slide"].astype(int) == int(slide_id)].tolist()
        if not matching:
            return float(self.df["timestamp"].max()) if len(self.df) else float("nan")

        current_idx = matching[0]
        if current_idx + 1 < len(slide_rows):
            return float(slide_rows.loc[current_idx + 1, "timestamp"])

        return float(self.df["timestamp"].max()) if len(self.df) else float(slide_rows.loc[current_idx, "timestamp"])
    
    def _convert_coordinates(self):
        """Convert relative coordinates to physical units."""
        
        # Table: relative [0,1] -> cm
        self.df["gaze_x_table_cm"] = self.df["gaze_x_table_rel"] * EnvironmentConstants.LEGO_WIDTH_CM
        self.df["gaze_y_table_cm"] = self.df["gaze_y_table_rel"] * EnvironmentConstants.LEGO_HEIGHT_CM
        
        # Screen: relative [0,1] -> cm
        self.df["gaze_x_screen_cm"] = self.df["gaze_x_screen_rel"] * EnvironmentConstants.DISPLAY_WIDTH_CM
        self.df["gaze_y_screen_cm"] = self.df["gaze_y_screen_rel"] * EnvironmentConstants.DISPLAY_HEIGHT_CM
        
        # Screen: cm -> pixels
        self.df["gaze_x_screen_px"] = (
            self.df["gaze_x_screen_cm"] * 
            (EnvironmentConstants.DISPLAY_WIDTH_PX / EnvironmentConstants.DISPLAY_WIDTH_CM)
        )
        self.df["gaze_y_screen_px"] = (
            self.df["gaze_y_screen_cm"] * 
            (EnvironmentConstants.DISPLAY_HEIGHT_PX / EnvironmentConstants.DISPLAY_HEIGHT_CM)
        )

        # Screen cm -> table affine-mapped cm
        self.df["gaze_x_table_affine_cm"] = np.nan
        self.df["gaze_y_table_affine_cm"] = np.nan

        screen_valid = self.df["gaze_x_screen_cm"].notna() & self.df["gaze_y_screen_cm"].notna()
        if screen_valid.any():
            screen_points = self.df.loc[screen_valid, ["gaze_x_screen_cm", "gaze_y_screen_cm"]].to_numpy(dtype=np.float32)
            transformed_points = cv2.transform(screen_points[:, np.newaxis, :], SCREEN_TO_TABLE_AFFINE_MATRIX).squeeze(axis=1)
            self.df.loc[screen_valid, "gaze_x_table_affine_cm"] = transformed_points[:, 0]
            self.df.loc[screen_valid, "gaze_y_table_affine_cm"] = transformed_points[:, 1]
        
        # Add target block and screen coordinates.
        self._add_target_coordinates()
        
        # Convert to float32 for memory efficiency
        coord_cols = [col for col in self.df.columns if "gaze_" in col or "target_" in col]
        for col in coord_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("float32")
    
    def _add_target_coordinates(self):
        """Add block origin and destination coordinates."""
        
        for corner in range(4):
            self.df[f"target_x_table_srt_cm{corner}"] = np.nan
            self.df[f"target_y_table_srt_cm{corner}"] = np.nan
            self.df[f"target_x_table_dst_cm{corner}"] = np.nan
            self.df[f"target_y_table_dst_cm{corner}"] = np.nan
            self.df[f"target_x_screen_cm{corner}"] = np.nan
            self.df[f"target_y_screen_cm{corner}"] = np.nan
            self.df[f"target_x_screen_px{corner}"] = np.nan
            self.df[f"target_y_screen_px{corner}"] = np.nan
        
        # Fill from instructions
        for step_id, instr in self.instructions.items():
            origin = relative_corners_to_physical(
                instr["origin_corners"],
                EnvironmentConstants.LEGO_WIDTH_CM,
                EnvironmentConstants.LEGO_HEIGHT_CM,
            )
            dest = relative_corners_to_physical(
                instr["dest_corners"],
                EnvironmentConstants.LEGO_WIDTH_CM,
                EnvironmentConstants.LEGO_HEIGHT_CM,
            )
            slide = self.slides.get(step_id, {})
            title_corners = slide.get("title_corners", tuple([np.nan] * 8))
            screen_cm = relative_corners_to_physical(
                title_corners,
                EnvironmentConstants.DISPLAY_WIDTH_CM,
                EnvironmentConstants.DISPLAY_HEIGHT_CM,
            )
            
            # Match rows with this step_id
            mask = self.df["slide_id"] == step_id
            
            for corner in range(4):
                x_idx = corner * 2
                y_idx = corner * 2 + 1
                self.df.loc[mask, f"target_x_table_srt_cm{corner}"] = origin[x_idx]
                self.df.loc[mask, f"target_y_table_srt_cm{corner}"] = origin[y_idx]
                self.df.loc[mask, f"target_x_table_dst_cm{corner}"] = dest[x_idx]
                self.df.loc[mask, f"target_y_table_dst_cm{corner}"] = dest[y_idx]
                self.df.loc[mask, f"target_x_screen_cm{corner}"] = screen_cm[x_idx]
                self.df.loc[mask, f"target_y_screen_cm{corner}"] = screen_cm[y_idx]
                self.df.loc[mask, f"target_x_screen_px{corner}"] = screen_cm[x_idx] * (EnvironmentConstants.DISPLAY_WIDTH_PX / EnvironmentConstants.DISPLAY_WIDTH_CM)
                self.df.loc[mask, f"target_y_screen_px{corner}"] = screen_cm[y_idx] * (EnvironmentConstants.DISPLAY_HEIGHT_PX / EnvironmentConstants.DISPLAY_HEIGHT_CM)
    
    def _compute_active_target_type(self):
        """Logic 2: Cognitive focus FSM (simplified)."""
        
        self.df["active_target_type"] = "BLINK"
        
        # Detect valid gazes
        table_valid = ~self.df["gaze_x_table_cm"].isna()
        screen_valid = ~self.df["gaze_x_screen_cm"].isna()

        slide_start_ts = self.df["slide_id"].map(self.slide_start_timestamp_by_slide).astype("float64")
        time_since_slide_change = self.df["timestamp"].astype("float64") - slide_start_ts
        screen_phase = screen_valid | time_since_slide_change.le(SCREEN_SWITCH_WINDOW_MS)
        
        # BLINK: both invalid
        blink_mask = ~table_valid & ~screen_valid
        self.df.loc[blink_mask, "active_target_type"] = "BLINK"
        
        # SCREEN: screen valid
        screen_mask = screen_phase & ~blink_mask
        self.df.loc[screen_mask, "active_target_type"] = "SCREEN"
        
        # SRT: table valid, no screen
        grasp_ts = self.df["slide_id"].map(self.grasp_timestamp_by_slide)
        dst_mask = table_valid & ~screen_mask & ~blink_mask & self.df["timestamp"].ge(grasp_ts.fillna(np.inf))
        srt_mask = table_valid & ~screen_mask & ~blink_mask & ~dst_mask
        self.df.loc[srt_mask, "active_target_type"] = "SRT"
        
        self.df.loc[dst_mask, "active_target_type"] = "DST"
        
        self.df["active_target_type"] = self.df["active_target_type"].astype("category")

    def _compute_is_success(self):
        """Broadcast slide-level success labels to all rows in each segment."""

        if self.df.empty:
            return

        self.df["is_success"] = self.df["slide_id"].map(self.success_by_slide).fillna(0).astype("int8")

    def _cast_dtypes(self):
        """Apply memory-friendly dtypes required by the spec."""

        if "timestamp" in self.df.columns:
            self.df["timestamp"] = self.df["timestamp"].astype("float64")

        for column in ["subject_id", "task", "event"]:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype("category")

        for column in ["slide_id", "step_id", "block_id", "is_success"]:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype("int16" if column != "is_success" else "int8")

        for column in [
            "table_pupil_timestamp",
            "screen_pupil_timestamp",
            "table_confidence_right",
            "table_confidence_left",
            "table_diameter_right",
            "table_diameter_left",
            "screen_confidence_right",
            "screen_confidence_left",
            "screen_diameter_right",
            "screen_diameter_left",
        ]:
            if column in self.df.columns:
                self.df[column] = self.df[column].astype("float32" if "confidence" in column or "diameter" in column else "float64")

        if "active_target_type" in self.df.columns:
            self.df["active_target_type"] = self.df["active_target_type"].astype("category")


# ============================================================================
# MAIN
# ============================================================================

def process_participant(subject_id: str, repo_root: Path, output_dir: Path) -> List[dict[str, str]]:
    """Process single participant across tasks and return export summaries."""

    participant_summaries: List[dict[str, str]] = []
    
    for task in TASK_NAMES:
        try:
            builder = MasterDataframeBuilder(subject_id, task, repo_root)
            df = builder.build()
            
            if len(df) == 0:
                logging.warning(f"[{subject_id}] {task}: Empty result")
                continue
            
            # Reorder columns to match exact user spec.
            base_cols = ["timestamp", "subject_id", "task", "slide_id", "step_id", "event", "block_id", "is_success", "active_target_type"]

            # Gaze columns in the exact order requested
            gaze_cols = []
            gaze_order = [
                "gaze_x_screen_rel", "gaze_y_screen_rel",
                "gaze_x_screen_cm", "gaze_y_screen_cm",
                "gaze_x_screen_px", "gaze_y_screen_px",
                "gaze_x_table_affine_cm", "gaze_y_table_affine_cm",
                "gaze_x_table_rel", "gaze_y_table_rel",
                "gaze_x_table_cm", "gaze_y_table_cm",
            ]
            for col in gaze_order:
                if col in df.columns:
                    gaze_cols.append(col)

            pupil_cols = []
            pupil_order = [
                "screen_pupil_timestamp",
                "screen_confidence_right",
                "screen_confidence_left",
                "screen_diameter_right",
                "screen_diameter_left",
                "table_pupil_timestamp",
                "table_confidence_right",
                "table_confidence_left",
                "table_diameter_right",
                "table_diameter_left",
            ]
            for col in pupil_order:
                if col in df.columns:
                    pupil_cols.append(col)

            # Target groups: each group lists corners 0..3 as x0,y0,x1,y1,...
            srt_cols = []
            dst_cols = []
            screen_cm_cols = []
            screen_px_cols = []
            for corner in range(4):
                x = f"target_x_table_srt_cm{corner}"
                y = f"target_y_table_srt_cm{corner}"
                if x in df.columns:
                    srt_cols.append(x)
                if y in df.columns:
                    srt_cols.append(y)

            for corner in range(4):
                x = f"target_x_table_dst_cm{corner}"
                y = f"target_y_table_dst_cm{corner}"
                if x in df.columns:
                    dst_cols.append(x)
                if y in df.columns:
                    dst_cols.append(y)

            for corner in range(4):
                x = f"target_x_screen_cm{corner}"
                y = f"target_y_screen_cm{corner}"
                if x in df.columns:
                    screen_cm_cols.append(x)
                if y in df.columns:
                    screen_cm_cols.append(y)

            for corner in range(4):
                x = f"target_x_screen_px{corner}"
                y = f"target_y_screen_px{corner}"
                if x in df.columns:
                    screen_px_cols.append(x)
                if y in df.columns:
                    screen_px_cols.append(y)

            target_group_cols = srt_cols + dst_cols + screen_cm_cols + screen_px_cols

            # Remaining columns preserve original order, excluding ones already used
            used = set(base_cols + gaze_cols + pupil_cols + target_group_cols)
            remaining = [col for col in df.columns if col not in used]

            df = df[base_cols + gaze_cols + pupil_cols + target_group_cols + remaining]
            
            # Save
            output_subdir = output_dir / subject_id
            output_subdir.mkdir(parents=True, exist_ok=True)
            output_file = output_subdir / f"{task}_master.csv"
            df.to_csv(output_file, index=False)

            failure_segments = extract_failure_segments(df)
            failure_output_file = None
            if not failure_segments.empty:
                failure_output_file = output_subdir / f"{task}_failed_segments.csv"
                failure_segments.to_csv(failure_output_file, index=False)
            
            logging.info(f"[{subject_id}] {task}: Saved to {output_file}")
            if failure_output_file is not None:
                logging.info(f"[{subject_id}] {task}: Saved failure segments to {failure_output_file}")
            else:
                logging.info(f"[{subject_id}] {task}: No failure segments to save")

            participant_summaries.append(
                build_failure_export_summary(
                    subject_id=subject_id,
                    task=task,
                    master_rows=len(df),
                    failure_segments=failure_segments,
                    failure_output_file=failure_output_file,
                )
            )
        
        except Exception as e:
            logging.error(f"[{subject_id}] {task}: Error - {e}", exc_info=True)

    return participant_summaries


def write_csv(csv_path: Path, rows: List[dict[str, str]], fieldnames: List[str]) -> None:
    """Write rows to CSV with stable headers."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="GAIPAT Master DataFrame Pipeline")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--subject-id", type=str, default=None, help="Single subject (optional)")
    
    args = parser.parse_args()
    
    repo_root = args.repo_root
    output_dir = args.output_dir or repo_root / "extract_unified_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "pipeline.log"),
            logging.StreamHandler(),
        ]
    )
    
    logging.info("="*60)
    logging.info("GAIPAT Master DataFrame Pipeline")
    logging.info(f"Repo: {repo_root}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*60)

    failure_export_rows: List[dict[str, str]] = []
    
    if args.subject_id:
        failure_export_rows.extend(process_participant(args.subject_id, repo_root, output_dir))
    else:
        participants_dir = repo_root / "participants"
        subject_ids = sorted([d.name for d in participants_dir.iterdir() if d.is_dir()])
        logging.info(f"Found {len(subject_ids)} subjects")
        
        for subject_id in subject_ids:
            failure_export_rows.extend(process_participant(subject_id, repo_root, output_dir))

    summary_file = output_dir / "failure_segments_summary.csv"
    total_master_rows = sum(int(row["master_rows"]) for row in failure_export_rows)
    total_failure_rows = sum(int(row["failure_rows"]) for row in failure_export_rows)
    total_failure_segments = sum(int(row["failure_segments"]) for row in failure_export_rows)
    total_exported = sum(int(row["failure_exported"]) for row in failure_export_rows)

    summary_rows = list(failure_export_rows)
    summary_rows.append(
        {
            "subject_id": "__TOTAL__",
            "task": "__TOTAL__",
            "master_rows": str(total_master_rows),
            "failure_rows": str(total_failure_rows),
            "failure_segments": str(total_failure_segments),
            "failure_exported": str(total_exported),
            "failure_file": "",
        }
    )

    write_csv(
        summary_file,
        summary_rows,
        [
            "subject_id",
            "task",
            "master_rows",
            "failure_rows",
            "failure_segments",
            "failure_exported",
            "failure_file",
        ],
    )

    logging.info(f"Saved failure summary to {summary_file}")
    
    logging.info("Pipeline complete!")


if __name__ == "__main__":
    main()

"""
python preprocess_script/01_extract_unified_data_pipeline.py --repo-root /root/autodl-tmp/shenxy/XDU/Dataset/gaipat
"""