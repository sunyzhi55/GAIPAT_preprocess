"""
Visualize GAIPAT task instructions and participant gazepoints on the table.
Exports the visualization as a video.

Usage:
    python visualize_task_gazepoints.py --task car --subject-id 69907732 --output output.mp4
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import Image
import io


@dataclass(frozen=False)
class BlockPosition:
    """Block position with corners and level."""
    corners: Tuple[float, float, float, float, float, float, float, float]
    level: int


def read_csv_rows(csv_path: Path) -> List[dict[str, str]]:
    """Read CSV file and return list of row dictionaries."""
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: str) -> float:
    """Parse string to float, handling NaN."""
    if value is None:
        return float("nan")
    text = value.strip()
    if not text or text.lower() == "nan":
        return float("nan")
    return float(text)


def parse_int(value: str) -> int:
    """Parse string to int, handling NaN."""
    if value is None:
        return 0
    text = value.strip()
    if not text or text.lower() == "nan":
        return 0
    return int(float(text))


def load_instructions(setup_dir: Path, task_name: str) -> Dict[int, Tuple[BlockPosition, BlockPosition]]:
    """
    Load instruction file and return dict mapping block_id to (origin, destination).
    Only includes steps 1+, skipping experimenter step 0.
    """
    instruction_path = setup_dir / f"instructions_{task_name}.csv"
    rows = read_csv_rows(instruction_path)
    
    block_positions: Dict[int, Tuple[BlockPosition, BlockPosition]] = {}
    
    for row in rows:
        step_id = parse_int(row["id"])
        if step_id == 0:
            continue
            
        block_id = parse_int(row["block"])
        
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
        origin_level = parse_int(row["origin_level"])
        
        dest_corners = (
            parse_float(row["destin_x0"]),
            parse_float(row["destin_y0"]),
            parse_float(row["destin_x1"]),
            parse_float(row["destin_y1"]),
            parse_float(row["destin_x2"]),
            parse_float(row["destin_y2"]),
            parse_float(row["destin_x3"]),
            parse_float(row["destin_y3"]),
        )
        dest_level = parse_int(row["destin_level"])
        
        block_positions[block_id] = (
            BlockPosition(corners=origin_corners, level=origin_level),
            BlockPosition(corners=dest_corners, level=dest_level),
        )
    
    return block_positions


def load_gazepoints(participant_dir: Path, task_name: str) -> List[Tuple[int, float, float]]:
    """
    Load gazepoints from table.
    Returns list of (timestamp, x, y) tuples, filtering out NaN values.
    """
    gazepoints_path = participant_dir / task_name / "table" / "gazepoints.csv"
    rows = read_csv_rows(gazepoints_path)
    
    gazepoints: List[Tuple[int, float, float]] = []
    for row in rows:
        timestamp = parse_int(row["timestamp"])
        x = parse_float(row["x"])
        y = parse_float(row["y"])
        
        # Skip if x or y is NaN
        if np.isnan(x) or np.isnan(y):
            continue
            
        gazepoints.append((timestamp, x, y))
    
    return gazepoints


def draw_quad(ax, corners: Tuple[float, ...], color: str, linestyle: str = "-", linewidth: float = 1.5, label: str = "") -> None:
    """
    Draw a quadrilateral given 4 corner points (x0, y0, x1, y1, x2, y2, x3, y3).
    Corners are: top-left, top-right, bottom-right, bottom-left.
    """
    x_coords = [corners[0], corners[2], corners[4], corners[6], corners[0]]
    y_coords = [corners[1], corners[3], corners[5], corners[7], corners[1]]
    
    ax.plot(x_coords, y_coords, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def get_block_colors(block_ids: List[int]) -> Dict[int, str]:
    """
    Assign a unique color to each block ID.
    Uses a predefined color palette.
    """
    # Extended color palette
    colors = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#FFA07A",  # Light Salmon
        "#98D8C8",  # Mint
        "#F7DC6F",  # Yellow
        "#BB8FCE",  # Purple
        "#85C1E2",  # Sky Blue
        "#F8B88B",  # Peach
        "#A2D5C6",  # Green-blue
        "#E6B89C",  # Tan
        "#90CAF9",  # Light Blue
        "#FFCC99",  # Light Orange
        "#B5EAD7",  # Seafoam
        "#FF9999",  # Light Red
        "#FFD700",  # Gold
        "#99CCFF",  # Periwinkle
        "#FFB3BA",  # Pink
        "#BAFFC9",  # Mint Green
        "#FFD8B3",  # Peach
    ]
    
    block_color_map = {}
    sorted_ids = sorted(block_ids)
    for idx, block_id in enumerate(sorted_ids):
        block_color_map[block_id] = colors[idx % len(colors)]
    
    return block_color_map


def render_frame(block_positions: Dict[int, Tuple[BlockPosition, BlockPosition]], 
                 gazepoints: List[Tuple[int, float, float]],
                 current_time: int,
                 task_name: str,
                 subject_id: str,
                 figsize: Tuple[int, int] = (10, 10)) -> np.ndarray:
    """
    Render a single frame as numpy array.
    
    Args:
        block_positions: Dict mapping block_id to (origin, destination) BlockPosition
        gazepoints: List of (timestamp, x, y) tuples
        current_time: Current timestamp in milliseconds
        task_name: Task name
        subject_id: Subject ID
        figsize: Figure size
    
    Returns:
        Frame as numpy array (H, W, 3) in BGR format
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("X (relative)", fontsize=10)
    ax.set_ylabel("Y (relative)", fontsize=10)
    ax.set_title(f"Task: {task_name}, Subject: {subject_id}", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Get color mapping for each block
    block_colors = get_block_colors(list(block_positions.keys()))
    
    # Draw blocks (origins and destinations)
    # First pass: draw origins (solid line)
    for block_id, (origin, dest) in sorted(block_positions.items()):
        color = block_colors[block_id]
        draw_quad(ax, origin.corners, color, linestyle="-", linewidth=2.0, 
                 label=f"Block {block_id} (solid=start, dash=target)")
    
    # Second pass: draw destinations (dashed line) without adding to legend
    for block_id, (origin, dest) in sorted(block_positions.items()):
        color = block_colors[block_id]
        draw_quad(ax, dest.corners, color, linestyle="--", linewidth=2.0)
    
    # Draw gazepoints up to current time
    gaze_color = "#00AA00"  # Green
    current_gazes = [(x, y) for t, x, y in gazepoints if t <= current_time]
    
    if current_gazes:
        gaze_x, gaze_y = zip(*current_gazes)
        ax.scatter(gaze_x, gaze_y, c=gaze_color, s=25, alpha=0.7, label="Gaze points", edgecolors="darkgreen", linewidth=0.5)
    
    # Create custom legend
    legend_elements = []
    
    # Add block entries to legend (only show first few to avoid clutter)
    max_blocks_in_legend = min(10, len(block_positions))
    for i, (block_id, color) in enumerate(sorted(block_colors.items())[:max_blocks_in_legend]):
        legend_elements.append(patches.Patch(facecolor=color, edgecolor="black", label=f"Block {block_id}"))
    
    if len(block_positions) > max_blocks_in_legend:
        legend_elements.append(patches.Patch(facecolor="white", label=f"... and {len(block_positions) - max_blocks_in_legend} more blocks"))
    
    # Add style indicators
    legend_elements.append(patches.Patch(facecolor="white", edgecolor="black", linewidth=2, label="─ = Start position"))
    legend_elements.append(patches.Patch(facecolor="white", edgecolor="black", linewidth=2, linestyle="--", label="─ ─ = Target position"))
    
    if current_gazes:
        legend_elements.append(patches.Patch(facecolor=gaze_color, edgecolor="darkgreen", label="Gaze points"))
    
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=1)
    
    # Add timestamp text
    ax.text(0.98, 0.02, f"Time: {current_time / 1000.0:.2f}s\n{len(current_gazes)} gazes", 
            transform=ax.transAxes, verticalalignment="bottom", horizontalalignment="right", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    
    # Render to numpy array
    fig.canvas.draw()
    image_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_rgb = image_rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert RGBA to BGR (drop alpha channel)
    image = cv2.cvtColor(image_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    
    plt.close(fig)
    
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GAIPAT task instructions and participant gazepoints."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["car", "tb", "house", "sc", "tc", "tsb"],
        help="Task name.",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        required=True,
        help="Participant subject ID.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output video file path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video output.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Maximum video duration in seconds. If None, use full gazepoint duration.",
    )
    
    args = parser.parse_args()
    
    repo_root = args.repo_root
    task_name = args.task
    subject_id = args.subject_id
    output_path = args.output
    fps = args.fps
    max_duration = args.duration
    
    setup_dir = repo_root / "setup"
    participant_dir = repo_root / "participants" / subject_id
    
    # Load instructions and gazepoints
    block_positions = load_instructions(setup_dir, task_name)
    gazepoints = load_gazepoints(participant_dir, task_name)
    
    if not gazepoints:
        print(f"No gazepoints found for subject {subject_id} task {task_name}")
        return
    
    if not block_positions:
        print(f"No instructions found for task {task_name}")
        return
    
    # Determine time range
    start_time = gazepoints[0][0]
    end_time = gazepoints[-1][0]
    
    if max_duration:
        end_time = min(end_time, start_time + int(max_duration * 1000))
    
    total_frames = int((end_time - start_time) / 1000.0 * fps) + 1
    
    # Render frames and write video
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get first frame to determine dimensions
    first_frame = render_frame(block_positions, gazepoints, start_time, task_name, subject_id)
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        # Fallback to MJPG codec if mp4v fails
        print("mp4v codec not available, using MJPG fallback...")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Writing {total_frames} frames to video...")
    print(f"  Output: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Duration: {total_frames / fps:.2f}s")
    
    for frame_idx in range(total_frames):
        if frame_idx % max(1, total_frames // 10) == 0:
            print(f"  Frame {frame_idx}/{total_frames}")
        
        current_time = start_time + (frame_idx / fps) * 1000.0
        frame = render_frame(block_positions, gazepoints, int(current_time), task_name, subject_id)
        writer.write(frame)
    
    writer.release()
    
    print(f"Done! Video saved to {output_path}")


if __name__ == "__main__":
    main()
