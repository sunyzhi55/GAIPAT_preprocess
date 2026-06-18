from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# 输出列定义（已去重，保留唯一值）
OUTPUT_COLUMNS = [
    "timestamp",
    "event",
    "gaze_x_screen_rel",
    "gaze_y_screen_rel",
    "gaze_x_screen_cm",
    "gaze_y_screen_cm",
    "gaze_x_screen_px",
    "gaze_y_screen_px",
    "screen_pupil_timestamp",
    "screen_confidence_right",
    "screen_confidence_left",
    "screen_diameter_right",
    "screen_diameter_left",
    "gaze_x_table_affine_cm",
    "gaze_y_table_affine_cm",
    "gaze_x_table_rel",
    "gaze_y_table_rel",
    "gaze_x_table_cm",
    "gaze_y_table_cm",
    "table_pupil_timestamp",
    "table_confidence_right",
    "table_confidence_left",
    "table_diameter_right",
    "table_diameter_left",
    "target_x_table_srt_cm0",
    "target_y_table_srt_cm0",
    "target_x_table_srt_cm1",
    "target_y_table_srt_cm1",
    "target_x_table_srt_cm2",
    "target_y_table_srt_cm2",
    "target_x_table_srt_cm3",
    "target_y_table_srt_cm3",
    "target_x_table_dst_cm0",
    "target_y_table_dst_cm0",
    "target_x_table_dst_cm1",
    "target_y_table_dst_cm1",
    "target_x_table_dst_cm2",
    "target_y_table_dst_cm2",
    "target_x_table_dst_cm3",
    "target_y_table_dst_cm3",
    "target_x_screen_cm0",
    "target_y_screen_cm0",
    "target_x_screen_cm1",
    "target_y_screen_cm1",
    "target_x_screen_cm2",
    "target_y_screen_cm2",
    "target_x_screen_cm3",
    "target_y_screen_cm3",
    "target_x_screen_px0",
    "target_y_screen_px0",
    "target_x_screen_px1",
    "target_y_screen_px1",
    "target_x_screen_px2",
    "target_y_screen_px2",
    "target_x_screen_px3",
    "target_y_screen_px3",
]

VALID_EVENT_VALUES = {"grasp", "release"}


def find_master_files(input_dir: Path) -> List[Path]:
    """查找目录下所有 master CSV 文件"""
    return sorted(path for path in input_dir.rglob("*_master.csv") if path.is_file())


def load_master_dataframe(master_path: Path) -> pd.DataFrame:
    """加载 CSV 数据为 DataFrame"""
    return pd.read_csv(master_path)


def iter_contiguous_segments(df: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """按 step_id + block_id 生成连续数据片段"""
    if df.empty:
        return

    ordered = df.sort_values(["timestamp", "step_id", "block_id"], kind="mergesort").reset_index(drop=True)
    segment_change = ordered[["step_id", "block_id"]].ne(ordered[["step_id", "block_id"]].shift()).any(axis=1)

    for _, segment in ordered.groupby(segment_change.cumsum(), sort=False):
        yield segment.reset_index(drop=True)


def infer_label(segment: pd.DataFrame) -> int:
    """从 is_success 推断标签"""
    if "is_success" not in segment.columns or segment.empty:
        return 0

    values = segment["is_success"].dropna().astype(int).unique()
    if not values.size:
        return 0
    if len(values) > 1:
        logging.warning("片段中存在混合 is_success 值，使用第一个有效值")
    return int(values[0])


def iter_event_segments(segment: pd.DataFrame) -> Iterable[pd.DataFrame]:
    """Split one step/block segment into contiguous grasp/release runs."""

    if segment.empty or "event" not in segment.columns:
        return

    ordered = segment.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    event_series = ordered["event"].fillna("").astype(str).str.strip().str.lower()
    valid_mask = event_series.isin(VALID_EVENT_VALUES)
    if not valid_mask.any():
        return

    event_group_ids = event_series.ne(event_series.shift()).cumsum()
    valid_ordered = ordered.loc[valid_mask].copy()
    valid_ordered["_event_group_id"] = event_group_ids.loc[valid_mask].to_numpy()

    for _, event_segment in valid_ordered.groupby("_event_group_id", sort=False):
        event_segment = event_segment.drop(columns=["_event_group_id"]).reset_index(drop=True)
        event_value = str(event_segment["event"].iloc[0]).strip().lower()
        if event_value not in VALID_EVENT_VALUES:
            continue
        yield event_segment


def filter_low_confidence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """删除 table 或 screen 两侧左右眼置信度同时低于 0.5 的行。"""

    required_confidence_columns = [
        "screen_confidence_right",
        "screen_confidence_left",
        "table_confidence_right",
        "table_confidence_left",
    ]
    if any(column not in df.columns for column in required_confidence_columns):
        return df

    screen_right = pd.to_numeric(df["screen_confidence_right"], errors="coerce")
    screen_left = pd.to_numeric(df["screen_confidence_left"], errors="coerce")
    table_right = pd.to_numeric(df["table_confidence_right"], errors="coerce")
    table_left = pd.to_numeric(df["table_confidence_left"], errors="coerce")

    screen_low_mask = screen_right.lt(0.5) & screen_left.lt(0.5)
    table_low_mask = table_right.lt(0.5) & table_left.lt(0.5)
    drop_mask = screen_low_mask | table_low_mask

    if drop_mask.any():
        dropped_rows = int(drop_mask.sum())
        logging.info("过滤低置信度行：%d 行", dropped_rows)
        return df.loc[~drop_mask].reset_index(drop=True)

    return df


def slice_master_file(master_path: Path, output_root: Path, min_segment_length: int) -> int:
    """切割单个 master 文件并输出分段 CSV"""
    df = load_master_dataframe(master_path)
    if df.empty:
        logging.info("跳过空文件：%s", master_path)
        return 0

    df = filter_low_confidence_rows(df)
    if df.empty:
        logging.info("过滤后为空文件：%s", master_path)
        return 0

    # 必需列（已合并去重，直接引用 OUTPUT_COLUMNS 避免重复）
    required_columns = ["timestamp", "step_id", "block_id", "is_success", "subject_id", "task", *OUTPUT_COLUMNS]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.warning("跳过 %s，缺失列：%s", master_path, ", ".join(missing_columns))
        return 0

    # 创建输出目录
    subject_id = str(df["subject_id"].iloc[0])
    task = str(df["task"].iloc[0])
    output_root.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for segment in iter_contiguous_segments(df):
        if segment.empty:
            continue

        step_id = int(segment["step_id"].iloc[0])
        if step_id == 0:
            continue

        block_id = int(segment["block_id"].iloc[0])
        if block_id < 0:
            continue

        for event_segment in iter_event_segments(segment):
            if len(event_segment) < min_segment_length:
                logging.info(
                    "跳过过短片段：%s step=%d block=%d event=%s length=%d",
                    master_path,
                    step_id,
                    block_id,
                    str(event_segment["event"].iloc[0]),
                    len(event_segment),
                )
                continue

            event_value = str(event_segment["event"].iloc[0]).strip().lower()
            label = infer_label(event_segment)
            output_frame = event_segment.reindex(columns=OUTPUT_COLUMNS)

            output_name = build_slice_file_name(subject_id, task, step_id, event_value, block_id, label)
            output_frame.to_csv(output_root / output_name, index=False)
            exported_count += 1

    logging.info("%s -> 导出 %d 个片段", master_path, exported_count)
    return exported_count


def build_slice_file_name(subject_id: str, task: str, step_id: int, event_value: str, block_id: int, label: int) -> str:
    """Build the new slice file name with event preserved."""

    return f"{subject_id}_{task}_{step_id}_{event_value}_{block_id}_{label}.csv"


def build_argument_parser() -> argparse.ArgumentParser:
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(description="将眼动数据按 step/block/event 切割为分段 CSV")
    parser.add_argument("--repo-root", type=Path, default='/root/autodl-tmp/shenxy/XDU/Dataset/gaipat-main', help="项目根目录")
    parser.add_argument("--input-dir", type=Path, help="输入目录，默认 <repo-root>/extract_unified_data")
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认 <repo-root>/slice_dataframes")
    parser.add_argument("--min-segment-length", type=int, default=5, help="切片最小行数，小于该值的片段将被丢弃")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    repo_root = args.repo_root
    input_dir = args.input_dir or repo_root / "extract_unified_data"
    output_dir = args.output_dir or repo_root / "slice_dataframes"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(output_dir / "slice.log"), logging.StreamHandler()],
    )

    # 执行处理
    master_files = find_master_files(input_dir)
    if not master_files:
        logging.warning("未找到任何 master 文件：%s", input_dir)
        return

    total_slices = sum(slice_master_file(path, output_dir, args.min_segment_length) for path in master_files)
    logging.info("处理完成，总计导出 %d 个片段", total_slices)


if __name__ == "__main__":
    main()