# GAIPAT master data 切片脚本使用说明

## 脚本

`02_slice_dataframes.py` - 将 `processed` 目录下已有的 master dataframe 按连续的 `step_id + block_id` 片段切片，并导出为单独的 CSV 文件。

## 功能说明

该脚本会递归扫描输入目录下所有 `*_master.csv` 文件，然后按时间顺序切成连续片段，并进一步按 `event` 拆分为 `grasp` 和 `release` 两类片段。每个片段对应一个装配步骤内的一个事件阶段，文件名按照以下格式保存：

`[subject_id]_[task]_[step]_[event]_[block_id]_[label].csv`

其中 `event` 仅取值为：

- `grasp`
- `release`

其中 `label` 表示该片段是否成功：

- `1`：成功
- `0`：失败

注意：`step_id = 0` 表示实验开始时的起始状态，这一段积木已经由实验员事先放好，不属于参与者操作，因此不会导出切片文件。

输出文件内容只保留你要求的特征列，并保留 `event` 列，用于下游区分抓取阶段与放置阶段。

另外，脚本会在导出前设置最小片段长度阈值，长度小于阈值的事件片段会被直接丢弃。

## 默认输入输出

- 输入目录：`<repo-root>/processed`
- 输出目录：`<repo-root>/processed/slice_dataframes`

脚本会把所有切片直接平铺保存到输出目录下，不再创建子文件夹：

```text
processed/slice_dataframes/
  ├── 10383544_sc_12_grasp_3_1.csv
  ├── 10383544_tc_9_release_2_0.csv
  └── ...
```

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/02_slice_dataframes.py
```

### 2. 指定仓库根目录

```bash
python preprocess_script/02_slice_dataframes.py --repo-root D:\code\gaze_attn
```

### 3. 指定输入和输出目录

```bash
python preprocess_script/02_slice_dataframes.py \
    --repo-root D:\code\gaze_attn \
    --input-dir D:\code\gaze_attn\processed \
    --output-dir D:\code\gaze_attn\processed\slice_dataframes
```

## 导出字段

每个切片 CSV 固定包含以下列:

- `timestamp`
- `gaze_x_screen_rel`, `gaze_y_screen_rel`
- `gaze_x_screen_cm`, `gaze_y_screen_cm`
- `gaze_x_screen_px`, `gaze_y_screen_px`
- `gaze_x_table_affine_cm`, `gaze_y_table_affine_cm`
- `gaze_x_table_rel`, `gaze_y_table_rel`
- `gaze_x_table_cm`, `gaze_y_table_cm`
- `target_x_srt_cm0` ~ `target_y_srt_cm3`
- `target_x_dst_cm0` ~ `target_y_dst_cm3`
- `target_x_screen_cm0` ~ `target_y_screen_cm3`
- `target_x_screen_px0` ~ `target_y_screen_px3`

## 注意事项

- 脚本依赖已有的 master dataframe，所以需要先完成 `gaipat_master_dataframe_pipeline.py` 的处理。
- 如果某个 master 文件缺少必须的列，脚本会跳过该文件并在日志中提示。
- 如果某个切片的 `block_id < 0`，脚本会忽略该片段。
- 如果某个切片的 `step_id = 0`，脚本会忽略该片段。
- 如果某个事件片段长度小于 `--min-segment-length`，脚本会忽略该片段。默认值为 `5`。

## 日志

运行日志会写入：

`processed/slice_dataframes/slice_processed_master_dataframes.log`
