# GAIPAT 第三步：偏差序列计算脚本使用说明

## 脚本

`03_compute_block_deviation_sequences.py` 用于读取第二步生成的切片 CSV，计算注视点到源目标 / 目标目标四边形的最短物理距离，并导出 JSONL 格式的偏差序列。

## 功能说明

脚本会递归扫描输入目录中的切片 CSV 文件，默认只处理文件名标签为 `1` 的成功切片。

对每个切片，脚本会：

- 找出 `gaze_x_table_affine_cm` 和 `gaze_y_table_affine_cm` 同时有效的行；
- 基于 `target_x_table_srt_cm0~3` / `target_y_table_srt_cm0~3` 构造源目标四边形；
- 基于 `target_x_table_dst_cm0~3` / `target_y_table_dst_cm0~3` 构造目标目标四边形；
- 使用 `Shapely` 计算注视点到两个四边形的最短距离；
- 生成：
  - `deviation_srt_cm`：到源目标的距离；
  - `deviation_dst_cm`：到目标目标的距离；
  - `deviation_cm`：`event == grasp` 时取 `deviation_srt_cm`，`event == release` 时取 `deviation_dst_cm`；
- 将左右眼相关特征合并为数组字段；
- 将 gaze 坐标和 source/destination 四边形坐标合并为数组字段；
- 对于已经合并过的特征，不再把原始单列重复写入 JSON。

## 默认输入输出

- 输入目录：`<repo-root>/slice_dataframes`
- 输出目录：`<repo-root>/block_deviation_sequences`

输出文件名格式：

```text
[subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
```

## JSONL 输出格式

JSONL 文件中每一行都是一个 JSON 对象，对应原来 CSV 中的一行有效特征。

### 保留的标量字段

- `timestamp`
- `event`
- `deviation_srt_cm`
- `deviation_dst_cm`
- `deviation_cm`

### 合并后的数组字段

- `screen_confidence_left_right`

```json
[screen_confidence_left, screen_confidence_right]
```

- `screen_diameter_left_right`

```json
[screen_diameter_left, screen_diameter_right]
```

- `table_confidence_left_right`

```json
[table_confidence_left, table_confidence_right]
```

- `table_diameter_left_right`

```json
[table_diameter_left, table_diameter_right]
```

- `gaze_x_y_table_affine_cm`

```json
[gaze_x_table_affine_cm, gaze_y_table_affine_cm]
```

- `target_x_y_table_srt_cm`

```json
[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
```

- `target_x_y_table_dst_cm`

```json
[[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
```

## 运行方式

### 1. 使用默认路径

```bash
python preprocess_script/03_compute_block_deviation_sequences.py
```

### 2. 同时处理失败切片

```bash
python preprocess_script/03_compute_block_deviation_sequences.py --include-failures
```

### 3. 指定仓库根目录

```bash
python preprocess_script/03_compute_block_deviation_sequences.py --repo-root D:\code\gaze_attn
```

### 4. 指定输入和输出目录

```bash
python preprocess_script/03_compute_block_deviation_sequences.py \
    --repo-root D:\code\gaze_attn \
    --input-dir D:\code\gaze_attn\slice_dataframes \
    --output-dir D:\code\gaze_attn\block_deviation_sequences
```

## 额外参数

- `--include-failures`：同时处理标签为 `0` 的失败切片。默认只处理成功切片。
- `--min-output-length`：偏差序列最小长度阈值，默认值为 `5`。小于该值的结果不会导出。

## 依赖

```bash
pip install shapely
```

## 日志

运行日志写入：

`block_deviation_sequences/deviation_sequences.log`
