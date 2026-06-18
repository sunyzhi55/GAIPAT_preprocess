# GAIPAT 第五步：JSONL 偏差序列定长化说明

## 脚本

`05_normalize_deviation_jsonl_sequences.py` 接在第三步 `03_compute_block_deviation_sequences.py` 之后运行。它读取第三步导出的 JSONL 偏差序列，把每个变长切片重采样为固定长度 `L`，并继续以 JSONL 格式保存。

与旧版 `compute_mean_deviation_sequence.py` 的主要区别：

- 输入是 `.jsonl`，不是 `.csv`。
- 输出仍是 `.jsonl`。
- 每个原始 JSONL 的字段都会尽量保留。
- 可以线性插值的数值字段会插值；不适合插值的字段会重复常量或写为 `null`。

## 默认输入输出

默认输入目录：

```text
<repo-root>/block_deviation_sequences
```

默认输出目录：

```text
<repo-root>/normalized_deviation_jsonl_sequences
```

输出结构：

```text
normalized_deviation_jsonl_sequences/
  arithmetic_mean/
    normalized_sequences/
      [subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
    mean_deviation_sequence.jsonl
    length_statistics.csv
    processing_summary.csv
    field_interpolation_summary.csv
  dba_mean/
    normalized_sequences/
      [subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
    mean_deviation_sequence.jsonl
    length_statistics.csv
    processing_summary.csv
    field_interpolation_summary.csv
  normalize_jsonl_sequences.log
```

## 输入文件格式

输入文件名必须符合第三步格式：

```text
[subject_id]_[task]_[step]_[event]_[block_id]_[label].jsonl
```

第三步 JSONL 典型字段包括：

- `timestamp`
- `event`
- `deviation_srt_cm`
- `deviation_dst_cm`
- `deviation_cm`
- `screen_confidence_left_right`
- `screen_diameter_left_right`
- `table_confidence_left_right`
- `table_diameter_left_right`
- `gaze_x_y_table_affine_cm`
- `target_x_y_table_srt_cm`
- `target_x_y_table_dst_cm`

## 字段处理规则

每个输出 JSONL 文件会新增：

- `normalized_progress`：归一化进度，范围为 `[0, 1]`。
- `sequence_id`
- `subject_id`
- `task`
- `step_id`
- `block_id`
- `label`

原始字段按以下规则处理：

### 1. 数值标量字段：线性插值

例如：

- `timestamp`
- `deviation_srt_cm`
- `deviation_dst_cm`
- `deviation_cm`

### 2. 固定形状数值数组字段：逐元素线性插值

例如：

- `screen_confidence_left_right`
- `screen_diameter_left_right`
- `table_confidence_left_right`
- `table_diameter_left_right`
- `gaze_x_y_table_affine_cm`

### 3. 几何目标字段：只保留常量，不插值

例如：

- `target_x_y_table_srt_cm`
- `target_x_y_table_dst_cm`

如果这些字段在同一个切片内始终一致，输出中会重复该常量。如果同一个切片内出现不一致，则输出为 `null`，避免插出没有物理意义的目标多边形。

### 4. 类别字段：不插值

例如：

- `event`

如果类别值在同一个切片内始终一致，输出中会重复该常量。如果不一致，则输出为 `null`。

### 5. 其他未知字段：自动判断

- 纯数值标量：线性插值。
- 固定形状数值数组：逐元素线性插值。
- 全程常量：重复常量。
- 混合类型、字符串变化、数组形状变化：输出 `null`。

每个方法目录下的 `field_interpolation_summary.csv` 会记录每个文件每个字段的处理策略和原因。

## 长度 L 的选择

### arithmetic_mean

`arithmetic_mean` 方法使用 progress-based linear resampling。默认用成功样本长度的中位数并四舍五入到 10 的倍数作为 `L`，最小为 20。

可用参数覆盖：

```bash
--target-length 100
--arithmetic-target-length 100
```

### dba_mean

`dba_mean` 方法使用 `tslearn.barycenters.dtw_barycenter_averaging` 计算 DBA 模板。默认由 DBA 自动确定长度，也可以手动覆盖：

```bash
--dba-target-length 120
```

注意：DBA 只用于计算平均模板和确定 DBA 方法目录下的输出长度；单个 JSONL 文件中所有字段的定长化仍使用 progress-based interpolation。这样可以保留原始字段结构，并避免对非距离字段使用 DTW 对齐造成语义混乱。

## 运行方式

### 1. 运行全部方法

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py
```

### 2. 只运行 arithmetic 方法

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py --method arithmetic_mean
```

### 3. 指定输入输出目录

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py \
    --repo-root /root/autodl-tmp/shenxy/Data/gaipat \
    --input-dir /root/autodl-tmp/shenxy/Data/gaipat/block_deviation_sequences \
    --output-dir /root/autodl-tmp/shenxy/Data/gaipat/normalized_deviation_jsonl_sequences
```

### 4. 手动指定固定长度

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py \
    --method arithmetic_mean \
    --arithmetic-target-length 100
```

### 5. 运行 DBA 方法

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py \
    --method dba_mean \
    --dba-target-length 120
```

## 依赖

基础定长化需要：

```bash
pip install numpy pandas
```

如果运行 `--method dba_mean` 或默认 `--method all`，还需要：

```bash
pip install tslearn
```

如果服务器暂时没有 `tslearn`，建议先运行：

```bash
python preprocess_script/05_normalize_deviation_jsonl_sequences.py --method arithmetic_mean
```

## 输出 JSONL 单行示例

```json
{
  "normalized_progress": 0.0,
  "sequence_id": "10383544_sc_1_grasp_7_1",
  "subject_id": "10383544",
  "task": "sc",
  "step_id": 1,
  "block_id": 7,
  "label": 1,
  "timestamp": 123456.0,
  "event": "grasp",
  "deviation_srt_cm": 3.2,
  "deviation_dst_cm": 12.1,
  "deviation_cm": 3.2,
  "screen_confidence_left_right": [0.8, 0.82],
  "gaze_x_y_table_affine_cm": [20.0, 11.0],
  "target_x_y_table_srt_cm": [[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]],
  "target_x_y_table_dst_cm": [[5.0, 6.0], [7.0, 6.0], [7.0, 8.0], [5.0, 8.0]]
}
```

## 使用建议

后续模型如果要求每条时序长度一致，建议优先使用：

```text
normalized_deviation_jsonl_sequences/arithmetic_mean/normalized_sequences
```

这个目录的每个 JSONL 文件长度一致，字段完整，且插值规则最容易解释。
